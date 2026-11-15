"""Maintaining user sessions from an updatable stream of access events.

The module implements an incremental *loading* of session reports.  Raw access
events are stored with their ``event_id``, ``source_version`` and a ``deleted``
flag.  A batch of changes is applied to that event store and the affected
sessions are recomputed and republished in the same database transaction.

Events that are backfilled, retracted or that change their event time can make
existing sessions grow, shrink, merge or split.  Instead of rebuilding all
sessions, only the sessions around the changed event times (plus their direct
neighbours) are recomputed.  Superseded sessions are physically withdrawn from
the published session table, so the table always contains exactly the sessions
that a full recomputation over the current event store would produce.

Main guarantees
----------------
* A change is applied only when its ``source_version`` is greater than the
  stored version.  Duplicates and stale (out of order) versions are ignored and
  do not cause any recomputation.
* Every batch is applied in one transaction.  Either all event changes and all
  retracted/published sessions become visible together, or none of them do.
* After every batch the published sessions equal a full recomputation from the
  current events, using the same user and idle-threshold semantics.
* Only the sessions around the changed event times are recomputed.  The
  recomputation window is extended by one neighbouring session on each side so
  that merges and splits at the window boundaries are captured.

The idle threshold defines a session as follows: events of one user are ordered
by event time and two consecutive events belong to the same session while the
distance between them is less than or equal to the idle threshold.  A larger
gap starts a new session.

Example
-------
::

    from datetime import timedelta
    from mara_pipelines.sessionization import EventChange, SessionLoader

    loader = SessionLoader(idle_threshold=timedelta(minutes=30))
    loader.load([
        EventChange(event_id='e1', source_version=1, user_id='alice',
                    event_time='2024-01-01T10:00:00Z', amount='9.99'),
        EventChange(event_id='e2', source_version=1, user_id='alice',
                    event_time='2024-01-01T10:05:00Z', amount='4.50'),
    ])

    # e2 was reported at the wrong time; the next batch retracts the old
    # session and publishes the corrected one.
    loader.load([
        EventChange(event_id='e2', source_version=2, user_id='alice',
                    event_time='2024-01-01T11:30:00Z', amount='4.50'),
    ])
"""

import collections.abc
import datetime
import decimal
import numbers
import re
import typing

import sqlalchemy
from sqlalchemy.orm import declarative_base

import mara_db.sqlalchemy_engine

from . import config

__all__ = [
    'Base',
    'SessionEvent',
    'UserSession',
    'SessionRetraction',
    'EventChange',
    'SessionData',
    'LoadResult',
    'SessionLoader',
    'create_tables',
    'load',
    'load_sessions',
    'apply_changes',
    'maintain_sessions',
    'recompute_sessions',
    'LoadSessions',
]

Base = declarative_base()

_UTC = datetime.timezone.utc
_ZERO = decimal.Decimal(0)


class SessionEvent(Base):
    """The current version of a raw access event.

    Rows are never physically removed.  A retracted event keeps its row with
    ``deleted`` set, which allows the loader to recognise and ignore a stale
    change that arrives after the retraction.
    """
    __tablename__ = 'data_integration_session_event'

    event_id = sqlalchemy.Column(sqlalchemy.Text, primary_key=True)
    user_id = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
    event_time = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=True)
    amount = sqlalchemy.Column(sqlalchemy.Numeric(38, 10), nullable=False, default=0)
    source_version = sqlalchemy.Column(sqlalchemy.BigInteger, nullable=False, default=0)
    deleted = sqlalchemy.Column(sqlalchemy.BOOLEAN, nullable=False, default=False)
    updated_at = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (
        sqlalchemy.Index('ix_data_integration_session_event_user_time', 'user_id', 'event_time'),
    )


class UserSession(Base):
    """A published session of one user.

    Only the current sessions are stored here.  When a change makes a session
    obsolete it is deleted (and recorded in :class:`SessionRetraction`) and the
    corrected session is inserted.
    """
    __tablename__ = 'data_integration_user_session'

    session_id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    user_id = sqlalchemy.Column(sqlalchemy.Text, nullable=False)
    started_at = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=False)
    ended_at = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=False)
    event_count = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    amount = sqlalchemy.Column(sqlalchemy.Numeric(38, 10), nullable=False, default=0)

    __table_args__ = (
        sqlalchemy.UniqueConstraint('user_id', 'started_at', 'ended_at',
                                    name='uq_data_integration_user_session_range'),
        sqlalchemy.Index('ix_data_integration_user_session_user_start', 'user_id', 'started_at'),
        sqlalchemy.Index('ix_data_integration_user_session_user_end', 'user_id', 'ended_at'),
    )


class SessionRetraction(Base):
    """An audit record of a published session that has been withdrawn.

    The session rows themselves are removed from :class:`UserSession`; this
    table keeps track of what was retracted, which is useful for monitoring and
    for consumers that need to propagate retractions downstream.
    """
    __tablename__ = 'data_integration_user_session_retraction'

    retraction_id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    user_id = sqlalchemy.Column(sqlalchemy.Text, nullable=False)
    started_at = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=False)
    ended_at = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=False)
    event_count = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    amount = sqlalchemy.Column(sqlalchemy.Numeric(38, 10), nullable=False, default=0)
    load_id = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
    retracted_at = sqlalchemy.Column(sqlalchemy.TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (
        sqlalchemy.Index('ix_data_integration_user_session_retraction_user', 'user_id'),
    )


class EventChange(typing.NamedTuple):
    """A change of a single raw access event.

    Args:
        event_id: The stable id of the event.  Other values are converted to
            their string form.
        source_version: The version of the change in its source system.  Only
            versions strictly greater than the stored version are applied.
        deleted: When true the event is retracted from its sessions.  The event
            time and user of an already known event are taken from the event
            store, so they may be omitted.
        user_id: The user (session key) of the event.
        event_time: The event time.  ``datetime`` values, ISO-8601 strings and
            unix timestamps in seconds are accepted.
        amount: The event amount.  ``Decimal``, ``int``, ``float`` and numeric
            strings are accepted.

    Besides constructing this tuple directly, :meth:`from_any` accepts
    dictionaries and arbitrary objects.
    """
    event_id: typing.Any
    source_version: typing.Any = 0
    deleted: bool = False
    user_id: typing.Any = None
    event_time: typing.Any = None
    amount: typing.Any = None

    @classmethod
    def from_any(cls, change: typing.Any) -> 'EventChange':
        """Coerces a mapping or object to an :class:`EventChange`."""
        if isinstance(change, cls):
            return change

        if isinstance(change, collections.abc.Mapping):
            def get(*names):
                for name in names:
                    if name in change:
                        return change[name]
                return None
        else:
            def get(*names):
                for name in names:
                    if hasattr(change, name):
                        return getattr(change, name)
                return None

        event_id = get('event_id', 'eventId', 'id')
        if event_id is None:
            raise ValueError(f'Event change without an event id: {change!r}')

        source_version = get('source_version', 'sourceVersion', 'version', 'revision')
        if source_version is None:
            source_version = 0

        deleted = get('deleted', 'is_deleted', 'isDeleted', 'delete', 'deleted_flag')
        if deleted is None:
            deleted = False

        return cls(event_id=event_id,
                   source_version=source_version,
                   deleted=bool(deleted),
                   user_id=get('user_id', 'userId', 'user', 'user_key'),
                   event_time=get('event_time', 'eventTime', 'timestamp', 'time', 'ts'),
                   amount=get('amount', 'value', 'revenue'))


class SessionData(typing.NamedTuple):
    """The immutable content of a session (without a database id)."""
    user_id: str
    started_at: datetime.datetime
    ended_at: datetime.datetime
    event_count: int
    amount: decimal.Decimal


class LoadResult(typing.NamedTuple):
    """Statistics of one applied batch.

    Args:
        changes: Number of changes passed to the loader.
        applied: Number of changes whose source version was newer.
        ignored: Number of duplicate or stale changes that were ignored.
        retracted_sessions: Number of published sessions that were withdrawn.
        published_sessions: Number of new sessions that were published.
        updated_sessions: Number of existing sessions whose event count or
            amount changed while their boundaries stayed the same.
        affected_users: The users whose sessions were recomputed.
        recomputed_ranges: The event-time ranges that were recomputed, by user.
        succeeded: Always true for a successfully committed batch.
    """
    changes: int
    applied: int
    ignored: int
    retracted_sessions: int
    published_sessions: int
    updated_sessions: int
    affected_users: typing.Tuple[str, ...]
    recomputed_ranges: typing.Dict[str, typing.Tuple[typing.Tuple[datetime.datetime, datetime.datetime], ...]]
    succeeded: bool = True

    def __bool__(self) -> bool:
        return bool(self.succeeded)

    @property
    def ignored_changes(self) -> int:
        """Alias for :attr:`ignored`."""
        return self.ignored


# ---------------------------------------------------------------------------
# value normalization
# ---------------------------------------------------------------------------

_DURATION_PATTERN = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$')
_DURATION_UNITS = {
    '': 1,
    's': 1, 'sec': 1, 'secs': 1, 'second': 1, 'seconds': 1,
    'm': 60, 'min': 60, 'mins': 60, 'minute': 60, 'minutes': 60,
    'h': 3600, 'hr': 3600, 'hrs': 3600, 'hour': 3600, 'hours': 3600,
    'd': 86400, 'day': 86400, 'days': 86400,
}


def _as_timedelta(value: typing.Any) -> datetime.timedelta:
    """Converts a number of seconds, a string or a timedelta to timedelta."""
    if isinstance(value, datetime.timedelta):
        return value
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return datetime.timedelta(seconds=float(value))
    if isinstance(value, str):
        match = _DURATION_PATTERN.match(value)
        if match:
            unit = match.group(2).lower()
            if unit in _DURATION_UNITS:
                return datetime.timedelta(seconds=float(match.group(1)) * _DURATION_UNITS[unit])
    raise ValueError(f'Cannot interpret "{value!r}" as a duration')


def _as_datetime(value: typing.Any) -> datetime.datetime:
    """Converts an event time to an aware UTC datetime."""
    if isinstance(value, datetime.datetime):
        result = value
    elif isinstance(value, datetime.date):
        result = datetime.datetime(value.year, value.month, value.day, tzinfo=_UTC)
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        result = datetime.datetime.fromtimestamp(float(value), tz=_UTC)
    elif isinstance(value, str):
        text = value.strip()
        try:
            result = datetime.datetime.fromisoformat(text.replace('Z', '+00:00'))
        except ValueError:
            try:
                from dateutil import parser  # python-dateutil is a dependency
                result = parser.parse(text)
            except Exception as exception:  # pragma: no cover - defensive
                raise ValueError(f'Cannot interpret "{value!r}" as an event time') from exception
    else:
        raise TypeError(f'Cannot interpret "{value!r}" as an event time')

    if result.tzinfo is None:
        return result.replace(tzinfo=_UTC)
    return result.astimezone(_UTC)


def _as_amount(value: typing.Any) -> decimal.Decimal:
    """Converts an amount to an exact Decimal."""
    if value is None:
        return _ZERO
    if isinstance(value, decimal.Decimal):
        return value
    if isinstance(value, bool):
        return decimal.Decimal(int(value))
    if isinstance(value, numbers.Integral):
        return decimal.Decimal(int(value))
    if isinstance(value, numbers.Real):
        # str() keeps the decimal representation that the caller wrote down
        # instead of the binary floating point approximation.
        return decimal.Decimal(str(value))
    if isinstance(value, str):
        return decimal.Decimal(value.strip())
    raise TypeError(f'Cannot interpret "{value!r}" as an amount')


def _as_version(value: typing.Any) -> int:
    """Converts a source version to an integer."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, decimal.Decimal):
        if value != value.to_integral_value():
            raise ValueError(f'Source version must be an integer, got "{value!r}"')
        return int(value)
    if isinstance(value, numbers.Real):
        if float(value) != int(value):
            raise ValueError(f'Source version must be an integer, got "{value!r}"')
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise TypeError(f'Cannot interpret "{value!r}" as a source version')


def _as_event_id(value: typing.Any) -> str:
    """Returns the canonical string form of an event id."""
    if value is None:
        raise ValueError('Event id must not be None')
    return str(value)


def _normalize_event_time(value: typing.Any) -> typing.Optional[datetime.datetime]:
    if value is None:
        return None
    return _as_datetime(value)


def _db_value(value: typing.Any) -> decimal.Decimal:
    """Normalizes an amount read from a database."""
    return _as_amount(value)


def _db_time(value: typing.Any) -> typing.Optional[datetime.datetime]:
    """Normalizes a timestamp read from a database (SQLite returns naive)."""
    if value is None:
        return None
    return _as_datetime(value)


# ---------------------------------------------------------------------------
# session computation
# ---------------------------------------------------------------------------

def _compute_sessions(user_id: str,
                      events: typing.Iterable[typing.Tuple[str, datetime.datetime, decimal.Decimal]],
                      idle_threshold: datetime.timedelta) -> typing.List[SessionData]:
    """Computes the sessions of one user from ``(event_id, time, amount)``.

    Events are ordered by event time and event id.  Consecutive events belong
    to the same session while the gap between them is at most ``idle_threshold``.
    """
    ordered = sorted(events, key=lambda event: (event[1], event[0]))

    sessions = []
    started_at = ended_at = None
    event_count = 0
    amount = _ZERO

    for _, event_time, event_amount in ordered:
        if started_at is None:
            started_at = ended_at = event_time
            event_count = 1
            amount = event_amount
        elif event_time - ended_at <= idle_threshold:
            ended_at = event_time
            event_count += 1
            amount += event_amount
        else:
            sessions.append(SessionData(user_id=user_id,
                                        started_at=started_at,
                                        ended_at=ended_at,
                                        event_count=event_count,
                                        amount=amount))
            started_at = ended_at = event_time
            event_count = 1
            amount = event_amount

    if started_at is not None:
        sessions.append(SessionData(user_id=user_id,
                                    started_at=started_at,
                                    ended_at=ended_at,
                                    event_count=event_count,
                                    amount=amount))

    return sessions


def _merge_ranges(ranges: typing.Iterable[typing.Tuple[datetime.datetime, datetime.datetime]],
                  idle_threshold: datetime.timedelta) -> typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]:
    """Merges affected event-time ranges that can influence each other."""
    ordered = sorted(ranges)
    merged = []
    for low, high in ordered:
        if merged and low - merged[-1][1] <= idle_threshold:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

class SessionLoader:
    """Applies event changes and maintains the published user sessions.

    Args:
        db_alias: The alias of the database that stores the event and session
            tables.  Defaults to :func:`mara_pipelines.config.session_db_alias`.
        idle_threshold: The longest gap between two consecutive events that
            still belongs to one session.  Accepts a ``timedelta``, a number of
            seconds, a string such as ``"30 minutes"`` or a callable/mapping
            that returns a threshold per user.  Defaults to
            :func:`mara_pipelines.config.session_idle_threshold`.
        engine: Optional SQLAlchemy engine (or an alias string).  Mainly useful
            for tests and for embedding the loader in another application.
        create_tables: When true (the default), missing tables are created at
            the beginning of a batch.  Set to false when tables are managed by
            ``mara_db.auto_migration``.
        record_retractions: When true (the default), withdrawn sessions are
            also recorded in :class:`SessionRetraction`.
        connection: Optional existing SQLAlchemy connection/transaction.  When
            given, the loader uses it and does not commit by itself.
    """

    def __init__(self,
                 db_alias: typing.Optional[str] = None,
                 idle_threshold: typing.Any = None,
                 engine: typing.Any = None,
                 create_tables: bool = True,
                 record_retractions: bool = True,
                 connection: typing.Any = None) -> None:
        if engine is not None and isinstance(engine, str):
            db_alias = engine
            engine = None

        self.db_alias = db_alias or config.session_db_alias()
        self._engine = engine
        self.create_tables = create_tables
        self.record_retractions = record_retractions
        self._connection = connection

        if idle_threshold is None:
            idle_threshold = config.session_idle_threshold()

        if callable(idle_threshold):
            self.idle_threshold = idle_threshold
        elif isinstance(idle_threshold, collections.abc.Mapping):
            self.idle_threshold = {user_id: _as_timedelta(value)
                                   for user_id, value in idle_threshold.items()}
        else:
            self.idle_threshold = _as_timedelta(idle_threshold)

        if isinstance(self.idle_threshold, datetime.timedelta) and self.idle_threshold < datetime.timedelta(0):
            raise ValueError('The idle threshold must not be negative')

    # -- public api ------------------------------------------------------

    @property
    def engine(self) -> sqlalchemy.engine.Engine:
        """The SQLAlchemy engine used by this loader."""
        if self._engine is not None:
            return self._engine
        return mara_db.sqlalchemy_engine.engine(self.db_alias)

    def threshold_for_user(self, user_id: str) -> datetime.timedelta:
        """The idle threshold that applies to ``user_id``."""
        threshold = self.idle_threshold
        if callable(threshold):
            threshold = threshold(user_id)
        elif isinstance(threshold, collections.abc.Mapping):
            threshold = threshold.get(user_id, datetime.timedelta(0))
        threshold = _as_timedelta(threshold)
        if threshold < datetime.timedelta(0):
            raise ValueError('The idle threshold must not be negative')
        return threshold

    def load(self,
             changes: typing.Iterable[typing.Any] = (),
             *,
             events: typing.Optional[typing.Iterable[typing.Any]] = None,
             connection: typing.Any = None,
             engine: typing.Any = None) -> LoadResult:
        """Applies a batch of event changes atomically.

        Args:
            changes: An iterable of :class:`EventChange` values, mappings or
                objects.  ``events`` is an alias for this argument.
            connection: An optional existing connection/transaction.  When it is
                given, committing is the caller's responsibility.
            engine: An optional engine override for this call.

        Returns:
            A :class:`LoadResult` with statistics about the committed batch.
        """
        if events is not None:
            if changes:
                raise ValueError('Pass either changes or events, not both')
            changes = events

        normalized = [EventChange.from_any(change) for change in changes]

        connection = connection if connection is not None else self._connection
        if connection is not None:
            if self.create_tables:
                Base.metadata.create_all(connection)
            return self._load(connection, normalized)

        target_engine = engine if engine is not None else self.engine
        if self.create_tables:
            create_tables(target_engine)
        with target_engine.begin() as transaction_connection:
            return self._load(transaction_connection, normalized)

    # common alternative spellings for the same operation
    apply = load
    apply_changes = load
    maintain = load
    load_events = load
    load_changes = load

    def recompute(self,
                  user_ids: typing.Optional[typing.Iterable[str]] = None,
                  *,
                  connection: typing.Any = None,
                  engine: typing.Any = None) -> LoadResult:
        """Recomputes all sessions from the current event store.

        This is the reference implementation that the incremental batches are
        guaranteed to match.  Args are the same as for :meth:`load`, except
        that ``user_ids`` restricts the rebuild to selected users.
        """
        connection = connection if connection is not None else self._connection
        if connection is not None:
            if self.create_tables:
                Base.metadata.create_all(connection)
            return self._recompute(connection, user_ids)

        target_engine = engine if engine is not None else self.engine
        if self.create_tables:
            create_tables(target_engine)
        with target_engine.begin() as transaction_connection:
            return self._recompute(transaction_connection, user_ids)

    # aliases that express the reconciliation use case
    rebuild = recompute
    reconcile = recompute

    def sessions(self,
                 user_id: typing.Optional[str] = None,
                 *,
                 connection: typing.Any = None,
                 engine: typing.Any = None) -> typing.List[SessionData]:
        """Returns the currently published sessions, optionally for one user."""
        if connection is not None:
            return self._read_sessions(connection, user_id)

        target_engine = engine if engine is not None else self.engine
        if self.create_tables:
            create_tables(target_engine)
        with target_engine.connect() as read_connection:
            return self._read_sessions(read_connection, user_id)

    def expected_sessions(self,
                          user_id: typing.Optional[str] = None,
                          *,
                          connection: typing.Any = None,
                          engine: typing.Any = None) -> typing.List[SessionData]:
        """Returns the sessions a full recomputation would publish now."""
        if connection is not None:
            return self._read_expected_sessions(connection, user_id)

        target_engine = engine if engine is not None else self.engine
        if self.create_tables:
            create_tables(target_engine)
        with target_engine.connect() as read_connection:
            return self._read_expected_sessions(read_connection, user_id)

    def retractions(self,
                    user_id: typing.Optional[str] = None,
                    *,
                    connection: typing.Any = None,
                    engine: typing.Any = None) -> typing.List[typing.Tuple[SessionData, str]]:
        """Returns the recorded retractions as ``(session, load_id)`` pairs."""
        def read(read_connection):
            table = SessionRetraction.__table__
            statement = sqlalchemy.select(table)
            if user_id is not None:
                statement = statement.where(table.c.user_id == _as_event_id(user_id))
            statement = statement.order_by(table.c.retraction_id)
            return [(SessionData(user_id=row.user_id,
                                 started_at=_db_time(row.started_at),
                                 ended_at=_db_time(row.ended_at),
                                 event_count=int(row.event_count),
                                 amount=_db_value(row.amount)),
                     row.load_id)
                    for row in read_connection.execute(statement).all()]

        if connection is not None:
            return read(connection)

        target_engine = engine if engine is not None else self.engine
        if self.create_tables:
            create_tables(target_engine)
        with target_engine.connect() as read_connection:
            return read(read_connection)

    def check_consistency(self,
                          *,
                          connection: typing.Any = None,
                          engine: typing.Any = None) -> typing.List[str]:
        """Compares published sessions with a full recomputation.

        Returns:
            A list of human readable differences.  An empty list means the
            published state is consistent.
        """
        published = self.sessions(connection=connection, engine=engine)
        expected = self.expected_sessions(connection=connection, engine=engine)
        if published == expected:
            return []

        differences = []
        for missing in expected:
            if missing not in published:
                differences.append(f'missing session {missing!r}')
        for extra in published:
            if extra not in expected:
                differences.append(f'unexpected session {extra!r}')
        return differences

    # -- batch internals -------------------------------------------------

    def _load(self, connection: sqlalchemy.engine.Connection,
              changes: typing.List[EventChange]) -> LoadResult:
        load_id = datetime.datetime.now(_UTC).isoformat()

        affected: typing.Dict[str, typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]] = {}
        applied = 0
        ignored = 0

        for change in changes:
            event_impacts = self._apply_change(connection, change)
            if event_impacts is None:
                ignored += 1
                continue

            applied += 1
            for user_id, event_time in event_impacts:
                if user_id is None or event_time is None:
                    # A tombstone without enough information to belong to a
                    # user cannot influence any session.
                    continue
                affected.setdefault(user_id, []).append((event_time, event_time))

        retracted = 0
        published = 0
        updated = 0
        recomputed_ranges = {}

        for user_id in sorted(affected):
            threshold = self.threshold_for_user(user_id)
            ranges = _merge_ranges(affected[user_id], threshold)
            recomputed_ranges[user_id] = tuple(ranges)
            for low, high in ranges:
                retracted_count, published_count, updated_count = \
                    self._recompute_range(connection, user_id, low, high, threshold, load_id)
                retracted += retracted_count
                published += published_count
                updated += updated_count

        return LoadResult(changes=len(changes),
                          applied=applied,
                          ignored=ignored,
                          retracted_sessions=retracted,
                          published_sessions=published,
                          updated_sessions=updated,
                          affected_users=tuple(sorted(affected)),
                          recomputed_ranges=recomputed_ranges,
                          succeeded=True)

    def _apply_change(self,
                      connection: sqlalchemy.engine.Connection,
                      change: EventChange
                      ) -> typing.Optional[typing.List[typing.Tuple[typing.Optional[str],
                                                                    typing.Optional[datetime.datetime]]]]:
        """Applies one event change.

        Returns:
            ``None`` when the change was ignored as duplicate/stale.  Otherwise
            a list of ``(user_id, event_time)`` positions that the change
            affected (the old and the new position for an update).
        """
        table = SessionEvent.__table__
        event_id = _as_event_id(change.event_id)
        source_version = _as_version(change.source_version)
        deleted = bool(change.deleted)

        row = connection.execute(
            sqlalchemy.select(table)
            .where(table.c.event_id == event_id)
            .with_for_update()
        ).first()

        now = datetime.datetime.now(_UTC)

        if row is None:
            if deleted:
                # Unknown event: remember the tombstone so that a stale insert
                # with a lower version cannot resurrect it later.
                connection.execute(sqlalchemy.insert(table).values(
                    event_id=event_id,
                    user_id=_as_event_id(change.user_id) if change.user_id is not None else None,
                    event_time=_normalize_event_time(change.event_time),
                    amount=_ZERO,
                    source_version=source_version,
                    deleted=True,
                    updated_at=now))
                return []

            user_id = _as_event_id(change.user_id) if change.user_id is not None else None
            event_time = _normalize_event_time(change.event_time)
            if user_id is None or event_time is None:
                raise ValueError(f'New event "{event_id}" needs a user_id and an event_time')

            connection.execute(sqlalchemy.insert(table).values(
                event_id=event_id,
                user_id=user_id,
                event_time=event_time,
                amount=_as_amount(change.amount),
                source_version=source_version,
                deleted=False,
                updated_at=now))
            return [(user_id, event_time)]

        stored_version = int(row.source_version or 0)
        if source_version <= stored_version:
            return None

        old_user_id = row.user_id
        old_event_time = _db_time(row.event_time)

        if deleted:
            connection.execute(
                sqlalchemy.update(table)
                .where(table.c.event_id == event_id)
                .values(source_version=source_version,
                        deleted=True,
                        updated_at=now))
            if old_user_id is None or old_event_time is None:
                return []
            return [(old_user_id, old_event_time)]

        new_user_id = _as_event_id(change.user_id) if change.user_id is not None else old_user_id
        new_event_time = _normalize_event_time(change.event_time) \
            if change.event_time is not None else old_event_time
        if new_user_id is None or new_event_time is None:
            raise ValueError(f'Event "{event_id}" needs a user_id and an event_time')
        new_amount = _as_amount(change.amount) if change.amount is not None else _db_value(row.amount)

        connection.execute(
            sqlalchemy.update(table)
            .where(table.c.event_id == event_id)
            .values(user_id=new_user_id,
                    event_time=new_event_time,
                    amount=new_amount,
                    source_version=source_version,
                    deleted=False,
                    updated_at=now))

        impacts = [(new_user_id, new_event_time)]
        if old_user_id is not None and old_event_time is not None:
            impacts.insert(0, (old_user_id, old_event_time))
        return impacts

    def _recompute_range(self,
                         connection: sqlalchemy.engine.Connection,
                         user_id: str,
                         low: datetime.datetime,
                         high: datetime.datetime,
                         idle_threshold: datetime.timedelta,
                         load_id: str) -> typing.Tuple[int, int, int]:
        """Retracts and republishes the sessions around ``[low, high]``.

        The recomputation window is expanded by one neighbouring session on
        each side, so that a change that merges or splits sessions at the
        window boundary is handled.  The outer boundaries of these neighbours
        are stable session boundaries, which makes the recomputed sessions
        inside the window final.
        """
        sessions_table = UserSession.__table__

        overlapping = connection.execute(
            sqlalchemy.select(sessions_table)
            .where(sessions_table.c.user_id == user_id,
                   sessions_table.c.ended_at >= low - idle_threshold,
                   sessions_table.c.started_at <= high + idle_threshold)
            .order_by(sessions_table.c.started_at, sessions_table.c.ended_at)
            .with_for_update()
        ).all()

        previous = self._neighbour(connection, user_id, low, before=True)
        following = self._neighbour(connection, user_id, high, before=False)

        window_low = low
        window_high = high
        for row in overlapping:
            window_low = min(window_low, _db_time(row.started_at))
            window_high = max(window_high, _db_time(row.ended_at))
        if previous is not None:
            window_low = min(window_low, _db_time(previous.started_at))
        if following is not None:
            window_high = max(window_high, _db_time(following.ended_at))

        events_table = SessionEvent.__table__
        events = connection.execute(
            sqlalchemy.select(events_table.c.event_id, events_table.c.event_time,
                              events_table.c.amount)
            .where(events_table.c.user_id == user_id,
                   events_table.c.deleted.is_(False),
                   events_table.c.event_time.is_not(None),
                   events_table.c.event_time >= window_low,
                   events_table.c.event_time <= window_high)
        ).all()

        new_sessions = _compute_sessions(
            user_id,
            ((str(event_id), _db_time(event_time), _db_value(amount))
             for event_id, event_time, amount in events),
            idle_threshold)

        # A session that is not identical to a previously published one is
        # withdrawn and a new one is published in its place.  Matching happens
        # by boundaries, which are unique per user within the window.
        existing = {}
        for row in connection.execute(
                sqlalchemy.select(sessions_table)
                .where(sessions_table.c.user_id == user_id,
                       sessions_table.c.started_at <= window_high,
                       sessions_table.c.ended_at >= window_low)
                .order_by(sessions_table.c.started_at, sessions_table.c.ended_at)
                .with_for_update()).all():
            existing[(_db_time(row.started_at), _db_time(row.ended_at))] = row

        computed = {(session.started_at, session.ended_at): session
                    for session in new_sessions}

        unchanged = set()
        for key, row in existing.items():
            replacement = computed.get(key)
            if replacement is not None \
                    and int(row.event_count) == replacement.event_count \
                    and _db_value(row.amount) == replacement.amount:
                unchanged.add(key)

        retracted = published = updated = 0
        for key, row in existing.items():
            if key in unchanged:
                continue
            self._retract_session(connection, row, load_id)
            retracted += 1
            if key in computed:
                updated += 1

        # Deletions are executed before the insertions below, so the unique
        # constraint on the session boundaries does not fire for replacements.
        for key, replacement in computed.items():
            if key in unchanged:
                continue
            self._insert_session(connection, replacement)
            published += 1

        return retracted, published, updated

    def _neighbour(self,
                   connection: sqlalchemy.engine.Connection,
                   user_id: str,
                   boundary: datetime.datetime,
                   before: bool) -> typing.Optional[sqlalchemy.engine.Row]:
        table = UserSession.__table__
        if before:
            statement = (sqlalchemy.select(table)
                         .where(table.c.user_id == user_id,
                                table.c.ended_at < boundary)
                         .order_by(table.c.ended_at.desc())
                         .limit(1))
        else:
            statement = (sqlalchemy.select(table)
                         .where(table.c.user_id == user_id,
                                table.c.started_at > boundary)
                         .order_by(table.c.started_at.asc())
                         .limit(1))
        return connection.execute(statement.with_for_update()).first()

    def _retract_session(self,
                         connection: sqlalchemy.engine.Connection,
                         row: sqlalchemy.engine.Row,
                         load_id: str) -> None:
        if self.record_retractions:
            retractions = SessionRetraction.__table__
            connection.execute(sqlalchemy.insert(retractions).values(
                user_id=row.user_id,
                started_at=row.started_at,
                ended_at=row.ended_at,
                event_count=row.event_count,
                amount=row.amount,
                load_id=load_id,
                retracted_at=datetime.datetime.now(_UTC)))
        sessions = UserSession.__table__
        connection.execute(sqlalchemy.delete(sessions)
                           .where(sessions.c.session_id == row.session_id))

    def _insert_session(self,
                        connection: sqlalchemy.engine.Connection,
                        session: SessionData) -> None:
        sessions = UserSession.__table__
        connection.execute(sqlalchemy.insert(sessions).values(
            user_id=session.user_id,
            started_at=session.started_at,
            ended_at=session.ended_at,
            event_count=session.event_count,
            amount=session.amount))

    # -- full recomputation ---------------------------------------------

    def _recompute(self,
                   connection: sqlalchemy.engine.Connection,
                   user_ids: typing.Optional[typing.Iterable[str]]) -> LoadResult:
        events_table = SessionEvent.__table__
        sessions_table = UserSession.__table__

        if user_ids is None:
            users = set(row[0] for row in connection.execute(
                sqlalchemy.select(events_table.c.user_id)
                .where(events_table.c.deleted.is_(False), events_table.c.user_id.is_not(None))
                .distinct()).all())
            users.update(row[0] for row in connection.execute(
                sqlalchemy.select(sessions_table.c.user_id).distinct()).all())
        else:
            users = {_as_event_id(user_id) for user_id in user_ids}

        retracted = published = 0
        recomputed_ranges = {}

        for user_id in sorted(users):
            threshold = self.threshold_for_user(user_id)

            existing = connection.execute(
                sqlalchemy.select(sessions_table)
                .where(sessions_table.c.user_id == user_id)
                .order_by(sessions_table.c.started_at)
                .with_for_update()
            ).all()

            events = connection.execute(
                sqlalchemy.select(events_table.c.event_id, events_table.c.event_time,
                                  events_table.c.amount)
                .where(events_table.c.user_id == user_id,
                       events_table.c.deleted.is_(False),
                       events_table.c.event_time.is_not(None))
            ).all()

            new_sessions = _compute_sessions(
                user_id,
                ((str(event_id), _db_time(event_time), _db_value(amount))
                 for event_id, event_time, amount in events),
                threshold)

            remaining = {(session.started_at, session.ended_at): session
                         for session in new_sessions}
            for row in existing:
                key = (_db_time(row.started_at), _db_time(row.ended_at))
                replacement = remaining.get(key)
                if replacement is not None \
                        and int(row.event_count) == replacement.event_count \
                        and _db_value(row.amount) == replacement.amount:
                    del remaining[key]
                    continue
                self._retract_session(connection, row, 'recompute')
                retracted += 1

            for replacement in remaining.values():
                self._insert_session(connection, replacement)
                published += 1

            if existing:
                low = min(_db_time(row.started_at) for row in existing)
                high = max(_db_time(row.ended_at) for row in existing)
                recomputed_ranges[user_id] = ((low, high),)

        return LoadResult(changes=0,
                          applied=0,
                          ignored=0,
                          retracted_sessions=retracted,
                          published_sessions=published,
                          updated_sessions=0,
                          affected_users=tuple(sorted(users)),
                          recomputed_ranges=recomputed_ranges,
                          succeeded=True)

    # -- read helpers ----------------------------------------------------

    def _read_sessions(self,
                       connection: sqlalchemy.engine.Connection,
                       user_id: typing.Optional[str]) -> typing.List[SessionData]:
        table = UserSession.__table__
        statement = sqlalchemy.select(table)
        if user_id is not None:
            statement = statement.where(table.c.user_id == _as_event_id(user_id))
        rows = connection.execute(
            statement.order_by(table.c.user_id, table.c.started_at, table.c.ended_at)
        ).all()
        return [SessionData(user_id=row.user_id,
                            started_at=_db_time(row.started_at),
                            ended_at=_db_time(row.ended_at),
                            event_count=int(row.event_count),
                            amount=_db_value(row.amount))
                for row in rows]

    def _read_expected_sessions(self,
                                connection: sqlalchemy.engine.Connection,
                                user_id: typing.Optional[str]) -> typing.List[SessionData]:
        events_table = SessionEvent.__table__
        statement = sqlalchemy.select(events_table.c.user_id).where(
            events_table.c.deleted.is_(False), events_table.c.user_id.is_not(None))
        if user_id is not None:
            statement = statement.where(events_table.c.user_id == _as_event_id(user_id))
        users = sorted(set(row[0] for row in connection.execute(statement.distinct()).all()))

        result = []
        for event_user_id in users:
            events = connection.execute(
                sqlalchemy.select(events_table.c.event_id, events_table.c.event_time,
                                  events_table.c.amount)
                .where(events_table.c.user_id == event_user_id,
                       events_table.c.deleted.is_(False),
                       events_table.c.event_time.is_not(None))
            ).all()
            result.extend(_compute_sessions(
                event_user_id,
                ((str(event_id), _db_time(event_time), _db_value(amount))
                 for event_id, event_time, amount in events),
                self.threshold_for_user(event_user_id)))
        return sorted(result, key=lambda session: (session.user_id, session.started_at,
                                                   session.ended_at))


# ---------------------------------------------------------------------------
# module level convenience api
# ---------------------------------------------------------------------------

def create_tables(engine: typing.Any = None,
                  db_alias: typing.Optional[str] = None) -> None:
    """Creates the event, session and retraction tables if they do not exist."""
    if engine is None:
        engine = mara_db.sqlalchemy_engine.engine(db_alias or config.session_db_alias())
    Base.metadata.create_all(engine)


def load(changes: typing.Iterable[typing.Any] = (),
         *,
         events: typing.Optional[typing.Iterable[typing.Any]] = None,
         db_alias: typing.Optional[str] = None,
         idle_threshold: typing.Any = None,
         engine: typing.Any = None,
         connection: typing.Any = None) -> LoadResult:
    """Applies a batch of event changes with a temporary :class:`SessionLoader`."""
    loader = SessionLoader(db_alias=db_alias,
                           idle_threshold=idle_threshold,
                           engine=engine,
                           connection=connection)
    return loader.load(changes, events=events)


# aliases for the different ways the operation is commonly called
load_sessions = load
apply_changes = load
maintain_sessions = load


def recompute_sessions(*,
                       user_ids: typing.Optional[typing.Iterable[str]] = None,
                       db_alias: typing.Optional[str] = None,
                       idle_threshold: typing.Any = None,
                       engine: typing.Any = None,
                       connection: typing.Any = None) -> LoadResult:
    """Rebuilds sessions from the event store with a temporary loader."""
    loader = SessionLoader(db_alias=db_alias,
                           idle_threshold=idle_threshold,
                           engine=engine,
                           connection=connection)
    return loader.recompute(user_ids)


# ---------------------------------------------------------------------------
# pipeline command
# ---------------------------------------------------------------------------

def _session_loader_command(loader: SessionLoader,
                            changes: typing.Iterable[typing.Any] = (),
                            recompute: bool = False):
    """Creates a pipeline :class:`~mara_pipelines.pipelines.Command`."""
    from . import pipelines
    from .logging import logger

    class _LoadSessions(pipelines.Command):
        def run(self) -> bool:
            result = loader.recompute() if recompute else loader.load(changes)
            logger.log(f'{result.applied} change(s) applied, {result.ignored} ignored, '
                       f'{result.retracted_sessions} session(s) retracted, '
                       f'{result.published_sessions} published, '
                       f'{result.updated_sessions} updated')
            return bool(result)

        def html_doc_items(self) -> typing.List[typing.Tuple[str, str]]:
            return [('db alias', loader.db_alias),
                    ('idle threshold', str(loader.idle_threshold)),
                    ('mode', 'recompute' if recompute else 'load changes')]

    return _LoadSessions()


def LoadSessions(changes: typing.Iterable[typing.Any] = (),
                 recompute: bool = False,
                 db_alias: typing.Optional[str] = None,
                 idle_threshold: typing.Any = None):
    """A pipeline command that applies a batch of session changes.

    Imports of :mod:`mara_pipelines` are deferred until the command is
    constructed, so the module stays usable without the pipeline machinery.

    Args:
        changes: The batch of :class:`EventChange` values, mappings or objects.
        recompute: Rebuild all sessions instead of applying a batch.
        db_alias: The database alias, defaults to the configured session db.
        idle_threshold: The idle threshold, defaults to the configured value.

    Returns:
        A :class:`~mara_pipelines.pipelines.Command` instance.
    """
    return _session_loader_command(
        SessionLoader(db_alias=db_alias, idle_threshold=idle_threshold),
        changes=changes,
        recompute=recompute)

