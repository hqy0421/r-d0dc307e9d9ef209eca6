"""Tests for the incremental user-session loader.

The tests run on an in-memory SQLite database, so they need neither docker nor
a PostgreSQL server.  The incremental batches are always compared against a
full recomputation from the event store, which is the consistency criterion the
loader guarantees.
"""

import datetime
import decimal
import random

import pytest
import sqlalchemy

from mara_pipelines.sessionization import (
    EventChange, LoadResult, SessionData, SessionLoader, SessionEvent, UserSession,
    SessionRetraction, create_tables, load,
)

UTC = datetime.timezone.utc
HOUR = datetime.timedelta(hours=1)
MINUTE = datetime.timedelta(minutes=1)


def at(minutes: float, day: int = 1) -> datetime.datetime:
    """A UTC event time on 2024-01-``day`` at ``minutes`` past midnight."""
    return datetime.datetime(2024, 1, day, tzinfo=UTC) + datetime.timedelta(minutes=minutes)


def event(event_id, user, minutes, amount=0, version=1, deleted=False, day=1):
    return EventChange(event_id=event_id,
                       source_version=version,
                       deleted=deleted,
                       user_id=user,
                       event_time=at(minutes, day),
                       amount=amount)


def summary(sessions):
    return [(s.user_id, s.started_at, s.ended_at, s.event_count, s.amount) for s in sessions]


def expected_summary(events, threshold):
    """Independent reference implementation of the session definition."""
    by_user = {}
    for e in events:
        if e.deleted:
            continue
        by_user.setdefault(e.user_id, []).append(
            (e.event_time if isinstance(e.event_time, datetime.datetime)
             else datetime.datetime.fromisoformat(str(e.event_time).replace('Z', '+00:00')),
             decimal.Decimal(str(e.amount or 0))))
    result = []
    for user in sorted(by_user):
        ordered = sorted(by_user[user])
        current = None
        for timestamp, amount in ordered:
            if current and timestamp - current[1] <= threshold:
                current[1] = timestamp
                current[2] += 1
                current[3] += amount
            else:
                if current:
                    result.append((user, current[0], current[1], current[2], current[3]))
                current = [timestamp, timestamp, 1, amount]
        if current:
            result.append((user, current[0], current[1], current[2], current[3]))
    return result


@pytest.fixture()
def engine():
    engine = sqlalchemy.create_engine('sqlite://')
    create_tables(engine)
    return engine


@pytest.fixture()
def loader(engine):
    return SessionLoader(idle_threshold=HOUR, engine=engine)


def assert_consistent(loader):
    assert loader.check_consistency() == [], 'published sessions differ from a full recompute'


# ---------------------------------------------------------------------------
# plain session computation
# ---------------------------------------------------------------------------

def test_first_batch_publishes_one_session_per_gap(loader):
    result = loader.load([
        event('e1', 'alice', 0), event('e2', 'alice', 10),
        event('e3', 'alice', 20),
        event('e4', 'alice', 100),   # more than an hour after e3
    ])

    assert result
    assert result.applied == 4
    assert result.ignored == 0
    assert result.published_sessions == 2
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(20), 3, decimal.Decimal('0.0000000000')),
        ('alice', at(100), at(100), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_threshold_is_inclusive(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 60)])
    assert len(loader.sessions('alice')) == 1

    loader.load([event('e2', 'alice', 61, version=2)])
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
        ('alice', at(61), at(61), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_amounts_are_exact_decimals(loader):
    loader.load([event('e1', 'alice', 0, '0.1'), event('e2', 'alice', 1, '0.2')])
    assert loader.sessions('alice')[0].amount == decimal.Decimal('0.3')


def test_users_are_sessionized_independently(loader):
    loader.load([event('a1', 'alice', 0), event('b1', 'bob', 0),
                 event('a2', 'alice', 30), event('b2', 'bob', 90)])
    assert summary(loader.sessions()) == [
        ('alice', at(0), at(30), 2, decimal.Decimal('0.0000000000')),
        ('bob', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
        ('bob', at(90), at(90), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_out_of_order_events_within_a_batch(loader):
    loader.load([event('e3', 'alice', 40), event('e1', 'alice', 0),
                 event('e2', 'alice', 20)])
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(40), 3, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


# ---------------------------------------------------------------------------
# backfill: adding an event can merge sessions
# ---------------------------------------------------------------------------

def test_backfilled_event_merges_two_sessions(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 120)])
    assert len(loader.sessions('alice')) == 2

    result = loader.load([event('e9', 'alice', 60)])

    assert result.retracted_sessions == 2
    assert result.published_sessions == 1
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(120), 3, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_backfilled_event_extends_session_without_merge(loader):
    loader.load([event('e1', 'alice', 0), event('e3', 'alice', 200)])
    result = loader.load([event('e2', 'alice', 50)])

    assert result.retracted_sessions == 1
    assert result.published_sessions == 1
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(50), 2, decimal.Decimal('0.0000000000')),
        ('alice', at(200), at(200), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_backfill_merges_only_affected_neighbourhood(loader):
    loader.load([
        event('a1', 'alice', 0), event('a2', 'alice', 300),
        event('a3', 'alice', 700), event('a4', 'alice', 1100),
        event('b1', 'bob', 0), event('b2', 'bob', 10),
    ])
    before = summary(loader.sessions('alice'))
    result = loader.load([event('a2b', 'alice', 320)])

    # only the session around 300 changes; the distant sessions keep their ids
    assert result.affected_users == ('alice',)
    after = summary(loader.sessions('alice'))
    assert after[0] == before[0]
    assert after[2:] == before[2:]
    assert len(loader.sessions('bob')) == 1
    assert_consistent(loader)


def test_unaffected_sessions_keep_their_rows(loader, engine):
    loader.load([event('a1', 'alice', 0), event('a2', 'alice', 30),
                 event('a3', 'alice', 600)])
    with engine.connect() as connection:
        table = UserSession.__table__
        rows_before = connection.execute(sqlalchemy.select(table)).all()

    loader.load([event('a4', 'alice', 45)])
    with engine.connect() as connection:
        rows_after = connection.execute(sqlalchemy.select(table)).all()

    distant_before = [row for row in rows_before if row.started_at.replace(tzinfo=UTC) == at(600)]
    distant_after = [row for row in rows_after if row.started_at.replace(tzinfo=UTC) == at(600)]
    assert [row.session_id for row in distant_before] == [row.session_id for row in distant_after]
    assert_consistent(loader)


# ---------------------------------------------------------------------------
# retraction
# ---------------------------------------------------------------------------

def test_retraction_splits_a_session(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 50),
                 event('e3', 'alice', 100)])

    result = loader.load([EventChange('e2', 2, True)])

    assert result.retracted_sessions == 1
    assert result.published_sessions == 2
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
        ('alice', at(100), at(100), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_retraction_removes_empty_session_and_records_it(loader):
    loader.load([event('e1', 'alice', 0)])
    result = loader.load([EventChange('e1', 2, True)])

    assert result.retracted_sessions == 1
    assert loader.sessions('alice') == []
    assert [s.user_id for s, _ in loader.retractions('alice')] == ['alice']
    assert_consistent(loader)


def test_retraction_of_unknown_event_is_a_tombstone(loader):
    loader.load([event('e1', 'alice', 0)])
    loader.load([EventChange('nope', 3, True)])
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_retracted_event_can_be_reinstated_with_newer_version(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 10)])
    loader.load([EventChange('e2', 2, True)])
    loader.load([event('e2', 'alice', 30, version=3)])

    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(30), 2, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


# ---------------------------------------------------------------------------
# event time corrections
# ---------------------------------------------------------------------------

def test_changed_event_time_splits_a_session(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 40),
                 event('e3', 'alice', 80)])
    assert len(loader.sessions('alice')) == 1

    result = loader.load([event('e2', 'alice', 300, version=2)])

    assert result.retracted_sessions == 1
    assert result.published_sessions == 3
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
        ('alice', at(80), at(80), 1, decimal.Decimal('0.0000000000')),
        ('alice', at(300), at(300), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_changed_event_time_merges_sessions(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 30),
                 event('e3', 'alice', 200)])
    result = loader.load([event('e3', 'alice', 50, version=2)])

    assert result.retracted_sessions == 2
    assert result.published_sessions == 1
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(50), 3, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_changed_user_moves_event_between_users(loader):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 10),
                 event('b1', 'bob', 0)])

    loader.load([event('e2', 'bob', 10, version=2)])

    assert summary(loader.sessions()) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
        ('bob', at(0), at(10), 2, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_changed_amount_republishes_the_same_session(loader):
    loader.load([event('e1', 'alice', 0, '10'), event('e2', 'alice', 10, '5')])

    result = loader.load([event('e2', 'alice', 10, '7', version=2)])

    # the boundaries stay the same, but the snapshot is withdrawn and the
    # corrected one is published
    assert result.retracted_sessions == 1
    assert result.updated_sessions == 1
    assert result.published_sessions == 1
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(10), 2, decimal.Decimal('17.0000000000')),
    ]
    assert_consistent(loader)


# ---------------------------------------------------------------------------
# duplicate and out-of-order versions
# ---------------------------------------------------------------------------

def test_duplicate_change_is_ignored(loader):
    loader.load([event('e1', 'alice', 0)])
    result = loader.load([event('e1', 'alice', 0)])

    assert result.ignored == 1
    assert result.applied == 0
    assert result.affected_users == ()
    assert_consistent(loader)


def test_identical_duplicate_in_first_batch(loader):
    result = loader.load([event('e1', 'alice', 0), event('e1', 'alice', 0)])
    assert result.applied == 1
    assert result.ignored == 1
    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
    ]


def test_older_version_cannot_override_a_newer_change(loader):
    loader.load([event('e1', 'alice', 100, version=5)])
    result = loader.load([event('e1', 'alice', 0, version=3)])

    assert result.ignored == 1
    assert summary(loader.sessions('alice')) == [
        ('alice', at(100), at(100), 1, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_stale_retraction_is_ignored(loader):
    loader.load([event('e1', 'alice', 0, version=1)])
    loader.load([event('e1', 'alice', 0, version=4)])
    result = loader.load([EventChange('e1', 2, True)])

    assert result.ignored == 1
    assert len(loader.sessions('alice')) == 1
    assert_consistent(loader)


def test_stale_insert_cannot_resurrect_a_retracted_event(loader):
    loader.load([event('e1', 'alice', 0, version=1)])
    loader.load([EventChange('e1', 2, True)])
    result = loader.load([event('e1', 'alice', 0, version=1)])

    assert result.ignored == 1
    assert loader.sessions('alice') == []
    assert_consistent(loader)


def test_stale_change_after_retraction_does_not_change_result(loader):
    loader.load([event('e1', 'alice', 0, version=1)])
    loader.load([EventChange('e1', 3, True)])

    # an older non-delete version arrives late
    result = loader.load([event('e1', 'alice', 0, version=2)])
    assert result.ignored == 1
    assert loader.sessions('alice') == []
    assert_consistent(loader)


def test_batch_order_does_not_matter(loader):
    changes = [
        event('e1', 'alice', 0, version=1),
        event('e2', 'alice', 30, version=1),
        event('e1', 'alice', 10, version=2),
        event('e1', 'alice', 5, version=1),   # stale within the same batch
    ]
    loader.load(list(reversed(changes)))
    assert_consistent(loader)
    assert summary(loader.sessions('alice')) == [
        ('alice', at(10), at(30), 2, decimal.Decimal('0.0000000000')),
    ]


# ---------------------------------------------------------------------------
# atomicity
# ---------------------------------------------------------------------------

def test_failed_batch_leaves_no_trace(loader, engine):
    loader.load([event('e1', 'alice', 0)])
    before = summary(loader.sessions())

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with engine.begin() as connection:
            loader.load([event('e2', 'alice', 30), event('e3', 'bob', 0)],
                        connection=connection)
            raise Boom()

    assert summary(loader.sessions()) == before
    assert_consistent(loader)


def test_invalid_change_in_batch_rolls_back_everything(loader):
    loader.load([event('e1', 'alice', 0)])

    with pytest.raises(ValueError):
        loader.load([
            event('e2', 'alice', 30),
            EventChange('e3', 1, False, 'alice', None, 1),   # missing event time
        ])

    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
    ]
    assert loader.engine is not None
    assert_consistent(loader)


# ---------------------------------------------------------------------------
# random batches compared with a full recomputation
# ---------------------------------------------------------------------------

def random_change(rng, event_ids, users):
    event_id = rng.choice(event_ids) if rng.random() < 0.6 else f'new-{rng.randrange(10**9)}'
    version = rng.randint(1, 8)
    if rng.random() < 0.15:
        return EventChange(event_id=event_id, source_version=version, deleted=True)
    return EventChange(event_id=event_id,
                       source_version=version,
                       user_id=rng.choice(users),
                       event_time=at(rng.randrange(0, 20 * 60)),
                       amount=rng.choice(['0', '1', '2.5', '10', '0.1']))


@pytest.mark.parametrize('seed', range(6))
def test_random_batches_match_full_recomputation(seed, engine):
    rng = random.Random(seed)
    loader = SessionLoader(idle_threshold=datetime.timedelta(minutes=45), engine=engine)

    event_ids = [f'e{i}' for i in range(25)]
    users = ['alice', 'bob', 'carol']

    # independent model of the event store: version, deleted, user, time, amount
    store = {}

    for _ in range(25):
        batch = [random_change(rng, event_ids, users) for _ in range(rng.randint(1, 8))]
        for change in batch:
            event_id = str(change.event_id)
            version = int(change.source_version)
            current = store.get(event_id)
            if current is not None and version <= current[0]:
                continue
            if change.deleted:
                if current is None:
                    store[event_id] = (version, True, None, None, decimal.Decimal(0))
                else:
                    store[event_id] = (version, True, current[2], current[3], current[4])
                continue
            user = str(change.user_id) if change.user_id is not None else (current[2] if current else None)
            moment = change.event_time if isinstance(change.event_time, datetime.datetime) \
                else datetime.datetime.fromisoformat(str(change.event_time).replace('Z', '+00:00'))
            amount = decimal.Decimal(str(change.amount if change.amount is not None
                                        else (current[4] if current else 0)))
            store[event_id] = (version, False, user, moment, amount)

        result = loader.load(batch)
        assert result.applied + result.ignored == len(batch)
        assert_consistent(loader)

        expected = expected_summary(
            [EventChange(event_id=k, source_version=v[0], deleted=v[1],
                         user_id=v[2], event_time=v[3], amount=v[4])
             for k, v in store.items()],
            datetime.timedelta(minutes=45))
        assert summary(loader.sessions()) == expected


def test_random_duplicates_and_reordering_are_idempotent(engine):
    rng = random.Random(1234)
    loader = SessionLoader(idle_threshold=datetime.timedelta(minutes=30), engine=engine)
    users = ['alice', 'bob']

    changes = [EventChange(event_id=f'e{i}',
                           source_version=rng.randint(1, 5),
                           user_id=rng.choice(users),
                           event_time=at(rng.randrange(0, 600)),
                           amount=rng.choice(['0', '1', '2.5']))
               for i in range(40)]

    loader.load(changes)
    state = summary(loader.sessions())

    # replaying the same changes in a different order changes nothing
    shuffled = list(changes)
    rng.shuffle(shuffled)
    result = loader.load(shuffled)
    assert result.applied == 0
    assert summary(loader.sessions()) == state
    assert_consistent(loader)


# ---------------------------------------------------------------------------
# input flexibility and configuration
# ---------------------------------------------------------------------------

def test_changes_can_be_mappings_and_objects(loader):
    class Change:
        event_id = 'e1'
        source_version = 1
        deleted = False
        user_id = 'alice'
        event_time = '2024-01-01T00:30:00Z'
        amount = '3.5'

    loader.load([Change()])
    loader.load([{'id': 'e2', 'version': 1, 'user': 'alice',
                  'time': 1_704_067_200, 'value': '1.5'}])

    assert summary(loader.sessions('alice')) == [
        ('alice', at(0), at(30), 2, decimal.Decimal('5.0000000000')),
    ]


def test_float_and_numeric_string_event_ids_are_equivalent(loader):
    loader.load([EventChange(event_id=1, source_version=1, user_id='alice',
                             event_time=at(0), amount=1)])
    result = loader.load([EventChange(event_id='1', source_version=2, user_id='alice',
                                      event_time=at(0), amount=2)])
    assert result.applied == 1
    assert loader.sessions('alice')[0].amount == decimal.Decimal('2.0000000000')


def test_per_user_idle_threshold(engine):
    loader = SessionLoader(idle_threshold={'alice': MINUTE,
                                           'bob': 5 * MINUTE}, engine=engine)
    loader.load([event('a1', 'alice', 0), event('a2', 'alice', 2),
                 event('b1', 'bob', 0), event('b2', 'bob', 2)])

    assert summary(loader.sessions()) == [
        ('alice', at(0), at(0), 1, decimal.Decimal('0.0000000000')),
        ('alice', at(2), at(2), 1, decimal.Decimal('0.0000000000')),
        ('bob', at(0), at(2), 2, decimal.Decimal('0.0000000000')),
    ]
    assert_consistent(loader)


def test_callable_idle_threshold(engine):
    loader = SessionLoader(idle_threshold=lambda user: MINUTE if user == 'alice' else HOUR,
                           engine=engine)
    loader.load([event('a1', 'alice', 0), event('a2', 'alice', 30),
                 event('b1', 'bob', 0), event('b2', 'bob', 30)])
    assert [s.event_count for s in loader.sessions()] == [1, 1, 2]
    assert_consistent(loader)


def test_string_and_numeric_thresholds(engine):
    for threshold in ['30 minutes', '30min', 1800, HOUR / 2]:
        loader = SessionLoader(idle_threshold=threshold, engine=engine)
        loader.load([event('e1', 'alice', 0), event('e2', 'alice', 30)])
        assert len(loader.sessions('alice')) == 1


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError):
        SessionLoader(idle_threshold=-1, engine=sqlalchemy.create_engine('sqlite://'))


def test_events_with_naive_datetimes_are_treated_as_utc(loader):
    loader.load([EventChange('e1', 1, False, 'alice',
                             datetime.datetime(2024, 1, 1, 0, 0), 1)])
    session = loader.sessions('alice')[0]
    assert session.started_at == at(0)


def test_timezones_are_normalized(loader):
    loader.load([EventChange('e1', 1, False, 'alice',
                             '2024-01-01T08:00:00+08:00', 1)])
    assert loader.sessions('alice')[0].started_at == at(0)


# ---------------------------------------------------------------------------
# recompute / reconcile
# ---------------------------------------------------------------------------

def test_recompute_repairs_a_manually_damaged_session_table(loader, engine):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 30),
                 event('e3', 'alice', 200)])
    expected = summary(loader.sessions())

    with engine.begin() as connection:
        connection.execute(sqlalchemy.delete(UserSession))

    assert summary(loader.sessions()) == []
    loader.recompute()
    assert summary(loader.sessions()) == expected
    assert_consistent(loader)


def test_recompute_restricted_to_users(engine):
    loader = SessionLoader(idle_threshold=HOUR, engine=engine)
    loader.load([event('a1', 'alice', 0), event('b1', 'bob', 0)])
    loader.recompute(user_ids=['alice'])
    assert len(loader.sessions('alice')) == 1
    assert len(loader.sessions('bob')) == 1


def test_module_level_load_function(engine):
    loader = SessionLoader(idle_threshold=HOUR, engine=engine)
    result = load([event('e1', 'alice', 0)], idle_threshold=HOUR, engine=engine)
    assert result.applied == 1
    assert len(loader.sessions('alice')) == 1


# ---------------------------------------------------------------------------
# exposure model registration
# ---------------------------------------------------------------------------

def test_models_are_registered_for_auto_migration():
    import mara_pipelines

    models = mara_pipelines.MARA_AUTOMIGRATE_SQLALCHEMY_MODELS()
    assert SessionEvent in models
    assert UserSession in models
    assert SessionRetraction in models


def test_session_table_only_contains_current_sessions(loader, engine):
    loader.load([event('e1', 'alice', 0), event('e2', 'alice', 30)])
    session_id_before = loader.sessions('alice')[0]

    loader.load([EventChange('e2', 2, True)])

    with engine.connect() as connection:
        rows = connection.execute(sqlalchemy.select(UserSession.__table__)).all()
    assert len(rows) == 1
    assert rows[0].ended_at.replace(tzinfo=UTC) == at(0)
    assert session_id_before != loader.sessions('alice')[0]
