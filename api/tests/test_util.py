#!/usr/bin/env python3

from datetime import date, datetime, timedelta
from dateutil import tz
import arrow

from api.util import round_down, xor


def test_round_down_timestamp_no_timezone():
    original = datetime(2022, 1, 1, 12, 30, 44)
    round_to = timedelta(hours=1)
    expected = datetime(2022, 1, 1, 12, 0, 0)
    assert round_down(original, round_to) == expected


def test_round_down_timestamp_local_time():
    original = datetime(2022, 1, 1, 12, 30, 44, tzinfo=tz.gettz('America/Los_Angeles'))
    round_to = timedelta(hours=1)
    expected = datetime(2022, 1, 1, 12, 0, 0, tzinfo=tz.gettz('America/Los_Angeles'))
    assert round_down(original, round_to) == expected


def test_round_down_timestamp_utc_time():
    original = datetime(2022, 1, 1, 12, 30, 44, tzinfo=tz.UTC)
    round_to = timedelta(hours=1)
    expected = datetime(2022, 1, 1, 12, 0, 0, tzinfo=tz.UTC)
    assert round_down(original, round_to) == expected


def test_round_down_timestamp_now_to_today():
    original = arrow.get(datetime.now()).datetime
    round_to = timedelta(days=1)
    expected = arrow.get(date.today()).datetime
    assert abs(round_down(original, round_to) - expected) < timedelta(seconds=1)


def test_logical_xor_two_operands():
    assert xor(True, False) is True
    assert xor(False, True) is True
    assert xor(True, True) is False
    assert xor(False, False) is False


def test_logical_xor_multiple_operands():
    assert xor(True, False, True) is False
    assert xor(True, True, True) is True
    assert xor(True, False, False) is True
    assert xor(False, True, False) is True
    assert xor(True, False, True, False) is False
    assert xor(True, False, True, False, True, False) is True
