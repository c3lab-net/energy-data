#!/usr/bin/env python3

from copy import deepcopy
from enum import Enum, IntEnum
from functools import wraps
import logging
import random
from typing import Any, Callable, Sequence, Union
from datetime import datetime, date, timedelta, time
from time import sleep, time
import yaml
import traceback
import psycopg2
import dataclasses
from flask import current_app
from json import JSONEncoder
from flask_caching import Cache
from werkzeug.exceptions import HTTPException


simple_cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache'
})

carbon_data_cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 15*60
})

def load_yaml_data(filepath):
    with open(filepath, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            current_app.logger.fatal('Failed to load YAML data from "%s"' % filepath)
            current_app.logger.fatal(e)
            current_app.logger.fatal(traceback.format_exc())
            return None


class CustomJSONEncoder(JSONEncoder):
    def default(self, o: object) -> Any:
        """This defines serialization for object types that `json` cannot handle by default."""
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, timedelta):
            return (datetime.min + o).time().isoformat()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.value
        raise TypeError("Type %s is not serializable" % type(o))


class DocstringDefaultException(HTTPException):
    """Subclass exception uses docstring as default message."""

    def __init__(self, message=None, *args: object, **kwargs):
        super().__init__(message or self.__doc__, *args, **kwargs)


class PSqlExecuteException(DocstringDefaultException):
    """Unknown database exception"""

    def __init__(self, message=None, *args: object, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.code = 500


class CustomHTTPException(HTTPException):
    def __init__(self, description: str = None, code: int = None) -> None:
        super().__init__(description)
        if code:
            self.code = code


def get_psql_connection(host='/var/run/postgresql/', database="electricity-data") -> psycopg2.extensions.connection:
    """Get a new postgresql connection."""
    try:
        conn = psycopg2.connect(host=host, database=database, user="restapi_ro")
        conn.autocommit = True
        return conn
    except Exception as e:
        current_app.logger.fatal(f"get_psql_connection: {e}")
        current_app.logger.fatal(traceback.format_exc())
        raise PSqlExecuteException("Failed to connect to database.")


def psql_execute_scalar(cursor: psycopg2.extensions.cursor, query: str, args: Sequence[Any] = None) -> Any | None:
    """Execute the psql query and return the first column of first row."""
    try:
        cursor.execute(query, args)
        result = cursor.fetchone()
    except psycopg2.Error as e:
        current_app.logger.error(f'psql_execute_scalar("{query}", {args}): {e}')
        current_app.logger.error(traceback.format_exc())
        raise PSqlExecuteException("Failed to execute SQL query.")
    return result[0] if result is not None else None


def psql_execute_list(cursor: psycopg2.extensions.cursor, query: str,
                      args: Union[Sequence[Any], dict[str, str]] = None) -> list[tuple]:
    """Execute the psql query and return all rows as a list of tuples."""
    try:
        cursor.execute(query, args)
        result = cursor.fetchall()
    except psycopg2.Error as e:
        current_app.logger.error(f'psql_execute_scalar("{query}", {args}): {e}')
        current_app.logger.error(traceback.format_exc())
        raise PSqlExecuteException("Failed to execute SQL query.")
    return result


def get_all_enum_values(enum_type):
    """Get all values of a particular Enum type."""
    return [e.value for e in enum_type]


def round_up(dt: datetime, round_to: timedelta) -> datetime:
    """Round up the given datetime to the specified interval."""
    dt = dt.replace(microsecond=0)
    # datetime.max has tzinfo=None
    total_seconds = (datetime.max - dt.replace(tzinfo=None)).total_seconds()
    remainder_seconds = total_seconds % round_to.total_seconds()
    return dt + timedelta(seconds=remainder_seconds)

def round_down(dt: datetime, round_to: timedelta) -> datetime:
    """Round down the given datetime to the specified interval."""
    dt = dt.replace(microsecond=0)
    # datetime.min has tzinfo=None
    total_seconds = (dt.replace(tzinfo=None) - datetime.min).total_seconds()
    remainder_seconds = total_seconds % round_to.total_seconds()
    return dt - timedelta(seconds=remainder_seconds)


def xor(*args):
    """Logical XOR of boolean values."""
    return sum(map(bool, args)) % 2 == 1


def timedelta_to_time(dt: timedelta) -> time:
    if dt >= timedelta(days=1):
        raise ValueError("time cannot be greater than or equal to a day.")
    return (datetime.min + dt).time()


class UnitPrefix(IntEnum):
    Base = 0
    Kilo = 10
    Mega = 20
    Giga = 30
    Tera = 40

    # def __sub__(self, other):
    #     if isinstance(other, UnitPrefix):
    #         return self.value - other.value
    #     raise ValueError("Incompatible type for subtraction")


class SizeUnit(IntEnum):
    Bytes = UnitPrefix.Base
    KB = UnitPrefix.Kilo
    MB = UnitPrefix.Mega
    GB = UnitPrefix.Giga
    TB = UnitPrefix.Tera


class RateUnit(IntEnum):
    bps = UnitPrefix.Base
    Kbps = UnitPrefix.Kilo
    Mbps = UnitPrefix.Mega
    Gbps = UnitPrefix.Giga
    Tbps = UnitPrefix.Tera


POWER_BASE = 2
BITS_PER_BYTE = 8


class ValueWithUnit:
    def __init__(self, value=1., unit=UnitPrefix.Base):
        self.value = value
        self.unit = unit

    def __eq__(self, other):
        return type(self) == type(other) and self.absolute_value() == other.absolute_value()

    def __lt__(self, other):
        if type(self) != type(other):
            raise ValueError("Incompatible type for comparison")
        return self.absolute_value() < other.absolute_value()

    def __le__(self, other):
        if type(self) != type(other):
            raise ValueError("Incompatible type for comparison")
        return self.absolute_value() <= other.absolute_value()

    def __gt__(self, other):
        if type(self) != type(other):
            raise ValueError("Incompatible type for comparison")
        return self.absolute_value() > other.absolute_value()

    def __ge__(self, other):
        if type(self) != type(other):
            raise ValueError("Incompatible type for comparison")
        return self.absolute_value() >= other.absolute_value()

    def __hash__(self):
        return hash(self.absolute_value())

    def _add_or_sub(self, other, sign=1):
        if type(self) != type(other):
            raise ValueError("Incompatible type for subtraction")
        smaller_unit = min(self.unit, other.unit)
        copy = deepcopy(self)
        if copy.unit > smaller_unit:
            unit_diff = copy.unit - other.unit
            copy.unit = smaller_unit
            copy.value *= pow(POWER_BASE, unit_diff)
            copy.value += sign * other.value
        else:
            unit_diff = other.unit - copy.unit
            copy.value += sign * other.value * pow(POWER_BASE, unit_diff)
        return copy

    def __add__(self, other):
        return self._add_or_sub(other, 1)

    def __sub__(self, other):
        return self._add_or_sub(other, -1)

    def __mul__(self, other):
        # Serial multiplication yields a value of the same type
        if isinstance(other, (int, float)):
            copy = deepcopy(self)
            copy.value *= other
            return copy
        raise ValueError("Incompatible type for multiplication")

    def __truediv__(self, other):
        # Serial division yields a value of the same type
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Cannot divide by zero")
            copy = deepcopy(self)
            copy.value /= other
            return copy
        # Same type division yields a singular value
        if type(self) != type(other) or not isinstance(other, ValueWithUnit):
            raise ValueError("Incompatible type for division")
        if other.value == 0:
            raise ValueError("Cannot divide by zero")
        return self.value / other.value * pow(POWER_BASE, self.unit - other.unit)

    def absolute_value(self):
        return self.value * pow(POWER_BASE, self.unit.value)

    def giga_value(self):
        return self.value * pow(POWER_BASE, self.unit.value - UnitPrefix.Giga.value)


class Size(ValueWithUnit):
    def __init__(self, value=1., unit: SizeUnit = SizeUnit.Bytes):
        super().__init__(value, UnitPrefix(unit))

    def bytes(self):
        return super().absolute_value()

    def gigabytes(self):
        return super().giga_value()

    def __truediv__(self, other):
        if isinstance(other, timedelta):
            value = BITS_PER_BYTE * self.value / other.total_seconds()
            unit = RateUnit(self.unit.value)
            return Rate(value, unit)
        elif isinstance(other, Rate):
            value = BITS_PER_BYTE * self.value / other.value
            unit = self.unit.value - other.unit.value
            return timedelta(seconds=value * pow(POWER_BASE, unit))
        else:
            return super().__truediv__(other)


class Rate(ValueWithUnit):
    def __init__(self, value=1., unit: RateUnit = RateUnit.bps):
        super().__init__(value, UnitPrefix(unit))

    def bps(self):
        return super().absolute_value()

    def gbps(self):
        return super().giga_value()

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return super().__mul__(other)
        if not isinstance(other, timedelta):
            raise ValueError("Can only bandwidth multiply with timedelta")
        value = self.value * other.total_seconds() / BITS_PER_BYTE
        unit = SizeUnit(self.unit.value)
        return Size(value, unit)

    def __str__(self) -> str:
        return f'{self.value} {self.unit.name}bps'

    # def __truediv__(self, other):
    #     if not isinstance(other, Rate):
    #         raise ValueError("Can only Rate divide with another Rate")
    #     if other.value == 0:
    #         raise ValueError("Cannot divide by zero")
    #     return self.value / other.value

def dict_min_key(d: dict, sort_key):
    return min(d.items(), key=sort_key)[0]

def exponential_backoff(max_retries: int = 3,
                        base_delay_s: int = 1,
                        should_retry: Callable[[Exception], bool] = lambda _: True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    result_func = func(*args, **kwargs)
                    return result_func
                except Exception as ex:
                    current_app.logger.debug(f"Attempt {retries + 1} failed: {ex}")
                    if retries < max_retries and should_retry(ex):
                        delay = (base_delay_s * 2 ** retries + random.uniform(0, 1))
                        current_app.logger.debug(f"Retrying in {delay:.2f} seconds...")
                        sleep(delay)
                        retries += 1
                    else:
                        raise
        return wrapper
    return decorator

LOGGING_LEVEL_RUNTIME = 25
logging.addLevelName(25, "RUNTIME")

def log_runtime(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        t_start = time()
        result = f(*args, **kwargs)
        t_end = time()


        def _modified_str(o):
            if any([isinstance(o, t) for t in (list, dict, set)]):
                return o if len(o) < 3 else f"{type(o)}[{len(o)}]"
            else:
                return o
        modified_args = [_modified_str(arg) for arg in args]
        modified_kwargs = {key: _modified_str(value) for key, value in kwargs.items()}
        current_app.logger.log(LOGGING_LEVEL_RUNTIME,
                               'func:%s.%s args:[%s, %s] took: %2.4f sec',
                               f.__module__, f.__name__, modified_args, modified_kwargs, t_end-t_start)
        return result
    return wrap
