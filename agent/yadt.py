from datetime import datetime, timedelta
import math
from pytz import timezone
import pytz
import pandas as pd
from dateutil.parser import parse

tz = timezone('America/Toronto')
tz_utc = timezone('UTC')


def ceil_dt(dt, delta):
    ceil = datetime.min + math.ceil(
        (dt.replace(tzinfo=None) - datetime.min) /
        timedelta(minutes=delta)) * timedelta(minutes=delta)

    return ceil.replace(tzinfo=tz)


def ceil_date(date, **kwargs):
    secs = timedelta(**kwargs).total_seconds()
    return datetime.fromtimestamp(date.timestamp() + secs -
                                  date.timestamp() % secs,
                                  tz=get_tz())


def floor_date(date, **kwargs):
    secs = timedelta(**kwargs).total_seconds()
    return datetime.fromtimestamp(date.timestamp() - date.timestamp() % secs,
                                  tz=get_tz())


def utcnow():
    return datetime.now(tz=tz)


def get_tz():
    return tz


def get_utc_tz():
    return tz_utc


def apply_tz_toronto(dt):
    dt = tz.localize(dt)
    return dt


def utc_to_toronto(dt):
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = tz_utc.localize(dt).astimezone(tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def parse_date(dt, toronto=False):
    parsed = parse(dt)
    if toronto:
        if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
            parsed = tz.localize(parsed)
    else:
        parsed = utc_to_toronto(parsed)

    return parsed


def scan_and_apply_tz(dictio):

    if isinstance(dictio, dict):
        for i in dictio.keys():
            dictio[i] = scan_and_apply_tz(dictio[i])
    if isinstance(dictio, list):
        for idx, i in enumerate(dictio):
            i2 = scan_and_apply_tz(i)
            dictio[idx] = i2

    if isinstance(dictio, datetime):
        dictio2 = utc_to_toronto(dictio)

        return dictio2
    else:
        try:
            dictio2 = parse_date(dictio).isoformat()
        except (ValueError, TypeError):
            return dictio

        return dictio2

    return dictio
