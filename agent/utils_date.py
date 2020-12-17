from datetime import datetime, timedelta
import math
from pytz import timezone
import pytz
import pandas as pd

tz = timezone('America/Montreal')


def ceil_dt(dt, delta):
    ceil = datetime.min + math.ceil(
        (dt.replace(tzinfo=None) - datetime.min) /
        timedelta(minutes=delta)) * timedelta(minutes=delta)

    return ceil.replace(tzinfo=tz)


def utcnow():
    return datetime.now(tz=timezone('America/Montreal'))
