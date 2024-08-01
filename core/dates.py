from datetime import datetime, timedelta, date


def round_date(date, h):
    """
    Round a date to the bracketing hours, by steps of `h` hours
    """
    assert isinstance(h, int)
    day = datetime(date.year, date.month, date.day)
    d0 = day + timedelta(hours=h*(date.hour//h))
    d1 = d0 + timedelta(hours=h)
    return (d0, d1)


def closest(date, h):
    """
    Round a date to the closest hour, by steps of `h` hours
    """
    assert isinstance(h, int)
    d0 = round_date(date, h)[0]
    if date - d0 < timedelta(hours=float(h)/2.):
        return d0
    else:
        return d0 + timedelta(hours=h)


def day_range(date_start: date, date_end: date) -> list[date]:
    """
    Returns a list of days starting from date_start, up to date_end included
    """
    days_list = [date_start + i*timedelta(days=1) for i in range((date_end - date_start).days)]
    return days_list


def time_range(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    """
    Returns a list of datetime starting from start, up to end included
    """
    assert end > start
    dt = (end - start) // step
    datelist = [start + i*step for i in range(dt)]
    if end not in datelist: datelist += [end]
    return datelist