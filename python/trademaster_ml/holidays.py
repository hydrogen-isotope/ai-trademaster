"""
US stock market (NYSE/Nasdaq) full-day holidays, by rule. Used for forecast dates only (v7 amendment B).

New Year's Day, Martin Luther King Jr. Day, Washington's Birthday, Good Friday, Memorial Day, Juneteenth (from 2022),
Independence Day, Labor Day, Thanksgiving, Christmas. A Saturday holiday is observed the Friday before (except New
Year's Day, which then has no weekday closure); a Sunday holiday the Monday after. Early closes are ordinary sessions.
"""
from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday(y, m, weekday, n):
    d = date(y, m, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y, m, weekday):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return date(y, month, (h + l - 7 * m + 114) % 31 + 1)


def _observed(d, new_year=False):
    if d.weekday() == 5:
        return None if new_year else d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def us_holidays(y):
    out = {
        _observed(date(y, 1, 1), new_year=True),
        _nth_weekday(y, 1, 0, 3),                  # MLK Day
        _nth_weekday(y, 2, 0, 3),                  # Washington's Birthday
        _easter(y) - timedelta(days=2),            # Good Friday
        _last_weekday(y, 5, 0),                    # Memorial Day
        _observed(date(y, 7, 4)),
        _nth_weekday(y, 9, 0, 1),                  # Labor Day
        _nth_weekday(y, 11, 3, 4),                 # Thanksgiving
        _observed(date(y, 12, 25)),
    }
    if y >= 2022:
        out.add(_observed(date(y, 6, 19)))
    out.discard(None)
    return frozenset(out)


def is_session(d):
    return d.weekday() < 5 and d not in us_holidays(d.year)


def next_sessions(after, n):
    """The next n US trading days after `after` (ISO date string), as ISO strings."""
    d, out = date.fromisoformat(after), []
    while len(out) < n:
        d += timedelta(days=1)
        if is_session(d):
            out.append(d.isoformat())
    return out
