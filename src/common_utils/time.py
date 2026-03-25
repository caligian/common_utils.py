import datetime

date = datetime.date
datetime = datetime.datetime
time = datetime.time


def strptime(
    fmt: str,
    date_str: str,
    *args,
    use: type = datetime,
    **kwargs,
) -> str:
    fn = use.strptime
    return fn(date_str, fmt)


def strftime(
    fmt: str,
    *args,
    use: type = datetime,
    **kwargs,
) -> str:
    return use(*args, **kwargs).strftime(fmt)


__all__ = ["date", "datetime", "time", "strftime", "strptime"]
