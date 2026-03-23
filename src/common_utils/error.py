import re
from typing import Callable, Sequence


def is_str_like(x: str | bytes) -> bool:
    return isinstance(x, (str, bytes))


def make_error(name: str) -> type[Exception]:
    def snake2camel(name: str) -> str:
        if "_" in name:
            name = name.split("_")
            name = [x.lstrip().rstrip().capitalize() for x in name]
            name = ("").join([x.capitalize() for x in name])
        else:
            return name

        return name

    name = snake2camel(name)
    name = f"{name}Error" if "error" not in name.lower() else name

    return type(name, (Exception,), dict())


def raise_error(
    x: type[Exception] | Exception,
    msg: str | None = None,
) -> None:
    if isinstance(x, Exception):
        if msg is not None:
            raise type(x)(msg)
        else:
            raise x
    elif msg is not None:
        raise x(msg)
    else:
        raise x


def raise_unless(
    cond: Callable[[], bool] | bool,
    error: Exception | type[Exception],
    msg: str | None = None,
) -> None:
    if isinstance(cond, bool):
        if not cond:
            raise_error(error, msg)
    elif not cond():
        raise_error(error, msg)


def raise_when(
    cond: Callable | bool,
    error: Exception | type[Exception],
    msg: str | None = None,
) -> None:
    if isinstance(cond, bool):
        if cond:
            raise_error(error, msg)
    elif cond():
        raise_error(error, msg)


def error_args(error: Exception) -> tuple | None:
    if is_error_instance(error):
        args = error.args
        if len(args) == 0:
            return
        else:
            return args


def set_error_args(
    error: Exception | type[Exception],
    *args: Sequence,
) -> Exception:
    if is_error_instance(error):
        error.args = tuple(args)
        return error
    else:
        return error(*args)


def error_msg(error: Exception) -> str | tuple | None:
    if is_error_instance(error):
        args = error.args
        if len(args) == 0:
            return
        elif len(args) == 1 and isinstance(args[0], (str, bytes)):
            return args[0]
        else:
            return args


def set_error_msg(
    error: Exception | type[Exception],
    msg: str,
) -> Exception:
    if is_error_instance(error):
        args = error.args
        if len(args) == 1 and isinstance(type(args[0]), (str, bytes)):
            args = list(args)
            args[0] = msg
            error.args = tuple(args)
            return error
        else:
            error.args = (msg,)
            return error
    else:
        return error(msg)


def is_error_instance(error: any) -> bool:
    return isinstance(error, Exception)


def is_error(error: any) -> bool:
    return is_error_instance(error) or is_error_class(error)


def is_error_class(error: any) -> bool:
    if is_error_instance(error):
        return False
    elif isinstance(error, type) and issubclass(error, Exception):
        return True
    else:
        return False


def error_class(error: Exception) -> type[Exception]:
    if is_error_instance(error):
        return type(error)
    else:
        return error


get_error_args = error_args
get_error_class = error_class
get_error_msg = error_msg

__all__ = [
    # Make error type
    "make_error",
    #
    # Raise errors
    "raise_when",
    "raise_unless",
    "raise_error",
    #
    # Get error type
    "error_class",
    "get_error_class",
    #
    # Check type | instance
    "is_error",
    "is_error_instance",
    "is_error_class",
    #
    # Error message
    "error_msg",
    "set_error_msg",
    "get_error_msg",
    #
    # Error arguments
    "error_args",
    "get_error_args",
    "set_error_args",
]
