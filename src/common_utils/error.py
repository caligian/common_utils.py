import re
from typing import Callable, Sequence


def is_str_like(x: str | bytes) -> bool:
    return isinstance(x, (str, bytes))


def make_error(name: str) -> Exception:
    def snake2camel(name: str) -> str:
        if "_" in name:
            name = name.split("_")
            name = [x.lstrip().rstrip().capitalize() for x in name]
            name = ("").join(map(str.capitalize, name))
        else:
            return name

        return name

    name = snake2camel(name)
    name = f"{name}Error" if "error" not in name.lower() else name

    return type(name, (Exception,), dict())


def raise_error(
    x: type | Exception | BaseException,
    msg: str | None = None,
) -> None:
    if is_error(x):
        if type(x) is type:
            if is_str_like(msg):
                raise x(msg)
            else:
                raise x()
        elif is_error_instance(x):
            if is_str_like(msg):
                raise set_error_args(x, msg)
            else:
                raise x
    else:
        raise Exception(dict(object=x, message=msg))


def raise_unless(
    cond: Callable | bool,
    error: Exception | BaseException | type,
    message: str | None = None,
) -> None:
    if type(cond) is bool:
        if not cond:
            raise_error(error, msg=message)
    elif not cond():
        raise_error(error, msg=message)


def raise_when(
    cond: Callable | bool,
    error: Exception | BaseException | type,
    message: str | None = None,
) -> None:
    if type(cond) is bool:
        if cond:
            raise_error(error, msg=message)
    elif cond():
        raise_error(error, msg=message)


def is_error(x: any) -> bool:
    if isinstance(x, BaseException):
        return True
    elif isinstance(x, Exception):
        return True
    elif type(x) is type:
        if "Error" in x.__name__:
            return True
        elif "Exception" in x.__name__:
            return True
        else:
            return False
    else:
        return False


def error_args(error: Exception | BaseException) -> tuple | None:
    if is_error_instance(error):
        args = error.args
        if len(args) == 0:
            return
        else:
            return args


def set_error_args(
    error: Exception | BaseException | type,
    *args: Sequence,
) -> Exception | BaseException | type | None:
    if is_error_instance(error):
        error.args = tuple(args)
        return error
    elif is_error_class(error):
        return error(*args)


def error_message(error: Exception | BaseException) -> str | tuple | None:
    if is_error_instance(error):
        args = error.args
        if len(args) == 0:
            return
        elif len(args) == 1 and isinstance(args[0], (str, bytes)):
            return args[0]
        else:
            return args


def set_error_message(
    error: Exception | BaseException | type,
    message: str,
) -> Exception | BaseException:
    if is_error_instance(error):
        args = error.args
        if len(args) == 1 and isinstance(type(args[0]), str):
            args = list(args)
            args[0] = message
            error.args = tuple(args)
            return error
        else:
            error.args = (message,)
            return error
    elif is_error_class(error):
        return set_error_message(error(), message)


def is_error_instance(error: Exception | BaseException) -> bool:
    return isinstance(error, (BaseException, Exception))


def is_error_type(error: Exception | BaseException) -> bool:
    if is_error_instance(error):
        return True
    elif type(error) is not type:
        return False
    elif re.search(r"error|exception$", error.__name__, flags=re.I):
        return True


def is_error_class(error: Exception | BaseException) -> type | None:
    if is_error_instance(error):
        return False
    elif is_error_type(error):
        return error
    else:
        return


def error_class(error: Exception | BaseException) -> type | None:
    if is_error_instance(error):
        return type(error)
    elif is_error_type(error):
        return error
    else:
        return


def as_error(error: Exception | BaseException | type) -> type:
    return error_class(error)


as_exception = as_error
exception_args = error_args
get_error_args = error_args
get_exception_args = error_args
set_exception_args = set_error_args
is_exception_class = is_error_class
get_error_class = error_class
get_exception_class = error_class
make_exception = make_error
is_exception = is_error
raise_exception = raise_error
set_exception_message = set_error_message
exception_message = error_message
get_exception_message = error_message
get_error_message = error_message
exception_class = error_class
is_exception_type = is_error_type
is_exception_instance = is_error_instance

__all__ = [
    # Make error type
    "make_error",
    "make_exception",
    # Raise errors
    "raise_when",
    "raise_unless",
    "raise_error",
    "raise_exception",
    #
    # Get error type
    "as_error",
    "error_class",
    "exception_class",
    "get_exception_class",
    "get_error_class",
    #
    # Check type | instance
    "is_error",
    "is_error_type",
    "is_error_instance",
    "is_error_class",
    "is_exception",
    "is_exception_type",
    "is_exception_instance",
    "is_exception_class",
    #
    # Error message
    "error_message",
    "set_error_message",
    "get_error_message",
    "exception_message",
    "set_exception_message",
    "get_exception_message",
    #
    # Error arguments
    "error_args",
    "get_error_args",
    "exception_args",
    "set_exception_args",
    "get_exception_args",
]
