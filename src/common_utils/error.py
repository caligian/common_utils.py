import re
from typing import Callable


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


def raise_error(x: type, msg: str | None = None) -> None:
    if is_error(x):
        if type(x) is type:
            if type(msg) is str:
                return x(msg)
            else:
                return x()
        elif isinstance(msg, str):
            raise_error(x, msg)
        else:
            raise_error(type(x), msg)
    elif isinstance(msg, str):
        raise Exception(msg)
    else:
        raise Exception(dict(argument=x, message=msg))


def raise_unless(cond: Callable | bool, error: Exception, message: str) -> None:
    if type(cond) is bool:
        if not cond:
            raise as_error(error)(message)
    elif not cond():
        raise as_error(error)(message)


def raise_when(cond: Callable | bool, error: Exception, message: str) -> None:
    if type(cond) is bool:
        if cond:
            raise as_error(error)(message)
    elif cond():
        raise as_error(error)(message)


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


def as_error(x: type | Exception) -> type:
    ok = is_error(x)
    if not ok:
        raise ValueError(f"Expected an error object, got {str(x)}")

    if isinstance(x, Exception):
        return type(x)
    else:
        return x


def bind_error(
    on_failure: Callable[[Exception, tuple, dict], any] = lambda e, a, k: (e, a, k),
) -> Callable[[...], any]:
    def decorator(f: Callable[[...], any]):
        def function(*args, **kwargs) -> any:
            try:
                value = f(*args, **kwargs)
            except Exception as error:
                return on_failure(as_error(error), args, kwargs)

            return value

        return function

    return decorator


def error_message(self, error: Exception | BaseException) -> str | tuple | None:
    if is_error_instance(error):
        args = error.args
        if len(args) == 1 and isinstance(type(args[0]), (str, bytes)):
            return args[0]
        else:
            return args


def set_error_message(
    self,
    error: Exception | BaseException,
    message: str,
) -> Exception | BaseException:
    if is_error_instance(error):
        args = error.args
        if len(args) == 1 and isinstance(type(args[0]), str):
            args = list(args)
            args[0] = message
            error.args = tuple(args)

            return error


def is_error_instance(error: Exception | BaseException) -> bool:
    return isinstance(error, (BaseException, Exception))


def is_error_type(error: Exception | BaseException) -> bool:
    if is_error_instance(error):
        return True
    elif type(error) is not type:
        return False
    elif re.search(r"error|exception$", error.__name__, flags=re.I):
        return True


def error_class(error: Exception | BaseException) -> type | None:
    if is_error_instance(error):
        return type(error)
    elif is_error_type(error):
        return error
    else:
        return


make_exception = make_error
is_exception = is_error
as_exception = as_error
bind_exception = bind_error
raise_exception = raise_error
set_exception_message = set_error_message
exception_message = error_message
get_exception_message = error_message
get_error_message = error_message
exception_class = error_class
is_exception_type = is_error_type
is_exception_instance = is_error_instance

__all__ = [
    "is_error",
    "as_error",
    "make_error",
    "bind_error",
    "raise_error",
    "is_exception",
    "as_exception",
    "make_exception",
    "bind_exception",
    "raise_exception",
    "raise_when",
    "raise_unless",
    "set_error_message",
    "error_message",
    "get_error_message",
    "set_exception_message",
    "exception_message",
    "get_exception_message",
    "exception_class",
    "is_exception_type",
    "is_exception_instance",
]
