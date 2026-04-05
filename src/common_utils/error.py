from __future__ import annotations

import re

from typing import Callable, Sequence, Any, TypeVar
from dataclasses import dataclass, field
from enum import Enum


T = TypeVar("T")


def to_camel_case(s: str) -> str:
    if "_" in s:
        s: list[str] = s.split("_")
        s = [x.capitalize() for x in s]

        return ("").join(s)
    elif " " in s:
        return to_camel_case(s.replace(" ", "_"))
    else:
        return s


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
    args: list | tuple | dict | None = None,
) -> None:
    if isinstance(x, Exception):
        if msg and args:
            raise type(x)((msg, args))
        elif msg is not None:
            raise type(x)(msg)
        elif args is not None:
            raise type(x)(args)
        else:
            raise x
    elif msg and args:
        raise x((msg, args))
    elif msg is not None:
        raise x(msg)
    elif args is not None:
        raise x(args)
    else:
        raise x


def raise_unless(
    cond: Callable[[], bool] | bool,
    error: Exception | type[Exception],
    msg: str | None = None,
    args: str | None = None,
) -> None:
    match callable(cond):
        case True:
            raise_error(error, msg, args)
        case False:
            if not cond:
                raise_error(error, msg, args)


def raise_when(
    cond: Callable[[], bool] | bool,
    error: Exception | type[Exception],
    msg: str | None = None,
    args: str | None = None,
) -> None:
    match callable(cond):
        case True:
            raise_error(error, msg, args)
        case _:
            if not cond:
                raise_error(error, msg, args)


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

ErrorSpec = dict[str, dict[str, Any] | Callable[["Error"], str] | str | None]


@dataclass
class ErrorGroup(dict):
    def __getitem__(self, err: str) -> Error | None:
        return self.errors.get(err)

    def __setitem__(self, err: str, spec: ErrorSpec | Error) -> None:
        if isinstance(spec, dict):
            self.errors[err] = Error(**spec)
        else:
            self.errors[err] = spec

    def __init__(
        self,
        name: str | None = None,
        *specs: ErrorSpec | Error,
    ) -> None:
        self.name = name if isinstance(name, str) else self.parent.__name__
        self.errors: dict[str, Error] = {}

        for spec in specs:
            if isinstance(spec, Error):
                self.errors[spec.name] = spec
                setattr(self, spec.name, spec)
            else:
                self.add(
                    spec["name"],
                    msg=spec.get("name"),
                    metadata=spec.get("metadata"),
                    formatter=spec.get("formatter"),
                )

    def add(
        self,
        name: str,
        msg: str | None = None,
        metadata: dict[str | int, Any] | None = None,
        formatter: Callable[[Error], str] | None = None,
    ) -> ErrorGroup:
        self.errors[name] = Error.new(
            name,
            msg=msg,
            metadata=metadata,
            formatter=formatter,
        )
        setattr(self, name, self.errors[name])
        return self


@dataclass
class Error(Exception):
    def __init__(
        self,
        msg: str | None = None,
        metadata: dict[str | int, Any] | None = None,
        formatter: Callable[[Error], str] | None = None,
    ) -> None:
        self.msg = msg if msg is not None else self.msg
        self.metadata = metadata if metadata is not None else self.metadata
        self.formatter = formatter if formatter is not None else self.formatter
        self.parent = type(self)
        self.name = self.parent.__name__

        super().__init__(self.msg, self.metadata)

    def __getitem__(self, key: str) -> Any | None:
        if self.metadata:
            return self.metadata.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if self.metadata is None:
            self.metadata = {key: value}
        else:
            self.metadata[key] = value

    def throw(
        self,
        *args,
        formatter: Callable[[Error], str] | None = None,
        **kwargs,
    ) -> None:
        raise self.parent(self.format(*args, formatter=formatter, **kwargs))

    def throwf(
        self,
        *args,
        formatter: Callable[[Error], str] | None = None,
        **kwargs,
    ) -> None:
        self.throw(*args, formatter=formatter, **kwargs)

    def map(self, f: Callable[[Error], T], *args, **kwargs) -> T:
        return f(self, *args, **kwargs)

    def map_unless(
        self,
        cond: bool | Callable,
        f: Callable[[Error], T],
        cond_args: list | None = None,
        cond_kwargs: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> T | None:
        cond_args = [] if cond_args is None else cond_args
        cond_kwargs = [] if cond_kwargs is None else cond_kwargs
        ok = cond if isinstance(cond, bool) else cond(*cond_args, **cond_kwargs)

        if not ok:
            return f(self, *args, **kwargs)

    def map_when(
        self,
        cond: bool | Callable,
        f: Callable[[Error], T],
        cond_args: list | None = None,
        cond_kwargs: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> T | None:
        cond_args = [] if cond_args is None else cond_args
        cond_kwargs = [] if cond_kwargs is None else cond_kwargs
        ok = cond if isinstance(cond, bool) else cond(*cond_args, **cond_kwargs)

        if ok:
            return f(self, *args, **kwargs)

    def throw_when(
        self,
        cond: bool | Callable,
        formatter: Callable[[Error], str] | None = None,
        format: bool = False,
        format_args: list[Any] | None = None,
        format_kwargs: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        format_args = format_args if isinstance(format_args, list) else []
        format_kwargs = format_kwargs if isinstance(format_kwargs, dict) else {}

        if isinstance(cond, bool):
            if cond:
                if format:
                    self.throwf(*format_args, formatter=formatter, **format_kwargs)
                else:
                    self.throw()
        elif cond(*args, **kwargs):
            if format:
                self.throwf(*format_args, formatter=formatter, **format_kwargs)
            else:
                self.throw()

    def throw_unless(
        self,
        cond: bool | Callable,
        formatter: Callable[[Error], str] | None = None,
        format: bool = False,
        format_args: list[Any] | None = None,
        format_kwargs: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(cond, bool):
            self.throw_when(
                not cond,
                formatter=formatter,
                format=format,
                format_args=format_args,
                format_kwargs=format_kwargs,
                *args,
                **kwargs,
            )
        else:
            self.throw_when(
                lambda *_args, **_kwargs: not cond(*_args, **_kwargs),
                formatter=formatter,
                format=format,
                format_args=format_args,
                format_kwargs=format_kwargs,
                *args,
                **kwargs,
            )

    def format(
        self,
        *args,
        formatter: Callable[[Error], str] | None = None,
        **kwargs,
    ) -> None:
        formatter = formatter if formatter else self.formatter
        if formatter:
            return formatter(self, *args, **kwargs)
        elif self.metadata is not None and self.msg is not None:
            return f"{(self.msg, self.metadata)}"
        elif self.msg:
            return self.msg
        elif self.metadata:
            return self.metadata
        else:
            return ""

    def __str__(self) -> None:
        return self.format()

    @classmethod
    def as_error(
        cls,
        obj: Exception | type[Exception],
        msg: str | None = None,
        metadata: dict[str, Any] | None = None,
        formatter: Callable[[Error], str] | None = None,
    ) -> Error | type[Error]:
        if isinstance(obj, Error):
            return obj
        elif isinstance(obj, Exception):
            args = obj.args
            args_len = len(args)
            not_empty = args_len > 0
            more_than_one = args_len > 1
            msg = args[0] if not_empty else None
            metadata = (
                args[1] if (more_than_one and isinstance(args[1], dict)) else metadata
            )

            return Error.new(
                type(obj).__name__,
                msg=msg,
                metadata=metadata,
                formatter=formatter,
            )
        else:
            return Error.new(
                obj.__name__,
                msg=msg,
                metadata=metadata,
                formatter=formatter,
            )

    @classmethod
    def new(
        cls,
        name: str,
        msg: str | None = None,
        metadata: dict[str, Any] | None = None,
        formatter: Callable[[Error], str] | None = None,
    ) -> type[Error]:
        cls = type(
            to_camel_case(name),
            (cls,),
            dict(
                name=name,
                msg=msg,
                metadata=metadata,
                formatter=formatter,
            ),
        )

        return cls


err = Error.as_error(Exception("hello"))
err().throwf()

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
    "Error",
    "ErrorGroup",
]
