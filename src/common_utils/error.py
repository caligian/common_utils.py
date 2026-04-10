from __future__ import annotations

import re

from typing import Callable, Sequence, Any, TypeVar, Iterable
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum


T = TypeVar("T")
U = TypeVar("U")

errorArgs = dict[str, Any] | None
errorMessage = str | None
errorType = Exception | type[Exception]
errorFormatter = Callable[[Exception], str] | None,
errorMetadata = dict[str, Any] | None
errorParents = list[type[Exception]]


def to_camel_case(s: str) -> str:
    if "_" in s:
        s: list[str] = s.split("_")
        s = [x.capitalize() for x in s]
        s = [x for x in s if x != ""]

        return ("").join(s)
    elif " " in s:
        return to_camel_case(s.replace(" ", "_"))
    else:
        return s.rstrip().lstrip()


def is_str_like(x: str | bytes) -> bool:
    return isinstance(x, (str, bytes))


def raise_error(
    x: type[Exception] | Exception,
    msg: errorMessage = None,
    args: errorArgs = None,
) -> None:
    if isinstance(x, Exception):
        if msg and args:
            raise type(x)(msg, args)
        elif msg is not None:
            raise type(x)(msg)
        elif args is not None:
            raise type(x)(args)
        else:
            raise type(x)
    elif msg and args:
        raise x(msg, args)
    elif msg is not None:
        raise x(msg)
    elif args is not None:
        raise x(args)
    else:
        raise x


throw = raise_error


def deferror(
    name: str,
    fix_name: bool = True,
    parents: errorParents = None,
    metadata: errorMetadata = None,
    formatter: errorFormatter = None,
    **attributes,
) -> type[Exception]:
    if fix_name:
        name = to_camel_case(name)

    parents = (parents) if not isinstance(parents, Sequence) else parents
    cls = type(name, tuple(parents), metadata)

    if formatter:
        attributes["__str__"] = formatter

    for k, v in attributes.items():
        setattr(cls, k, v)

    return cls


def raise_unless(
    cond: Callable[[], bool] | bool,
    error: errorType,
    msg: errorMessage = None,
    args: str | None = None,
) -> bool:
    ok: bool
    if callable(cond):
        ok = cond()
    else:
        ok = cond

    if not ok:
        throw(error, msg=msg, args=args)

    return True


def raise_when(
    cond: Callable[[], bool] | bool,
    error: errorType,
    msg: errorMessage = None,
    args: str | None = None,
) -> None:
    ok: bool
    if callable(cond):
        ok = cond()
    else:
        ok = cond

    if ok:
        throw(error, msg=msg, args=args)

    return True


raise_if = raise_when


def error_args(error: Exception, msg: bool = True, metadata: bool = True) -> tuple[str, dict[str, Any]] | str | dict[str, Any]:
    args = error.args
    args_len = len(args)

    if args_len == 0:
        return (None, None)
    elif msg and metadata:
        return (msg, metadata)
    elif msg and args_len >= 1:
        return (msg, None)
    elif metadata and args_len  >= 1:
        return (None, metadata)



def set_error_args(
    error: errorType,
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
    error: errorType,
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

ErrorSpec = dict[str, str | dict[str, Any]]


@dataclass
class ErrorGroup(dict[str, Exception]):
    def __init__(self, *specs: ErrorSpec | Exception) -> None:
        self.errors: dict[str, Exception] = {}

        for spec in specs:
            if isinstance(spec, Exception):
                self.errors[spec.name] = spec
                setattr(self, spec.name, spec)
            else:
                self.add(
                    spec["name"],
                    msg=spec.get("msg"),
                    metadata=spec.get("metadata"),
                )

    def __getitem__(self, err: str) -> type[Exception] | None:
        return self.errors.get(err)

    def __setitem__(self, name: str, err: type[Exception]) -> None:
        self.errors[name] = err

    def __iter__(self) -> Iterable[tuple[str, Exception]]:
        yield from self.items()

    def keys(self) -> list[str]:
        return list(self.errors.keys())

    def values(self) -> list[Exception]:
        return list(self.errors.values())

    def items(self) -> dict[str, Exception]:
        return list(self.errors.items())

    def has(self, err: str) -> bool:
        return self.errors.get(err) is not None

    def get(self, err: str) -> Exception | None:
        return self.errors.get(err)

    def set(
        self,
        err: str,
        attr: str,
        value: Any | None = None,
        update: Callable[[Any], Any] | None = None, 
    ) -> bool:
        try:
            use = update() if update else value
            setattr(self.errors[err], attr, use)
            return True
        except (KeyError, AttributeError):
            return False

    def update(
        self,
        err: str, 
        msg: errorMessage = None,
    ) -> None:
        pass

    def add(
        self,
        name: str,
        fix_name: bool = True,
        parents: errorParents = None,
        metadata: errorMetadata = None,
        formatter: errorFormatter = None,
        **attributes,
    ) -> ErrorGroup:
        self.errors[name] = deferror(
            name,
            parents=parents,
            fix_name=fix_name,
            metadata=metadata,
            formatter=formatter,
            **attributes,
        )

        return self


__all__ = [
    "ErrorGroup",
    "error_args",
    "error_class",
    "error_msg",
    "get_error_args",
    "get_error_class",
    "get_error_msg",
    "is_error",
    "is_error_class",
    "is_error_instance",
    "deferror",
    "raise_error",
    "raise_unless",
    "raise_when",
    "set_error_args",
    "set_error_msg",
]
