from __future__ import annotations

import re

from typing import Callable, Sequence, Any, TypeVar, Iterable
from collections import namedtuple
from dataclasses import dataclass, field
from .types import defined


T = TypeVar("T")
U = TypeVar("U")
ErrorSpec = dict[str, str | dict[str, Any]]

errorCond = Callable[[], bool] | bool | None
errorType = Exception | type[Exception]
errorMessage = str | None
errorFormatter = Callable[[Exception], str] | None
errorMetadata = dict[str, Any] | None
errorParents = list[type[Exception]] | None
errorState = tuple[errorMessage, errorMetadata]


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
    metadata: errorMetadata = None,
) -> None:
    if isinstance(x, Exception):
        if msg and metadata:
            raise type(x)(msg, metadata)
        elif msg is not None:
            raise type(x)(msg, None)
        elif metadata is not None:
            raise type(x)(None, metadata)
        else:
            raise type(x)(None, None)
    elif msg and metadata:
        raise x(msg, metadata)
    elif msg is not None:
        raise x(msg, None)
    elif metadata is not None:
        raise x(None, metadata)
    else:
        raise x


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
    cls = type(name, (parents,), attributes)

    if formatter:
        attributes["__str__"] = formatter

    for k, v in metadata.items():
        setattr(cls, k, v)

    return cls


def raise_unless(
    cond: errorCond,
    error: errorType,
    msg: errorMessage = None,
    metadata: errorMetadata = None,
) -> bool:
    ok: bool
    if callable(cond):
        ok = cond()
    else:
        ok = cond

    if not ok:
        throw(error, msg=msg, metadata=metadata)

    return True


def raise_when(
    cond: errorCond,
    error: errorType,
    msg: errorMessage = None,
    metadata: errorMetadata = None,
) -> None:
    ok: bool
    if callable(cond):
        ok = cond()
    else:
        ok = cond

    if ok:
        throw(error, msg=msg, metadata=metadata)

    return True


raise_if = raise_when


def error_set(
    error: Exception,
    msg: errorMessage = None,
    metadata: errorMetadata = None,
) -> Exception:
    msg_def = defined(msg)
    metadata_def = defined(metadata)
    args = error.args
    args_empty = len(args) == 0
    default_msg = args[0] if not args_empty else None
    default_metadata = args[1] if len(args) >= 1 else None
    use_msg = None
    use_metadata = None

    if msg_def and metadata_def:
        use_msg = msg
        use_metadata = metadata
    elif msg_def:
        use_msg = msg
        use_metadata = default_metadata
    elif metadata_def:
        use_metadata = metadata
        use_msg = default_msg

    error.args = (use_msg, use_metadata)
    return error


def set_error_msg(error: Exception, msg: errorMessage) -> Exception:
    return error_set(error, msg=msg)


def set_error_metadata(error: Exception, metadata: errorMetadata) -> Exception:
    return error_set(error, metadata=metadata)


def error_args(
    error: Exception,
    default_msg: errorMessage = None,
    default_metadata: errorMetadata = None,
) -> errorState:
    args = error.args
    args_len = len(args)
    is_one = args_len == 1
    none = args_len == 0

    if none:
        return (default_msg, None)
    elif is_one:
        current = args[0]
        if isinstance(current, dict):
            return (default_msg, current)
        else:
            return (current, default_metadata)
    elif args[0] is None and args[1] is None:
        return (default_msg, default_metadata)
    elif args[0] is None:
        return (default_msg, None)
    elif args[1] is None:
        return (None, default_metadata)
    else:
        return (default_msg, default_metadata)


def error_msg(error: Exception) -> str | None:
    msg, _ = error_args(error)
    return msg


def error_metadata(error: Exception) -> dict[str, Any] | None:
    _, metadata = error_args(error)
    return metadata


def is_error(
    error: Any,
    cls: bool | None = None,
    instance: bool | None = None,
) -> bool:
    cls = False if cls is None else cls
    instance = True if instance is None else instance

    if cls:
        if type(error) is not type:
            return False
        elif "Error" in error.__name__:
            return True
        else:
            return False
    else:
        if type(error) is type:
            return False
        elif isinstance(error, Exception):
            return True
        else:
            return False


def error_class(error: Exception) -> type[Exception]:
    if is_error(error, cls=True):
        return type(error)
    else:
        return error


throw = raise_error
set_error_args = error_set
get_error_args = error_args
get_error_class = error_class
get_error_msg = error_msg
get_error_metadata = error_metadata


@dataclass
class ErrorGroup(dict[str, Exception]):
    __match_args__ = ("errors",)

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


def is_error_class(x: Any) -> bool:
    return is_error(x, cls=True)


def is_error_instance(x: Any) -> bool:
    return is_error(x, instance=True)


__all__ = [
    "ErrorGroup",
    "deferror",
    "error_args",
    "error_class",
    "error_metadata",
    "error_msg",
    "get_error_args",
    "get_error_class",
    "get_error_metadata",
    "get_error_msg",
    "is_error",
    "is_error_class",
    "is_error_instance",
    "raise_error",
    "raise_unless",
    "raise_when",
    "set_error_args",
    "set_error_metadata",
    "set_error_msg",
]
