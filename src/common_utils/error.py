from __future__ import annotations

import re

from typing import Callable, Sequence, Any, TypeVar, Iterable, Literal
from collections import namedtuple
from dataclasses import dataclass, field
from .types import defined, module


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
errorSpec = dict[
    str,
    errorCond
    | errorType
    | errorMessage
    | errorFormatter
    | errorMetadata
    | errorParents
    | errorState,
]


def is_str_like(x: str | bytes) -> bool:
    return isinstance(x, (str, bytes))


def is_bytes(x: str) -> bool:
    return isinstance(x, bytes)


def is_str(x: str) -> bool:
    return isinstance(x, str)


class exception:
    def __init__(_, *__, **___) -> None:
        raise AssertionError("This is a classmodule!")

    @staticmethod
    def new(
        name: str,
        parents: tuple | list[type] | type | None = None,
        **attributes,
    ) -> type[Exception]:
        "WARNING: Do not add new attributes as that can mess up __match_args__ or set __match_args__ already"

        if parents is None:
            parents = tuple()
        elif type(parents) is type:
            parents = (parents,)

        cls = type(name, parents, attributes)
        dunder = set(k for k in attributes.keys() if k[0:2] == "__")

        if "__match_args__" not in dunder:
            attribs = set(attributes.keys())
            attribs = attribs - dunder
            attribs.discard("args")

            cls.__match_args__ = ("args", *tuple(attribs))

        return cls

    @staticmethod
    def new_group(
        name: str,
        *specs: tuple[str, tuple | list[type] | None, dict | None] | type[Exception],
    ) -> module:
        error_module = module(name)
        for spec in specs:
            if type(spec) is type:
                error_module.add_method(spec)
            elif len(spec) != 3:
                raise ValueError(
                    f"Expected tuple[str, tuple | list[type] | None, dict | None], got {spec}"
                )
            else:
                err_name, parents, attribs = spec
                parents = tuple() if parents is None else parents
                attribs = dict() if attribs is None else attribs
                error_module.add_method(exception.new(err_name, parents, **attribs))

        return error_module

    @staticmethod
    def throw(
        x: type[Exception] | Exception,
        msg: errorMessage = None,
        metadata: errorMetadata = None,
        unless: errorCond = None,
        when: errorCond = None,
    ) -> Literal[True] | None:
        def _throw() -> None:
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

        unless_given = unless is not None
        when_given = when is not None
        check = lambda cond: cond() if callable(cond) else cond

        if not unless_given and not when_given:
            _throw()
        elif when_given and check(when):
            _throw()
        elif unless_given and check(unless):
            _throw()
        else:
            return True

    @staticmethod
    def update(
        error: Exception,
        msg: errorMessage = None,
        metadata: errorMetadata = None,
    ) -> tuple:
        msg_def = defined(msg)
        metadata_def = defined(metadata)
        args = error.args
        args_len = len(args)

        if not msg_def and not metadata_def:
            if args_len == 0:
                error.args = (None, None)
            elif args_len == 1:
                error.args = (args[0], None)

            return error.args

        args_msg = args[0] if args_len >= 1 else None
        args_metadata = args[1] if args_len >= 2 else None

        if msg_def and metadata_def:
            error.args = (msg, metadata)
        elif msg_def:
            error.args = (msg, args_metadata)
        elif metadata_def:
            error.args = (args_msg, metadata)

        return error.args

    @staticmethod
    def set_msg(error: Exception, msg: errorMessage) -> Exception:
        return exception.update(error, msg=msg)

    @staticmethod
    def set_metadata(error: Exception, metadata: errorMetadata) -> Exception:
        return exception.update(error, metadata=metadata)

    @staticmethod
    def args(
        error: Exception,
        default_msg: errorMessage = None,
        default_metadata: errorMetadata = None,
    ) -> errorState:
        args = error.args
        args_len = len(args)

        if args_len == 2:
            metadata = args[1]
            if isinstance(metadata, dict):
                return args
            else:
                return (args[0], {"args": (args[1],)})
        elif args_len > 2:
            metadata = args[1]
            if isinstance(metadata, dict):
                metadata = metadata.copy()
                metadata["args"] = args[2:]
                return (args[0], metadata)
            else:
                return (args[0], {"args": args[1:]})
        elif args_len == 0:
            return (None, None)
        else:
            return (args[0], None)

    @staticmethod
    def msg(error: Exception) -> str | None:
        args = exception.args(error)
        if args is None:
            return

        args_len = len(args)
        if args_len >= 1:
            return args[0]

    @staticmethod
    def message(error: Exception) -> str | None:
        return exception.msg(error)

    @staticmethod
    def metadata(error: Exception) -> dict[str, Any] | None:
        args = exception.args(error)
        if args is None:
            return

        args_len = len(args)
        if args_len < 2:
            return

        metadata = args[1]
        if isinstance(metadata, dict):
            return metadata
        else:
            return dict(args=args[1:])

    @staticmethod
    def get_class(error: Exception) -> type[Exception]:
        return type(error)

    @staticmethod
    def type(error: Exception) -> type[Exception]:
        return type(error)

    @staticmethod
    def is_class(error: Any) -> bool:
        if type(error) is type:
            return issubclass(error, (Exception, BaseException))

    @staticmethod
    def is_instance(error: Any) -> bool:
        return isinstance(error, Exception)

    @staticmethod
    def is_object(error: Any) -> bool:
        return exception.is_class(error) or exception.is_instance(error)


__all__ = ["exception"]
