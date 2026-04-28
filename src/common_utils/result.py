from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Self,
    Any,
    Callable,
    Generic,
    TypeVar,
    NoReturn,
    Union,
    overload,
    ParamSpec,
)
from functools import partial
from termcolor import cprint
from .error import is_error, is_error_class, is_error_instance

T = TypeVar("T")
E = TypeVar("E", bound=Exception)
E2 = TypeVar("E2", bound=Exception)
U = TypeVar("U")
P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Result(Generic[T, E]):
    value: T | E = field(default=None)
    metadata: dict[str, Any] | None = field(default=None)
    type: str = field(default="Ok")

    __slots__ = ("_value", "metadata", "type")
    __match_args__ = ("value", "metadata")

    def __init__(
        self,
        value: T | E,
        metadata: dict[str, any] | None = None,
        result_type: str = "Ok",
    ) -> None:
        self.value = value
        self.metadata: dict[str, any] | None = metadata
        self.type: str = result_type

    def __str__(self) -> str:
        return f"{self.type}: {str(self.value)}"

    def __repr__(self) -> str:
        return f"{self.type}:\n<value>:\n\t{str(self.value)}\n<metadata>:\n\t{self.metadata}"

    def __bool__(self) -> bool:
        return self.is_ok()

    def __getitem__(self, metadata_key: str) -> None | any:
        return self.metadata.get(metadata_key) if self.metadata else None

    def __setitem__(self, key: str, value: any) -> None:
        if self.metadata:
            self.metadata[key] = value

    def copy(
        self,
        skip_value: bool = False,
        skip_metadata: bool = False,
    ) -> Result:
        cls = type(self)
        if skip_value and skip_metadata:
            return cls(None, {}, self.type)
        elif skip_value:
            return cls(None, self.metadata, self.type)
        elif skip_metadata:
            return cls(self.value, {}, self.type)
        else:
            return cls(self.value, self.metadata, self.type)

    def is_ok(self) -> bool:
        return self.type == "Ok"

    def is_err(self) -> bool:
        return self.type == "Err"

    def throw(
        self,
        formatter: Callable[[Result], str] | None = None,
    ) -> None:
        if not self.is_err():
            return
        elif formatter:
            raise type(self.value)(formatter(self))
        else:
            raise self.value

    def args(self) -> tuple | None:
        if self.is_err():
            return self.value.args

    def msg(
        self,
        elem: int = 0,
        formatter: Callable[[Result], str] | None = None,
    ) -> str | None:
        if not self.is_err():
            return
        elif formatter:
            return formatter(self)
        elif len(self.value.args) == 0:
            return

        try:
            msg = self.value.args[elem]
            if isinstance(msg, str):
                return msg
        except IndexError:
            return

    def format(
        self,
        s: str | None = None,
        *metadata_keys: str,
        formatter: Callable[[Result], str] | None = None,
    ) -> str:
        if formatter:
            return formatter(self)
        elif s is None:
            return str(self)
        elif len(metadata_keys) == 0:
            return str(self)

        use = {}
        for k, v in self.metadata.items():
            if k in metadata_keys:
                use[k] = v

        return s.format_map(use)

    def print(
        self,
        s: str | None = None,
        *metadata_keys: str,
        formatter: Callable[[Result], str] | None = None,
        err_color: str = "red",
        ok_color: str = "green",
        **print_kwargs,
    ) -> None:
        msg = self.format(s, *metadata_keys, formatter=formatter)
        color = self.is_err() and err_color or ok_color
        cprint(msg, color=color, **print_kwargs)

    def unwrap(
        self,
        pcall: bool = False,
        true: Callable[[T], U] | None = None,
        false: Callable[[E], U] | None = None,
        formatter: Callable[[Result], str] | None = None,
    ) -> T | E:
        if self.is_ok():
            if true:
                return true(self.value)
            else:
                return self.value
        elif not pcall:
            self.throw(formatter=formatter)
        elif false:
            return false(self.value)
        else:
            return self.value

    def and_then(self, f: Callable[[T], Result], *args, **kwargs) -> Result:
        if self.is_err():
            return Err(self.value, self.metadata)

        try:
            return f(self.value, *args, **kwargs)
        except Exception as error:
            return Err(error, self.metadata)

    def map(self, f: Callable[[T], U], *args, **kwargs) -> Result:
        if self.is_err():
            return Err(self.value, self.metadata)
        else:
            try:
                return Ok(f(self.value, *args, **kwargs))
            except Exception as error:
                return Err(error, self.metadata)

    def merge(self, result: Result) -> Result:
        if self.metadata is None:
            return result

        metadata = result.metadata
        metadata = metadata.copy()

        for k, v in self.metadata.items():
            if metadata.get(k) is None:
                metadata[k] = v

        result.metadata = metadata
        return result

    def include(self, result: Result) -> Result:
        if self.metadata is None:
            return self

        metadata = self.metadata
        metadata = metadata.copy()

        for k, v in result.metadata.items():
            if metadata.get(k) is None:
                metadata[k] = v

        self.metadata = metadata
        return self

    def map_err(self, f: Callable[[E], U | E2]) -> T | U:
        if self.is_err():
            return f(self.value)
        else:
            return self.value

    def unwrap_or_else(
        self,
        default_factory: Callable[[E], U],
    ) -> T | U:
        if self.is_err():
            return default_factory(self.value)
        else:
            return self.value

    def unwrap_or(self, default: any) -> T | U:
        if self.is_err():
            return default
        else:
            return self.value


class Err(Result[NoReturn, E]):
    __match_args__ = ("value", "metadata")

    def __init__(
        self,
        value: E,
        metadata: dict[str, any] | None = None,
    ) -> None:
        assert isinstance(value, Exception)
        super().__init__(value, metadata, "Err")


class Ok(Result[T, NoReturn]):
    __match_args__ = ("value", "metadata")

    def __init__(
        self,
        value: T,
        metadata: dict[str, any] | None = None,
    ) -> None:
        assert not isinstance(value, Exception)
        super().__init__(value, metadata, "Ok")


def rsafe(f: Callable) -> Callable[[...], Ok | Err]:
    def function(*args, **kwargs) -> any:
        try:
            result = f(*args, **kwargs)
            t_result = isinstance(result, Result)

            if t_result:
                return result
            elif isinstance(result, Exception):
                return Err(result)
            else:
                return Ok(result)
        except Exception as error:
            return Err(error)

    return function


def rpcall(
    f: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Result[R, Exception]:
    try:
        result = f(*args, **kwargs)
        t_result = isinstance(result, Result)

        if t_result:
            return result
        elif isinstance(result, Exception):
            return Err(result)
        else:
            return Ok(result)
    except Exception as error:
        return Err(error)


def is_result(obj: Any) -> bool:
    return isinstance(obj, Result)


def if_result(obj: Any, f: Callable[[Result], Any]) -> Any:
    if is_result(obj):
        return f(obj)


def is_ok(obj: Any) -> bool:
    return isinstance(obj, Ok)


def if_ok(obj: Any, f: Callable[[Result], Any]) -> Any:
    if is_ok(obj):
        return f(obj)


def if_err(obj: Any, f: Callable[[Result], Any]) -> Any:
    if is_err(obj):
        return f(obj)


def is_err(obj: Any) -> bool:
    return isinstance(obj, Err)


def rifelse(
    obj: Result[T, E],
    true: Callable[[T], U],
    false: Callable[[E], U] = lambda error: error,
    raise_on_error: bool = False,
    formatter: Callable[[Result], str] | None = None,
) -> U:
    if obj.is_ok():
        return true(obj.value)
    elif raise_on_error:
        obj.throw(formatter)
    else:
        return false(obj.value)


def runless(
    obj: Result[T, E],
    false: Callable[[E], U],
    true: Callable[[T], U] = lambda value: value,
) -> U:
    if obj.is_ok():
        return true(obj.value)
    else:
        return false(obj.value)


def runwrap(
    obj: Result,
    f: Callable | None = None,
    default: any = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
    formatter: Callable[[Result], str] | None = None,
) -> any:
    if is_ok(obj):
        return obj.unwrap(pcall=pcall, true=f)
    elif not pcall:
        obj.throw(formatter)
    elif default_factory:
        return obj.unwrap_or_else(default_factory)
    else:
        return obj.unwrap_or(default)


def rthread(
    value: Result[T, E] | T | Exception,
    *fs: Callable[[T], U | Result[U, E] | Exception],
) -> Result[U, E]:
    metadata = {}

    if isinstance(value, Exception):
        return Err(value, {"err_index": -1})
    elif is_err(value):
        return Err(value.value, {"err_index": -1})
    elif is_ok(value):
        metadata = value.metadata.copy()
        value = value.value

    fs = [] if fs is None else fs
    for i, f in enumerate(fs):
        try:
            res = f(value)
            if is_result(res):
                if is_ok(res):
                    metadata.update(res.metadata)
                    value = res.value
                else:
                    return Err(res.value, {"err_index": i, **metadata, **res.metadata})
            elif isinstance(res, Exception):
                return Err(res, {"err_index": i, **metadata})
            else:
                value = res
        except Exception as error:
            return Err(error, {"err_index": i, **metadata})

    if is_result(value):
        return value.include(metadata)
    elif isinstance(value, Exception):
        return Err(value, metadata)
    else:
        return Ok(value, metadata)


__all__ = [
    "E",
    "Err",
    "Ok",
    "Result",
    "T",
    "if_result",
    "if_ok",
    "if_err",
    "is_err",
    "is_ok",
    "is_result",
    "rifelse",
    "runless",
    "rpcall",
    "rsafe",
    "rthread",
    "runless",
    "runwrap",
]
