from dataclasses import dataclass
from typing import Self, Callable, Generic, TypeVar
from functools import partial
from termcolor import cprint

T = TypeVar("T")
Error = TypeVar("Error", bound=Exception)


class Result(Generic[T]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: T, metadata: dict[str, any] | None = None) -> None:
        self.value = value
        self.metadata: dict[str, any] = {} if not metadata else metadata

    def __str__(self) -> str:
        s = str(self.value)
        if isinstance(self.value, Exception):
            s = s.lstrip().strip()
            s = len(s) == 0 and s or type(self.value).__name__

            if len(self.value.args) == 1 and isinstance(
                self.value.args[0], (str, bytes)
            ):
                s = f"{s}: {self.value.args[0]}"
            else:
                s = f"{s}: {self.value.args}"

            return s
        else:
            return s

    def __getitem__(self, metadata_key: str) -> None | any:
        return self.metadata.get(metadata_key)

    def __setitem__(self, key: str, value: any) -> None:
        self.metadata[key] = value

    def ok(self) -> bool:
        return not isinstance(self.value, Exception)

    def not_ok(self) -> bool:
        return isinstance(self.value, Exception)

    def err(self) -> bool:
        return self.not_ok()

    def is_ok(self) -> bool:
        return self.ok()

    def is_err(self) -> bool:
        return self.not_ok()

    def throw(self, pcall: bool = False, value: T | None = None) -> Exception | None:
        value = self.value if value is None else value
        not_ok = self.not_ok()

        if not_ok:
            if not pcall:
                raise value
            else:
                return value

    def unwrap(self, pcall: bool = False) -> any:
        if self.not_ok():
            return self.throw(pcall=pcall)
        else:
            return self.value

    def merge_metadata(self, result: Self) -> Self:
        metadata = result.metadata
        for k, v in self.metadata.items():
            if metadata.get(k) is None:
                metadata[k] = v
        return result

    def map(self, f: Callable[[any], any], *args, **kwargs) -> Self:
        cls = type(self)
        try:
            res = f(self.value, *args, **kwargs)
            if isinstance(res, cls):
                return self.merge_metadata(res)
            else:
                return cls(res, self.metadata)
        except Exception as error:
            return cls(error, self.metadata)

    def unwrap_or(
        self,
        f: Callable[[Exception], any],
        or_else: Callable[[T], any] = lambda x: x,
    ) -> Self[Error] | Self[T]:
        if self.err():
            return self.map(f)
        else:
            return self.map(or_else)

    def unwrap_and(
        self,
        f: Callable,
        or_else: Callable[[Error], any] = lambda error: error,
    ) -> Self[Error] | Self[T]:
        if self.ok():
            return self.map(f)
        else:
            return self.map(or_else)

    def bind(self, f: Callable) -> None:
        self.apply.append(f)


class Failure(Result[Error]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: Error, metadata: dict[str, any] | None = None) -> None:
        assert isinstance(value, Exception)
        super().__init__(value, metadata)

    def args(self) -> tuple:
        return self.value.args

    def message(
        self,
        tostr: bool = False,
        format_with_metadata: bool = True,
    ) -> str | None:
        args = self.value.args
        s: str | None = None

        if tostr:
            s = str(self.value)
            if format_with_metadata:
                s = args[0].format(*self.metadata)
            else:
                s = args[0]
        elif isinstance(args[0], (list, bytes)) and len(args) == 1:
            if format_with_metadata:
                s = args[0].format(*self.metadata)
            else:
                s = args[0]

        return s

    def print(
        self,
        should_raise: bool = False,
        format_with_metadata: bool = True,
        color: str = "red",
        tostr: bool = False,
    ) -> None:
        if should_raise:
            self.unwrap()

        msg = self.message(tostr=tostr, format_with_metadata=format_with_metadata)
        cprint(msg, color)


class Success(Result[T]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: T, metadata: dict[str, any] | None = None) -> None:
        assert not isinstance(value, Exception)
        super().__init__(value, metadata)


def safe(f: Callable) -> Callable[[...], Success | Failure]:
    def function(*args, **kwargs) -> any:
        try:
            result = f(*args, **kwargs)
            t_result = isinstance(result, Result)

            if t_result:
                return result
            elif isinstance(result, Exception):
                return Failure(result)
            else:
                return Success(result)
        except Exception as error:
            return Failure(error)

    return function


__all__ = ["Success", "Failure", "safe", "Result", "Error", "T"]
