from dataclasses import dataclass
from typing import Self, Callable, Generic, TypeVar
from functools import partial
from termcolor import cprint

T = TypeVar("T")
Error = TypeVar("Error", bound=Exception)


class Result(Generic[T]):
    __match_args__ = ("value", "metadata")

    def __init__(
        self,
        value: T,
        metadata: dict[str, any] | None = None,
        result_type: str = "Success",
    ) -> None:
        self.value = value
        self.metadata: dict[str, any] = {} if not metadata else metadata
        self.type: str = result_type

        if isinstance(self.value, Exception):
            self.type = "Failure"
        else:
            self.type = "Success"

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

    def err_args(self) -> tuple | None:
        if self.not_ok():
            return self.value.args

    def args(self) -> tuple | None:
        return self.err_args()

    def error_args(self) -> tuple | None:
        return self.err_args()

    def err_message(
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
        elif len(args) > 0 and isinstance(args[0], (str, bytes)) and len(args) == 1:
            if format_with_metadata:
                s = args[0].format(*self.metadata)
            else:
                s = args[0]

        return s

    def error_message(
        self,
        tostr: bool = False,
        format_with_metadata: bool = True,
    ) -> str | None:
        return self.err_message(tostr, format_with_metadata)

    def msg(
        self,
        tostr: bool = False,
        format_with_metadata: bool = True,
    ) -> str | None:
        return self.err_message(tostr, format_with_metadata)

    def message(
        self,
        tostr: bool = False,
        format_with_metadata: bool = True,
    ) -> str | None:
        return self.err_message(tostr, format_with_metadata)

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

    def unwrap(self, pcall: bool = False) -> any:
        if self.not_ok():
            return self.throw(pcall=pcall)
        else:
            return self.value

    def merge_metadata(self, result: Self) -> Self:
        metadata = result.metadata
        metadata = metadata.copy()

        for k, v in self.metadata.items():
            if metadata.get(k) is None:
                metadata[k] = v

        result.metadata = metadata
        return result


    def map_err(self, f: Callable[[Exception, ...], any], *args, **kwargs) -> Self:
        cls = type(self)
        if cls.__name__ != "Result":
            cls = self.__base__

        if self.err():
            try:
                res = f(self.value, *args, **kwargs)
                if isinstance(res, cls):
                    return self.merge_metadata(res)
                else:
                    return self.merge_metadata(cls(res))
            except Exception as error:
                return self.merge_metadata(cls(error))
        else:
            return 

    def map(self, f: Callable[[T, ...], any], *args, **kwargs) -> Self:
        cls = type(self)
        if cls.__name__ != "Result":
            cls = self.__base__

        try:
            res = f(self.value, *args, **kwargs)
            if isinstance(res, cls):
                res = self.merge_metadata(res)
                res.type = "Success"
                return res
            else:
                return cls(res, self.metadata, "Success")
        except Exception as error:
            return cls(error, self.metadata, "Failure")

    def unwrap_or(
        self,
        default: any = None,
        default_factory: Callable[[Error], any] = lambda err: None,
    ) -> Self[Error] | Self[T]:
        if self.err():
            if default is not None:
                return default
            elif default_factory and callable(default_factory):
                return default_factory(self.value)
        else:
            return self.value

    def cbind(self, f: Callable) -> None:
        self.apply.append(f)


class Failure(Result[Error]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: Error, metadata: dict[str, any] | None = None) -> None:
        assert isinstance(value, Exception)
        super().__init__(value, metadata, "Failure")


class Success(Result[T]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: T, metadata: dict[str, any] | None = None) -> None:
        assert not isinstance(value, Exception)
        super().__init__(value, metadata, "Success")


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


def is_result(obj: Result) -> bool:
    return isinstance(obj, Result)


def is_ok(obj: Result) -> bool:
    return isinstance(obj, Success) or is_result(obj) and obj.type == "Success"


def is_err(obj: Result) -> bool:
    return isinstance(obj, Failure) or is_result(obj) and obj.type == "Failure"


def unwrap(
    obj: Result,
    f: Callable | None = None,
    default: any = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> any:
    if is_ok(obj):
        return obj.unwrap()
    elif default is not None:
        return default
    elif default_factory is not None:
        return default_factory()


__all__ = [
    "Error",
    "Failure",
    "Result",
    "Success",
    "T",
    "is_err",
    "is_ok",
    "is_result",
    "safe",
    "unwrap",
]
