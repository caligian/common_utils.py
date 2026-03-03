from dataclasses import dataclass
from typing import Self, Callable, Generic, TypeVar
from src.common_utils.error import is_error_instance
from functools import partial

T = TypeVar("T")
Error = TypeVar("Error", bound=Exception)


@dataclass
class UnwrapError(Exception):
    __match_args__ = ("value",)

    def __init__(self, value: Exception) -> None:
        self.value = value
        super().__init__(value)

    def throw(self) -> None:
        raise self

    def unwrap(self, pcall: bool = False) -> None:
        if not pcall:
            self.throw()
        else:
            return self.value


class Result(Generic[T]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: T, metadata: dict[str, any] | None = None) -> None:
        self.value = value
        self.metadata: dict[str, any] = {} if not metadata else metadata

    def __getitem__(self, metadata_key: str) -> None | any:
        return self.metadata.get(metadata_key)

    def __setitem__(self, key: str, value: any) -> None:
        self.metadata[key] = value

    def ok(self) -> bool:
        return not isinstance(self.value, Exception)

    def not_ok(self) -> bool:
        return isinstance(self.value, Exception)

    def throw(self, pcall: bool = False, value: T | None = None) -> UnwrapError | None:
        value = self.value if value is None else value
        not_ok = self.not_ok()

        if not_ok:
            if not pcall:
                raise UnwrapError(value)
            else:
                return UnwrapError(value)

    def check(self, f: Callable, value: T) -> tuple[bool, T]:
        try:
            value = f(value)
            if is_error_instance(value):
                return (False, UnwrapError(value))
            else:
                return (True, value)
        except Exception as error:
            return (False, UnwrapError(error))

    def unwrap(self, pcall: bool = False) -> any:
        if self.not_ok():
            return self.throw(pcall=pcall)
        else:
            return self.value

    def unwrap_or(self, f: Callable, or_else: Callable = lambda x: x) -> T | Error:
        if not self.not_ok():
            return f(UnwrapError(self.value))
        else:
            return or_else(self.value)

    def unwrap_and(self, f: Callable, or_else: Callable = lambda x: x) -> T | Error:
        if self.ok():
            return f(self.value)
        else:
            return or_else(UnwrapError(self.value))

    def when(self, when: Callable, unless: Callable) -> T | Error:
        return self.unwrap_and(when, unless)

    def unless(self, unless: Callable, when: Callable) -> T | Error:
        return self.when(when, unless)

    def bind(self, f: Callable) -> None:
        self.apply.append(f)


class Failure(Result[Error]):
    __match_args__ = ("value", "metadata")

    def __init__(self, value: Error, metadata: dict[str, any] | None = None) -> None:
        assert isinstance(value, Exception)
        super().__init__(value, metadata)


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
            elif isinstance(result, UnwrapError):
                return Failure(result.value)
            elif isinstance(result, Exception):
                return Failure(result)
            else:
                return Success(result)
        except UnwrapError as error:
            return Failure(error.value)
        except Exception as error:
            return Failure(error)

    return function


__all__ = ["Success", "Failure", "UnwrapError", "safe", "Result"]
