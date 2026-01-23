from typing import Self, Callable, Generic, TypeVar
from src.common_utils.error import is_error_instance

T = TypeVar("T")
Error = TypeVar("Error", bound=Exception)


class UnwrapError(Exception):
    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__(error)

    def throw(self) -> None:
        raise self.error


class Failure(Generic[Error]):
    __match_args__ = ("error",)

    def __init__(self, error: Exception) -> None:
        assert isinstance(error, Exception)
        self.error = error
        super().__init__(error)

    def throw(self) -> None:
        raise self.error

    def unwrap(self, pcall: bool = False) -> Self | UnwrapError:
        if pcall:
            return UnwrapError(self.error)
        else:
            raise self.error


class Success(Generic[T]):
    __match_args__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value
        self.apply: list[Callable] = []

    def unwrap(self, pcall: bool = False) -> Failure | Self:
        cls = type(self)
        value = self.value

        for f in self.apply:
            value = f(value)
            if is_error_instance(value):
                value = Failure(value)
                return value.unwrap(pcall=pcall)

        return cls(value)

    def unwrap_and(self, f: Callable, *args, **kwargs) -> Failure | Self:
        cls = type(self)
        res = f(self.value, *args, **kwargs)

        if is_error_instance(res):
            return Failure(res)
        else:
            return cls(res)

    def bind(self, f: Callable) -> None:
        self.apply.append(f)


def safe(f: Callable) -> Callable[[...], Success | Failure]:
    def function(*args, **kwargs) -> any:
        try:
            return Success(f(*args, **kwargs))
        except Exception as error:
            return Failure(error)

    return function


__all__ = ["Success", "Failure", "UnwrapError", "safe"]
