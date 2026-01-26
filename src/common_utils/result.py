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
        super().__init__()
        self.error = error
        self.args = (self.error,)

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

    def unwrap(self, pcall: bool = False) -> any:
        value = self.value

        for f in self.apply:
            value = f(value)
            if is_error_instance(value):
                value = Failure(value)
                return value.unwrap(pcall=pcall)

        return value

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
            ok = f(*args, **kwargs)
            t_ok = type(ok)

            if (t_ok is Success) or (t_ok is Failure):
                return ok
            elif isinstance(ok, Exception):
                return Failure(ok)
            else:
                return Success(ok)
        except Exception as error:
            return Failure(error)

    return function


Result = Failure | Success
__all__ = ["Success", "Failure", "UnwrapError", "safe", "Result"]
