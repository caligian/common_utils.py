import re

from dataclasses import dataclass, field
from typing import Self, Callable, Generic, TypeVar
from src.common_utils.error import (
    is_error,
    error_message,
    error_class,
    is_error_instance,
)

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    ok: bool
    value: T
    message: str | None = field(default=None)

    def __post_init__(self) -> None:
        if is_error_instance(self.value):
            self.message = error_message(self.value)
            self.value = type(self.value)

    def unwrap(self, message: str | None = None) -> any:
        if self.is_error():
            raise self.errorf(message=message)
        else:
            return self.value

    def errorf(self, message: str | None = None) -> None:
        message = message if message else self.message
        str_message = isinstance(message, (str, bytes))

        if self.is_error():
            error = error_class(self.value)
            if str_message:
                raise error(message)
            else:
                raise error(str(message))
        elif str_message:
            raise Exception(message)
        else:
            raise Exception(str(message))

    def is_error(self) -> bool:
        return is_error(self.value)

    def is_ok(self) -> bool:
        return not self.is_error() and self.ok

    def map(
        self,
        f: Callable[[any], any],
        *args,
        pcall: bool = False,
        **kwargs,
    ) -> Self:
        cls = type(self)

        try:
            value = f(self.value, *args, **kwargs)
            return cls(True, value, None)
        except Exception as error:
            value = type(error)
            message = error_message(error)

            if not pcall:
                raise error
            else:
                return self(False, value, message)

    def when(
        self,
        f: Callable[[any], any],
        *args,
        pcall: bool = False,
        message: str | None = None,
        **kwargs,
    ) -> Self:
        ok = self.ok
        message = message if not self.message else self.message

        if ok:
            return self.map(f, *args, pcall=pcall, **kwargs)
        elif not pcall:
            self.errorf(message=message)
        else:
            return Result(False, self.value, message)

    @classmethod
    def Failure(cls, value: any, message: str | None = None) -> Self:
        if not message:
            if is_error(value):
                message = error_message(value)
                value = (
                    type(value)
                    if isinstance(value, (BaseException, Exception))
                    else value
                )

        return cls(False, value, message)

    @classmethod
    def Success(cls, value: any) -> Self:
        return cls(True, value)

    @classmethod
    def bind(cls, f: Callable) -> Callable[[...], Self]:
        def function(*args, **kwargs):
            try:
                value = f(*args, **kwargs)
                return cls(True, value)
            except Exception as error:
                return cls(False, error)

        return function

    @classmethod
    def bind_all(cls, *f: Callable) -> list[Callable[[...], Self]]:
        return [cls.bind(x) for x in f]


result = Result.Failure(AssertionError("some message"))
