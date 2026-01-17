import re

from dataclasses import dataclass, field
from typing import Self, Callable


@dataclass
class Result:
    ok: bool
    value: any
    message: str | None = field(default=None)

    def unwrap(self) -> any:
        if self.is_error():
            raise self.errorf()
        else:
            return self.value

    def errorf(self, **kwargs) -> None:
        def raise_with_message():
            if isinstance(type(self.message), str):
                msg = self.message
                raise self.value(msg.format(**kwargs))
            else:
                raise self.value(dict(message=self.message, kwargs=kwargs))

        if self.is_error():
            raise_with_message()
        elif isinstance(type(self.message), str):
            msg = self.message
            raise Exception(msg.format(**kwargs))
        else:
            raise Exception(dict(message=self.message, value=self.value))

    def is_error(self) -> bool:
        if isinstance(self.value, BaseException):
            return True
        elif isinstance(self.value, Exception):
            return True
        elif type(self.value) is type and re.search(
            "error|exception", self.value.__name__, re.I
        ):
            return True
        else:
            return False

    def map(self, f: Callable[[any], any], pcall: bool = False, **kwargs) -> Self:
        try:
            value = f(self.value)
            return self(True, value, None)
        except Exception as error:
            value = type(error)
            message = None

            if len(error.args) == 1 and isinstance(error.args[0], str):
                message = error.args[0]
            else:
                message = error.args

            if not pcall:
                raise error
            else:
                return self(False, value, message)

    def apply(
        self,
        f: Callable[[any], any],
        *args,
        pcall: bool = False,
        **kwargs,
    ) -> Self:
        try:
            self.value = f(self.value, *args, **kwargs)
            self.ok = True

            return self
        except Exception as error:
            self.ok = False
            self.value = type(error)

            if len(error.args) == 1 and isinstance(error.args[0], str):
                self.message = error.args[0]
            else:
                self.message = error.args

            if not pcall:
                raise error
            else:
                return self

    def when(
        self,
        cond: bool | Callable,
        f: Callable[[any], any],
        *args,
        pcall: bool = False,
        apply: bool = False,
        **kwargs,
    ) -> Self:
        ok = False
        if type(cond) is bool:
            ok = cond
        else:
            ok = cond(self.value)

        if ok and apply:
            return self.apply(f, *args, pcall=pcall, **kwargs)
        elif ok:
            return self.map(f, *args, pcall=pcall, **kwargs)
        elif not pcall:
            self.errorf(**kwargs)
        else:
            return Result(False, self.value, self.message)

    @classmethod
    def from_value(
        cls,
        value: any,
        ok: bool = True,
        message: str | None = None,
    ) -> Self:
        return cls(ok, value, message)

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
