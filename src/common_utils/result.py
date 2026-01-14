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
        if type(self.message) is str:
            if self.is_error():
                if type(self.value) is type:
                    raise self.value(self.message)
                else:
                    raise self.value
            else:
                raise Exception(self.message.format(**kwargs))
        elif isinstance(self.value, Exception):
            raise self.value

        try:
            if re.search("error|exception", self.value.__name__, re.I):
                raise self.value(self.message.format(**kwargs))
            else:
                raise Exception(self.value)
        except Exception:
            raise Exception(self.value)

    def is_error(self) -> bool:
        if isinstance(self.value, Exception):
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
            message = error.args[0]

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
            self.message = error.args[0]

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
