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
            return Result(True, value, None)
        except Exception as error:
            self.value = type(error)
            self.message = error.args[0]

            if not pcall:
                raise error
            else:
                return Result(False, self.value, self.message)

    def when(self, f: Callable[[any], any], pcall: bool = False, **kwargs) -> Self:
        if self.ok:
            return self.map(f, pcall=pcall, **kwargs)
        elif not pcall:
            self.errorf(**kwargs)
        else:
            return Result(False, self.value, self.message)
