import os
import re
import sys

from argparse import ArgumentParser
from typing import Callable, Self
from termcolor import cprint, COLORS

# from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import FuzzyWordCompleter, WordCompleter, NestedCompleter
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

from src.common_utils.result import Result


class Validators:
    def __init__(self) -> None:
        self.validators: dict[str, Validator] = {}

    def __getitem__(self, name: str) -> Validator | None:
        return self.validators.get(name)

    def add(self, name: str, *fs: Callable[[str], Result]) -> None:
        def snake_to_camelcase(s: str) -> str:
            s = s.split("_")
            s = [x.capitalize() for x in s]
            s = ("_").join(s)

            return s

        def validator(this, document) -> None:
            value = document.text
            res = fs[0](value)

            for f in fs[1:]:
                if not res.ok:
                    raise ValidationError(message=res.message)
                else:
                    res = f(res.value)

            if not res.ok:
                raise ValidationError(message=res.message)

        validator_cls = type(f"{snake_to_camelcase(name)}Validator", (Validator,))
        validator_cls.validate = validator
        self.validators[name] = validator_cls()
        self.completers: dict = {}

    @classmethod
    def with_defaults(cls) -> Self:
        def is_float(x: str) -> Result:
            if re.search(r"^[0-9]+[.][0-9]+$", x, re.I):
                return Result(True)
            else:
                return Result(False, None, "Decimal expected")

        def is_int(x: str) -> Result:
            if re.search(r"^-?[0-9]+[.][0-9]+$", x, re.I):
                return Result(True)
            else:
                return Result(False, None, "Integer expected")

        def is_natural_number(x: str) -> Result:
            if re.search(r"^[0-9]+", x, re.I):
                return Result(True)
            else:
                return Result(False, None, "Natural number expected")

        def is_file(x: str) -> Result:
            if os.path.isfile(x):
                return Result(True)
            else:
                return Result(False, None, f"{x} is not a valid filename")

        def is_dir(x: str) -> Result:
            if os.path.isdir(x):
                return Result(True)
            else:
                return Result(False, None, f"{x} is not a valid directory name")

        def is_path(x: str) -> Result:
            if os.path.exists(x):
                return Result(True)
            else:
                return Result(False, None, f"{x} does not exist on filesystem")

        def is_non_empty(x: str) -> Result:
            if len(x) != 0:
                return Result(True)
            else:
                return Result(False, None, "No input provided")

        def add_validator(obj: Validators, name: str, f: Callable) -> None:
            obj.add(name, is_non_empty, f)

        obj: Validators = cls()
        obj.add("non_empty", is_non_empty)

        add_validator(obj, "path", is_path)
        add_validator(obj, "dir", is_dir)
        add_validator(obj, "file", is_file)
        add_validator(obj, "number", is_float)
        add_validator(obj, "float", is_float)
        add_validator(obj, "int", is_int)
        add_validator(obj, "natural_number", is_natural_number)

        return obj


class Prompt:
    def __init__(
        self,
        history: str | None = None,
        prompt: str = "%",
    ) -> None:
        self.prompt = prompt
        self.history_file = history
        self.history: FileHistory | None

        if self.history_file:
            self.history = FileHistory(self.history_file)
        else:
            self.history = None

        self.session: PromptSession | None = None
        self.init_done = False
        self.validators = Validators.with_defaults()
        self.mkstyle = Style.from_dict
        self.styles = {}
        self.completers: dict = {}
        self.input: Callable | None = None

    def _raise_unless_init(self) -> None:
        if not self.init_ok:
            raise AssertionError("<object>.init(*args, **kwargs) has not been run")

    def init(self, *args, **kwargs) -> None:
        if self.history:
            self.session = PromptSession(*args, history=self.history, **kwargs)
            self.init_done = True
            self.input = self.session.prompt

    def add_validator(self, name: str, *fs: Callable[[str], Result]) -> None:
        self.validators.add(name, *fs)

    def add_completer(
        self,
        *cmd: str,
        children: dict[str, dict | set] | None = None,
    ) -> None:
        cmds = self.completers
        for c in cmd[1:]:
            cmds = cmds[c]

        cmds[cmd[-1]] = children

    def make_completer(self) -> None:
        return NestedCompleter.from_nested_dict(self.completers)

    def input(
        self,
        message: str = "deepseek # ",
        multiline: bool = False,
        default: any = None,
        validator: Validator | str | None = "non_empty",
        apply: Callable[[str], any] = lambda x: x,
        on_eof: Callable = lambda: None,
        on_interrupt: Callable = lambda: None,
    ) -> str | None:
        self._raise_unless_init()

        def get_multiline_message() -> list:
            return [
                ("class:multiline", "[multiline] "),
                ("class:prompt", message),
            ]

        def get_message() -> list:
            return [("class:prompt", message)]

        response: str
        style = self.mkstyle(
            {"frame.border": "#ffffff", "multiline": "#0f52ba", "prompt": "#ff0000"}
        )
        validator = (
            self.validators[validator] if isinstance(validator, str) else validator
        )
        validator = self.validators["non_empty"] if validator is None else validator
        completer = self.make_completer()

        try:
            if multiline:
                response = self.session.prompt(
                    get_multiline_message(),
                    style=style,
                    multiline=True,
                    prompt_continuation=">> ",
                    completer=completer,
                    validator=self.validators[validator],
                    validate_while_typing=True,
                )
            else:
                response = self.session.prompt(
                    get_message(),
                    style=style,
                    completer=completer,
                    validator=self.validators[validator],
                    validate_while_typing=True,
                )
        except KeyboardInterrupt:
            sys.stdout.flush()
            on_interrupt()
            return
        except EOFError:
            sys.stdout.flush()
            on_eof()
            raise EOFError

        response = response.strip()
        if len(response) == 0:
            return None
        else:
            return apply(response)
