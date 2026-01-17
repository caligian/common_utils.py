import os
import re
import sys

from typing import Callable, Self
from termcolor import cprint

# from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.validation import Validator
from prompt_toolkit.styles import Style

from src.common_utils.result import Result


class Validators:
    def __init__(self) -> None:
        self.validators: dict[str, Validator] = {}

    def __getitem__(self, name: str) -> Validator | None:
        return self.validators.get(name)

    def add(
        self,
        name: str,
        *fs: Callable[[str], Result],
        error_message: str | None = None,
        move_cursor_to_end: bool | None = None,
    ) -> None:
        assert error_message
        error_message = error_message is None and "Validation error" or error_message

        if move_cursor_to_end is None:
            move_cursor_to_end = True

        def validator(text) -> bool:
            value = text
            for f in fs:
                value = f(value)
                if not value:
                    return False

            return True

        self.validators[name] = Validator.from_callable(
            validator,
            error_message=error_message,
            move_cursor_to_end=move_cursor_to_end,
        )

    @classmethod
    def with_defaults(cls) -> Self:
        def is_float(x: str) -> bool:
            if re.search(r"^-?[0-9]+[.][0-9]+$", x, re.I):
                return x
            else:
                return False

        def is_int(x: str) -> bool:
            if re.search(r"^-?[0-9]+$", x, re.I):
                return x
            else:
                return False

        def is_natural_number(x: str) -> bool:
            if re.search(r"^[0-9]+", x, re.I):
                return x
            else:
                return False

        def is_number(x: str) -> bool:
            return (is_int(x) or is_float(x)) and x

        def is_file(x: str) -> bool:
            if os.path.isfile(x):
                return x
            else:
                return False

        def is_dir(x: str) -> bool:
            if os.path.isdir(x):
                return x
            else:
                return False

        def is_path(x: str) -> bool:
            if os.path.exists(x):
                return x
            else:
                return False

        def is_non_empty(x: str) -> bool:
            if len(x) > 0:
                return x
            else:
                return False

        def add_validator(
            obj: Validators,
            name: str,
            *f: Callable,
            error_message: str | None = None,
        ) -> None:
            obj.add(name, is_non_empty, *f, error_message=error_message)

        obj: Validators = cls()
        obj.add(
            "non_empty",
            is_non_empty,
            error_message="Input cannot be empty",
        )
        add_validator(
            obj,
            "path",
            is_path,
            error_message="Valid path expected",
        )
        add_validator(
            obj,
            "dir",
            is_dir,
            error_message="Nonexistent directory given",
        )
        add_validator(
            obj,
            "file",
            is_file,
            error_message="Nonexistent filename",
        )
        add_validator(
            obj,
            "number",
            is_number,
            error_message="Decimal number or integer expected",
        )
        add_validator(
            obj,
            "float",
            is_float,
            error_message="Decimal number expected",
        )
        add_validator(
            obj,
            "int",
            is_int,
            error_message="Integer expected",
        )
        add_validator(
            obj,
            "natural_number",
            is_natural_number,
            error_message="Natural number expected",
        )

        return obj


class Prompt:
    def __init__(
        self,
        history: str | None = None,
        prompt: str = "%",
    ) -> None:
        if history is None:
            history = os.path.join(
                os.getenv("HOME"),
                ".local",
                "state",
                "common_utils_default_session.history",
            )

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

    def _raise_unless_init(self) -> None:
        if not self.init_done:
            raise AssertionError("<object>.init(*args, **kwargs) has not been run")

    def init(self, *args, **kwargs) -> None:
        if self.history:
            self.session = PromptSession(*args, history=self.history, **kwargs)
        else:
            self.session = PromptSession(*args, **kwargs)

        self.init_done = True
        return self

    def add_validator(
        self,
        name: str,
        *fs: Callable[[str], Result],
        error_message: str | None = None,
        move_cursor_to_end: bool | None = None,
    ) -> None:
        self.validators.add(
            name,
            *fs,
            error_message=error_message,
            move_cursor_to_end=move_cursor_to_end,
        )

    def add_completer(
        self,
        *cmd: str,
        children: dict[str, dict | set] | None = None,
    ) -> None:
        cmds = self.completers
        for c in cmd[:-1]:
            if not cmds.get(c):
                cmds[c] = {}
            cmds = cmds[c]

        cmds[cmd[-1]] = children

    def make_completer(self) -> None:
        return NestedCompleter.from_nested_dict(self.completers)

    def input(
        self,
        message: str | None = None,
        prompt: str = "%",
        multiline: bool = False,
        default: any = None,
        validator: Validator | str | None = "non_empty",
        apply: Callable[[str], any] = lambda x: x,
    ) -> str | None:
        self._raise_unless_init()

        def get_multiline_message() -> list:
            return [
                ("class:multiline", "[multiline] "),
                ("class:prompt", prompt),
            ]

        def get_message() -> list:
            return [("class:prompt", prompt)]

        prompt: str
        if not message:
            prompt = f"{prompt} "
        else:
            prompt = f"{message} {prompt} "

        response: str
        style = self.mkstyle(
            {"frame.border": "#ffffff", "multiline": "#0f52ba", "prompt": "#ff0000"}
        )
        validator = (
            self.validators[validator] if isinstance(validator, str) else validator
        )
        validator = self.validators["non_empty"] if validator is None else validator
        completer = self.make_completer()

        def get_response() -> str:
            if multiline:
                return self.session.prompt(
                    get_multiline_message(),
                    style=style,
                    multiline=True,
                    prompt_continuation=">> ",
                    completer=completer,
                    validator=validator,
                    validate_while_typing=True,
                )
            else:
                return self.session.prompt(
                    get_message(),
                    style=style,
                    completer=completer,
                    validator=validator,
                    validate_while_typing=True,
                )

        try:
            response = get_response()
        except KeyboardInterrupt:
            sys.stdout.flush()
            cprint("KeyboardInterrupt", "red")

            return self.input(
                prompt=prompt,
                multiline=multiline,
                default=default,
                validator=validator,
                apply=apply,
            )
        except EOFError:
            sys.stdout.flush()
            return

        response = response.strip()
        if len(response) == 0:
            return
        else:
            return apply(response)

