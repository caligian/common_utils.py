import os
import re
import argparse
import shlex

from prompt_toolkit.validation import Validator
from typing import Self, Callable

from src.common_utils.cmdline import Argv
from src.common_utils.prompt import Prompt
from src.common_utils.result import Result


class CLINoSuchCommandError(Exception):
    pass


class CLIInvalidArgumentsError(Exception):
    pass


class CLICommand:
    def __init__(self, name: str) -> None:
        self.name = name
        self.children: dict = {}
        self.parser = Argv(prog=name)
        self.parent: Self | None = None

    def on(self, *args, **kwargs):
        return self.parser.on(*args, **kwargs)

    def get(
        self,
        *command: str,
        parent: Self | None = None,
        create: bool = False,
    ) -> None | Self:
        if len(command) == 0:
            return parent

        first = command[0]
        rest = command[1:]
        parent: CLICommand = self if parent is None else parent
        command: CLICommand = parent.children.get(first)

        if command is None:
            if create:
                command = type(self)(first)
                command.parent = parent
                parent.children[first] = command
            else:
                return None
        else:
            return self.get(*rest, parent=command, create=create)

    def __getitem__(self, command: str | tuple) -> None | Self:
        command = [command] if type(command) is str else command
        return self.get(*command)

    def parse(self, args: list[str]) -> argparse.Namespace | SystemExit:
        return self.parser.parse(args, pcall=True)

    def add(self, *command: str) -> Self:
        return self.get(*command, create=True)


class CLIParser:
    def __init__(
        self,
        prompt: str = "%",
        history: str | None = None,
        banner: str | None = None,
    ) -> None:
        self.user_prompt = prompt
        self.history_file = history
        self.banner = banner
        self.commands: dict[str, CLICommand] = {}
        self.prompt: Prompt = Prompt(self.history_file, prompt=prompt)
        self.input = self.prompt.input

    def add_command(self, *name: str, parent=None) -> CLICommand:
        if len(name) == 0:
            return parent

        first = name[0]

        if parent:
            command = parent.children.get(first)
            command = CLICommand(first) if command is None else command
            parent.children[first] = command
            return self.add_command(*name[1:], parent=command)
        else:
            self.commands[first] = CLICommand(first)
            return self.add_command(*name[1:], parent=self.commands[first])

    def strip_args(self, args: list[str]) -> list[str]:
        return [x.lstrip().rstrip() for x in args]

    def get_command_from_args(
        self,
        args: list[str],
        prefix: str = "",
        parse: bool = False,
    ) -> tuple[CLICommand, list[str]] | None:
        till_non_word = -1
        for i, x in enumerate(args):
            if not re.search(r"[a-z_]", x[0]):
                till_non_word = i
                break

        before_non_word = args[:till_non_word]
        after_non_word = args[till_non_word:]
        first = before_non_word[0]
        command = self.commands.get(first)

        if command is None:
            if prefix:
                return Result(
                    False,
                    CLINoSuchCommandError,
                    f"No such command: {first}",
                )
            else:
                return Result(
                    False,
                    CLINoSuchCommandError,
                    f"{prefix}: No such command {first}",
                )

        if len(before_non_word) == 0:
            if parse:
                return self
            else:
                return command.parse(after_non_word)

        subcommand = None
        subcommand_index = -1
        before_non_word = before_non_word[1:]
        index = list(range(1, len(before_non_word) + 1))

        try:
            while i := index.pop():
                subcommand = command.get(*before_non_word[:i])
                if subcommand:
                    subcommand_index = i
                    break
        except IndexError:
            pass

        if subcommand_index == -1:
            return command.parse(after_non_word)
        else:
            args = before_non_word[subcommand_index:]
            args += after_non_word
            return subcommand.parse(args)

    def get(self, *command: str) -> CLICommand | None:
        first = command[0]
        if not self.commands.get(first):
            return None
        else:
            return self.commands[first].get(*command[1:])

    def __getitem__(self, command: str | tuple) -> CLICommand | None:
        command = [command] if isinstance(command, str) else command
        return self.get_command(*command)

    def on(self, command: str | list[str], *args, **kwargs) -> None:
        command = [command] if isinstance(command, str) else command
        command: CLICommand = self.get_command(*command)
        command.on(*args, **kwargs)

    def parse(
        self,
        from_input: bool = False,
        args: list[str] | str | None = None,
        message: str | None = None,
        prompt: str = "%",
        multiline: bool = False,
        default: any = None,
        validator: Validator | str | None = "non_empty",
        apply: Callable[[str], any] = lambda x: x,
    ) -> any:
        assert from_input or args, "Expected either user input or args to be supplied"
        args = (
            from_input
            and self.input(
                message=message,
                prompt=prompt,
                multiline=multiline,
                default=default,
                validator=validator,
                apply=apply,
            )
            or args
        )

        if isinstance(type(args), str):
            args = shlex.split(args)

        assert len(args) > 0, "Empty input"


cli = CLIParser()
cli.add_command("kaushik", "krunal", "brishti")
res = cli.get_command_from_args(["kaushik", "krunal", "-i", "1", "2"])
print(res.value)
