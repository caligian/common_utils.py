import os
import re
import argparse
import shlex

from prompt_toolkit.validation import Validator
from typing import Self, Callable

from src.common_utils.cmdline import Argv
from src.common_utils.prompt import Prompt
from src.common_utils.result import Result


class CLICommand:
    def __init__(self, name: str) -> None:
        self.name = name
        self.children: dict = {}
        self.parser = Argv(prog=name)

    def get(self, *command: str) -> None | Self:
        final = self.children.get(command[0])
        if final is None:
            return 

        children = final
        for cmd in command[1:-1]:
            final = children.get(cmd)
            if final is None:
                return

        return final.get(command[-1])

    def __getitem__(self, command: str | tuple) -> None | Self:
        command = [command] if type(command) is str else command
        return self.get(*command)

    def parse(self, args: list[str]) -> argparse.Namespace | SystemExit:
        return self.parser.parse(args, pcall=True)

    def add(self, *command: str) -> Self:
        cls = type(self)
        children = self.children
        prev = children

        for name in command[:-1]:
            children = children.get(name)

            if not children:
                children = cls(name)

            prev[name] = children
            prev = children.children

        last = children[command[-1]]
        if not last:
            last = cls(command[-1])

        prev[command[-1]] = last
        return self


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

    def add_command(self, *name: str) -> CLICommand:
        n = name[0]
        command = self.commands.get(n)
        command = CLICommand(n) if type(command) is None else command
        self.commands[n] = command

        if len(name[1:]) > 0:
            return self.add_command(*name[1:])
        else:
            return command

    def get_command_from_args(self, args: list[str]) -> 









    def is_valid_command(self, *command: str) -> bool:
        pass


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
        args = from_input and self.input(
            message=message,
            prompt=prompt,
            multiline=multiline,
            default=default,
            validator=validator,
            apply=apply,
        ) or args

        if isinstance(type(args), str):
            args = shlex.split(args)

        assert len(args) > 0, "Empty input"

        main_command: str = args[0]
        main_command: CLICommand = self.commands[main_command]
        subcommands: list[str] = []

        for arg in args[1:]:
            if args[0] == '-':
                break
            else:
                subcommands.append(arg)


        for sub in subcommands:






        return main_command.parse







command = CLICommand("kaushik")
command.add("krunal", "karun", "brishti")

parser = argparse.ArgumentParser(prog="ask")
sub = parser.add_subparsers(title="with")
sub.add_parser("table")
