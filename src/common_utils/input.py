import re

from sspipe import p, px
from typing import Callable
from dataclasses import dataclass, field
from termcolor import cprint
from pyfzf import FzfPrompt
from src.common_utils.result import Result


class InvalidCommandError(Exception):
    pass


class InvalidIndexError(Exception):
    pass


class NotEnoughArgumentsError(Exception):
    pass


class TooManyArgumentsError(Exception):
    pass


class VoidCommandError(Exception):
    pass


class NoChoicesMatchedError(Exception):
    pass


class InvalidNargsError(Exception):
    pass


class InvalidArgumentError(Exception):
    pass


def default_items_formatter(key: str | int, value: any, key_width: int = 100) -> str:
    key = str(key)
    value = str(value)

    return f"{key:<{key_width}} | {value}"


def max_key_width(xs: list[str]):
    return range(0, len(xs)) | p(list) | p(map, str, px) | p(map, len, px) | p(max)


def parse_range(n: int | list[int], inp: str) -> list[int]:
    inp = inp.lstrip().rstrip()
    exclude = inp[0] == "^"
    inp = exclude and inp[1:] or inp
    is_range = "-" in inp
    index: list[int] = []
    n: list[int] = list(range(1, n + 1)) if type(n) is not list else n

    if is_range:
        inp_ = inp
        inp = inp.split("-")

        try:
            inp = [int(x) for x in inp]
        except ValueError:
            return Result(
                False,
                InvalidArgumentError,
                f"expected {{start_index}}-{{end_index}}, got {inp_}",
            )
        start, end = inp
        start, end = int(inp[0]), int(inp[1])

        if start == 0 or end == 0:
            return Result(
                False,
                InvalidArgumentError,
                "menu items are not zero-indexed",
            )

        start = n + start if start < 0 else start
        end = n + end if end < 0 else end

        if start >= end:
            return Result(
                False,
                InvalidArgumentError,
                f"expected {{start_index}} < {{end_index}}, got start={start}, end={end}",
            )

        index = list(range(start, end + 1))
    else:
        try:
            index = [int(inp)]
            if index[0] == 0:
                return Result(
                    False,
                    InvalidArgumentError,
                    "menu items are not zero index while selection",
                )
        except ValueError:
            return Result(
                False,
                InvalidArgumentError,
                f"expected an integer , got {inp}",
            )

    if exclude:
        return Result(True, list(filter(lambda x: x not in index, n)))
    else:
        return Result(True, index)


@dataclass
class MenuCommand:
    name: str
    desc: str
    nargs: str | int = field(default=1)
    aliases: list[str] | None = field(default=None)
    cond: Callable[[str], bool] = field(default=lambda s=None: True)
    process: Callable[[str], str] = field(default=lambda s=None: s)

    def __post_init__(self) -> None:
        assert (self.nargs in ["+", "*", "?"]) or (type(self.nargs) is int)
        self.help: str = self.make_help()

    def make_help(self) -> str:
        res = [self.name, " "]
        match self.nargs:
            case "+":
                res.append("{arg1} [arg2] [arg3] ...")
            case "*":
                res.append("[arg1] [arg2] [arg3] ...")
            case "?":
                res.append("[arg1]")
            case 0:
                res.append("")
            case 1:
                res.append("{arg1}")
            case _:
                res.append("{arg1}...{arg" + str(self.nargs) + "}")

        res.append("\n")
        res.append(self.desc)

        return ("").join(res)

    def print_help(self) -> None:
        print(self.help)

    def process_value(self, x: str | list[str] | None = None) -> any:
        if x is None:
            return

        functions = [self.process] if type(self.process) is not list else self.process
        for f in functions:
            x = f(x)

        return x

    def parse(self, user_input: str | None = None) -> Result:
        user_input = "" if user_input is None else user_input
        user_input = user_input.lstrip().rstrip()
        ok = self.cond(user_input)

        if not ok:
            raise AssertionError(user_input)

        match self.nargs:
            case 1:
                return Result(True, self.process(user_input))
            case 0:
                if user_input != "":
                    return Result(False, VoidCommandError, self.name)
                else:
                    return Result(True, self.process_value())
            case nargs:
                user_input = re.split(r"\s+", user_input, flags=re.M)
                is_empty = len(user_input) == 0

                match nargs:
                    case "*":
                        return Result(True, self.process_value(user_input))

                    case "?":
                        if len(user_input) > 1:
                            return Result(
                                False,
                                TooManyArgumentsError,
                                f"expected 0 or 1 argument, got {len(user_input)}",
                            )
                        else:
                            return Result(True, user_input)
                    case "+":
                        if is_empty:
                            return Result(
                                False,
                                NotEnoughArgumentsError,
                                "expected at least 1 argument",
                            )
                        else:
                            return Result(True, self.process_value(user_input))
                    case n if type(nargs) is int:
                        if len(user_input) != n:
                            return Result(
                                False,
                                NotEnoughArgumentsError,
                                f"expected {nargs} arguments, got {n}",
                            )
                        else:
                            return Result(True, self.process_value(user_input))
                    case _:
                        raise NotImplementedError(user_input)


class Menu:
    def __init__(self, items: list[str]) -> None:
        self.commands: dict[str, MenuCommand] = {}
        self.items: list[str] = items
        self.items_: list[str] = items
        self.max_key_width = max_key_width(items)
        self.command_aliases: dict[str, MenuCommand] = {}
        self.history = []

        self.on(
            "filter",
            "Filter items by regular expressions",
            aliases=["/", "f"],
            cond=lambda s: len(s) != 0,
            nargs=1,
        )
        self.on(
            "fzf",
            "Select items using fzf",
            aliases=["z", "//"],
            nargs="?",
        )
        self.on(
            "select",
            "Select items by index.\nPrefix index with `^` in order to exclude that index or range of index. Ranges are considered to be end-inclusive\nExample:\n\t^1-9: Exclude indices from 1 till 10\n\t1-9: Select indices from 1 till 10\n\t1 Select the first item\n\t^2: Exclude the second item",
            aliases=["s"],
            cond=lambda s: re.search(r"^[0-9\^ ]+$", s) and re.search("[0-9]", s),
            process=lambda s: list(map(lambda x: x.lstrip().rstrip(), s)),
            nargs="+",
        )
        self.on(
            "filter",
            "Filter items by regular expressions",
            aliases=["/", "f"],
            cond=lambda s: len(s) != 0,
            nargs=1,
        )
        self.on(
            "help",
            "Display this help",
            aliases=["h"],
            nargs=0,
        )
        self.on(
            "clear",
            "Remove the latest filter and revert back to previous set of items",
            aliases=["c"],
            nargs=0,
        )
        self.on(
            "print",
            "Print items",
            aliases=["p"],
            nargs=0,
        )
        self.on(
            "quit",
            "Exit menu and return None",
            aliases=["q"],
            nargs=0,
        )

    def filter(self, pattern: str | re.Pattern) -> None:
        items = self.items
        current = items
        pattern = re.compile(pattern, flags=re.I)
        res = list(filter(pattern.search, items))

        if len(res) == 0:
            return Result(
                False,
                NoChoicesMatchedError,
                f"pattern `{pattern}` did not match any items",
            )

        self.items = res
        self.history.append(current)

        return Result(True, res)

    def clear_filter(self) -> None:
        if len(self.history) == 0:
            return self.items_
        else:
            self.items = self.history.pop()

    def print_help(self) -> None:
        commands = list(self.commands.values())
        for command in commands[:-1]:
            command.print_help()
            print()

        commands[-1].print_help()

    def fzf(self, pattern: str | re.Pattern | None = None) -> Result:
        items = self.items
        if pattern:
            items = [x for x in items if re.search(pattern, x, flags=re.M + re.I)]

        if len(items) == 0:
            return Result(False, NoChoicesMatchedError, "nothing selected")

        prompt = FzfPrompt().prompt
        selected = prompt(items, "--multi")

        if len(selected) == 0:
            return Result(False, NoChoicesMatchedError, "nothing selected")
        else:
            return Result(True, selected)

    def print(self) -> None:
        items = self.items
        key_width = max_key_width(items)

        for i, x in enumerate(items):
            cprint(f"{i + 1:<{key_width}} |", color="yellow", end=" ")
            cprint(str(x), color="yellow")

    def select(self, *index: str | int) -> Result:
        choices = [x.strip() for x in index]
        choices = [x for x in choices if len(x) > 0]
        n = list(range(1, len(self.items) + 1))

        if len(choices) == 0:
            return Result(False, NotEnoughArgumentsError, "no choices provided")

        selected = set()
        for choice in choices:
            match parse_range(n, choice):
                case Result(ok=False) as res:
                    return res
                case Result(ok=True, value=indices):
                    selected.update(set(indices))

        return Result(True, list(selected))

    def on(
        self,
        name: str | MenuCommand,
        desc: str,
        nargs: str | int = 1,
        aliases: list[str] | None = None,
        cond: Callable[[str], bool] = lambda _: True,
        process: Callable[[str], str] = lambda s=None: s,
    ) -> None:
        command = (
            name
            if type(name) is MenuCommand
            else MenuCommand(name, desc, nargs, aliases, cond, process)
        )
        self.commands[command.name] = command
        self.command_aliases[command.name] = command

        for alias in command.aliases:
            self.command_aliases[alias] = command

    def input(self) -> Result:
        inp = ""

        try:
            inp = input("% ")
        except KeyboardInterrupt:
            cprint("^C", "red")
            return self.input()
        except EOFError as error:
            print()
            cprint("Exit (y for yes) % ", color="red", end="")
            should_quit = input()

            if "y" in should_quit:
                return Result(False, error)
            else:
                return self.input()

        inp = inp.lstrip().rstrip()
        if len(inp) == 0:
            cprint("No input provided", "red")
            return self.input()

        cmd = inp.split(maxsplit=1)
        if cmd[0] not in self.command_aliases:
            cprint(f"Invalid command provided: {cmd[0]}", "red")
            cprint("Pass 'help' to display help", "blue")
            return self.input()
        elif len(cmd) == 1:
            return Result(True, (self.command_aliases[cmd[0]], ''))
        else:
            return Result(True, (self.command_aliases[cmd[0]], cmd[1]))

    def pop_history(self) -> list[str]:
        if len(self.history) == 0:
            return self.items_
        else:
            return self.history.pop()

    def cli(
        self,
        items: list[str] | None = None,
        print_items: bool = True,
    ) -> list[str] | None:
        items = self.items if self.items == [] else items
        if print_items:
            self.print()

        res = self.input()
        if not res.ok:
            return

        cmd, args = res.value
        if cmd.name == "quit":
            return

        cmd: MenuCommand
        parsed = cmd.parse(args)

        if parsed.is_error():
            cprint(parsed.message, "red")
            return self.cli(items=items, print_items=False)

        value = parsed.value
        value = [value] if not isinstance(value, (tuple, list)) else value

        match cmd.name:
            case "print":
                self.print()
                return self.cli(print_items=False)
            case "select":
                match self.select(*value):
                    case Result(ok=False, message=message):
                        cprint(message, "red")
                    case result:
                        return [self.items[index - 1] for index in result.value]
            case "filter":
                match self.filter(*value):
                    case Result(ok=False) as result:
                        cprint(result.message, "red")
                        return self.cli(print_items=False)
                    case _:
                        return self.cli(print_items=True)
            case "fzf":
                match self.fzf(*value):
                    case Result(ok=False) as result:
                        cprint(result.message, "red")
                        return self.cli(print_items=True)
                    case result:
                        return result.value
            case "help":
                self.print_help()
                return self.cli(print_items=False)
            case "clear":
                self.clear_filter()
                return self.cli(print_items=True)
            case command:
                raise NotImplementedError(command)
