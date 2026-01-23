import re

from sspipe import p, px
from typing import Callable
from dataclasses import dataclass, field
from termcolor import cprint
from pyfzf import FzfPrompt
from src.common_utils.result import Success, Failure, T
from src.common_utils.error import error_message


MenuCommandApplyFunction = Callable[[str], Success[T] | Failure[ValueError]]
MenuCommandCondFunction = Callable[[str], bool | Failure[ValueError]]


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


def parse_range(
    n: int | list[int],
    inp: str,
) -> Success[list[int]] | Failure[InvalidArgumentError]:
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
            error = InvalidArgumentError(
                f"expected {{start_index}}-{{end_index}}, got {inp_}"
            )
            return Failure(error)
        start, end = inp
        start, end = int(inp[0]), int(inp[1])

        if start == 0 or end == 0:
            return Failure(InvalidArgumentError("menu items are not zero-indexed"))

        start = n + start if start < 0 else start
        end = n + end if end < 0 else end

        if start >= end:
            return Failure(
                InvalidArgumentError(
                    f"expected {{start_index}} < {{end_index}}, got start={start}, end={end}"
                )
            )

        index = list(range(start, end + 1))
    else:
        try:
            index = [int(inp)]
            if index[0] == 0:
                return Failure(
                    InvalidArgumentError(
                        "menu items are not zero index while selection"
                    )
                )

        except ValueError:
            return Failure(InvalidArgumentError(f"expected an integer, got {inp}"))

    if exclude:
        return Success(list(filter(lambda x: x not in index, n)))
    else:
        return Success(index)


@dataclass
class MenuCommand:
    name: str
    desc: str
    nargs: str | int = field(default=1)
    aliases: list[str] | None = field(default=None)
    cond: list[MenuCommandApplyFunction] | MenuCommandApplyFunction = field(
        default_factory=lambda: [lambda s=None: True]
    )
    apply: list[MenuCommandCondFunction] | MenuCommandCondFunction = field(
        default_factory=lambda: [lambda s: Success(s)]
    )

    def __post_init__(self) -> None:
        assert (self.nargs in ["+", "*", "?"]) or (type(self.nargs) is int)
        self.help: str = self.make_help()

        if isinstance(self.cond, list):
            self.cond = [self.cond]

        if isinstance(self.apply, list):
            self.apply = [self.apply]

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

    def check(self, x: str | list[str] | None) -> bool | Failure[ValueError]:
        if x is None:
            return Success(x)

        x = Success(x)
        for f in self.apply:
            try:
                match f(x.value):
                    case Failure() as failure:
                        return failure
                    case Success(value):
                        x = value
                    case value:
                        x = value
            except ValueError as error:
                return Failure(error)

    def process(
        self, x: str | list[str] | None = None
    ) -> Success | Failure[ValueError]:
        match self.check(x):
            case Failure() as failure:
                return failure
            case _:
                pass

        if x is None:
            return Success(x)

        x = Success(x)
        for f in self.apply:
            try:
                match f(x.value):
                    case Success() as success:
                        x = success
                    case Failure() as failure:
                        return failure
                    case value:
                        x = value
            except ValueError as error:
                return Failure(error)

        return x

    def parse(
        self,
        user_input: str | None = None,
    ) -> (
        Success
        | Failure[
            VoidCommandError
            | TooManyArgumentsError
            | NotEnoughArgumentsError
            | ValueError
        ]
    ):
        user_input = "" if user_input is None else user_input
        user_input = user_input.lstrip().rstrip()

        match self.nargs:
            case 1:
                return self.process(user_input)
            case 0:
                if user_input != "":
                    return Failure(
                        VoidCommandError(
                            f"Command {self.name} does not accept any arguments"
                        )
                    )
                else:
                    return Success([])
            case nargs:
                user_input = re.split(r"\s+", user_input, flags=re.M)
                is_empty = len(user_input) == 0

                match nargs:
                    case "*":
                        return self.process(user_input)

                    case "?":
                        if len(user_input) > 1:
                            return Failure(
                                TooManyArgumentsError(
                                    f"Expected 0 or 1 argument, got {len(user_input)}"
                                )
                            )
                        else:
                            return self.process(user_input)
                    case "+":
                        if is_empty:
                            return Failure(
                                NotEnoughArgumentsError("Expected at least 1 argument"),
                            )
                        else:
                            return self.process(user_input)
                    case n if type(nargs) is int:
                        if len(user_input) != n:
                            return Failure(
                                NotEnoughArgumentsError(
                                    f"expected {nargs} arguments, got {n}"
                                ),
                            )
                        else:
                            return self.process(user_input)
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
            return Failure(
                NoChoicesMatchedError(f"pattern `{pattern}` did not match any items"),
            )

        self.items = res
        self.history.append(current)

        return Success(res)

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

    def fzf(
        self, pattern: str | re.Pattern | None = None
    ) -> Success[T] | Failure[NoChoicesMatchedError]:
        items = self.items
        if pattern:
            items = [x for x in items if re.search(pattern, x, flags=re.M + re.I)]

        if len(items) == 0:
            return Failure(NoChoicesMatchedError("Nothing selected"))

        prompt = FzfPrompt().prompt
        selected = prompt(items, "--multi")

        if len(selected) == 0:
            return Failure(NoChoicesMatchedError("Nothing selected"))
        else:
            return Success(selected)

    def print(self) -> None:
        items = self.items
        key_width = max_key_width(items)

        for i, x in enumerate(items):
            cprint(f"{i + 1:<{key_width}} |", color="yellow", end=" ")
            cprint(str(x), color="yellow")

    def select(self, *index: str | int) -> Success | Failure:
        choices = [x.strip() for x in index]
        choices = [x for x in choices if len(x) > 0]
        n = list(range(1, len(self.items) + 1))

        if len(choices) == 0:
            return Failure(NotEnoughArgumentsError("no choices provided"))

        selected = set()
        for choice in choices:
            match parse_range(n, choice):
                case Failure() as failure:
                    return failure
                case Success(indices):
                    selected.update(set(indices))

        return Success(list(selected))

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

    def input(self) -> Success[str] | Failure[EOFError]:
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
                return Failure(error)
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
            return Success((self.command_aliases[cmd[0]], ""))
        else:
            return Success((self.command_aliases[cmd[0]], cmd[1]))

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
        if not isinstance(res, Success):
            return

        cmd, args = res.value
        if cmd.name == "quit":
            return

        cmd: MenuCommand
        parsed = cmd.parse(args)

        if isinstance(parsed, Failure):
            cprint(error_message(parsed.value), "red")
            return self.cli(items=items, print_items=False)

        value = parsed.value
        value = [value] if not isinstance(value, (tuple, list)) else value

        match cmd.name:
            case "print":
                self.print()
                return self.cli(print_items=False)
            case "select":
                match self.select(*value):
                    case Failure(error):
                        cprint(error_message(error), "red")
                    case result:
                        return [self.items[index - 1] for index in result.value]
            case "filter":
                match self.filter(*value):
                    case Failure(error):
                        cprint(error_message(error), "red")
                        return self.cli(print_items=False)
                    case _:
                        return self.cli(print_items=True)
            case "fzf":
                match self.fzf(*value):
                    case Failure(error):
                        cprint(error_message(error), "red")
                        return self.cli(print_items=True)
                    case Success(value):
                        return value
            case "help":
                self.print_help()
                return self.cli(print_items=False)
            case "clear":
                self.clear_filter()
                return self.cli(print_items=True)
            case command:
                raise NotImplementedError(command)


menu = Menu(list(map(str, [1, 2, 3, 4, 5])))
menu.cli()
