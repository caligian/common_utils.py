import re

from sspipe import p, px
from typing import Callable, Self
from functools import partial
from dataclasses import dataclass, field
from termcolor import cprint
from pyfzf import FzfPrompt
from .result import Success, Failure, T, safe
from .error import error_message
from .prompt import Prompt


Index = list[int]
Input = list[str] | str
CommandCondition = Callable[
    [str], str | bool | ValueError | Success[str | bool] | Failure[ValueError]
]
CommandMapper = Callable[[any], Success[str] | Failure[ValueError] | str | list[str]]


class Utils:
    @staticmethod
    def condition(**default_kwargs) -> Callable:
        def decorator(f: Callable) -> Callable[[any], bool | ValueError]:
            def function(*args, **kwargs) -> bool | ValueError:
                nonlocal default_kwargs
                default_kwargs = default_kwargs.copy()
                default_kwargs.update(kwargs)
                kwargs = default_kwargs
                ok = f(*args, **kwargs)
                t_ok = type(ok)
                is_bool = t_ok is bool
                is_err = t_ok is ValueError
                is_not_ok = (is_bool and not ok) or (not ok) or is_err

                if is_err:
                    return ok
                elif is_not_ok:
                    return ValueError(f"Assertion error. Function provided: {f}")
                elif ok:
                    return True

            return function

        return decorator

    @staticmethod
    def apply(**default_kwargs) -> Callable:
        def decorator(f: Callable) -> Callable[[any], Success | Failure]:
            def function(*args, **kwargs) -> bool | ValueError:
                nonlocal default_kwargs
                nonlocal f

                default_kwargs = default_kwargs.copy()
                default_kwargs.update(kwargs)
                kwargs = default_kwargs
                f = safe(f)

                return f(*args, **kwargs)

            return function

        return decorator


class Condition:
    @staticmethod
    def index(s: str | list[str]) -> bool:
        s = re.split(r"\s+", s) if type(s) is str else s
        s = [x for x in s if len(x) > 0]

        for string in s:
            exclude = string[0] == "^"
            string = string[1:] if exclude else string

            if string[0] == 0:
                return ValueError("Selection is 1-based, not 0-based")
            elif (
                re.search("^[0-9]+-[0-9]+$", string) or re.search("^[0-9]+$", string)
            ) is not None:
                pass
            else:
                return ValueError(
                    f"Invalid index `{string}`. Valid syntax: ^?[1-9]+-?[1-9]*"
                )

        return True

    @staticmethod
    def non_empty(s: str) -> bool | ValueError:
        if len(s) == 0:
            return ValueError("Input is empty")
        else:
            return True

    @staticmethod
    def natural_number(s: str | list[str]) -> bool | ValueError:
        for string in s:
            ok = re.search(r"^[0-9]+$", string)
            if ok:
                pass
            else:
                return ValueError(f"{string} is not a natural number")

        return True

    @staticmethod
    def integer(s: str | list[int]) -> bool | ValueError:
        for string in s:
            ok = re.search(r"^-?[0-9]+$", string)
            if ok:
                pass
            else:
                return ValueError(f"{string} is not an integer")
        return True


class Map:
    @staticmethod
    def strip(s: str | list[str]) -> Success[str | list[str]]:
        if type(s) is str:
            return Success(s.strip())
        else:
            return Success([x.strip() for x in s])


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
                f"Expected {{start_index}}-{{end_index}} OR {{index}} OR ^{{index}} OR ^{{start_index}}-{{end_index}} where index are natural numbers, got {inp_}"
            )
            return Failure(error)
        start, end = inp
        start, end = int(inp[0]), int(inp[1])

        if start == 0 or end == 0:
            return Failure(InvalidArgumentError(" items are not zero-indexed"))

        start = n + start if start < 0 else start
        end = n + end if end < 0 else end

        if start >= end:
            return Failure(
                InvalidArgumentError(
                    f"Expected {{start_index}} < {{end_index}}, got start={start}, end={end}"
                )
            )

        index = list(range(start, end + 1))
    else:
        try:
            index = [int(inp)]
            if index[0] == 0:
                return Failure(
                    InvalidArgumentError(" items are not zero indexed while selection")
                )
        except ValueError:
            return Failure(InvalidArgumentError(f"Expected an integer, got {inp}"))

    if exclude:
        return Success(list(filter(lambda x: x not in index, n)))
    else:
        return Success(index)


@dataclass
class Command:
    name: str
    desc: str
    nargs: str | int = field(default=1)
    aliases: list[str] | None = field(default=None)
    cond: list[CommandCondition] | CommandCondition = field(default_factory=lambda: [])
    apply: list[CommandMapper] | CommandMapper = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        assert (self.nargs in ["+", "*", "?"]) or (type(self.nargs) is int)
        self.help: str = self.make_help()

        if not isinstance(self.cond, list):
            self.cond = [self.cond]

        if not isinstance(self.apply, list):
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

    def print_help(self, print_nl: bool = True) -> None:
        cprint(self.name + " ", "red", end="")
        match self.nargs:
            case "+":
                cprint("{arg1} [arg2] [arg3] ...", "green")
            case "*":
                cprint("[arg1] [arg2] [arg3] ...", "green")
            case "?":
                cprint("[arg1]", "green")
            case 0:
                cprint("")
            case 1:
                cprint("{arg1}", "green")
            case _:
                cprint("{arg1}...{arg" + str(self.nargs) + "}", "green")

        cprint(self.desc, "white")
        if print_nl:
            print()

    def check(self, x: str | list[str] | None) -> bool | Failure[ValueError]:
        if x is None:
            return Success(x)

        x = Success(x)
        for f in self.cond:
            res = f(x.value)
            try:
                match res:
                    case ValueError() as error:
                        return Failure(error)
                    case Failure(error) as failure:
                        return failure
                    case Success(value) as success:
                        res = success
                    case str(value):
                        res = Success(value)
                    case True:
                        res = Success(x)
                    case failure:
                        raise ValueError(f"Invalid return value: {failure}")
            except ValueError as error:
                return Failure(error)

        return x

    def process(
        self, x: str | list[str] | None = None
    ) -> Success | Failure[ValueError]:
        match self.check(x):
            case Failure() as failure:
                return failure
            case Success(str(value)):
                x = value

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
                    case value if value:
                        x = Success(value)
                    case value:
                        raise ValueError(f"Invalid return value: {value}")
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
    Command = Command
    Utils = Utils

    def __init__(
        self,
        items: list[str],
        prompt_history: str | None = None,
    ) -> None:
        self.commands: dict[str, Command] = {}
        self.items: list[str] = items
        self.items_: list[str] = items
        self.max_key_width = max_key_width(items)
        self.command_aliases: dict[str, Command] = {}
        self.history = []
        self.prompt = Prompt(prompt_history)
        self.prompt.init()
        self.hooks: list[Callable[[], Success[bool] | Failure | Exception]] = []

        self.on(
            "filter",
            "Filter items by regular expressions",
            aliases=["/", "f"],
            cond=Condition.non_empty,
            nargs=1,
        )
        self.on(
            "history",
            "Show history or filter history with regex pattern",
            aliases=["hist", "?"],
            nargs="?",
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
            cond=Condition.index,
            nargs="+",
        )
        self.on(
            "filter",
            "Filter items by regular expressions",
            aliases=["/", "f"],
            cond=Condition.non_empty,
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

    def cmd_filter(self, pattern: str | re.Pattern) -> None:
        items = self.items
        current = items
        pattern = re.compile(pattern, flags=re.I)
        res = list(filter(pattern.search, items))

        if len(res) == 0:
            return Failure(
                NoChoicesMatchedError(
                    f"pattern `{repr(pattern)}` did not match any items"
                ),
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

        commands[-1].print_help(print_nl=False)

    def cmd_fzf(
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

    def cmd_print(self) -> None:
        items = self.items
        key_width = max_key_width(items)

        for i, x in enumerate(items):
            cprint(f"{i + 1:<{key_width}} |", color="yellow", end=" ")
            cprint(str(x), color="yellow")

    def cmd_select(self, *index: str | int) -> Success | Failure:
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
        name: str | Command,
        desc: str,
        nargs: str | int = 1,
        aliases: list[str] | None = None,
        cond: Callable[[str], bool] = lambda _: True,
        process: Callable[[str], str] = lambda s=None: s,
    ) -> None:
        aliases = [] if aliases is None else aliases
        command = (
            name
            if type(name) is Command
            else Command(name, desc, nargs, aliases, cond, process)
        )
        self.commands[command.name] = command
        self.command_aliases[command.name] = command

        for alias in command.aliases:
            self.command_aliases[alias] = command

    def input(self) -> Success[str] | Failure[EOFError]:
        inp = ""

        try:
            inp = self.prompt.input()
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

        inp = "" if not inp else inp
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
            print(cmd)
            return Success((self.command_aliases[cmd[0]], cmd[1]))

    def pop_history(self) -> list[str]:
        if len(self.history) == 0:
            return self.items_
        else:
            return self.history.pop()

    def cmd_help(self, *_) -> Success:
        self.print_help()
        return Success(True)

    def cmd_clear(self, *_) -> Success:
        self.clear_filter()
        return Success(True)

    def add_hook(self, f: Callable[[Self], any]) -> None:
        self.hooks.append(partial(f, self))

    def run_hooks(self) -> Success[bool] | Failure:
        for h in self.hooks:
            try:
                match h():
                    case Failure() as failure:
                        return failure
                    case Success():
                        continue
                    case Exception() as error:
                        return Failure(error)
                    case _:
                        ValueError(
                            "returned value is not any of Failure | Success | Exception "
                        )
            except Exception as error:
                return Failure(error)

        return Success(True)

    def cli(
        self,
        items: list[str] | None = None,
        print_items: bool = True,
    ) -> list[str] | None:
        match self.run_hooks():
            case Success() as success:
                if success["completed"]:
                    return
                else:
                    pass
            case Failure(error):
                cprint(error_message(error), "red")

        items = self.items if self.items == [] else items
        if print_items:
            self.cmd_print()

        res = self.input()
        if not isinstance(res, Success):
            return

        cmd, args = res.value
        if cmd.name == "quit":
            return

        cmd: Command
        parsed = cmd.parse(args)

        if isinstance(parsed, Failure):
            cprint(error_message(parsed.value), "red")
            return self.cli(items=items, print_items=False)

        value = parsed.value
        value = [value] if not isinstance(value, (tuple, list)) else value

        match cmd.name:
            case "print":
                self.cmd_print()
                return self.cli(print_items=False)
            case "select":
                match self.cmd_select(*value):
                    case Failure(error):
                        cprint(error_message(error), "red")
                    case result:
                        return [self.items[index - 1] for index in result.value]
            case "filter":
                match self.cmd_filter(*value):
                    case Failure(error):
                        cprint(error_message(error), "red")
                        return self.cli(print_items=False)
                    case _:
                        return self.cli(print_items=True)
            case "fzf":
                match self.cmd_fzf(*value):
                    case Failure(error):
                        cprint(error_message(error), "red")
                        return self.cli(print_items=True)
                    case Success(value):
                        return value
            case "help":
                self.cmd_help()
                return self.cli(print_items=False)
            case "clear":
                self.cmd_clear()
                return self.cli(print_items=True)
            case other:
                f = getattr(self, f"cmd_{other}")
                assert f, f".cmd_{other}() is not defined in class"

                match f(*value):
                    case Success(value) as success:
                        if success["completed"]:
                            return value
                        else:
                            return self.cli(print_items=success["print_items"])
                    case Failure(error) as failure:
                        cprint(error_message(error), "red")
                        if failure["completed"]:
                            return
                        else:
                            return self.cli(print_items=failure["print_items"])


"""
items = [chr(i) + ' ' + str(66 - i) for i in range(65, 97)]


# Subclass menu instead of adding methods to instances because it is much cleaner
# Adding methods to instances can lead to weird problems as they are unbound
class MyMenu(Menu):
    def cmd_print_value(self, *value):
        print(type(self))
        print(value)
        return Success(True, {"completed": False})

    def cmd_history(self, *value):
        print(self)
        print(self.history)
        return Success(True, {"completed": False})


menu = MyMenu(items)


@menu.add_hook
def _(self):
    print(self.__dict__)
    return Success(True)


# Here you define how the input is parsed on reaching here
menu.on("print_value", "Print all the values", nargs="+")

# Invoking menu CLI but you can manipulate menu state by directly calling cmd_ functions
menu.cli()
"""

menu = Menu

__all__ = ["menu", "Menu"]
