import re

from termcolor import cprint
from typing import Callable

COMMANDS: dict[str, dict[str, list[str] | int]] = {
    "filter": dict(aliases=["f", "/"], nargs=1),
    "select": dict(aliases=["s"], nargs=1),
    "help": dict(aliases=["h"], nargs=0),
    "print": dict(aliases=["p"], nargs=0),
    "quit": dict(aliases=["q"], nargs=0),
    "clear": dict(aliases=["c"], nargs=0),
}
COMMANDS_WITH_ALIASES: list[str] = [
    "filter",
    "f",
    "/",
    "select",
    "s",
    "help",
    "h",
    "print",
    "p",
    "quit",
    "q",
    "clear",
    "c",
]
ALIASES: dict[str, str] = {
    "f": "filter",
    "/": "filter",
    "s": "select",
    "q": "quit",
    "h": "help",
    "c": "clear",
    "p": "print",
}
invalid_command_message = "Invalid valid command provided: {cmd}\nValid commands are filter [f, /], select [s], help [h], print [p], quit [q], clear [c]\nInput `help` | `h` to display valid commands"
no_argument_message = "No argument provided for command: {cmd}"
void_command_message = "No arguments are required for this command: {cmd}"
invalid_index_message = "Invalid index provided: {index}"


def default_menu_formatter(key: str | int, value: any, key_width: int = 100) -> str:
    key = str(key)
    value = str(value)

    return f"{key:<{key_width}} | {value}"


def calc_max_key_width(xs: list[str]):
    keys = list(range(0, len(xs)))
    keys = map(str, keys)
    keys = map(len, keys)

    return max(keys)


def press_enter_to_continue():
    cprint("Press enter to continue", "blue")
    input()


def invalid_input(msg: str):
    cprint(msg + "\nPress enter to continue", "red")
    input()


def get_nargs(cmd: str) -> int | str:
    return COMMANDS[cmd]["nargs"]


def valid_command(cmd: str) -> tuple[bool, str | None]:
    if cmd in COMMANDS_WITH_ALIASES:
        return (True, None)
    else:
        return (False, invalid_command_message.format(cmd=cmd))


def valid_nargs(cmd: str, args: str | None = None) -> tuple[bool, str]:
    if cmd not in ALIASES:
        return (False, invalid_command_message.format(cmd=cmd))

    args = "" if args is None else args
    args = args.lstrip().rstrip()
    is_blank = len(args) == 0

    match valid_command(cmd):
        case (True, _):
            match cmd:
                case x if x in ("filter", "select"):
                    if is_blank:
                        (False, no_argument_message.format(cmd=cmd))
                    else:
                        return (True, None)
                case _ if not is_blank:
                    return (False, void_command_message.format(cmd=cmd))

        case (False, msg):
            return (False, msg)


def parse_select_args(args: str) -> str | list[int]:
    _args = args
    args = args.lstrip().rstrip()
    args = args.split(" ")
    args = [x.lstrip().rstrip() for x in args]
    is_pattern = False not in [re.search(r"^[0-9]+$", x) is not None for x in args]

    if is_pattern:
        return args
    else:
        return list(map(int, args))


def select_options(
    options: list[str], index: list[int]
) -> tuple[bool, list[str] | str]:
    options_len = len(options)
    res = []

    for a in index:
        a = a + options_len if a < 0 else a
        if a < 0 or a > options_len:
            return (False, invalid_index_message.format(index=str(a)))
        else:
            res.append(options[a])

    return res


def menu(
    xs: Options,
    formatter: Formatter = default_menu_formatter,
    sep: str = r"\n",
    f: Callable[[str | list[str]], any] = lambda x: x,
    prompt: str = ">",
) -> any:
    menu_help = """Valid commands:
        filter | f <regex>
        Filter using this regular expression

        select | s <regex>
        Select all items that match this regex

        select | s <index1> [index2] ...
        Select these options specified by index separated by whitespace

        clear  | c
        clear current filter

        print  | p
        Print selections

        help   | h
        Show this help

        quit   | q
        Quit menu and return None"""
    command_aliases: dict[str, str] = {
        "f": "filter",
        "s": "select",
        "c": "clear",
        "q": "quit",
        "h": "help",
        "filter": "filter",
        "select": "select",
        "clear": "clear",
        "quit": "quit",
        "help": "help",
    }

    def format_options(options: list[str]) -> str:
        pass

    def press_enter():
        cprint("Press enter to continue", "blue")
        input()
        return

    def get_options(pattern: str | list[int], options: list[str]) -> list[str]:
        pass

    def parse(
        s: str,
        multiple: bool = False,
        sep: bool = r"\s+",
        options: list[str] = [],
    ) -> tuple[str, str | list[str]] | None:
        s = re.split(sep, s, maxsplit=1)
        if len(s) == 0:
            cprint("No command provided", "red")
            press_enter()
            return

        cmd = s[0]

        if cmd not in command_aliases:
            cprint(
                "red",
            )
            press_enter()
            return

        cmd = command_aliases[cmd]
        if cmd in ("filter", "select") and len(s) == 1:
            cprint(f"No arguments provided for command {cmd}", "red")
            press_enter()
        elif cmd == "help":
            cprint(menu_help, "white")
            press_enter()
            return

        args = s[1]
        if grep(args, "^[0-9 ]+$"):
            args = re.split(r"\s+", args)
            args = filter(is_int, args)
            args = map(parse_int, args)
            limit = len(options)
            failed = False

            for index in args:
                if index < 1 or index > limit:
                    cprint(f"Invalid index provided: {index}", "red")
                    failed = True

            if failed:
                cprint("Cannot proceed with incorrect indices of options", "red")
                press_enter()
                return
            else:
                return
        else:
            pass


__all__ = [
    "menu",
]
