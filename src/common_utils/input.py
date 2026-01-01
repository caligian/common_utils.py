import re

from termcolor import cprint
from typing import Callable
from pyfzf import FzfPrompt

COMMANDS: dict[str, dict[str, list[str] | int]] = {
    "filter": dict(aliases=["f", "/"], nargs=1),
    "select": dict(aliases=["s"], nargs=1),
    "help": dict(aliases=["h"], nargs=0),
    "print": dict(aliases=["p"], nargs=0),
    "quit": dict(aliases=["q"], nargs=0),
    "clear": dict(aliases=["c"], nargs=0),
    "fzf": dict(aliases=["z"], nargs=0),
}

COMMANDS_WITH_ALIASES: list[str] = [
    "fzf",
    "z",
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
    "",
]
ALIASES: dict[str, str] = {
    "z": "fzf",
    "f": "filter",
    "/": "filter",
    "s": "select",
    "q": "quit",
    "h": "help",
    "c": "clear",
    "p": "print",
    "fzf": "fzf",
    "filter": "filter",
    "select": "select",
    "quit": "quit",
    "help": "help",
    "clear": "clear",
    "print": "print",
}
HELP = """Valid commands:
    filter | f <regex>
    Filter using this regular expression

    select | s <index1> [index2] ...
    Select these options specified by index separated by whitespace and return choices

    fzf    | z
    Select items using fzf and return choices

    clear  | c
    clear current filter

    print  | p
    Print selections

    help   | h
    Show this help

    quit   | q
    Quit menu and return None"""

invalid_command_message = "Invalid valid command provided: {cmd}\nValid commands are filter [f, /], select [s], help [h], print [p], quit [q], fzf [z], clear [c]\nInput `help` | `h` to display valid commands"
no_argument_message = "No argument provided for command: {cmd}"
void_command_message = "No arguments are required for this command: {cmd}"
invalid_index_message = "Invalid index provided: {index}"
no_choices_matched = "No choices matched with `{pattern}`"


def no_matched_choices(pattern: str | re.Pattern) -> tuple[bool, str]:
    return (False, no_choices_matched.format(pattern=pattern))


def invalid_command(cmd: str) -> tuple[bool, str]:
    return (False, invalid_command_message.format(cmd=cmd))


def no_argument(cmd: str) -> tuple[bool, str]:
    return (False, no_argument_message.format(cmd=cmd))


def void_command(cmd: str) -> tuple[bool, str]:
    return (False, void_command_message.format(cmd=cmd))


def invalid_index(index: str) -> tuple[bool, str]:
    return (False, invalid_index_message.format(index=index))


def mkmessage(msg: str, cmd: str) -> tuple[bool, str]:
    return (False, msg.format(cmd=cmd))


def default_menu_formatter(
    key: str | int,
    value: any,
    key_width: int = 100,
) -> str:
    key = str(key)
    value = str(value)

    return f"{key:<{key_width}} | {value}"


def max_key_width(xs: list[str]):
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
        return invalid_command(cmd)


def valid_nargs(cmd: str, args: str | None = None) -> tuple[bool, str]:
    if cmd not in ALIASES:
        return invalid_command(cmd)
    else:
        cmd = ALIASES[cmd]

    args = "" if args is None else args
    args = args.lstrip().rstrip()
    is_blank = len(args) == 0

    match valid_command(cmd):
        case (True, _):
            match cmd:
                case x if x in ("filter", "select"):
                    if is_blank:
                        return no_argument(cmd)
                    else:
                        return (True, cmd)
                case _ if not is_blank:
                    return void_command(cmd)
                case _:
                    return (True, cmd)

        case (False, msg):
            return (False, msg)


def parse_select(items: list[str], choices: str) -> tuple[bool, list[str] | str]:
    def parse_choice(items: list[str], choice: str) -> tuple[bool, list[int] | str]:
        s = choice
        if m := re.search(r"^([0-9]+)-([0-9]+)$", s):
            start, end = int(m.group(1)), int(m.group(2))
            return (True, list(range(start, end + 1)))
        elif m := re.search(r"^\^([0-9]+)-([0-9]+)$", s):
            start, end = m.group(1), m.group(2)
            start, end = int(start), int(end)
            ignore = list(range(start, end + 1))
            return (True, [x for x in range(1, len(items) + 1) if x not in ignore])
        elif m := re.search(r"^([0-9]+)$", s):
            return (True, [int(m.group(1))])
        else:
            return invalid_index(choice)

    def parse_choices(
        items: list[str], choices: list[str] | str
    ) -> tuple[bool, list[int] | str]:
        choices = [choices] if not isinstance(choices, (list, tuple)) else choices
        res: list[int] = []
        items_len = len(items)

        for choice in choices:
            result = parse_choice(items, choice)
            match result:
                case (False, msg):
                    return (False, msg)
                case (True, index):
                    for ind in index:
                        if ind < 1 or ind > items_len:
                            return invalid_index(ind)
                        else:
                            res.append(ind - 1)
                case _:
                    raise NotImplementedError

        return (True, res)

    choices = re.split(r"\s+", choices)
    choices = [x.strip() for x in choices]
    choices = [x for x in choices if len(x) > 0]

    if len(choices) == 0:
        return no_argument("select")

    match parse_choices(items, choices):
        case (True, indices):
            return (True, [items[x] for x in indices])
        case (False, msg):
            return (False, msg)


def parse_filter(
    items: list[str],
    pattern: str | re.Pattern,
) -> tuple[bool, list[str] | str]:
    res = [x for x in items if re.search(pattern, x, flags=re.I + re.M)]
    ok = len(res) > 0

    if not ok:
        return no_matched_choices(pattern)
    else:
        return (True, res)


def print_items(items: list[str], key_width: int | None = None) -> None:
    key_width = max_key_width(items) if not key_width else key_width
    for i, x in enumerate(items):
        cprint(f"{i + 1:<{key_width}} |", color="yellow", end=" ")
        cprint(str(x), color="yellow")


def parse_fzf(items: list[str]) -> tuple[bool, list[str]]:
    prompt = FzfPrompt().prompt
    selected = prompt(items, "--multi")

    if len(selected) == 0:
        return (False, "No choices made")
    else:
        return (True, selected)


def parse_input(
    items: list[str],
    input: str,
) -> tuple[bool, tuple[str, list[str]] | str]:
    input = input.lstrip().rstrip()
    input = input.split(" ", maxsplit=1)
    input[0] = input[0].lstrip().rstrip()
    res = valid_nargs(*input)

    match res:
        case (True, cmd):
            match cmd:
                case "select":
                    match parse_select(items, input[1]):
                        case (True, xs):
                            return (True, ("select", xs))
                        case (False, msg):
                            return (False, msg)
                case "filter":
                    match parse_filter(items, input[1]):
                        case (True, items):
                            return (True, ("filter", items))
                        case (False, msg):
                            return (False, msg)
                case "help":
                    return (True, ("help", HELP))
                case "clear":
                    return (True, ("clear", None))
                case "quit":
                    return (True, ("quit", None))
                case "print":
                    print_items(items)
                    return (True, ("print", None))
                case "fzf":
                    match parse_fzf(items):
                        case (True, items):
                            return (True, ("fzf", items))
                        case (False, msg):
                            return (False, msg)
                case cmd:
                    return invalid_command(cmd)
        case (False, msg):
            return (False, msg)


def menu(
    items: list[str],
    history: list[list[str]] | None = None,
    depth: int = 1,
    first: bool = True,
) -> list[str]:
    key_width = max_key_width(items)
    if first:
        print_items(items, key_width=key_width)

    if not history or len(history) == 0:
        history = [items]

    if depth < 0:
        depth = 1

    try:
        inp = input("% ")
        inp = inp.lstrip().rstrip()

        if len(inp) == 0:
            cprint("No input provided", "red")
            return menu(items, history, depth, False)

        user_input = parse_input(items, inp)
        match user_input:
            case (False, msg):
                cprint(msg, "red")
                return menu(items, history, depth, False)
            case (True, ("select", items)):
                return items
            case (True, ("filter", items)):
                history.append(items)
                return menu(items, history, depth + 1, True)
            case (True, ("help", help_str)):
                cprint(help_str, "green")
                return menu(items, history, depth, False)
            case (True, ("print", _)):
                return menu(items, history, depth, False)
            case (True, ("clear", _)):
                history.pop()
                return menu(history[-1], history, depth - 1, True)
            case (True, ("quit", _)):
                return
            case (True, ("fzf", choices)):
                return choices
            case _:
                raise NotImplementedError
    except KeyboardInterrupt as error:
        cprint(str(error), "red")
        return menu(items, history, depth, False)
    except EOFError:
        cprint("Quit (y/n) ? ", "red", end="")
        inp = input()
        inp = inp.lstrip().rstrip()

        if "y" in inp:
            return
        else:
            return menu(items, history, depth, False)


__all__ = ["all"]
