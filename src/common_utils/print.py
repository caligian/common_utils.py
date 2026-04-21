from termcolor import cprint
from pprint import pprint, pp


def message(*s: str, color: str = "blue", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_err(*s: str, color: str = "red", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_ok(*s: str, color: str = "green", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_warn(*s: str, color: str = "yellow", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


__all__ = ["message", "msg_err", "msg_ok", "msg_warn", "pprint", "pp"]
