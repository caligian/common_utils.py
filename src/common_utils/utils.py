import os
import re
import subprocess
import datetime
import sys
import shutil
import copy

from termcolor import cprint
from functools import partial
from typing import Callable
from glob import glob
from collections import namedtuple
from functools import reduce
from pyfzf import FzfPrompt
from sspipe import p, px

from src.common_utils.menu import *
from src.common_utils.process import *
from src.common_utils.table import *
from src.common_utils.file import *
from src.common_utils.result import *
from src.common_utils.cmdline import *
from src.common_utils.error import *

StrLike = str | bytes

pipe = p
it = px
deepcopy = copy.deepcopy
shallowcopy = copy.copy
date = datetime.date
datetime = datetime.datetime
time = datetime.time


def whenNone(x: any, success: any, failure: any = None) -> any:
    if x is None:
        if isinstance(success, Callable):
            return success(x)
        else:
            return success
    elif failure:
        if isinstance(failure, Callable):
            return failure(x)
        else:
            return failure


def unlessNone(x: any, success: any, failure: any = None) -> any:
    if x is not None:
        if isinstance(success, Callable):
            return success(x)
        else:
            return success
    elif failure:
        if isinstance(failure, Callable):
            return failure(x)
        else:
            return failure


def when(x: any, success: any, failure: any = None) -> any:
    if x:
        if isinstance(success, Callable):
            return success(x)
        else:
            return success
    elif failure:
        if isinstance(failure, Callable):
            return failure(x)
        else:
            return failure


def unless(x: any, success: any, failure: any = None) -> any:
    if not x:
        if isinstance(success, Callable):
            return success(x)
        else:
            return success
    elif failure:
        if isinstance(failure, Callable):
            return failure(x)
        else:
            return failure


def when(x: any, success: any, failure: any = None) -> any:
    if x:
        if isinstance(success, Callable):
            return success(x)
        else:
            return success
    elif failure:
        if isinstance(failure, Callable):
            return failure(x)
        else:
            return failure


def is_str_like(s: any) -> bool:
    return isinstance(s, (str, bytes))


def isa(x, *types: type) -> bool:
    return isinstance(x, tuple(types))


def ARGV() -> list[str]:
    return sys.argv


def has_argv() -> bool:
    return len(sys.argv) != 1


def strptime(
    fmt: str,
    date_str: str,
    *args,
    use: type = datetime,
    **kwargs,
) -> str:
    fn = use.strptime
    return fn(date_str, fmt)


def strftime(
    fmt: str,
    *args,
    use: type = datetime,
    **kwargs,
) -> str:
    return use(*args, **kwargs).strftime(fmt)


def message(*s: str, color: str = "blue", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_failure(*s: str, color: str = "red", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_success(*s: str, color: str = "green", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_warn(*s: str, color: str = "yellow", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def ifelse(
    value: any,
    truthy: Callable = lambda x: x,
    falsy: Callable | None = None,
    *args,
    **kwargs,
) -> any:
    if value:
        return truthy(value, *args, **kwargs)
    elif falsy:
        return falsy(value, *args, **kwargs)
    else:
        return value


def pcall(f, *args, **kwargs) -> Success | Failure:
    try:
        output = f(*args, **kwargs)
        return Success(output)
    except Exception as error:
        return Failure(error)


msg_ok = msg_success

__all__ = []
