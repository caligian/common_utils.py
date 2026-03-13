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
from src.common_utils.prompt import *

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
    elif failure is not None:
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
    elif failure is not None:
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
    elif failure is not None:
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
commandparser = Argv
strlike = is_str_like

__all__ = [
    "abspath",
    "agrep",
    "all",
    "andgrep",
    "ARGV",
    "as_error",
    "as_exception",
    "as_float",
    "asfloat",
    "as_int",
    "asint",
    "as_list",
    "aslist",
    "assoc",
    "basename",
    "blank",
    "butlast",
    "car",
    "cdr",
    "commandparser",
    "container",
    "Container",
    "cp",
    "cpstat",
    "date",
    "datetime",
    "deepcopy",
    "dirname",
    "empty",
    "endswith",
    "Error",
    "error_args",
    "error_class",
    "error_message",
    "exception_args",
    "exception_class",
    "exception_message",
    "exists",
    "extend",
    "file_extension",
    "flatten",
    "foreach",
    "fzf",
    "get_error_args",
    "get_error_class",
    "get_error_message",
    "get_exception_args",
    "get_exception_class",
    "get_exception_message",
    "grep",
    "grepv",
    "has_argv",
    "has_extension",
    "head",
    "identity",
    "ifelse",
    "isa",
    "is_dir",
    "isdir",
    "is_error",
    "is_error_class",
    "is_error_instance",
    "is_error_type",
    "is_exception",
    "is_exception_class",
    "is_exception_instance",
    "is_exception_type",
    "is_file",
    "isfile",
    "is_float",
    "isfloat",
    "is_int",
    "isint",
    "is_junction",
    "isjunction",
    "is_link",
    "is_mount",
    "ismount",
    "ispath",
    "is_str_like",
    "it",
    "keep",
    "lextend",
    "ls",
    "lstrip",
    "make_error",
    "make_exception",
    "make_prompt",
    "menu",
    "Menu",
    "message",
    "mimetype",
    "mkdefault",
    "mkdir",
    "msg_failure",
    "msg_ok",
    "msg_success",
    "msg_warn",
    "not_blank",
    "not_empty",
    "ogrep",
    "orgrep",
    "paste",
    "paste0",
    "Pattern",
    "pcall",
    "pipe",
    "pop",
    "popn",
    "push",
    "raise_error",
    "raise_exception",
    "raise_unless",
    "raise_when",
    "read_csv",
    "read_json",
    "readlines",
    "read_pkl",
    "reverse",
    "rm",
    "rmtree",
    "rstrip",
    "safe",
    "sed",
    "seq",
    "seq_along",
    "sequence",
    "Sequence",
    "set_error_args",
    "set_error_message",
    "set_exception_args",
    "set_exception_message",
    "shallowcopy",
    "shift",
    "shiftn",
    "slurp",
    "some",
    "spit",
    "split",
    "splitlines",
    "startswith",
    "stat",
    "strfind",
    "strftime",
    "strip",
    "strlike",
    "StrLike",
    "strmatch",
    "strptime",
    "system",
    "systemlist",
    "T",
    "tail",
    "tapply",
    "tbl_apply",
    "tbl_exclude",
    "tbl_filter",
    "tbl_get",
    "tbl_grep",
    "tbl_grepv",
    "tbl_has",
    "tbl_keep",
    "tbl_map",
    "tbl_set",
    "texclude",
    "tfilter",
    "tget",
    "tgrep",
    "tgrepv",
    "time",
    "tkeep",
    "tmap",
    "tset",
    "unless",
    "unlessNone",
    "unpush",
    "unwrap",
    "when",
    "whenNone",
    "whereis",
    "write_csv",
    "write_json",
    "writelines",
    "write_pkl",
]
