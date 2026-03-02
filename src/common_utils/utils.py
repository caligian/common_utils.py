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

from src.common_utils.menu import Menu
from src.common_utils.process import system, systemlist
from src.common_utils.table import (
    all,
    andgrep,
    as_float,
    asfloat,
    as_int,
    asint,
    as_list,
    aslist,
    assoc,
    blank,
    butlast,
    car,
    cdr,
    container,
    Container,
    empty,
    endswith,
    extend,
    flatten,
    foreach,
    fzf,
    grep,
    head,
    identity,
    is_float,
    isfloat,
    is_int,
    isint,
    keep,
    lextend,
    lstrip,
    not_blank,
    not_empty,
    orgrep,
    paste,
    paste0,
    Pattern,
    pop,
    popn,
    push,
    reverse,
    rstrip,
    sed,
    seq,
    seq_along,
    sequence,
    Sequence,
    shift,
    shiftn,
    some,
    split,
    splitlines,
    startswith,
    strfind,
    strip,
    tail,
    tbl_apply,
    tbl_exclude,
    tbl_filter,
    tbl_get,
    tbl_grep,
    tbl_grepv,
    tbl_has,
    tbl_keep,
    tbl_map,
    tbl_set,
    unpush,
    unwrap,
)
from src.common_utils.file import (
    abspath,
    basename,
    cp,
    cpstat,
    dirname,
    file_extension,
    has_extension,
    is_dir,
    isdir,
    is_file,
    isfile,
    is_junction,
    isjunction,
    is_link,
    is_mount,
    ismount,
    ispath,
    ls,
    mimetype,
    mkdir,
    path_exists,
    read_csv,
    read_json,
    readlines,
    read_pkl,
    rm,
    rmtree,
    slurp,
    spit,
    stat,
    whereis,
    write_csv,
    write_json,
    writelines,
    write_pkl,
)
from src.common_utils.result import (
    Success,
    Failure,
    UnwrapError,
    Result,
    Error,
    T,
    safe,
)
from src.common_utils.cmdline import (
    Argv,
    mkdefault,
)
from src.common_utils.error import (
    set_error_args,
    set_exception_args,
    make_error,
    make_exception,
    raise_when,
    raise_unless,
    raise_error,
    raise_exception,
    as_error,
    as_exception,
    error_class,
    exception_class,
    get_exception_class,
    get_error_class,
    is_error,
    is_error_type,
    is_error_instance,
    is_error_class,
    is_exception,
    is_exception_type,
    is_exception_instance,
    is_exception_class,
    error_message,
    set_error_message,
    get_error_message,
    exception_message,
    set_exception_message,
    get_exception_message,
    error_args,
    get_error_args,
    exception_args,
    set_exception_args,
    get_exception_args,
)

StrLike = str | bytes

pipe = p
it = px
deepcopy = copy.deepcopy
shallowcopy = copy.copy
date = datetime.date
datetime = datetime.datetime
time = datetime.time


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


def unless(
    value: any,
    falsy: Callable = lambda x: x,
    truthy: Callable | None = None,
    *args,
    **kwargs,
) -> any:
    if not value:
        return falsy(value, *args, **kwargs)
    elif truthy:
        return truthy(value, *args, **kwargs)
    else:
        return value


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


def unlessNone(
    value: any,
    when_not_none: Callable,
    when_none: Callable | None = None,
) -> any:
    if value is not None:
        return when_not_none(value)
    elif when_none is not None:
        return when_none()
    else:
        return value


def ifNone(
    value: any,
    when_none: Callable,
    when_not_none: Callable | None = None,
) -> any:
    if value is None:
        return when_none()
    elif when_not_none is not None:
        return when_not_none(value)
    else:
        return value


def pcall(f, *args, **kwargs) -> Success | Failure:
    try:
        output = f(*args, **kwargs)
        return Success(output)
    except Exception as error:
        return Failure(error)


msg_ok = msg_success

__all__ = [
    "abspath",
    "all",
    "andgrep",
    "ARGV",
    "Argv",
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
    'glob',
    "grep",
    "has_argv",
    "has_extension",
    "head",
    "identity",
    "ifelse",
    "ifNone",
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
    "message",
    "mimetype",
    "mkdefault",
    "mkdir",
    "msg_failure",
    "msg_ok",
    "msg_success",
    "msg_warn",
    'namedtuple',
    "not_blank",
    "not_empty",
    "orgrep",
    "paste",
    "paste0",
    "path_exists",
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
    "StrLike",
    "strptime",
    "system",
    "systemlist",
    "T",
    "tail",
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
    "time",
    "unless",
    "unlessNone",
    "unpush",
    "unwrap",
    "whereis",
    "write_csv",
    "write_json",
    "writelines",
    "write_pkl",
    "Success",
    "Failure",
    "Result",
    "UnwrapError",
    "Menu",
]
