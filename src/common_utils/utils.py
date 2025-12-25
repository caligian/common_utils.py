import os
import re
import subprocess
import datetime
import sys
import shutil
import copy

from pickle import (
    load as pkl_load,
    dump as pkl_dump,
    loads as _load_pkl,
    dumps as _dump_pkl,
)
from csv import reader as csv_reader, writer as csv_writer
from termcolor import cprint
from functools import partial as _partial
from typing import Callable
from glob import glob
from collections import namedtuple as nt
from json import (
    load as json_load,
    dump as json_dump,
    loads as _load_json,
    dumps as _dump_json,
)
from functools import reduce as reduce_
from pyfzf import FzfPrompt
# from .input import menu

Pattern = re.Pattern
Container = list | tuple | dict
Sequence = list | tuple

deepcopy = copy.deepcopy
shallowcopy = copy.copy
isa = isinstance
load_pkl = _load_pkl
dump_pkl = _dump_pkl
load_json = _load_json
dump_json = _dump_json
partial = _partial
namedtuple = nt
reduce = reduce_
mkdir = os.makedirs
is_dir = os.path.isdir
is_file = os.path.isfile
is_mount = os.path.ismount
is_link = os.path.islink
is_junction = os.path.isjunction
path_exists = os.path.exists
rmtree = shutil.rmtree
date = datetime.date
datetime = datetime.datetime
time = datetime.time


def file_extension(filename: str) -> str:
    return filename.rsplit(".", maxsplit=1)[-1]


def has_extension(filename: str, *pattern: str | re.Pattern) -> bool:
    extension = file_extension(filename)
    for p in pattern:
        if re.search(p, extension, flags=re.I):
            return True

    return False


def mime_type(filename: str) -> str | None:
    out = subprocess.check_output(["file", filename])
    out = out.decode()
    out = out.split(":")
    out = out[-1]
    out = out.strip()

    if startswith(out, "cannot open"):
        return

    return out


def blank(s: str | list | tuple | dict) -> bool:
    return len(s) == 0


def not_blank(s: str | list | tuple | dict) -> bool:
    return len(s) > 0


def ARGV() -> list[str]:
    return sys.argv


def has_argv() -> bool:
    return len(sys.argv) != 1


def rm(path: str, **kwargs) -> bool:
    if not os.path.exists(path):
        return False
    elif os.path.isdir(path):
        shutil.rmtree(path, **kwargs)
    else:
        os.remove(path, **kwargs)

    return True


def strptime(
    fmt: str,
    date_str: str,
    *args,
    use_date: bool = False,
    use_datetime: bool = True,
    use_time: bool = False,
    **kwargs,
) -> str:
    cls = None
    if use_date:
        cls = datetime.date
    elif use_datetime:
        cls = datetime.datetime
    elif use_time:
        cls = datetime.time

    fn = cls.strptime
    return fn(date_str, fmt)


def strftime(
    fmt: str,
    *args,
    use_date: bool = False,
    use_datetime: bool = True,
    use_time: bool = False,
    **kwargs,
) -> str:
    cls = None
    if use_date:
        cls = datetime.date
    elif use_datetime:
        cls = datetime.datetime
    elif use_time:
        cls = datetime.time

    return cls(*args, **kwargs).strftime(fmt)


def read_json(filename: str) -> any:
    with open(filename, "r") as fh:
        return json_load(fh)


def write_json(filename: str, obj: any) -> None:
    with open(filename, "w") as fh:
        json_dump(obj, fh)


def read_pkl(filename: str) -> any:
    with open(filename, "rb") as fh:
        return pkl_load(fh)


def write_pkl(filename: str, obj: any) -> any:
    with open(filename, "wb") as fh:
        return pkl_dump(obj, fh)


def read_csv(filename: str, read_all: bool = True, **kwargs) -> list[str]:
    with open(filename) as fh:
        if read_all:
            return [line for line in csv_reader(fh, **kwargs)]
        else:
            return csv_reader(fh)


def write_csv(
    filename: str,
    lines: str | list[str],
    sep: str = r"\n",
    **kwargs,
) -> int:
    with open(filename, "w") as fh:
        writer = csv_writer(fh, **kwargs)
        size = 0

        if type(lines) is str and r"\n" in lines:
            size = len(size)
            lines = lines.split(r"\n")
        elif type(lines) is list:
            size = sum(map(len, lines))

        writer.writerows(lines)
        return size


def seq_along(xs: Container) -> list[int]:
    if isa(xs, (list, tuple)):
        return list(range(len(xs)))
    else:
        return list(xs.keys())


def failure(*s: str, color: str = "red", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def message(*s: str, color: str = "blue", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def success(*s: str, color: str = "green", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def foreach(
    tbl: Container,
    apply: Callable[[int | str, any], any] = lambda _, x: x,
    keep: Callable[[int | str, any], any] = lambda _, __: True,
    exclude: Callable[[int | str], any] = lambda _, __: False,
) -> Container:
    if type(tbl) is dict:
        res = {}
        for k, v in tbl.items():
            if keep(k, v) and not exclude(k, v):
                res[k] = apply(k, v)

        return res

    return type(tbl)(
        apply(i, x) for i, x in enumerate(tbl) if keep(i, x) and not exclude(i, x)
    )


def keep(tbl: Container, f: Callable[[int | str, any], any]) -> Container:
    return foreach(tbl, keep=f)


def tbl_apply(tbl: Container, f: Callable[[int | str, any], any]) -> Container:
    return foreach(tbl, apply=f)


def tbl_keep(tbl: Container, f: Callable[[int | str, any], any]) -> Container:
    return foreach(tbl, keep=f)


def tbl_exclude(tbl: Container, f: Callable[[int | str, any], any]) -> Container:
    return foreach(tbl, exclude=f)


def tbl_get(
    xs: Container,
    *ks: int | str | list[int | str],
    pcall: bool = True,
) -> list[any]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case (True, x, _):
                res.append(x)
            case (False, error, _):
                if not pcall:
                    raise error
                else:
                    res.append(error)

    return res


def tbl_has(xs: Container, *ks: int | str | list[int | str]) -> list[any]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case (True, _, _):
                res.append(True)
            case (False, _, _):
                res.append(False)

    return res


def tbl_set(
    xs: Container,
    *keys_and_values: tuple[any, any],
    pcall: bool = True,
) -> Container | Exception:
    if len(keys_and_values) == 0:
        return xs

    for k, v in keys_and_values:
        match assoc(xs, k, value=v):
            case (True, _value, _level):
                continue
            case (False, error, _level):
                if not pcall:
                    raise error

    return xs


def split(
    s: str,
    pattern: Pattern | str,
    **kwargs: str | int,
) -> list[str]:
    return re.split(pattern, s, **kwargs)


def splitlines(
    s: str,
    pattern: str | Pattern = r"\n",
    **kwargs,
) -> list[str]:
    return split(s, pattern, **kwargs)


def grep(
    s: str,
    pattern: str | Pattern,
    **kwargs,
) -> re.Match | None:
    return re.search(pattern, s, **kwargs)


def tbl_grep(
    tbl: Container,
    pattern: str | Pattern,
    **kwargs,
) -> dict[any, re.Match] | list[re.Match]:
    return tbl_keep(tbl, lambda _, v: re.search(pattern, v, **kwargs))


def tbl_grepv(
    tbl: Container,
    pattern: str | Pattern,
    **kwargs,
) -> dict[any, re.Match] | list[re.Match]:
    return tbl_exclude(tbl, lambda _, v: re.search(pattern, v, **kwargs))


def startswith(s: str, pattern: str | Pattern, **kwargs) -> re.Match | None:
    return re.match(pattern, s, **kwargs)


def endswith(s: str, pattern: str | Pattern, **kwargs) -> re.Match | None:
    return re.search(pattern + "$", s, **kwargs)


def str_is_int(s: str, strip_whitespace: bool = False) -> bool:
    if strip_whitespace:
        s = re.sub(r"\s+", "", s)

    return grep(s, "^[0-9]+$") is not None


def as_int(s: str, strip_whitespace: bool = False) -> int | None:
    if str_is_int(s, strip_whitespace=strip_whitespace):
        s = re.sub(r"\s+", "", s)
        try:
            return int(s)
        except Exception:
            return


def str_is_float(s: str, strip_whitespace: bool = False) -> bool:
    if strip_whitespace:
        s = re.sub(r"\s+", "", s)

    return grep(s, "^[0-9]+[.][0-9]+$") is not None


def as_float(s: str, strip_whitespace: bool = False) -> float | None:
    if strip_whitespace:
        s = re.sub(r"\s+", "", s)

    try:
        return float(s)
    except Exception:
        return


def sed(
    s: str,
    *patterns_and_replacements: tuple[str, str],
    **kwargs,
) -> str:
    for pattern, repl in patterns_and_replacements:
        s = re.sub(pattern, repl, s, **kwargs)

    return s


def system(
    cmd: list[str] | str,
    capture: bool = True,
    split_nl: bool = False,
    pcall: bool = True,
    chomp: bool = True,
    **kwargs,
) -> list[str] | str | Exception | subprocess.CompletedProcess:
    if type(cmd) is str:
        kwargs["shell"] = True

    def run_and_capture() -> list[str] | str:
        out = subprocess.check_output(cmd, **kwargs)
        out = out.decode()
        out = chomp and strip(out) or out
        out = split_nl and out.split("\n") or out

        return out

    if pcall and capture:
        try:
            return run_and_capture()
        except subprocess.CalledProcessError as error:
            return error
    elif capture:
        return run_and_capture()
    elif pcall:
        try:
            return subprocess.run(cmd, **kwargs)
        except subprocess.CalledProcessError as error:
            return error
    else:
        return subprocess.run(cmd, **kwargs)


def systemlist(
    cmd: list[str] | str,
    pcall: bool = True,
    chomp: bool = True,
    **kwargs,
) -> list[str] | Exception:
    return system(cmd, capture=True, split_nl=True, chomp=chomp, pcall=pcall, **kwargs)


def strip(s: str, lhs: bool = True, rhs: bool = True) -> str:
    if lhs:
        s = lstrip(s)

    if rhs:
        s = rstrip(s)

    return s


def lstrip(s: str) -> str:
    return re.sub(r"^\s+", "", s, flags=re.M)


def rstrip(s: str) -> str:
    return re.sub(r"\s+$", "", s, flags=re.M)


def slurp(
    filename: str,
    mode: str = "r",
    filetype: str = "text",
    reader: Callable[[str | bytes], any] | None = None,
    chomp: bool = True,
) -> list[str] | str:
    ft_is_text = filetype in ("json", "csv", "text")
    ft_is_pkl = filetype in ("pickle", "pkl")

    if ft_is_text:
        mode = "r"
    elif ft_is_pkl:
        mode = "rb"
    else:
        raise NotImplementedError(
            "filetype should be any of json, csv, text, pickle, pkl"
        )

    with open(filename, mode) as fh:
        if ft_is_text:
            match filetype:
                case "json":
                    return read_json(filename)
                case "csv":
                    return read_csv(filename)
                case _:
                    if chomp:
                        return rstrip(fh.read())
                    else:
                        return fh.read()
        elif ft_is_pkl:
            return read_pkl(filename)
        elif callable(reader):
            return reader(fh)


def spit(
    filename: str,
    text: str | list[str],
    mode: str = "w",
    format: str = "text",
) -> int:
    with open(
        filename,
    ) as fh:
        text_len = None
        match text:
            case list():
                fh.writelines(text)
                text_len = sum([len(x) for x in text])
            case str():
                fh.write(text)
                text_len = len(text)

        return text_len


def sequence(xs: list | tuple) -> bool:
    return isa(xs, (tuple, list))


def container(xs: dict | list | tuple) -> bool:
    return isa(xs, (tuple, int, dict))


def flatten(xs: list | tuple, maxdepth: int = -1) -> list:
    def vector(lst: list | tuple, current_depth: int = 0, result: list = []) -> list:
        if current_depth != -1 and current_depth == maxdepth:
            return result

        for i, x in enumerate(lst):
            if sequence(x):
                current = len(result)
                vector(x, current_depth + 1, result=result)
                if current == len(result):
                    result.append(x)
            else:
                result.append(x)

    result = []
    vector(xs, result=result)

    return result


def assoc(
    d: Container,
    ks: any,
    value: any = None,
) -> tuple[bool, any, Container]:
    v = d
    for k in ks[:-1]:
        try:
            v = v[k]
        except Exception as error:
            return (False, error, v)

    k = ks[-1]
    try:
        if value is not None:
            v[k] = value
        return (True, v[k], v)
    except Exception as error:
        return (False, error, v)


def as_list(xs: any, force: bool = False) -> list:
    if force:
        return [xs]
    elif sequence(xs):
        return list(xs)
    else:
        return [xs]


def ifelse(
    value: any,
    when_truthy: Callable,
    when_falsy: Callable | None = None,
) -> any:
    if value:
        return when_truthy(value)
    elif callable(when_falsy):
        return when_falsy(value)
    else:
        return value


def unless(
    value: any,
    when_falsy: Callable,
    when_truthy: Callable | None = None,
) -> any:
    if value:
        return when_falsy(value)
    elif callable(when_truthy):
        return when_truthy(value)
    else:
        return value


def ifNone(
    value: any,
    when_none: Callable,
    when_not_none: Callable | None = None,
) -> any:
    if value is None:
        return when_none(value)
    elif callable(when_not_none):
        return when_not_none(value)
    else:
        return value


def unlessNone(
    value: any,
    when_not_none: Callable | None,
    when_none: Callable | None = None,
) -> any:
    if type(value) is not None:
        return when_not_none(value)
    elif callable(when_none):
        return when_none(value)
    else:
        return value


def pcall(f, *args, **kwargs) -> tuple[bool, any]:
    try:
        output = f(*args, **kwargs)
        return (True, output)
    except Exception as error:
        return (False, error)


def whereis(binary: str) -> list[str]:
    out = system(f"whereis {binary}")
    out = split(out, r"\s+")
    out[0] = out[0][:-1]
    out.pop(0)

    return [x for x in out if os.access(x, os.X_OK)]


def ls(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
    include: str = "dflmj",
    stat: bool = False,
    follow_symlinks: bool = False,
) -> list[str] | list[tuple[str, os.stat_result]]:
    pattern = re.compile(pattern, flags=re.I + re.M)
    files: list[str] = glob(f"{d}/*") + glob(f"{d}/.*")
    exclude = exclude and re.compile(exclude, flags=re.I + re.M) or None
    files = (
        foreach(
            files,
            keep=lambda _, s: pattern.search(s),
        )
        if not exclude
        else foreach(
            files,
            keep=lambda _, s: pattern.search(s),
            exclude=lambda _, s: exclude.search(s),
        )
    )

    res = []
    d = "d" in include
    f = "f" in include
    lnk = "l" in include
    m = "m" in include
    j = "j" in include

    def append_file(filename: str) -> None:
        if stat:
            res.append((filename, os.stat(filename)))
        else:
            res.append(filename)

    for file in files:
        if (
            (d and is_dir(file))
            or (f and is_file(file))
            or (lnk and is_link(file))
            or (m and is_mount(file))
            or (j and is_junction(file))
        ):
            append_file(file)

    return res


def unwrap(xs: Sequence) -> any:
    assert len(xs) == 1
    return xs[0]


def push(xs: Sequence, *elements: any, index: int | None = None) -> Sequence:
    cls = type(xs)
    xs = list(xs)

    for e in elements:
        if not index:
            xs.append(e)
        else:
            xs.append(e)

    return cls(xs)


def reverse(xs: tuple | list | str) -> tuple | list | str:
    return xs[::-1]


def unpush(xs: Sequence, *elements: any) -> Sequence:
    return push(xs, *elements[::-1], index=0)


def extend(xs: Sequence, *elements: any) -> Sequence:
    cls = type(xs)
    xs = list(xs)

    for e in elements:
        if sequence(e):
            xs.extend(list(e))
        else:
            xs.append(e)

    return cls(xs)


def lextend(xs: list, *elements: any) -> Sequence:
    cls = type(xs)
    xs = list(xs)

    for e in elements[::-1]:
        if sequence(e):
            unpush(xs, e)
        else:
            xs.insert(0, e)

    return cls(xs)


def identity(element: any) -> any:
    return element


def pop(
    xs: list | dict,
    index: int | str = -1,
    default: Callable | None = None,
    pcall: bool = True,
) -> any:
    if type(xs) is dict and type(index) is int:
        index = list(xs.keys())[index]

    try:
        return xs.pop(index)
    except (IndexError, KeyError) as error:
        if default:
            return default()
        elif pcall:
            return error
        else:
            raise error


def shift(
    xs: list,
    default: Callable | None = None,
    pcall: bool = True,
) -> list:
    return pop(
        xs,
        index=0,
        default=default,
        pcall=pcall,
    )


def popn(
    xs: list,
    n: int = 1,
    index: int | str = -1,
    reverse: bool = False,
    pcall: bool = True,
    default: Callable | None = None,
) -> list[any]:
    res = []
    for i in range(n):
        res.append(
            pop(
                xs,
                index=index,
                default=default,
                pcall=pcall,
            )
        )

    if reverse:
        return res[::-1]
    else:
        return res


def shiftn(
    xs: list,
    n: int = 1,
    index: int = -1,
    reverse: bool = False,
    pcall: bool = True,
    default: Callable | None = None,
) -> list[any]:
    return popn(
        xs,
        n,
        index=0,
        reverse=reverse,
        pcall=pcall,
        default=default,
    )


def fzf(
    tbl: dict[str, any] | list | tuple,
    lalign: bool = True,
    ralign: bool = False,
    center: bool = False,
    bin: str | None = None,
    skip_index: bool = False,
) -> list:
    lookup = dict()
    _tbl = {}
    longest = 0
    display = []
    index = seq_along(tbl)
    _dict = isa(tbl, dict)

    if not skip_index:
        for k in index:
            k = str(k)
            k_len = len(k)

            if longest < k_len:
                longest = k_len

    for k in index:
        v = tbl[k]
        if not skip_index:
            if ralign:
                k = f"{str(k):>{longest}} | {str(v)}"
            elif lalign:
                k = f"{str(k):<{longest}} | {str(v)}"
            else:
                k = f"{str(k):^{longest}} | {str(v)}"

            lookup[k] = v
            display.append(k)
        elif _dict:
            str_k = str(k)
            lookup[str_k] = k
            display.append(str_k)
        else:
            v = str(v)
            lookup[v] = k
            display.append(v)

    _fzf = FzfPrompt(executable_path=bin)
    choice = _fzf.prompt(display, fzf_options="--multi")

    return [tbl[lookup[k]] for k in choice]


def plist_get1(xs: list[tuple[any, any]])


def plist_get(
    xs: list[tuple[any, any]], 
    *key: any,
) -> any:
    for k, v in xs:
        if key 


__all__ = [
    "ARGV",
    "rm",
    "strptime",
    "strftime",
    "read_json",
    "write_json",
    "read_pkl",
    "write_pkl",
    "read_csv",
    "write_csv",
    "seq_along",
    "failure",
    "message",
    "success",
    "foreach",
    "keep",
    "tbl_apply",
    "tbl_keep",
    "tbl_exclude",
    "tbl_get",
    "tbl_set",
    "tbl_has",
    "split",
    "splitlines",
    "grep",
    "tbl_grep",
    "tbl_grepv",
    "startswith",
    "endswith",
    "str_is_int",
    "as_int",
    "str_is_float",
    "as_float",
    "sed",
    "system",
    "systemlist",
    "strip",
    "lstrip",
    "rstrip",
    "slurp",
    "spit",
    "sequence",
    "container",
    "flatten",
    "assoc",
    "as_list",
    "ifelse",
    "unless",
    "ifNone",
    "unlessNone",
    "pcall",
    "whereis",
    "ls",
    "unwrap",
    "push",
    "reverse",
    "unpush",
    "extend",
    "lextend",
    "identity",
    "pop",
    "shift",
    "popn",
    "shiftn",
    "fzf",
    "blank",
    "not_blank",
    "has_argv",
    "deepcopy",
    "shallowcopy",
    "isa",
    "load_pkl",
    "dump_pkl",
    "load_json",
    "dump_json",
    "partial",
    "namedtuple",
    "reduce",
    "mkdir",
    "is_dir",
    "is_file",
    "is_mount",
    "is_link",
    "is_junction",
    "path_exists",
    "rmtree",
    "date",
    "datetime",
    "time",
    "file_extension",
    "has_extension",
]
