import os
import re
import subprocess
import datetime
import shutil

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
# from .input import menu

Pattern = re.Pattern
Container = list | tuple | dict
Sequence = list | tuple

load_pkl = _load_pkl
dump_pkl = _dump_pkl
load_json = _load_json
dump_json = _dump_json
partial = _partial
namedtuple = nt
reduce = reduce_
mkdir = os.makedirs


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


def write_csv(filename: str, lines: str | list[str], sep: str = r"\n", **kwargs) -> int:
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


def seq_along(xs: Sequence) -> list[int]:
    return list(range(len(xs)))


def failure(*s: str, color: str = "red", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def message(*s: str, color: str = "blue", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def success(*s: str, color: str = "green", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def foreach(
    s: Container,
    apply: Callable[[int | str, any], any] = lambda _, x: x,
    keep: Callable[[int | str, any], any] = lambda _, __: True,
    exclude: Callable[[int | str], any] = lambda _, __: False,
) -> Container:
    if type(s) is dict:
        res = {}
        for k, v in s.items():
            if keep(k, v) and not exclude(k, v):
                res[k] = apply(k, v)

        return res

    return type(s)(
        apply(i, x) for i, x in enumerate(s) if keep(i, x) and not exclude(i, x)
    )


def tbl_get(
    xs: Container,
    *ks: int | str | list[int | str],
    pcall: bool = True,
) -> list[any]:
    res = []
    for k in ks:
        if sequence(k):
            match assoc(xs, k):
                case (True, x, _):
                    res.append(x)
                case (Exception(error), _, _):
                    if not pcall:
                        raise error
                    else:
                        return None
        else:
            try:
                res.append(xs[k])
            except (KeyError, IndexError) as error:
                if not pcall:
                    raise error
                else:
                    res.append(None)

    return res


def tbl_set(xs: Container, *keys_and_values: any, pcall: bool = True) -> Container:
    keys_and_values_len = len(keys_and_values)
    if keys_and_values_len == 0:
        return xs
    elif keys_and_values_len % 2 != 0:
        raise AssertionError(
            f"keys_and_values_len {keys_and_values_len} should be even in number"
        )

    for i in range(0, keys_and_values_len, 2):
        k = keys_and_values[i]
        v = keys_and_values[i + 1]

        try:
            assoc(xs, k, value=v)
        except Exception as error:
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
    pattern: str | Pattern,
    repl: str,
    **kwargs,
) -> re.Match | None:
    return re.sub(pattern, repl, s, **kwargs)


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
    binary: bool = False,
    split_nl: bool = True,
) -> list[str] | str:
    with open(filename, "r" if not binary else "rb") as fh:
        text = fh.read()
        text = split_nl and text.split("\n") or text
        return text


def spit(
    filename: str,
    text: str | list[str],
    binary: bool = False,
) -> int:
    with open(filename, "w" if not binary else "wb") as fh:
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
    return isinstance(xs, (tuple, list))


def container(xs: dict | list | tuple) -> bool:
    return isinstance(xs, (tuple, int, dict))


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
) -> tuple[bool | Exception, any, Container | None]:
    v = d
    for k in ks[:-1]:
        try:
            v = v[k]
        except Exception as error:
            return (error, None)

    k = ks[-1]
    try:
        if value is not None:
            v[k] = value
        return (True, v[k], v)
    except Exception as error:
        return (error, None, None)


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


def pcall(f, *args, **kwargs) -> tuple[bool | Exception, any]:
    try:
        output = f(*args, **kwargs)
        return (True, output)
    except Exception as error:
        return (error, None)


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
    dirs_only: bool = False,
    files_only: bool = False,
    links_only: bool = False,
    mounts_only: bool = False,
) -> list[str]:
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

    if dirs_only or files_only or links_only or mounts_only:
        res = []
        for f in files:
            if dirs_only and os.path.isdir(f):
                res.append(f)
            elif files_only and os.path.isfile(f):
                res.append(f)
            elif links_only and os.path.islink(f):
                res.append(f)
            elif mounts_only and os.path.ismount(f):
                res.append(f)
        return res
    else:
        return files


def ls_links(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
) -> list[str]:
    return ls(d, pattern, links_only=True)


def ls_mounts(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
) -> list[str]:
    return ls(d, pattern, mounts_only=True)


def ls_dirs(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
) -> list[str]:
    return ls(d, pattern, dirs_only=True)


def ls_files(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
) -> list[str]:
    return ls(d, pattern, files_only=True)


def unwrap(xs: Sequence) -> any:
    assert len(xs) == 1
    return xs[0]


def push(xs: list, *elements: any, index: int | None = None) -> list:
    for e in elements:
        if index:
            xs.append(e)
        else:
            xs.insert(index, e)


def reverse(xs: tuple | list | str) -> tuple | list | str:
    return xs[::-1]


def unpush(xs: list, *elements: any) -> list:
    for e in elements[::-1]:
        xs.insert(0, e)


def extend(xs: list, *elements: any) -> list:
    for e in elements:
        if sequence(e):
            xs.extend(list(e))
        else:
            xs.append(e)


def lextend(xs: list, *elements: any) -> list:
    for e in elements:
        if sequence(e):
            for _e in reverse(e):
                unpush(xs, _e)
        else:
            xs.insert(0, e)

    return xs


def identity(element: any) -> any:
    return element


def pop(
    xs: list | dict,
    index: int | str = -1,
    default: Callable = lambda: None,
) -> list | dict:
    if type(index) is int and index < 0:
        index += len(xs)

    if isinstance(xs, list):
        xs: list
        return xs.pop(index)
    else:
        xs: dict
        return xs.pop(index)


def shift(xs: list) -> list:
    return pop(xs, 0)


def popn(xs: list, n: int = 1, index: int = -1, reverse: bool = False) -> list[any]:
    res = []
    for i in range(n):
        res.append(xs.pop(index))

    if reverse:
        return res[::-1]
    else:
        return res


def shiftn(xs: list, n: int = 1, index: int = -1, reverse: bool = False) -> list[any]:
    return popn(xs, n, index=0, reverse=reverse)


__all__ = [
    "popn",
    "pop",
    "shift",
    "shiftn",
    "identity",
    "seq_along",
    "failure",
    "message",
    "success",
    "foreach",
    "tbl_get",
    "tbl_set",
    "split",
    "splitlines",
    "grep",
    "startswith",
    "endswith",
    "str_is_int",
    "as_int",
    "as_float",
    "str_is_float",
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
    "ls_files",
    "ls_links",
    "ls_mounts",
    "ls_dirs",
    "unwrap",
    "partial",
    "read_csv",
    "write_csv",
    "read_pkl",
    "write_pkl",
    "read_json",
    "write_json",
    "partial",
    "namedtuple",
    "reduce",
    "push",
    "unpush",
    "extend",
    "lextend",
    "strftime",
    "strptime",
    "mkdir",
    "rm",
    "load_pkl",
    "dump_pkl",
    "load_json",
    "dump_json",
]
