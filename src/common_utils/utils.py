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
from sspipe import p as _p, px as _px

from src.common_utils.menu import Menu as _menu
from src.common_utils.result import Result as _result
from src.common_utils.cmdline import Argv as _cmdparser
from src.command_utils.error import (
    make_exception as _make_exception,
    is_exception as _is_exception,
    as_exception as _as_exception,
    make_error as _make_error,
    is_error as _is_error,
    as_error as _as_error,
    bind_error as _bind_error,
    bind_exception as _bind_exception,
    raise_exception as _raise_exception,
    raise_error as _raise_error,
    raise_when as _raise_when,
    raise_unless as _raise_unless,
    get_error_message as _get_error_message,
    set_error_message as _set_error_message,
    error_class as _error_class,
    is_error_instance as _is_error_instance,
    is_error_type as _is_error_type
)

Pattern = re.Pattern
Container = list | tuple | dict
Sequence = list | tuple
Result = _result

error_class = _error_class
is_error_type = _is_error_type
is_error_instance = _is_error_instance 
exception_class = _error_class
is_exception_type = _is_error_type
is_exception_instance = _is_error_instance 
error_message = _get_error_message
get_error_message = _get_error_message
set_error_message = _set_error_message
exception_message = error_message
get_exception_message = get_error_message
set_exception_message = set_error_message
raise_when = _raise_when
raise_unless = _raise_unless
raise_error = _raise_error
raise_exception = _raise_exception
make_exception = _make_exception
is_exception = _is_exception
as_exception = _as_exception
bind_exception = _bind_exception
make_error = _make_error
is_error = _is_error
as_error = _as_error
bind_error = _bind_error
cmdparser = _cmdparser
menu = _menu
p = _p
it = _px
deepcopy = copy.deepcopy
shallowcopy = copy.copy
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
basename = os.path.basename
dirname = os.path.dirname
abspath = os.path.abspath
stat = os.stat
cpstat = shutil.copystat


def some(x: Container) -> bool:
    if isinstance(x, dict):
        for value in x.values():
            if value:
                return True
    else:
        for value in x:
            if value:
                return x

    return False


def all(x: Container) -> bool:
    if isinstance(x, dict):
        for value in x.values():
            if not value:
                return False
        return True
    else:
        for value in x:
            if not value:
                return False
        return True


def isa(x, *types: type) -> bool:
    return isinstance(x, tuple(types))


def cp(src: str, dest: str, **kwargs) -> str:
    if os.path.isdir(src):
        shutil.copytree(src, dest, **kwargs)
    else:
        shutil.copy(src, dest, **kwargs)

    return dest


def file_extension(filename: str) -> str:
    return filename.rsplit(".", maxsplit=1)[-1]


def has_extension(filename: str, *pattern: str | re.Pattern) -> bool:
    extension = file_extension(filename)
    for p in pattern:
        if re.search(p, extension, flags=re.I):
            return True

    return False


def mime_type(filename: str) -> str | None:
    out = subprocess.check_output(["file", "--mime-encoding", filename])
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
    if isa(xs, list, tuple):
        return list(range(len(xs)))
    else:
        return list(xs.keys())


def message(*s: str, color: str = "blue", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_failure(*s: str, color: str = "red", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_success(*s: str, color: str = "green", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


def msg_warn(*s: str, color: str = "yellow", **kwargs) -> None:
    cprint(*s, color=color, **kwargs)


msg_ok = msg_success


def foreach(
    tbl: Container,
    apply: Callable[[int | str, any], any] | Callable[[any], any] | None = None,
    keep: Callable[[int | str, any], any] | Callable[[any], any] | None = None,
    exclude: Callable[[int | str], any] | Callable[[any], any] | None = None,
    stop_when: Callable[[int, str], bool] | Callable[[any], any] | None = None,
    ignore_errors: bool = True,
    index: bool = False,
) -> Container:
    if index:
        if not apply:

            def apply(_, x):
                return x

        if not keep:

            def keep(_, x):
                return True

        if not exclude:

            def exclude(_, x):
                return False

        if not stop_when:

            def stop_when(_, x):
                return False
    else:
        if not apply:

            def apply(x):
                return x

        if not keep:

            def keep(x):
                if x:
                    return True

        if not exclude:

            def exclude(x):
                return False

        if stop_when:

            def stop_when(x):
                return False

    res = {}
    it = None

    if index:
        if sequence(tbl):
            it = enumerate(tbl)
        else:
            it = tbl.items()
    elif sequence(tbl):
        it = tbl
    else:
        it = tbl.values()

    if index:
        for k, v in it:
            if keep(k, v) and not exclude(k, v) and not stop_when(k, v):
                if ignore_errors:
                    if not is_error(v):
                        res.append(apply(k, v))
                    else:
                        pass
                elif is_error(v):
                    raise_error(v, f"Key passed: {k}")
                else:
                    res.append(apply(k, v))
    else:
        for x in it:
            if keep(x) and not exclude(x) and not stop_when(x):
                if ignore_errors:
                    if not is_error(v):
                        res.append(apply(v))
                    else:
                        pass
                elif is_error(v):
                    raise_error(v, f"Key passed: {k}")
                else:
                    res.append(apply(k, v))

    return type(tbl)(res)


def keep(
    tbl: Container,
    f: Callable[[int | str, any], any] | Callable[[any], any],
    index: bool = False,
) -> Container:
    return foreach(tbl, keep=f, index=index)


def tbl_apply(
    tbl: Container,
    f: Callable[[int | str, any], any] | Callable[[any], any],
    index: bool = False,
) -> Container:
    return foreach(tbl, apply=f, index=index)


def tbl_keep(
    tbl: Container,
    f: Callable[[int | str, any], any] | Callable[[any], any],
    index: bool = False,
) -> Container:
    return foreach(tbl, keep=f, index=index)


def tbl_exclude(
    tbl: Container,
    f: Callable[[int | str, any], any] | Callable[[any], any],
    index: bool = False,
) -> Container:
    return foreach(tbl, exclude=f, index=index)


def tbl_get(
    xs: Container,
    *ks: int | str | list[int | str],
    pcall: bool = False,
) -> list[any]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case Result(ok=True, value=x):
                res.append(x)
            case Result(ok=False, value=error):
                if not pcall:
                    raise error
                else:
                    res.append(error)

    return res


def tbl_has(
    xs: Container,
    *ks: int | str | list[int | str],
) -> list[any]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case Result(ok=True, value=value):
                res.append(value)
            case Result(ok=False):
                res.append(False)

    return res


def tbl_set(
    xs: Container,
    *keys_and_values: tuple[any, any],
    pcall: bool = False,
) -> Container | Exception:
    if len(keys_and_values) == 0:
        return xs

    for k, v in keys_and_values:
        match assoc(xs, k, value=v):
            case Result(ok=True):
                continue
            case Result(ok=False, value=error):
                if not pcall:
                    raise error
                else:
                    return error

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
    *pattern: str | Pattern,
    **kwargs,
) -> re.Match | None:
    for p in pattern:
        return re.search(p, s, **kwargs)


def tbl_grep(
    tbl: Container,
    *pattern: str | Pattern,
    **kwargs,
) -> dict[any, re.Match] | list[re.Match]:
    return tbl_keep(tbl, lambda v: grep(str(v), *pattern, **kwargs))


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


def is_int(s: str, strip_whitespace: bool = False) -> bool:
    if strip_whitespace:
        s = re.sub(r"\s+", "", s)

    return grep(s, "^[0-9]+$") is not None


def as_int(s: str, strip_whitespace: bool = False) -> int | None:
    if is_int(s, strip_whitespace=strip_whitespace):
        s = re.sub(r"\s+", "", s)
        try:
            return int(s)
        except Exception:
            return


def is_float(s: str, strip_whitespace: bool = False) -> bool:
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
    splitlines: bool = False,
    pcall: bool = False,
    chomp: bool = True,
    no_stdout: bool = False,
    no_stderr: bool = False,
    **kwargs,
) -> (
    subprocess.CompletedProcess
    | subprocess.CalledProcessError
    | FileNotFoundError
    | tuple[str, str]
    | tuple[list[str], list[str]]
    | bool
):
    kwargs = kwargs.copy()
    kwargs["check"] = True
    kwargs["capture_output"] = capture

    if type(cmd) is str:
        kwargs["shell"] = True

    if no_stdout:
        kwargs["stdout"] = subprocess.DEVNULL

    if no_stderr:
        kwargs["stderr"] = subprocess.DEVNULL

    try:
        proc = subprocess.run(cmd, **kwargs)
        if capture:
            stdout = proc.stdout.decode()
            stderr = proc.stderr.decode()
            stdout = strip(stdout, lhs=False) if chomp else stdout
            stderr = strip(stderr, lhs=False) if chomp else stderr
            stdout = splitlines and stdout.split("\n") or stdout
            stderr = splitlines and stderr.split("\n") or stderr

            return (stdout, stderr)
        else:
            return True
    except Exception as error:
        if pcall:
            return error
        else:
            raise error


def systemlist(
    cmd: list[str] | str,
    pcall: bool = False,
    chomp: bool = True,
    **kwargs,
) -> list[str] | Exception:
    return system(
        cmd,
        capture=True,
        splitlines=True,
        chomp=chomp,
        pcall=pcall,
        **kwargs,
    )


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
    format: str = "text",
    reader: Callable[[str | bytes], any] | None = None,
    newlines: bool = False,
    chomp: bool = True,
) -> list[str] | str:
    match format:
        case ft if ft in ("json", "j"):
            return read_json(filename)
        case ft if ft in ("text", "txt", "t"):
            with open(filename, mode) as fh:
                text = fh.read()
                text = chomp and text.strip() or text

                if newlines:
                    return text.split("\n")
                else:
                    return text
        case ft if ft in ("pickle", "pkl", "p"):
            return read_pkl(filename)
        case reader if callable(reader):
            return reader(filename)
        case ft:
            raise NotImplementedError(f"{ft} reader is not implemented")


def spit(
    filename: str,
    obj: any,
    mode: str = "w",
    format: str = "text",
) -> int:
    match format:
        case ft if ft in ("json", "j"):
            return write_json(filename, obj)
        case ft if ft in ("text", "txt", "t"):
            with open(filename, mode) as fh:
                fh.write(str(obj))
        case ft if ft in ("pickle", "pkl", "p"):
            return write_pkl(filename, obj)
        case writer if callable(writer):
            return writer(filename, obj)
        case ft:
            raise NotImplementedError(f"{ft} writer is not implemented")


def readlines(filename: str) -> list[str]:
    return slurp(filename, newlines=True)


def writelines(
    filename: str,
    *text: list[str],
    append_newline: bool = True,
) -> int:
    with open(filename, "w") as fh:
        size = 0
        for line in text:
            if append_newline:
                fh.write(line + "\n")
                size += len(line) + 1
            else:
                fh.write(line)
                size += len(line)

        return size


def sequence(xs: list | tuple) -> bool:
    return isa(xs, tuple, list)


def container(xs: dict | list | tuple) -> bool:
    return isa(xs, tuple, int, dict)


def flatten(xs: list | tuple, maxdepth: int = -1) -> list:
    def vector(lst: list | tuple, current_depth: int = 0, result: list = []) -> list:
        if current_depth != -1 and current_depth == maxdepth:
            return result

        for i, x in enumerate(lst):
            if sequence(x):
                current = len(result)
                vector(x, current_depth + 1, result=result)

                if len(result) == current:
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
            return Result(False, error, v)

    k = ks[-1]
    try:
        if value is not None:
            v[k] = value
        return Result(True, v[k], v)
    except Exception as error:
        return Result(False, error, v)


def as_list(xs: any, force: bool = False) -> list:
    if force:
        return [xs]
    elif sequence(xs):
        return list(xs)
    else:
        return [xs]


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


def pcall(f, *args, **kwargs) -> Result:
    try:
        output = f(*args, **kwargs)
        return Result(True, output)
    except Exception as error:
        return Result(False, error)


def whereis(binary: str) -> list[str]:
    out = system(f"whereis {binary}")
    out = split(out, r"\s+")
    out[0] = out[0][:-1]
    out.pop(0)

    return [x for x in out if os.access(x, os.X_OK)]


def paste0(
    *s: str | list[str],
    collapse: str = "",
) -> str:
    return (collapse).join(flatten(s))


def paste(
    *s: str | list[str],
    collapse: str = " ",
) -> str:
    return (collapse).join(flatten(s))


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
            keep=lambda s: pattern.search(s),
            index=False,
        )
        if not exclude
        else foreach(
            files,
            keep=lambda s: pattern.search(s),
            exclude=lambda s: exclude.search(s),
            index=False,
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


def push(
    xs: Sequence,
    *elements: any,
    index: int | None = None,
) -> Sequence:
    cls = type(xs)
    xs = list(xs)
    xs_len = len(xs)
    index = xs_len - 1 if index is None else index
    index = xs_len + index if index < 0 else index

    if index == xs_len - 1:
        for e in elements:
            xs.append(e)
    else:
        for e in elements[::-1]:
            xs.insert(index, e)

    return cls(xs)


def reverse(xs: tuple | list | str) -> tuple | list | str:
    return xs[::-1]


def unpush(xs: Sequence, *elements: any) -> Sequence:
    return push(xs, *elements, index=0)


def extend(
    xs: Sequence,
    *elements: any,
    index: int | None = None,
) -> Sequence:
    cls = type(xs)
    xs = list(xs)
    xs_len = len(xs)
    index = xs_len - 1 if index is None else index
    index = xs_len + index if index < 0 else index

    if index == xs_len - 1:
        for e in elements:
            if sequence(e):
                xs.extend(list(e))
            else:
                xs.append(e)
    else:
        for e in elements[::-1]:
            if sequence(e):
                xs = push(xs, *e, index=index)
            else:
                xs.insert(index, e)

    return cls(xs)


def lextend(xs: list, *elements: any) -> Sequence:
    cls = type(xs)
    xs = list(xs)

    for e in elements[::-1]:
        if sequence(e):
            xs = unpush(xs, *e)
        else:
            xs.insert(0, e)

    return cls(xs)


def identity(element: any) -> any:
    return element


def pop(
    xs: list | dict,
    index: int | str = -1,
    default: Callable | None = None,
    pcall: bool = False,
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
    pcall: bool = False,
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
    pcall: bool = False,
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
    pcall: bool = False,
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


def andgrep(
    s: str,
    *pattern: str | re.Pattern,
    flags=re.I,
) -> bool:
    for p in pattern:
        if not re.search(p, s, flags=flags):
            return False

    return True


def orgrep(
    s: str,
    *pattern: str | re.Pattern,
    flags=re.I,
) -> str | re.Pattern | None:
    for p in pattern:
        if re.search(p, s, flags=flags):
            return True

    return False


def strfind(
    s: str,
    pattern: re.Pattern | str,
    flags=re.I,
    start: int = 0,
) -> tuple[int, int] | None:
    s_required = s[start:]
    if m := re.search(pattern, s_required, flags=flags):
        span = m.span(0)
        return (span[0] + start, span[1] + start)


tbl_map = tbl_apply

__all__ = [
    # misc stuff
    "ARGV",
    "some",
    "ifelse",
    "unless",
    "ifNone",
    "unlessNone",
    "pcall",
    "has_argv",
    "isa",
    "p",
    "it",
    "identity",
    "partial",
    "is_error",
    "blank",
    "not_blank",
    "deepcopy",
    "shallowcopy",
    "Result",
    "cmdparser",
    #
    # file operations
    "rm",
    "cp",
    "cpstat",
    "slurp",
    "spit",
    "read_json",
    "write_json",
    "read_pkl",
    "write_pkl",
    "read_csv",
    "write_csv",
    "load_pkl",
    "dump_pkl",
    "load_json",
    "dump_json",
    "mkdir",
    "is_dir",
    "is_file",
    "is_mount",
    "is_link",
    "is_junction",
    "path_exists",
    "rmtree",
    "file_extension",
    "has_extension",
    "basename",
    "dirname",
    "abspath",
    "stat",
    "whereis",
    "ls",
    #
    # colored print
    "message",
    "msg_failure",
    "msg_success",
    "msg_ok",
    #
    # date and time stuff
    "strptime",
    "strftime",
    "date",
    "datetime",
    "time",
    #
    # string utils
    "grep",
    "startswith",
    "endswith",
    "split",
    "splitlines",
    "is_int",
    "as_int",
    "is_float",
    "as_float",
    "sed",
    "strip",
    "lstrip",
    "rstrip",
    "andgrep",
    "orgrep",
    "strfind",
    #
    # shell calls
    "system",
    "systemlist",
    #
    # container stuff
    "seq_along",
    "foreach",
    "keep",
    "sequence",
    "tbl_map",
    "tbl_apply",
    "tbl_keep",
    "tbl_exclude",
    "tbl_get",
    "tbl_set",
    "tbl_has",
    "tbl_grep",
    "tbl_grepv",
    "container",
    "flatten",
    "assoc",
    "as_list",
    "unwrap",
    "push",
    "reverse",
    "unpush",
    "extend",
    "lextend",
    "pop",
    "shift",
    "popn",
    "shiftn",
    "namedtuple",
    "reduce",
    #
    # Exception
    "make_exception",
    "is_exception",
    "as_exception",
    "make_error",
    "is_error",
    "as_error",
    "error_message",
    "get_error_message",
    "set_error_message",
    "exception_message",
    "get_exception_message",
    "set_exception_message",
    "error_class",
    "is_error_type",
    "is_error_instance",
    "exception_class",
    "is_exception_type",
    "is_exception_instance",
    #
    # menu
    "fzf",
    "menu",
]
