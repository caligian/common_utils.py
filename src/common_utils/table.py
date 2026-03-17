import re

from typing import Callable
from pyfzf import FzfPrompt


from .result import (
    Success,
    Failure,
)
from .error import (
    raise_error,
    is_error,
)
from functools import reduce

Pattern = re.Pattern
Container = list | tuple | dict
Sequence = list | tuple


def treduce(
    tbl: Container,
    init: any = None,
    fn: Callable[[any, any], any] = lambda elem, acc: (elem, acc),
) -> Container:
    pass


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
        assert isinstance(x, (tuple, list))
        for value in x:
            if not value:
                return False

        return True


def blank(s: str | list | tuple | dict) -> bool:
    return len(s) == 0


def not_blank(s: str | list | tuple | dict) -> bool:
    return len(s) > 0


def seq_along(xs: Container) -> list[int]:
    if isinstance(xs, (list, tuple)):
        return list(range(len(xs)))
    else:
        return list(xs.keys())


def foreach(
    tbl: Container,
    apply: Callable[[int | str, any], any] | Callable[[any], any] | None = None,
    keep: Callable[[int | str, any], any] | Callable[[any], any] | None = None,
    exclude: Callable[[int | str], any] | Callable[[any], any] | None = None,
    until: Callable[[int, str], bool] | Callable[[any], any] | None = None,
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

        if not until:

            def until(_, x):
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

        if not until:

            def until(x):
                return False

    def append(res, k=None, v=None):
        if isinstance(res, dict):
            assert k
            res[k] = v
        elif k:
            res.append((k, v))
        else:
            res.append(v)

    res = {} if isinstance(tbl, dict) else []
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
            if keep(k, v) and not exclude(k, v) and not until(k, v):
                if ignore_errors:
                    if not is_error(v):
                        new_value = apply(k, v)
                        append(res, k=k, v=new_value)
                elif is_error(v):
                    raise_error(v, f"Key passed: {k}")
                else:
                    new_value = apply(k, v)
                    append(res, k=k, v=new_value)
    else:
        for x in it:
            if keep(x) and not exclude(x) and not until(x):
                if ignore_errors:
                    if not is_error(x):
                        new_value = apply(x)
                        append(res, k=None, v=new_value)
                elif is_error(x):
                    raise_error(x, f"Key passed: {k}")
                else:
                    new_value = apply(x)
                    append(res, k=None, v=new_value)

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
            case Success(x):
                res.append(x)
            case Failure(error):
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
            case Success(value):
                res.append(value)
            case Failure():
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
            case Success():
                continue
            case Failure(error):
                if not pcall:
                    raise error
                else:
                    return error

    return xs


def assoc(
    d: Container,
    ks: any,
    value: any = None,
) -> Success | Failure:
    ks = [ks] if not sequence(ks) else ks
    v = d

    for k in ks[:-1]:
        try:
            v = v[k]
        except Exception as error:
            return Failure(error)

    k = ks[-1]
    try:
        if value is not None:
            v[k] = value
        return Success(v[k])
    except Exception as error:
        return Failure(error)


def as_list(xs: any, force: bool = False) -> list:
    if force:
        return [xs]
    elif sequence(xs):
        return list(xs)
    else:
        return [xs]


def sequence(xs: list | tuple) -> bool:
    return isinstance(xs, (tuple, list))


def container(xs: dict | list | tuple) -> bool:
    return isinstance(xs, (tuple, int, dict))


def flatten(xs: list | tuple, maxdepth: int = -1) -> list:
    result = []

    def vector(lst: list | tuple, current_depth: int = 0) -> list:
        nonlocal result
        if current_depth != -1 and current_depth == maxdepth:
            result.append(lst)
            return

        for i, x in enumerate(lst):
            if sequence(x):
                vector(x, current_depth + 1)
            else:
                result.append(x)

    vector(xs, 0)
    return result


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


def andgrep(
    s: str,
    *pattern: str | re.Pattern,
    flags=re.I,
) -> bool:
    for pat in pattern:
        if not re.search(pat, s, flags=flags):
            return False

    return True


def orgrep(
    s: str,
    *pattern: str | re.Pattern,
    flags=re.I,
) -> str | re.Pattern | None:
    for pat in pattern:
        if re.search(pat, s, flags=flags):
            return pat


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


def strmatch(
    s: str,
    *pattern: str | Pattern,
    invert: bool = False,
    **kwargs,
) -> re.Match | None:
    for pat in pattern:
        if m := re.search(pat, s, **kwargs):
            if invert:
                return False
            else:
                return m

    if invert:
        return False
    else:
        return True


def grep(
    x: Container,
    *pattern,
    invert: bool = False,
    **kwargs,
) -> Container:
    if isinstance(x, dict):
        res = {}
        for k, v in x.items():
            if isinstance(v, str):
                for p in pattern:
                    if re.search(p, v, **kwargs) and not invert:
                        res[k] = v

        return res
    else:
        assert isinstance(x, (list, tuple))

        res = []
        dst_type = type(x)

        for elem in x:
            if isinstance(elem, str):
                for p in pattern:
                    if re.search(p, elem, **kwargs) and not invert:
                        res.append(elem)

        return dst_type(res)


def grepv(x: Container, *pattern, **kwargs) -> Container:
    return grep(x, *pattern, invert=True, **kwargs)


def car(x: Sequence) -> any:
    try:
        return x[0]
    except Exception:
        return


def cdr(x: Sequence) -> Sequence:
    return x[1:]


def butlast(x: Sequence) -> any:
    return x[: len(x) - 1]


def head(x: Sequence, n: int = 1) -> Sequence:
    assert n >= 0
    if len(x) <= n:
        return x

    return [x[i] for i in range(n)]


def tail(x: Sequence, n: int = 1) -> Sequence:
    assert n >= 0

    x_len = len(x)
    if x_len <= n:
        return x

    return [x[i] for i in range(x_len - n, x_len)]


def seq(start: int, end: int, step: int = 1) -> list[int]:
    return [x for x in range(start, end, step)]


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


tbl_grep = grep
tbl_grepv = grepv


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
    _dict = isinstance(tbl, dict)

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


tbl_map = tbl_apply
tbl_filter = tbl_keep
tbl_reduce = treduce
tmap = tbl_map
tfilter = tbl_filter
aslist = as_list
empty = blank
not_empty = not_blank
isint = is_int
isfloat = is_float
asint = as_int
asfloat = as_float
agrep = andgrep
ogrep = orgrep
tapply = tbl_apply
texclude = tbl_exclude
tget = tbl_get
tset = tbl_set
tfilter = tbl_filter
tmap = tbl_map
tkeep = tbl_keep
tgrep = tbl_grep
tgrepv = tbl_grepv

__all__ = [
    "asint",
    "asfloat",
    "as_int",
    "as_float",
    "isint",
    "isfloat",
    "is_int",
    "is_float",
    "all",
    "andgrep",
    "as_list",
    "aslist",
    "assoc",
    "blank",
    "empty",
    "not_empty",
    "container",
    "extend",
    "flatten",
    "foreach",
    "grep",
    "grepv",
    "identity",
    "keep",
    "lextend",
    "not_blank",
    "orgrep",
    "paste",
    "paste0",
    "pop",
    "popn",
    "push",
    "reverse",
    "seq_along",
    "sequence",
    "shift",
    "shiftn",
    "some",
    "split",
    "splitlines",
    "startswith",
    "endswith",
    "strfind",
    "tbl_apply",
    "tbl_exclude",
    "tbl_filter",
    "tbl_get",
    "tbl_has",
    "tbl_keep",
    "tbl_map",
    "tbl_set",
    "tbl_grep",
    "tbl_grepv",
    "tmap",
    "tapply",
    "texclude",
    "treduce",
    "tget",
    "tset",
    "tfilter",
    "tmap",
    "tkeep",
    "tgrep",
    "tgrepv",
    "unpush",
    "head",
    "tail",
    "unwrap",
    "butlast",
    "car",
    "seq",
    "cdr",
    "strip",
    "lstrip",
    "rstrip",
    "reduce",
    "fzf",
    "agrep",
    "ogrep",
    "strmatch",
    #
    # Other classes
    "Container",
    "Sequence",
    "Pattern",
    "sed",
]
