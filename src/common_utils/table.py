import re

from typing import Callable, overload, Any, Literal
from pyfzf import FzfPrompt
from functools import reduce
from .result import Ok, Err, Result, T, E
from .error import (
    raise_error,
    is_error,
)

Pattern = re.Pattern
Container = list | tuple | dict
Sequence = list | tuple
ReducerWithIndex = Callable[[int | str, Any, Any], Any]
TransformerWithIndex = Callable[[Any], Any]
FilterWithIndex = Callable[[Any], bool]
ReducerWithoutIndex = Callable[[Any, Any], Any]
TransformerWithoutIndex = Callable[[Any], Any]
FilterWithoutIndex = Callable[[Any], bool]
Reducer = ReducerWithIndex | ReducerWithoutIndex
Transformer = TransformerWithIndex | TransformerWithoutIndex
Filter = FilterWithIndex | FilterWithoutIndex



@overload
def treduce(
    tbl,
    init,
    fn: Callable[[Any, Any], Any],
    index=False,
    ignore_errors=True,
    raise_on_error=False,
): ...


@overload
def treduce(
    tbl,
    init,
    fn: Callable[[int | str, Any, Any], Any],
    index=True,
    ignore_errors=True,
    raise_on_error=False,
): ...


def treduce(
    tbl: Container,
    init: Any | None = None,
    fn: Reducer | None = None,
    index: bool = False,
    ignore_errors: bool = True,
    raise_on_error: bool = False,
) -> Any:
    if fn is None:
        if index:
            fn = lambda k, elem, acc: acc
        else:
            fn = lambda elem, acc: acc

    def check(v) -> bool:
        if isinstance(v, Exception):
            if raise_on_error:
                raise v
            else:
                return not ignore_errors
        else:
            return True

    res: Any = init
    if index:
        it = tbl.items() if isinstance(tbl, dict) else enumerate(tbl)
        for k, v in it:
            if check(v):
                res = fn(k, v, res)
    else:
        it = tbl.values() if isinstance(tbl, dict) else tbl
        for v in it:
            if check(v):
                res = fn(v, res)

    return res


def tsome(x: Container) -> bool:
    if isinstance(x, dict):
        for value in x.values():
            if value:
                return True
    else:
        for value in x:
            if value:
                return True

    return False


def tall(x: Container) -> bool:
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


def tblank(s: str | list | tuple | dict) -> bool:
    return len(s) == 0


def tkeys(xs: Container) -> list[int | str]:
    if isinstance(xs, (list, tuple)):
        return list(range(len(xs)))
    else:
        return list(xs.keys())


def titems(xs: Container) -> list[tuple[int | str, Any]]:
    if isinstance(xs, (list, tuple)):
        return list(enumerate(xs))
    else:
        return list(xs.items())


def tvalues(xs: Container) -> list[Any]:
    if isinstance(xs, (list, tuple)):
        return xs
    else:
        return list(xs.values())


@overload
def tfor(
    tbl,
    index=False,
    apply: Callable[[Any], Any] | None = None,
    keep: Callable[[Any], bool] | None = None,
    exclude: Callable[[Any], bool] | None = None,
    until: Callable[[Any], bool] | None = None,
    when: Callable[[int | str, Any], bool] | None = None,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
    cast: bool = False,
) -> Container: ...


@overload
def tfor(
    tbl,
    index=True,
    apply: Callable[[int | str, Any], Any] | None = None,
    keep: Callable[[int | str, Any], bool] | None = None,
    exclude: Callable[[int | str, Any], bool] | None = None,
    until: Callable[[int | str, Any], bool] | None = None,
    when: Callable[[int | str, Any], bool] | None = None,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
    cast: bool = False,
) -> Container: ...


def tfor(
    tbl: Container,
    apply: Transformer | None = None,
    keep: Filter | None = None,
    exclude: Filter | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
    index: bool = False,
    cast: bool = False,
) -> Container:
    if index:
        apply = apply or (lambda _, x: x)
        keep = keep or (lambda _, x: True)
        exclude = exclude or (lambda _, x: False)
        until = until or (lambda _, x: False)
        when = when or (lambda _, x: True)
    else:
        apply = apply or (lambda x: x)
        keep = keep or (lambda x: True)
        exclude = exclude or (lambda x: False)
        until = until or (lambda x: False)
        when = when or (lambda x: True)

    def append(
        res: Container,
        k: str | int | None = None,
        v: Any | None = None,
    ) -> None:
        if isinstance(res, dict):
            res[k] = v
        elif k is not None:
            res.append((k, v))
        else:
            res.append(v)

    def check(v) -> bool:
        if isinstance(v, Exception):
            if raise_on_error:
                raise v
            elif ignore_errors:
                return False
            else:
                return True
        else:
            return True

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
            if not check(v):
                continue
            elif when(k, v) and keep(k, v) and not exclude(k, v) and not until(k, v):
                append(res, k=k, v=apply(k, v))
    else:
        for x in it:
            if not check(x):
                continue
            elif when(x) and keep(x) and not exclude(x) and not until(x):
                append(res, v=apply(x))

    if cast:
        return type(tbl)(res)
    else:
        return res


@overload
def tkeep(
    tbl,
    f: Callable[[Any], bool],
    exclude: Callable[[Any], bool] | None = None,
    index=False,
) -> Container: ...


@overload
def tkeep(
    tbl,
    f: Callable[[int | str, Any], bool] = None,
    exclude: Callable[[int | str, Any], bool] | None = None,
    index=True,
) -> Container: ...


def tkeep(
    tbl: Container,
    f: Filter,
    exclude: Filter | None = None,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Container:
    return tfor(
        tbl,
        keep=f,
        exclude=exclude,
        index=index,
        ignore_errors=ignore_errors,
        raise_on_error=raise_on_error,
    )


@overload
def texclude(
    tbl,
    f: Callable[[Any], bool],
    keep: Callable[[Any], bool] | None = None,
    index=False,
) -> Container: ...


@overload
def texclude(
    tbl,
    f: Callable[[int | str, Any], bool] = None,
    keep: Callable[[int | str, Any], bool] | None = None,
    index=True,
) -> Container: ...


def texclude(
    tbl: Container,
    f: Filter,
    keep: Filter | None = None,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Container:
    return tfor(
        tbl,
        exclude=f,
        keep=keep,
        index=index,
        ignore_errors=ignore_errors,
        raise_on_error=raise_on_error,
    )


@overload
def tapply(
    tbl,
    f: Callable[[Any], bool],
    index=False,
) -> Container: ...


@overload
def tapply(
    tbl,
    f: Callable[[int | str, Any], bool] = None,
    index=True,
) -> Container: ...


def tapply(
    tbl: Container,
    f: Transformer,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Container:
    return tfor(
        tbl,
        apply=f,
        index=index,
        ignore_errors=ignore_errors,
        raise_on_error=raise_on_error,
    )


def tget(
    xs: Container,
    *ks: int | str | list[int | str],
    pcall: bool = False,
) -> list[Any | Exception]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case Ok(x):
                res.append(x)
            case Err(error):
                if not pcall:
                    raise error
                else:
                    res.append(error)

    return res


def thas(
    xs: Container,
    *ks: int | str | list[int | str],
) -> list[bool]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case Ok():
                res.append(True)
            case Err():
                res.append(False)

    return res


def tset(
    xs: Container,
    *keys_and_values: tuple[Any, Any],
    pcall: bool = False,
) -> Container | Exception:
    if len(keys_and_values) == 0:
        return xs

    for k, v in keys_and_values:
        match assoc(xs, k, value=v):
            case Ok():
                continue
            case Err(error):
                if not pcall:
                    raise error
                else:
                    return error

    return xs


def assoc(
    d: Container,
    ks: int | str | list[int | str],
    value: Any = None,
    set: bool = False,
) -> Result[T, E]:
    ks = [ks] if not sequence(ks) else ks
    v = d

    for i, k in enumerate(ks[:-1]):
        try:
            v = v[k]
        except Exception as error:
            return Err(
                error,
                dict(obj=d, ks=ks, k=k, index=i),
            )

    k = ks[-1]
    try:
        if set:
            v[k] = value
        return Ok(v[k])
    except Exception as error:
        return Err(
            error,
            dict(obj=d, ks=ks, k=ks[-1], index=len(ks) - 1),
        )


def as_list(xs: Any, force: bool = False) -> list:
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


def push(
    xs: Sequence,
    *elements: Any,
    index: int | None = None,
) -> Sequence:
    cls = type(xs)
    xs = list(xs)
    xs_len = len(xs)

    if index is None:
        index = xs_len
    elif index < 0:
        index = max(0, xs_len + index)

    if index == xs_len - 1:
        for e in elements:
            xs.append(e)
    else:
        for e in elements[::-1]:
            xs.insert(index, e)

    return cls(xs)


def reverse(xs: tuple | list | str) -> tuple | list | str:
    return xs[::-1]


def unpush(xs: Sequence, *elements: Any) -> Sequence:
    return push(xs, *elements, index=0)


@overload
def extend(
    xs: list, *elements: Any, index: int | None = None, cast: Literal[False] = False
) -> list: ...


@overload
def extend(
    xs: tuple, *elements: Any, index: int | None = None, cast: Literal[False] = False
) -> list: ...


@overload
def extend(
    xs: list, *elements: Any, index: int | None = None, cast: Literal[True]
) -> list: ...


@overload
def extend(
    xs: tuple, *elements: Any, index: int | None = None, cast: Literal[True]
) -> tuple: ...


def extend(
    xs: Sequence,
    *elements: Any,
    index: int | None = None,
    cast: bool = False,
) -> list | tuple:
    cls = type(xs)
    xs = list(xs)
    xs_len = len(xs)
    index = xs_len - 1 if index is None else index
    index = xs_len + index if index < 0 else index

    if index >= xs_len - 1:
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

    if cast:
        return cls(xs)
    else:
        return xs


@overload
def lextend(xs: tuple, *elements: Any, cast: Literal[False] = False) -> list: ...


@overload
def lextend(xs: list, *elements: Any, cast: Literal[True]) -> list: ...


@overload
def lextend(xs: tuple, *elements: Any, cast: Literal[True]) -> tuple: ...


def lextend(xs: Sequence, *elements: Any, cast: bool = False) -> list | tuple:
    cls = type(xs)
    xs = list(xs)

    for e in elements[::-1]:
        if sequence(e):
            xs = unpush(xs, *e)
        else:
            xs.insert(0, e)

    if cast:
        return cls(xs)
    else:
        return xs


def identity(element: Any) -> Any:
    return element


def pop(
    xs: list | dict,
    index: int | str = -1,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Any | KeyError | IndexError:
    if isinstance(xs, dict) and isinstance(index, int):
        try:
            index = list(xs.keys())[index]
        except (IndexError, KeyError) as error:
            if default_factory:
                return default_factory()
            elif default is not None:
                return default
            elif pcall:
                return error
            else:
                raise error

    try:
        return xs.pop(index)
    except (IndexError, KeyError) as error:
        if default_factory:
            return default_factory()
        elif default is not None:
            return default
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
) -> list[Any]:
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
) -> list[Any]:
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


def car(x: Sequence) -> Any:
    try:
        return x[0]
    except Exception:
        return


def cdr(x: Sequence) -> Sequence:
    return x[1:]


def butlast(x: Sequence) -> Any:
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
    tbl: dict[str, Any] | list | tuple,
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
    index = tkeys(tbl)
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


__all__ = [
    "Container",
    "Pattern",
    "Sequence",
    "andgrep",
    "as_float",
    "as_int",
    "as_list",
    "assoc",
    "butlast",
    "car",
    "cdr",
    "container",
    "endswith",
    "extend",
    "flatten",
    "fzf",
    "grep",
    "grepv",
    "head",
    "identity",
    "is_float",
    "is_int",
    "lextend",
    "lstrip",
    "orgrep",
    "paste",
    "paste0",
    "pop",
    "popn",
    "push",
    "reduce",
    "reverse",
    "rstrip",
    "sed",
    "seq",
    "tkeys",
    "tvalues",
    "titems",
    "sequence",
    "shift",
    "shiftn",
    "split",
    "splitlines",
    "startswith",
    "strfind",
    "strip",
    "strmatch",
    "tail",
    "tall",
    "tapply",
    "tblank",
    "texclude",
    "tfor",
    "tget",
    "thas",
    "tkeep",
    "treduce",
    "tset",
    "tsome",
    "unpush",
]
