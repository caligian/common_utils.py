import re
import copy

from typing import Callable, overload, Any, Literal
from pyfzf import FzfPrompt
from functools import reduce
from math import ceil

from .result import Ok, Err, Result, T, E, is_ok, is_err
from .error import (
    error_msg,
    raise_error,
    is_error,
)
from .types import is_a, container, sequence, module

# from src.common_utils.result import Ok, Err, Result, T, E, is_ok, is_err
# from src.common_utils.error import (
#     error_msg,
#     raise_error,
#     is_error,
# )
# from src.common_utils.types import is_a, container, sequence


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

deepcopy = copy.deepcopy
shallowcopy = copy.copy


def make_empty_table(x_type: type[list | dict | tuple]) -> list | dict | tuple:
    if x_type is list:
        return []
    elif x_type is dict:
        return dict()
    else:
        return tuple()


@overload
def make_table(
    x_type: type[dict],
    size: int | None = None,
    index: list[int | str] | None = None,
    values: list[Any] | None = None,
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
) -> dict: ...


@overload
def make_table(
    x_type: type[list | tuple],
    size: int | None = None,
    index: list[int | str] | None = None,
    values: list[Any] | None = None,
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
) -> list | tuple: ...


def make_table(
    x_type: type[list | dict | tuple],
    size: int | None = None,
    index: list[int | str] | None = None,
    values: list[Any] | None = None,
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
) -> list | dict | tuple:
    size_given = size is not None

    if x_type is dict:
        res = {}
        if size_given and index and values:
            raise AssertionError(
                "Cannot pass size_given, index and value with dict input"
            )
        elif size_given and not (index and values):
            for i in range(size):
                if default_factory:
                    res[i] = default_factory()
                else:
                    res[i] = default
        elif (size_given and index) and not values:
            assert len(index) == size
            for i in range(size):
                if default_factory:
                    res[index[i]] = default_factory()
                else:
                    res[index[i]] = default
        elif size_given and values:
            assert len(index) == size
            for i in range(size):
                if default_factory:
                    res[i] = values[i]
                else:
                    res[i] = values[i]
        elif index and values:
            assert len(index) == len(values)
            return dict(zip(index, values))
        elif index:
            for i in index:
                if default_factory:
                    res[i] = default_factory()
                else:
                    res[i] = default
        elif values:
            for i, x in enumerate(values):
                res[i] = x

        return res
    elif index or values:
        raise AssertionError("Cannot pass index and/or value with non-dict input")
    elif size == 0:
        return make_empty_table(x_type)
    elif size:
        res = [(default_factory() if default_factory else default) for _ in range(size)]

        if x_type is tuple:
            return tuple(res)
        else:
            return res
    else:
        return make_empty_table()


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
    fn: ReducerWithIndex,
    index: Literal[True] = True,
    ignore_errors=True,
    raise_on_error=False,
) -> Any: ...


def treduce(
    tbl: Container,
    init: Any,
    fn: ReducerWithoutIndex,
    index: Literal[False] = False,
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


def tsome(
    x: Container,
    cond: Callable[[Any], bool] = lambda x: True if x else False,
) -> bool:
    it = x.values() if isinstance(x, dict) else x
    for value in it:
        if cond(value):
            return True
    return False


def tall(
    x: Container,
    cond: Callable[[Any], bool] = lambda x: True if x else False,
) -> bool:
    it = x.values() if isinstance(x, dict) else x
    for value in it:
        if not cond(value):
            return False
    return True


def tblank(s: str | list | tuple | dict | set) -> bool:
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
    index: Literal[False] = False,
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
    index: Literal[True] = True,
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
    index: Literal[False] = False,
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
    index: Literal[False] = False,
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
    tbl: Container,
    f: Callable[[Any], bool],
    index: Literal[False] = False,
) -> Container: ...


@overload
def tapply(
    tbl: Container,
    f: Callable[[int | str, Any], bool] = None,
    index: Literal[True] = True,
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


@overload
def thas(
    xs: Container,
    *ks: int | str | list[int, str],
    index: Literal[False] = False,
) -> list[bool]: ...


@overload
def thas(
    xs: Container,
    *ks: int | str | list[int, str],
    index: Literal[True] = True,
) -> list[tuple[int | str | list[int | str], bool]]: ...


def thas(
    xs: Container,
    *ks: int | str | list[int | str],
    index: bool = False,
) -> list[tuple[int | str | list[int | str]], bool] | list[bool]:
    res = []

    for k in ks:
        match assoc(xs, k):
            case Ok():
                if index:
                    res.append((k, True))
                else:
                    res.append(True)
            case Err():
                if index:
                    res.append((k, False))
                else:
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
        match assoc(xs, k, value=v, set=True):
            case Ok():
                continue
            case Err(error):
                if not pcall:
                    raise error
                else:
                    return error

    return xs


@overload
def assoc(
    d: dict,
    ks: int | str | list[int | str],
    value: Any = None,
    set: bool = False,
) -> Ok[dict] | Err[KeyError]: ...


@overload
def assoc(
    d: list | tuple,
    ks: int | str | list[int | str],
    value: Any = None,
    set: bool = False,
) -> Ok[list | tuple] | Err[IndexError]: ...


@overload
def assoc(
    d: dict,
    ks: int | str | list[int | str],
    value: Any = None,
    set: Literal[True] = False,
) -> Ok[Container] | Err[KeyError]: ...


def assoc(
    d: Container,
    ks: int | str | list[int | str],
    value: Any = None,
    set: bool = False,
) -> Ok[Any] | Err[KeyError | IndexError]:
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
            return Ok(d, dict(obj=d, level=v, k=ks[-1], index=len(ks) - 1))
        else:
            return Ok(v[k], dict(obj=d, level=v, k=ks[-1], index=len(ks) - 1))
    except Exception as error:
        return Err(
            error,
            dict(obj=d, ks=ks, level=v, k=ks[-1], index=len(ks) - 1),
        )


fassocTable = list | dict[int | str, Any]
fassocIndex = str | int
fassocIndices = list[str | int]
fassocDefaultContainer = type[dict] | type[list]
fassocDict = dict[int | str, Any]
fassocErrors = KeyError | IndexError | TypeError | ValueError


def fassoc(
    d: fassocTable,
    ks: fassocIndex | fassocIndices,
    value: Any,
    use: fassocDefaultContainer | None = None,
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
) -> Ok[fassocTable] | Err[fassocErrors]:
    use_cls = use if use is not None else dict
    ks = [ks] if is_a(ks, str, bytes, int) else ks

    if not is_a(d, dict):
        for i, k in enumerate(ks):
            if k < 0:
                return Err(
                    IndexError(f"Cannot use negative index `{k}` with list/tuple"),
                    dict(i=i, k=k, ks=ks, remaining=ks[i + 1 :], d=d, d_type=type(d)),
                )

    def get_default() -> Any:
        if default_factory:
            return default_factory()
        else:
            return default

    def set_dict_value(x: fassocDict, k: fassocIndex, v: Any) -> dict[str | int, Any]:
        x[k] = v
        return x

    def set_list_value(x: list, k: int, v: Any) -> list:
        x_len = len(x)

        if k >= x_len:
            for i in range((k - x_len) + 1):
                x.append(get_default())

        x[k] = v
        return x

    def set_value(x: fassocTable, k: fassocIndex, v: Any) -> fassocTable:
        if not is_a(x, list, dict):
            return Err(
                ValueError(
                    f"Only mutable data structures such as list and dict are allowed, got {type(x)}"
                ),
                dict(x=x, k=k, v=v, ks=ks, d=d),
            )
        elif is_a(x, list):
            try:
                return Ok(set_list_value(x, k, v))
            except IndexError as error:
                return Err(error, dict(d=d, ks=ks, x=x, k=k, v=v))
        else:
            return Ok(set_dict_value(x, k, v))

    def index_into(
        x: fassocTable,
        k: fassocIndex,
        use: fassocDefaultContainer,
    ) -> Ok[fassocTable] | Err[IndexError]:
        try:
            inside = x[k]
            if container(inside):
                return Ok(inside)
            else:
                return set_value(x, k, use())
        except (IndexError, KeyError, TypeError):
            return set_value(x, k, use())

    tbl = d
    for k in ks[:-1]:
        inner = index_into(tbl, k, use_cls)

        if is_err(inner):
            return inner
        else:
            tbl = inner.unwrap()

    last_key = ks[-1]
    res = index_into(tbl, last_key, use_cls)

    if is_err(res):
        return res
    else:
        set_value(tbl, last_key, value)
        return Ok(d, res.metadata)


def as_list(xs: Any, force: bool = False) -> list:
    if force:
        return [xs]
    elif sequence(xs):
        return list(xs)
    else:
        return [xs]


def flatten(xs: list | tuple, maxdepth: int = -1) -> list | tuple:
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
    if isinstance(xs, tuple):
        return tuple(result)
    else:
        return result


def paste0(*s: str | list[str]) -> str:
    return ("").join(flatten(s))


def paste(*s: str | list[str], collapse: str = " ") -> str:
    return (collapse).join(flatten(s))


def cat(*seqs: list | tuple) -> list | tuple:
    res = []

    for x in seqs:
        res.extend(list(x))

    return res


def merge(*dicts: dict, overwrite: bool = True) -> dict:
    res = {}

    for d in dicts:
        for k, v in d.items():
            if overwrite or (res.get(k) is None):
                res[k] = v
    return res


def lmerge(*dicts: dict) -> dict:
    return merge(*dicts, overwrite=False)


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


def reverse(xs: dict | tuple | list | str) -> tuple | list | str:
    if isinstance(xs, dict):
        keys = tkeys(xs)
        res = {}

        for k in keys[::-1]:
            res[k] = xs[k]

        return res
    else:
        return xs[::-1]


def unpush(xs: Sequence, *elements: Any) -> Sequence:
    return push(xs, *elements, index=0)


@overload
def extend(
    xs: list,
    *elements: Any,
    index: int | None = None,
    cast: Literal[False] = False,
) -> list: ...


@overload
def extend(
    xs: tuple,
    *elements: Any,
    index: int | None = None,
    cast: Literal[False] = False,
) -> list: ...


@overload
def extend(
    xs: list,
    *elements: Any,
    index: int | None = None,
    cast: Literal[True],
) -> list: ...


@overload
def extend(
    xs: tuple,
    *elements: Any,
    index: int | None = None,
    cast: Literal[True],
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
def lextend(
    xs: tuple,
    *elements: Any,
    cast: Literal[False] = False,
) -> list: ...


@overload
def lextend(
    xs: list,
    *elements: Any,
    cast: Literal[True],
) -> list: ...


@overload
def lextend(
    xs: tuple,
    *elements: Any,
    cast: Literal[True],
) -> tuple: ...


def lextend(
    xs: Sequence,
    *elements: Any,
    cast: bool = False,
) -> list | tuple:
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


@overload
def pop(
    xs: list | tuple,
    index: int | str = -1,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Ok[Any] | Err[IndexError]: ...


@overload
def pop(
    xs: dict,
    index: int | str = -1,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Ok[Any] | Err[KeyError]: ...


def pop(
    xs: Container,
    index: int | str = -1,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Ok[Any] | Err[KeyError | IndexError]:
    ys: list
    if isinstance(xs, tuple):
        ys = list(xs)
    else:
        ys = xs

    metadata = dict(x=xs, index=index)
    if isinstance(ys, dict) and isinstance(index, int):
        try:
            index = list(ys.keys())[index]
        except (IndexError, KeyError) as error:
            if default_factory:
                return Ok(default_factory(), metadata)
            elif default is not None:
                return Ok(default, metadata)
            elif pcall:
                return Err(error, metadata)
            else:
                raise_error(error, msg=error_msg(error), args=metadata)

    try:
        return Ok(ys.pop(index), metadata)
    except (IndexError, KeyError) as error:
        if default_factory:
            return Ok(default_factory(), metadata)
        elif default is not None:
            return Ok(default, metadata)
        elif pcall:
            return Err(error, metadata)
        else:
            raise_error(error, msg=error_msg(error), args=metadata)


@overload
def shift(
    xs: dict,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Ok[Any] | Err[KeyError]: ...


@overload
def shift(
    xs: list | tuple,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Ok[Any] | Err[IndexError]: ...


def shift(
    xs: Container,
    default: Any | None = None,
    default_factory: Callable | None = None,
    pcall: bool = False,
) -> Ok[Any] | Err[KeyError | IndexError]:
    return pop(
        xs,
        index=0,
        default=default,
        default_factory=default_factory,
        pcall=pcall,
    )


def popn(
    xs: Container,
    n: int = 1,
    index: int | str = -1,
    reverse: bool = False,
    pcall: bool = False,
    default_factory: Callable | None = None,
    default: Any | None = None,
) -> Ok[list[Any]] | Err[KeyError | IndexError]:
    res = []
    for i in range(n):
        match pop(
            xs,
            index=index,
            default=default,
            default_factory=default_factory,
            pcall=pcall,
        ):
            case Ok(value):
                res.append(value)
            case Err() as err:
                if not pcall:
                    raise_error(
                        err.value,
                        msg=error_msg(err.value),
                        args=err.metadata,
                    )
                else:
                    return err

    metadata = dict(x=xs, index=index, n=n)
    if reverse:
        return Ok(res[::-1], metadata)
    else:
        return Ok(res, metadata)


def shiftn(
    xs: list | tuple | dict,
    n: int = 1,
    index: int = -1,
    reverse: bool = False,
    pcall: bool = False,
    default: Callable | None = None,
) -> Ok[list[Any]] | Err[IndexError | KeyError | ValueError]:
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
    **kwargs,
) -> re.Match | None:
    for pat in flatten(pattern, -1):
        if m := re.search(pat, s, **kwargs):
            return m


@overload
def grep(
    x: list[T],
    *pattern: str | Pattern,
    value: Literal[True] = False,
    invert: Literal[False] = False,
    **kwargs,
) -> list[T]: ...


@overload
def grep(
    x: dict[str, T],
    *pattern: str | Pattern,
    value: Literal[True] = False,
    invert: Literal[False] = False,
    **kwargs,
) -> dict[str, T | None]: ...


@overload
def grep(
    x: list[T],
    *pattern: str | Pattern,
    value: Literal[False] = False,
    invert: Literal[True] = False,
    **kwargs,
) -> list[bool]: ...


@overload
def grep(
    x: dict[str, T],
    *pattern: str | Pattern,
    value: Literal[False] = False,
    invert: Literal[True] = False,
    **kwargs,
) -> dict[str, bool]: ...


@overload
def grep(
    x: dict[str, T],
    *pattern: str | Pattern,
    value: Literal[True] = False,
    invert: Literal[False] = False,
    **kwargs,
) -> dict[str, T | None]: ...


@overload
def grep(
    x: tuple,
    *pattern: str | Pattern,
    value: Literal[True] = False,
    invert: bool = False,
    **kwargs,
) -> tuple: ...


def grep(
    x: tuple | list[T] | dict[str, T],
    *pattern: str | Pattern,
    value: bool = False,
    invert: bool = False,
    **kwargs,
) -> tuple | list[T | None] | dict[str, T | None] | list[bool] | dict[str, bool]:
    if isinstance(x, dict):
        res = {}

        for k, elem in x.items():
            ok = isinstance(elem, str) and strmatch(elem, *pattern)
            ok = not ok and invert

            if ok:
                if value:
                    res[k] = elem
                else:
                    res[k] = True
            elif not value:
                res[k] = False
            else:
                res[k] = None

        return res

    res = []
    dst_type = type(x)

    for elem in x:
        ok = isinstance(elem, str) and strmatch(elem, *pattern)
        ok = not ok and invert

        if ok:
            if value:
                res.append(elem)
            else:
                res.append(True)
        elif not value:
            res.append(False)
        else:
            res.append(None)

    return dst_type(res)


@overload
def grepv(
    x: list[T],
    *pattern: str | Pattern,
    value: Literal[True] = False,
    invert: Literal[False] = False,
    **kwargs,
) -> list[T]: ...


@overload
def grepv(
    x: dict[str, T],
    *pattern: str | Pattern,
    value: Literal[True] = False,
    invert: Literal[False] = False,
    **kwargs,
) -> dict[str, T | None]: ...


@overload
def grepv(
    x: list[T],
    *pattern: str | Pattern,
    value: Literal[False] = False,
    invert: Literal[True] = False,
    **kwargs,
) -> list[bool]: ...


@overload
def grepv(
    x: dict[str, T],
    *pattern: str | Pattern,
    value: Literal[False] = False,
    **kwargs,
) -> dict[str, bool]: ...


@overload
def grepv(
    x: dict[str, T],
    *pattern: str | Pattern,
    value: Literal[True] = False,
    **kwargs,
) -> dict[str, T | None]: ...


@overload
def grepv(
    x: tuple,
    *pattern: str | Pattern,
    value: Literal[True] = False,
    **kwargs,
) -> tuple: ...


def grepv(
    x: tuple | list[T] | dict[str, T],
    *pattern: str | Pattern,
    value: bool = False,
    **kwargs,
) -> tuple | list[T | None] | dict[str, T | None] | list[bool] | dict[str, bool]:
    return grep(x, *pattern, invert=True, **kwargs)


def cut(
    x: Container,
    start: int = 0,
    end: int = None,
    step: int = 1,
) -> Container:
    x_len = len(x)
    end = x_len if end is None else end
    end = (x_len + end) if end < 0 else end
    start = (x_len + start) if start < 0 else start

    if (start < 0 or end < 0) or (start > end or start > x_len) or (end > x_len):
        raise IndexError(dict(x=x, start=start, end=end))

    if not isinstance(x, dict):
        return x[start:end:step]
    else:
        ks = tuple(x.keys())
        res = {}

        for k in ks[start:end:step]:
            res[k] = x[k]

        return res


@overload
def partition(x: list[T], n: int) -> list[list[T]]: ...


@overload
def partition(x: tuple, n: int) -> list[tuple]: ...


@overload
def partition(
    x: dict[str | int, T],
    n: Callable[[Any], bool],
) -> list[dict[str | int, T]]: ...


@overload
def partition(
    x: list[T],
    n: Callable[[Any], bool],
) -> tuple[list[T], list[T]]: ...


@overload
def partition(
    x: tuple,
    n: Callable[[Any], bool],
) -> tuple[tuple, tuple]: ...


@overload
def partition(
    x: dict[str | int, T],
    n: Callable[[Any], bool],
) -> tuple[dict[str | int, T], dict[str | int, T]]: ...


def partition(
    x: Container,
    n: int | Callable[[Any], bool],
) -> (
    tuple[dict[str | int, T], dict[str | int, T]]
    | tuple[tuple, tuple]
    | tuple[list[T], list[T]]
    | list[dict[str | int, T]]
    | list[list[T]]
    | list[tuple]
):
    is_dict = isinstance(x, dict)
    if not is_dict and callable(n):
        res = ([], [])
        for elem in x:
            if not n(elem):
                res[1].append(elem)
            else:
                res[0].append(elem)

        return res
    elif not is_dict:
        return chunk(x, ceil(len(x) / n))

    ks = tuple(x.keys())
    partitioned = partition(ks, n)
    res = []

    for ks in partitioned:
        res.append(dict(zip(ks, [x[k] for k in ks])))

    return res


def chunk(x: Container, chunk_size: int) -> list[Container]:
    if isinstance(x, dict):
        ks = tuple(x.keys())
        chunked = chunk(ks, chunk_size=chunk_size)
        return [dict(zip(ks, [x[k] for k in ks])) for ks in chunked]

    res = []
    for i in range(0, len(x), chunk_size):
        res.append(x[i : i + chunk_size])

    return res


def car(x: Container) -> Any | None:
    if isinstance(x, dict):
        try:
            return x[tuple(x.keys())[0]]
        except Exception:
            return

    try:
        return x[0]
    except Exception:
        return


def cdr(x: Container) -> Sequence:
    return cut(x, start=1)


def butlast(x: Container) -> Any:
    is_dict = isinstance(x, dict)
    if len(x) < 1:
        if is_dict:
            return {}
        elif isinstance(x, tuple):
            return tuple()
        else:
            return []

    return cut(x, end=-1)


def head(x: Container, n: int = 1) -> Sequence:
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


def _test():
    print(table(list, size=10))
    print("---")
    print(table(dict, size=10))
    print("---")
    print(
        table(
            dict,
            size=20,
            default=True,
        )
    )
    print(cut(table(list, size=10), 0, -2))
    print(cut(table(dict, size=10), 0, -2, 2))
    print(chunk(table(dict, size=11), 2))
    print(partition(table(dict, size=15), 3))
    print(partition(table(tuple, size=19), 2))
    print(grep(["a", "b", "c", "d"], "[a-b]", value=True))
    print(grep(["a", "b", "c", "d"], "[a-c]", value=False))
    print(grepv(["a", "b", "c", "d"], "[a-c]", value=False))
    print(grepv(dict(zip([0, 1, 2, 3], ["a", "b", "c", "d"])), "[a-c]", value=False))


# fix pop and do not autoconvert int to str for dict index - it is incorrect
# fix fassoc as well

# _test()

# x = dict(a=1, b=[2, 3, 4], c=[1, 2, 3])
# print(fassoc(x, ("b", 1, 1), 10).unwrap()

table = module("table")
table.add_methods(
    assoc=assoc,
    butlast=butlast,
    car=car,
    cat=cat,
    cdr=cdr,
    chunk=chunk,
    cut=cut,
    empty=make_empty_table,
    new=make_table,
    extend=extend,
    fassoc=fassoc,
    flatten=flatten,
    fzf=fzf,
    grep=grep,
    grepv=grepv,
    head=head,
    lextend=lextend,
    lmerge=lmerge,
    lstrip=lstrip,
    merge=merge,
    partition=partition,
    pop=pop,
    popn=popn,
    push=push,
    reverse=reverse,
    seq=seq,
    shift=shift,
    shiftn=shiftn,
    tail=tail,
    all=tall,
    apply=tapply,
    blank=tblank,
    exclude=texclude,
    map=tfor,
    foreach=tfor,
    get=tget,
    has=thas,
    items=titems,
    keep=tkeep,
    keys=tkeys,
    reduce=treduce,
    set=tset,
    some=tsome,
    values=tvalues,
    unpush=unpush,
)

__all__ = [
    "table",
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
    "cat",
    "cdr",
    "chunk",
    "cut",
    "make_empty_table",
    "make_table",
    "endswith",
    "extend",
    "fassoc",
    "flatten",
    "fzf",
    "grep",
    "grepv",
    "head",
    "identity",
    "is_float",
    "is_int",
    "lextend",
    "lmerge",
    "lstrip",
    "merge",
    "orgrep",
    "partition",
    "paste",
    "paste0",
    "pop",
    "popn",
    "push",
    "reverse",
    "rstrip",
    "sed",
    "seq",
    "shift",
    "shiftn",
    "split",
    "splitlines",
    "startswith",
    "strfind",
    "strip",
    "strmatch",
    "table",
    "tail",
    "tall",
    "tapply",
    "tblank",
    "texclude",
    "tfor",
    "tget",
    "thas",
    "titems",
    "tkeep",
    "tkeys",
    "treduce",
    "tset",
    "tsome",
    "tvalues",
    "unpush",
]
