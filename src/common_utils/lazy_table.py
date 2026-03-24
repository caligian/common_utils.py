import re

from typing import Callable, overload, Any, Literal, Iterator, Generator
from pyfzf import FzfPrompt
from functools import reduce
from .result import Ok, Err, Result, T, E
from .error import (
    raise_error,
    is_error,
)
from .table import (
    sequence,
    container,
    Pattern,
    Container,
    Sequence,
    Reducer,
    Transformer,
    Filter,
    FilterWithIndex,
    FilterWithoutIndex,
    ReducerWithIndex,
    ReducerWithoutIndex,
    TransformerWithIndex,
    TransformerWithoutIndex,
)


def ltvalues(xs: Container) -> Iterator[int | str]:
    if isinstance(xs, (list, tuple)):
        yield from xs
    else:
        yield from xs.values()


def ltkeys(xs: Container) -> Iterator[int | str]:
    if isinstance(xs, (list, tuple)):
        yield from range(len(xs))
    else:
        yield from xs.keys()


def ltitems(xs: Container) -> Iterator[tuple[int | str, Any]]:
    if isinstance(xs, (list, tuple)):
        yield from enumerate(xs)
    else:
        yield from xs.items()


@overload
def ltfor(
    tbl,
    index=False,
    apply: Callable[[Any], Any] | None = None,
    keep: Callable[[Any], bool] | None = None,
    exclude: Callable[[Any], bool] | None = None,
    until: Callable[[Any], bool] | None = None,
    when: Callable[[int | str, Any], bool] | None = None,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator: ...


@overload
def ltfor(
    tbl,
    index=True,
    apply: Callable[[int | str, Any], Any] | None = None,
    keep: Callable[[int | str, Any], bool] | None = None,
    exclude: Callable[[int | str, Any], bool] | None = None,
    until: Callable[[int | str, Any], bool] | None = None,
    when: Callable[[int | str, Any], bool] | None = None,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator: ...


def ltfor(
    tbl: Container,
    apply: Transformer | None = None,
    keep: Filter | None = None,
    exclude: Filter | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
    index: bool = False,
) -> Iterator:
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

    it: Iterator
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
            if when(k, v) and keep(k, v) and not exclude(k, v) and not until(k, v):
                try:
                    yield apply(k, v)
                except Exception as error:
                    if raise_on_error:
                        raise error
                    elif ignore_errors:
                        continue
                    else:
                        yield error
    else:
        for v in it:
            if when(v) and keep(v) and not exclude(v) and not until(v):
                try:
                    yield apply(v)
                except Exception as error:
                    if raise_on_error:
                        raise error
                    elif ignore_errors:
                        continue
                    else:
                        yield error


@overload
def ltkeep(
    tbl: Container,
    f: Callable[[Any], bool],
    exclude: Callable[[Any], bool] | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    index=False,
) -> Iterator: ...


@overload
def ltkeep(
    tbl: Container,
    f: Callable[[int | str, Any], bool] = None,
    exclude: Callable[[int | str, Any], bool] | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    index=True,
) -> Iterator: ...


def ltkeep(
    tbl: Container,
    f: Filter,
    exclude: Filter | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    apply: Transformer | None = None,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator:
    return ltfor(
        tbl,
        index=index,
        keep=f,
        apply=apply,
        exclude=exclude,
        until=until,
        when=when,
        ignore_errors=ignore_errors,
        raise_on_error=raise_on_error,
    )


@overload
def ltapply(
    tbl,
    f: Callable[[int | str, Any], Any] = None,
    exclude: Callable[[Any], bool] | None = None,
    keep: Callable[[Any], bool] | None = None,
    until: Callable[[Any], bool] | None = None,
    when: Callable[[Any], bool] | None = None,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator: ...


@overload
def ltapply(
    tbl,
    f: Callable[[int | str, Any], Any] = None,
    exclude: Callable[[int | str, Any], bool] | None = None,
    keep: Callable[[int | str, Any], bool] | None = None,
    until: Callable[[int | str, Any], bool] | None = None,
    when: Callable[[int | str, Any], bool] | None = None,
    index=True,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator: ...


def ltapply(
    tbl: Iterator,
    f: Transformer,
    exclude: Filter | None = None,
    keep: Filter | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator:
    return ltfor(
        tbl,
        apply=f,
        exclude=exclude,
        keep=keep,
        index=index,
        until=until,
        when=when,
        ignore_errors=ignore_errors,
        raise_on_error=raise_on_error,
    )


@overload
def ltexclude(
    tbl,
    f: Callable[[Any], bool],
    apply: Transformer | None = None,
    keep: Callable[[Any], bool] | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    index=False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator: ...


@overload
def ltexclude(
    tbl,
    f: Callable[[int | str, Any], bool] = None,
    apply: Transformer | None = None,
    keep: Callable[[int | str, Any], bool] | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    index=True,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator: ...


def ltexclude(
    tbl: Iterator,
    f: Filter,
    apply: Transformer | None = None,
    keep: Filter | None = None,
    until: Filter | None = None,
    when: Filter | None = None,
    index: bool = False,
    ignore_errors: bool = False,
    raise_on_error: bool = False,
) -> Iterator:
    return ltfor(
        tbl,
        exclude=f,
        keep=keep,
        index=index,
        until=until,
        when=when,
        apply=apply,
        ignore_errors=ignore_errors,
        raise_on_error=raise_on_error,
    )


def lreverse(xs: tuple | list | str) -> Iterator:
    yield from xs[::-1]


def lgrep(
    x: Container,
    *pattern: str | re.Pattern,
    invert: bool = False,
    **kwargs,
) -> Iterator[str | tuple]:
    def check(v: str, *ps: str | re.Pattern) -> bool:
        for p in ps:
            if re.search(p, v, **kwargs):
                if invert:
                    continue
                else:
                    return True
        return not invert

    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, str):
                if check(v, *pattern):
                    yield (k, v)
    else:
        assert isinstance(x, (list, tuple))
        for elem in x:
            if isinstance(elem, str):
                if check(elem, *pattern):
                    yield elem


def lgrepv(x: Container, *pattern: str | re.Pattern, **kwargs) -> Iterator[str | tuple]:
    return lgrep(x, *pattern, invert=True, **kwargs)


def lcdr(x: Sequence) -> Iterator:
    yield from x[1:]


def lbutlast(x: Sequence) -> Iterator:
    yield from x[: len(x) - 1]


def lhead(x: Sequence, n: int = 1) -> Iterator:
    assert n >= 0
    if n <= 0:
        return iter([])

    x_len = len(x)
    if x_len <= n:
        return iter(x)

    for i in range(n):
        yield x[i]


def ltail(x: Sequence, n: int = 1) -> Iterator:
    assert n >= 0
    if n <= 0:
        return iter([])

    x_len = len(x)
    if x_len <= n:
        return iter(x)

    for i in range(x_len - n, x_len):
        yield x[i]


def lseq(start: int, end: int, step: int = 1) -> Iterator[int]:
    yield from range(start, end, step)


__all__ = [
    "Iterator",
    "Generator",
    "ltfor",
    "ltkeep",
    "ltapply",
    "ltexclude",
    "lcdr",
    "lbutlast",
    "lhead",
    "ltail",
    "lreverse",
    "lgrep",
    "lgrepv",
    "lseq",
]
