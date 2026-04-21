from typing import Callable, Any, Iterator, Generator, TypeVar, overload, Literal

T = TypeVar("T")
R = TypeVar("R")

DefaultFactory = Callable[[], Any]
Mapper = Callable[[T], R]


@overload
def defined(x: T, f: Mapper) -> R: ...


@overload
def defined(x: None, f: Mapper | None) -> Literal[False]: ...


@overload
def defined(x: T, f: None) -> Literal[True]:
    pass


def defined(x: T | None, f: Mapper | None = None) -> bool | R:
    if x is not None:
        if f:
            return f(x)
        else:
            return True
    else:
        return False


@overload
def undefined(x: T, f: Mapper | None) -> Literal[False]: ...


@overload
def undefined(x: None, f: Mapper) -> R: ...


@overload
def undefined(x: None, f: None) -> Literal[True]: ...


def undefined(x: T | None, f: Mapper | None = None) -> bool | R:
    if x is None:
        if f:
            return f(x)
        else:
            return True
    else:
        return False


def is_range(x: Any) -> bool:
    return isinstance(x, range)


def literal(x: Any) -> Callable[[], Any]:
    return lambda *_args, **_kwargs: x


def when(
    cond: bool | None,
    true: DefaultFactory | None = None,
    false: DefaultFactory | None = None,
) -> Any:
    if isa(cond, bool):
        if cond:
            if true:
                return true()
            else:
                return True
        elif false:
            return false()
    elif cond():
        return true()
    elif false:
        return false()


def unless(
    cond: Any | None,
    true: DefaultFactory | None = None,
    false: DefaultFactory | None = None,
) -> Any:
    if isa(cond, bool):
        if not cond:
            if true:
                return true()
            else:
                return True
        elif false:
            return false()
    elif not cond():
        return true()
    elif false:
        return false()


@overload
def ifelse(
    x: dict[int | str, Any],
    cond: Callable[[Any], bool],
    true: Callable[[Any], Any],
    false: Callable[[Any], Any],
    invert: bool = False,
    inplace: bool = False,
) -> dict: ...


@overload
def ifelse(
    x: list,
    cond: Callable[[Any], bool],
    true: Callable[[Any], Any],
    false: Callable[[Any], Any],
    invert: bool = False,
    inplace: bool = False,
) -> list: ...


@overload
def ifelse(
    x: tuple,
    cond: Callable[[Any], bool],
    true: Callable[[Any], Any],
    false: Callable[[Any], Any],
    invert: bool = False,
    inplace: bool = False,
) -> tuple: ...


def ifelse(
    x: list | tuple | dict[int | str, Any],
    cond: Callable[[Any], bool],
    true: Callable[[Any], Any],
    false: Callable[[Any], Any],
    invert: bool = False,
    inplace: bool = False,
) -> list | tuple | dict:
    use = x
    index: list[int | str] | None = None
    index = list(x.keys()) if is_dict(x) else index

    if is_range(x) or is_generator(x):
        use = [elem for elem in x]
    elif is_dict(x):
        use = list(x.values())
    else:
        use = list(use)

    if inplace:
        use = use.copy()

    for i in range(len(use)):
        elem = use[i]
        _success = cond(elem)
        success = (_success and invert) if invert else _success

        if success:
            use[i] = true(elem)
        else:
            use[i] = false(elem)

    if isa(x, dict):
        return dict(zip(index, use))
    elif isa(x, list):
        return use
    else:
        return tuple(use)


def is_bytes(s: Any) -> bool:
    return isinstance(s, (bytes,))


def is_str(s: Any) -> bool:
    return isinstance(s, (str,))


def is_str_like(s: Any) -> bool:
    return isinstance(s, (str, bytes))


def isa(x: Any, *types: type) -> bool:
    return isinstance(x, tuple(types))


def is_dict(x: Any) -> bool:
    return isa(x, dict)


def is_list(x: Any) -> bool:
    return isa(x, list)


def is_tuple(x: Any) -> bool:
    return isa(x, tuple)


def is_container(x: Any) -> bool:
    return isa(x, dict, tuple, list)


def is_sequence(
    x: Any,
    with_str: bool = False,
    with_bytes: bool = False,
    with_str_like: bool = False,
) -> bool:
    if with_str_like:
        return isa(x, list, tuple, str, bytes)
    elif with_bytes:
        return isa(x, list, tuple, bytes)
    elif with_str:
        return isa(x, list, tuple, str)
    else:
        return isa(x, list, tuple)


def is_iterable(x: Any) -> bool:
    return isinstance(x, Iterator)


def is_number(x: Any) -> bool:
    return isa(x, int, float)


def is_int(x: Any) -> bool:
    return isa(x, int)


def is_float(x: Any) -> bool:
    return isa(x, float)


def is_callable(x: Any) -> bool:
    return isa(x, Callable)


def is_generator(x: Any) -> bool:
    return isa(x, Generator)


def is_error(x: Any) -> bool:
    return isa(x, Exception)


def is_type(x: Any) -> bool:
    return isinstance(x, type)


def defguard(*x_type: type) -> Callable[[Any], bool]:
    return lambda obj: isinstance(obj, x_type)


container = is_container
sequence = is_sequence
is_exception = is_error
is_a = isa

__all__ = [
    "container",
    "defined",
    "defguard",
    "ifelse",
    "is_bytes",
    "is_callable",
    "is_container",
    "is_dict",
    "is_error",
    "is_exception",
    "is_float",
    "is_generator",
    "is_int",
    "is_iterable",
    "is_list",
    "is_number",
    "is_sequence",
    "is_str",
    "is_str_like",
    "is_tuple",
    "is_type",
    "is_a",
    "isa",
    "literal",
    "sequence",
    "unless",
    "when",
    "undefined",
    "defined",
]
