from typing import Callable, Any, Iterator, Generator

DefaultFactory = Callable[[], Any]


def is_range(x: Any) -> bool:
    return isinstance(x, range)


def literal(x: Any) -> Callable[[], Any]:
    return lambda *_args, **_kwargs: x


def unlessNone(
    cond: Any | None,
    true: DefaultFactory = None,
    false: DefaultFactory | None = None,
) -> Any:
    if isa(cond, bool):
        if cond is not None:
            if true:
                return true()
            else:
                return True
        elif false:
            return false()

    is_none = cond() is None
    if is_none:
        return false()
    elif true:
        return true()


def whenNone(
    cond: Any | None,
    true: DefaultFactory = None,
    false: DefaultFactory | None = None,
) -> Any:
    if isa(cond, bool):
        if cond is None:
            if true:
                return true()
            else:
                return True
        elif false:
            return false()

    is_none = cond() is None
    if is_none:
        return true()
    elif false:
        return false()


def when(
    cond: Any | None,
    true: DefaultFactory = None,
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
    true: DefaultFactory = None,
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


def ifelse(
    x: list | tuple,
    cond: Callable[[Any], bool],
    true: Callable[[Any], Any] = lambda x: x,
    false: Callable[[Any], Any] = lambda x: x,
    invert: bool = False,
) -> list | tuple:
    if is_iter(x) or is_range(x) or is_generator(x):
        x = [elem for elem in x]

    res = x.copy()
    for i, elem in enumerate(x):
        _success = cond(elem)
        success = _success and invert if invert else _success

        if success:
            res[i] = true(elem)
        else:
            res[i] = false(elem)

    if isa(x, list):
        return res
    else:
        return tuple(res)


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


def is_sequence(x: Any) -> bool:
    return isa(x, list, tuple)


def is_str(x: Any) -> bool:
    return isa(x, str)


def is_bytes(x: Any) -> bool:
    return isa(x, bytes)


def is_number(x: Any) -> bool:
    return isa(x, int, float)


def is_int(x: Any) -> bool:
    return isa(x, int)


def is_float(x: Any) -> bool:
    return isa(x, float)


def is_callable(x: Any) -> bool:
    return isa(x, Callable)


def is_iter(x: Any) -> bool:
    return isa(x, Iterator)


def is_generator(x: Any) -> bool:
    return isa(x, Generator)


def is_error(x: Any) -> bool:
    return isa(x, Exception)


def guard(*x_type: type) -> Callable[[Any], bool]:
    return lambda obj: isinstance(obj, x_type)


container = is_container
sequence = is_sequence
Error = Exception

__all__ = [
    "Error",
    "container",
    "guard",
    "ifelse",
    "is_bytes",
    "is_callable",
    "is_container",
    "is_dict",
    "is_error",
    "is_float",
    "is_generator",
    "is_int",
    "is_iter",
    "is_list",
    "is_number",
    "is_sequence",
    "is_str",
    "is_str_like",
    "is_tuple",
    "isa",
    "literal",
    "sequence",
    "unless",
    "unlessNone",
    "when",
    "whenNone",
]

# print(when(False, true=literal(100), false=literal("OK")))
# print(unless(False, true=literal(100), false=literal("OK")))
