from __future__ import annotations
from typing import Callable, Sequence, Any

from collections.abc import Mapping
import collections.abc as abc


def assert_no_false(variable: bool, method: bool) -> None:
    if not variable and not method:
        raise ValueError(
            "Variable and method cannot be both False",
            dict(variable=variable, method=method),
        )


def assert_no_true(variable: bool, method: bool) -> None:
    if variable and method:
        raise ValueError(
            "Variable and method cannot be both True",
            dict(variable=variable, method=method),
        )


def named_assert_no_false(
    attrib: str,
    variable: bool = False,
    method: bool = False,
) -> None:
    if not variable and not method:
        raise ValueError(
            f"{attrib}: Variable and method cannot be both False",
            dict(attrib=attrib, variable=variable, method=method),
        )


def named_assert_no_true(
    attrib: str,
    variable: bool = False,
    method: bool = False,
) -> None:
    if variable and method:
        raise ValueError(
            f"{attrib}: Variable and method cannot be both True",
            dict(attrib=attrib, variable=variable, method=method),
        )


class ModuleMeta(type):
    """
    Auto-wraps all callables as staticmethod -> no self pollution.
    Prevents instantiation -> pure namespace.
    Enforces closed hierarchy -> only subclasses of ModuleBase can be mixed.
    Makes the class itself a Mapping -> dict‑like access + match/case support.
    """

    def __new__(metaclass, name, bases, namespace):
        # if not all(b.__name__ == "Module" for b in bases):
        if not all(issubclass(b, Module) for b in bases):
            raise TypeError(
                f"'{name}' can only inherit from Module class "
                f"(subclasses of ModuleBase)"
            )

        for key, value in namespace.items():
            if callable(value) and not key.startswith("__"):
                value.__name__ = key
                namespace[key] = staticmethod(value)

        cls = super().__new__(metaclass, name, bases, namespace)
        abc.Mapping.register(cls)

        return cls

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"'{cls.__name__}' is a static module and cannot be instantiated"
        )

    def __getitem__(cls, key):
        return cls.__dict__[key]

    def __iter__(cls):
        return (k for k in cls.__dict__ if not k.startswith("__"))

    def __len__(cls) -> int:
        return sum(1 for _ in cls.__iter__())

    def __dir__(cls) -> list[str]:
        attribs = set(super().__dir__())
        user_attribs = (k for k in cls.__dict__ if not k.startswith("__"))

        return list(attribs.union(user_attribs))

    def items(
        cls,
        variable: bool = False,
        method: bool = False,
    ) -> list[tuple[str, Any]]:
        assert_no_false(variable, method)

        attribs = cls.__dict__.items()
        attribs = [x for x in attribs if not x[0].startswith("__")]

        if variable and method:
            return list(attribs)
        elif variable:
            return list((k, v) for k, v in attribs if not callable(v))
        else:
            return list((k, v) for k, v in attribs if callable(v))

    def keys(
        cls,
        variable: bool = False,
        method: bool = False,
    ) -> list[str]:
        return [k for k, _ in cls.items(variable=variable, method=method)]

    def values(
        cls,
        variable: bool = False,
        method: bool = False,
    ) -> list[Any]:
        return [v for _, v in cls.items(variable=variable, method=method)]

    def get(
        cls,
        attrib: str,
        default: Any | None = None,
        default_factory: Callable[[], Any | None] | None = None,
    ) -> Any | None:
        if attrib not in cls.__dict__:
            if callable(default_factory):
                return default_factory()
            else:
                return default
        else:
            return cls.__dict__[attrib]

    def has(
        cls,
        attrib: str,
        variable: bool = False,
        method: bool = False,
    ) -> bool:
        named_assert_no_false(attrib, variable, method)

        if variable and method:
            return attrib in cls.__dict__
        elif attrib not in cls.__dict__:
            return False

        val = cls.__dict__[attrib]
        if variable:
            return not callable(val)
        else:
            return callable(val)

    def has_var(cls, var: str) -> bool:
        return cls.has(var, variable=True)

    def has_method(cls, method: str) -> bool:
        return cls.has(method, method=True)

    def has_vars(cls, *vars: str) -> list[bool]:
        return [cls.has_var(v) for v in vars]

    def has_methods(cls, *methods: str) -> list[bool]:
        return [cls.has_method(m) for m in methods]

    def get_var(
        cls,
        var: str,
        default: bool = False,
        default_factory=None,
    ) -> Any | None:
        if cls.has_var(var):
            return cls.__dict__[var]
        elif callable(default_factory):
            return default_factory()
        else:
            return default

    def get_method(
        cls,
        method: str,
        default: bool = False,
        default_factory=None,
    ) -> Any | None:
        if cls.has_method(method):
            return cls.__dict__[method]
        elif callable(default_factory):
            return default_factory()
        else:
            return default

    def get_vars(
        cls,
        *vars: str,
        default: bool = False,
        default_factory: bool = False,
    ) -> list[Any | None]:
        return [
            cls.get_var(
                v,
                default=default,
                default_factory=default_factory,
            )
            for v in vars
        ]

    def get_methods(
        cls,
        *methods: str,
        default: bool = False,
        default_factory: bool = False,
    ):
        return [
            cls.get_method(
                m,
                default=default,
                default_factory=default_factory,
            )
            for m in methods
        ]


class Module(metaclass=ModuleMeta):
    pass


#
#
__all__ = ["Module"]
