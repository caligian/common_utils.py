from __future__ import annotations

import sys
import re
import argparse
import shlex

from dataclasses import field, dataclass
from collections import defaultdict
from typing import Callable, Sequence, Iterator, Any, overload, Literal
from argparse import ArgumentParser

from .error import raise_error
from .result import Ok, Err, Result, is_ok, is_err, is_result
from .cond import is_error

OptParserParsedValue = str | bool | int | float
OptParserParsedDict = dict[str, OptParserParsedValue | list[OptParserParsedValue]]
OptParserProcessorCallable = Callable[[OptParserParsedValue], Any]
OptParserProcessorSpec = (
    tuple[
        OptParserProcessorCallable,
        tuple | list | None,
        dict | None,
    ]
    | tuple[OptParserProcessorCallable, tuple | list]
    | tuple[OptParserProcessorCallable, dict]
)
StoreAction = argparse._StoreAction


def ARGV() -> list[str]:
    return sys.argv


def has_argv() -> bool:
    return len(sys.argv) != 1


def fix_name(name: str) -> str:
    if name[0] != "-":
        return name

    name = re.sub(r"^-+", "", name)
    name = name.replace("-", "_")
    name = re.sub("_{2,}", "_", name)

    return name


def mkdefault(x: Any, y=None):
    if x is None:
        return y
    else:
        return x


@dataclass
class OptParserProcessor:
    variable: str
    f: OptParserProcessorCallable
    args: list | tuple = field(default_factory=lambda: [])
    kwargs: dict = field(default_factory=lambda: {})

    def check(self, value: OptParserParsedValue) -> bool:
        try:
            _value = self.f(value, *self.args, **self.kwargs)
            if is_err(_value):
                return False
            elif is_error(_value):
                return False
            elif not _value:
                return False
            else:
                return True
        except Exception:
            return False

    def process(
        self, value: OptParserParsedValue
    ) -> Result[OptParserParsedValue, ValueError]:
        try:
            _value = self.f(value, *self.args, **self.kwargs)
            if is_result(_value):
                if is_ok(_value):
                    return Ok(_value.value, _value.metadata)
                else:
                    return Err(
                        _value.value,
                        {"obj": self, "value": value, **_value.metadata},
                    )
            elif _value:
                return Ok(_value)
        except Exception as error:
            return Err(error, dict(value=value, obj=self))


@dataclass
class OptParserProcessors(list):
    variable: str
    processors: list[OptParserProcessor] = field(default_factory=lambda: [])

    def __getitem__(self, name: int) -> OptParserProcessor | None:
        try:
            return self.procesors[name]
        except IndexError:
            return

    def has(self, name: int) -> bool:
        try:
            return self.processors[name] and True
        except IndexError:
            return False

    def pop(self, name: int) -> OptParserProcessor | None:
        if self.has(name):
            return self.processors.pop(name)

    def __iter__(self) -> Iterator[OptParserProcessor]:
        return iter(self.processors)

    def on(
        self,
        f: OptParserProcessorCallable,
        *args,
        **kwargs,
    ) -> OptParserProcessor:
        processor = OptParserProcessor(self.variable, f, *args, **kwargs)
        self.processors.append(processor)
        return processor

    def add(
        self, *specs: OptParserProcessorSpec | OptParserProcessorCallable
    ) -> OptParserProcessors:
        specs: list[OptParserProcessorSpec | OptParserProcessorSpec] = list(specs)

        if isinstance(specs, Callable):
            specs = [specs]

        for spec in specs:
            if isinstance(spec, Callable):
                spec = [spec]

            spec_len = len(spec)
            assert spec_len >= 1 and spec_len <= 3, "Expected ({f} [args] [kwargs])"

            f, args, kwargs = spec[0], [], {}
            match spec_len:
                case 1:
                    pass
                case 2:
                    if isinstance(spec[1], (tuple, list)):
                        args = spec[1]
                    else:
                        kwargs = spec[1]
                case 3:
                    args, kwargs = spec[1:]

            self.on(f, *args, **kwargs)

        return self.processors[self.name]

    def process(
        self,
        value: OptParserParsedValue | list[OptParserParsedValue],
        pcall: bool = True,
    ) -> Result[OptParserParsedValue, ValueError]:
        not_list = isinstance(value, list)
        value = [value] if not isinstance(value, (list, tuple)) else value

        for processor in self.processors:
            res = Ok(value)

            for i, v in enumerate(value.copy()):
                match processor.process(v):
                    case Ok(final) as ok:
                        value[i] = final
                        res = Ok(ok.value, ok.metadata)
                    case Err() as err:
                        return err

        res = Ok(value)
        if not_list:
            if len(res.value) > 0:
                return Ok(res.value[0], res.metadata)
            else:
                return Ok(res.value, res.metadata)
        else:
            return Ok(res.unwrap(), res.metadata)


@dataclass
class OptParser(ArgumentParser):
    prog: str
    options: dict[str, bool] = field(default_factory=lambda: {"rest": True})
    parsed: OptParserParsedDict = field(default_factory=lambda: {})
    processors: dict[str, OptParserProcessors] = field(default_factory=lambda: {})
    aliases: dict[str, str] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        self._config_set: bool = False

    def config(self, *args, **kwargs) -> OptParser:
        if self._config_set:
            return self

        super().__init__(self.prog, *args, **kwargs)
        self.add_argument(
            "rest",
            nargs="*",
            help="Rest of the arguments passed",
        )

        self._config_set = True
        return self

    def _raise_if_config_not_set(self) -> None:
        if not self._config_set:
            raise AssertionError(".config(*args, **kwargs) is not called yet.")

    def add_alias(
        self,
        short: str | None = None,
        long: str | None = None,
    ) -> None:
        if short:
            short = fix_name(short)

        if long:
            long = fix_name(long)

        if short and long:
            self.aliases[long] = long
            self.aliases[short] = long
        elif short:
            self.aliases[short] = short
        elif long:
            self.aliases[long] = long

    def add_processors(
        self,
        name: str,
        specs: list[OptParserProcessorSpec]
        | OptParserProcessorSpec
        | OptParserProcessorCallable
        | None = None,
        f: Callable[[OptParserParsedValue], Any] | None = None,
        *args,
        **kwargs,
    ) -> OptParserProcessors:
        name = fix_name(name)
        processors = self.processors.get(name)

        if processors is None:
            processors = OptParserProcessors(name)
            self.processors[name] = processors

        if specs:
            if isinstance(specs, Callable):
                specs = [specs]

            assert len(specs) >= 1, "specs= is empty"
            for proc in specs:
                if isinstance(proc, Callable):
                    self.add_processors(name, f=proc)
                else:
                    f, args, kwargs = proc[0], [], {}
                    if len(proc) == 3:
                        what = proc[1]
                        if isinstance(what, (tuple, list)):
                            args = what
                            kwargs = {}
                        elif isinstance(what, dict):
                            kwargs = args
                            args = []
                    elif len(proc) == 3:
                        args = proc[1]
                        kwargs = proc[2]

                    return self.add_processors(f=f, *args, **kwargs)
        elif f:
            return processors.on(f, *args, **kwargs)
        else:
            raise AssertionError("Expected specs= OR f=")

    def add_rest_processors(
        self,
        specs: list[OptParserProcessorSpec]
        | OptParserProcessorSpec
        | OptParserProcessorCallable
        | None = None,
        f: Callable[[OptParserParsedValue], Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        return self.add_processors("rest", specs=specs, f=f, *args, **kwargs)

    def on(
        self,
        *option: str,
        processors: OptParserProcessorCallable
        | OptParserProcessorSpec
        | list[OptParserProcessorSpec]
        | None = None,
        **option_kwargs,
    ) -> OptParser:
        self._raise_if_config_not_set()
        self.add_alias(*option)

        if isinstance(processors, tuple):
            processors = [processors]
        elif isinstance(processors, Callable):
            processors = [(processors,)]

        name: str
        if len(option) == 1:
            name = option[0]
        else:
            name = option[-1]

        name = fix_name(name)

        if processors:
            self.add_processors(name, specs=processors)

        self.options[name] = True
        self.add_argument(*option, **option_kwargs)

        return self

    def __getitem__(self, var: str) -> OptParserParsedValue | None:
        return self.parsed.get(var)

    def process(
        self,
        parsed: OptParserParsedDict | None = None,
        pcall: bool = False,
    ) -> Result[OptParserParsedDict, AssertionError]:
        self._raise_if_config_not_set()

        parsed = self.parsed if parsed is None else parsed
        assert parsed is not None, (
            "No arguments were parsed. Have you run <obj>.parse(...)"
        )

        new: OptParserParsedDict = {}
        for k, v in parsed.items():
            processors = self.processors.get(k)
            if processors is not None:
                res = processors.process(v)
                new[k] = res.unwrap()

        self.parsed = new
        return Ok(new)

    @overload
    def parse(
        self,
        args: str | list[str] | None = None,
        pcall: bool = False,
        unwrap: Literal[True] = True,
    ) -> OptParserParsedDict: ...

    @overload
    def parse(
        self,
        args: str | list[str] | None = None,
        pcall: Literal[True] = False,
        unwrap: Literal[True] = True,
    ) -> Result[OptParserParsedDict, AssertionError]: ...

    def parse(
        self,
        args: str | list[str] | None = None,
        pcall: bool = False,
        unwrap: bool = True,
    ) -> Result[OptParserParsedDict, AssertionError] | OptParserParsedDict:
        self._raise_if_config_not_set()

        args: list[str] | str = ARGV() if args is None else args
        args: list[str] = shlex.split(args) if isinstance(args, str) else args
        has_help = ("-h" in args) or ("--help" in args)
        parsed = None

        try:
            # Parsing known_args will print the error message anyway. That is what we need
            if args:
                args = isinstance(args, str) and shlex.split(args) or args
                parsed, end_args = self.parse_known_args(args)
            else:
                parsed, end_args = self.parse_known_args()

            _parsed = {}
            for k in self.options.keys():
                _parsed[k] = getattr(parsed, k)

            parsed = _parsed
            parsed["rest"].extend(end_args)

            for i, x in enumerate(parsed["rest"]):
                parsed[i + 1] = x

            res = self.process(parsed, pcall=pcall)
            if is_err(res):
                if not pcall:
                    res.unwrap()
                else:
                    return res

            if unwrap:
                return self.parsed
            else:
                return Ok(self.parsed)
        except SystemExit as error:
            if has_help:
                return Ok({"help": True})
            elif not pcall:
                raise_error()
                raise error
            else:
                return Err(error)
        except Exception as error:
            if not pcall:
                raise error
            else:
                return Err(error)


parser = OptParser("Some CLI application")
parser.config("Do this, do that, blah blah")
parser.on(
    "-i",
    "--input-file",
    nargs="?",
)
parser.add_processors(
    "input_file", specs=(lambda x: x + "hello_", lambda x: x + " 1.5")
)

parser.on("-f", "--flag", action="store_true")
print(parser.parse(shlex.split("-i /home/caligian/.bashrc -f")))


__all__ = ["mkdefault", "OptParser", "ARGV", "has_argv"]
