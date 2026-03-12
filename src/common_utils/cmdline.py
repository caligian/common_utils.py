import re
import argparse
import shlex

from collections import defaultdict
from typing import Callable, Sequence
from argparse import ArgumentParser
from src.common_utils.result import Success, Failure

ParsedValue = str | bool | int | float
ParsedDict = dict[str, ParsedValue | list[ParsedValue]]
ProcessorCallable = Callable[
    [ParsedValue, ...],
    Success[str] | Failure[AssertionError] | AssertionError | ParsedValue,
]
ProcessorSpec = (
    tuple[ProcessorCallable, Sequence, dict]
    | tuple[ProcessorCallable, Sequence]
    | tuple[ProcessorCallable, dict]
    | tuple[ProcessorCallable]
    | ProcessorCallable
)

StoreAction = argparse._StoreAction


def fix_name(name: str) -> str:
    name = re.sub(r"^-+", "", name)
    name = name.replace("-", "_")
    name = re.sub("_{2,}", "_", name)

    return name


def mkdefault(x, y=None):
    if x is None:
        return y
    else:
        return x


class Processor:
    def __init__(
        self,
        variable: str,
        f: ProcessorCallable,
        *args,
        **kwargs,
    ) -> None:
        self.variable = variable
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def process(self, value: ParsedValue) -> Success[ParsedValue] | Failure[ValueError]:
        f = self.f
        try:
            value = f(value, *self.args, **self.kwargs)
        except Exception as error:
            return Failure(error)

        match value:
            case Success() as success:
                return success
            case Failure() as failure:
                return failure
            case Exception() as error:
                return Failure(error)
            case True:
                return Success(value)
            case _ if not value:
                return Failure(AssertionError(f"Validation error: {value}"))
            case _:
                return Success(value)


class Argv(ArgumentParser):
    def __init__(self, prog: str, *args, **kwargs) -> None:
        super().__init__(prog, *args, **kwargs)

        self.options: dict[str, bool] = {"rest": True}
        self.parsed: ParsedDict = {}
        self.processors: dict[str, list[Processor]] = defaultdict(lambda: [])
        self.aliases: dict[str, str] = {}

        self.add_argument(
            "rest",
            nargs="*",
            help="Rest of the arguments passed",
        )

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

    def add_processor(
        self,
        option: str,
        *processor: ProcessorSpec,
    ) -> None:
        option = self.aliases[option]
        option_processors = self.processors[option]

        def add(processor: ProcessorSpec) -> None:
            option_processors.append(processor)

        for spec in processor:
            match spec:
                case f if callable(f):
                    add(Processor(option, f))
                case (f,):
                    add(Processor(option, f))
                case (f, args, kwargs):
                    add(Processor(option, f, *args, **kwargs))
                case (f, args) if isinstance(args, (tuple, list)):
                    add(Processor(option, f, *args))
                case (f, kwargs) if isinstance(args, dict):
                    add(Processor(option, f, **kwargs))
                case _:
                    raise ValueError(f"Invalid form: {spec}")

    def add_rest_processor(self, *spec: ProcessorSpec) -> None:
        self.add_processor("rest", *spec)

    def on(
        self,
        *option: str,
        processor: ProcessorSpec | list[ProcessorSpec] | None = None,
        **option_kwargs,
    ) -> StoreAction:
        self.add_alias(*option)

        name: str
        if len(option) == 1:
            name = option[0]
        else:
            name = option[-1]

        name = fix_name(name)

        if processor:
            processor = (
                [processor] if not isinstance(processor, (list, tuple)) else processor
            )
            self.add_processor(name, *processor)

        self.options[name] = True
        return self.add_argument(*option, **option_kwargs)

    def __getitem__(self, var: str) -> ParsedValue | None:
        return self.parsed.get(var)

    def process(
        self,
        parsed: ParsedDict | None = None,
        pcall: bool = False,
    ) -> Success[ParsedDict] | Failure[AssertionError]:
        parsed = parsed is None and self.parsed or parsed
        assert parsed is not None, (
            "No arguments were parsed. Have you run <obj>.parse(...)"
        )

        new: ParsedDict = {}
        for k, v in parsed.items():
            if fs := self.processors.get(k):
                for f in fs:
                    res = f.process(v)
                    match res:
                        case Success(value):
                            v = value
                        case Failure() as failure:
                            if not pcall:
                                failure.unwrap(pcall=pcall)
                            else:
                                return failure
            new[k] = v

        self.parsed = new
        return Success(new)

    def parse(
        self,
        args: str | list[str] | None = None,
        pcall: bool = True,
        unwrap: bool = False,
    ) -> Success[ParsedDict] | Failure[AssertionError] | ParsedDict:
        parsed = None
        has_help = ('-h' in args) or ('--help' in args)

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
            match res:
                case Success(new):
                    parsed = new
                case Failure() as failure:
                    if not pcall:
                        failure.unwrap()
                    else:
                        return failure

            self.parsed = parsed
            if unwrap:
                return self.parsed
            else:
                return Success(self.parsed)
        except SystemExit as error:
            if has_help:
                return Success({'help': True})
            elif not pcall:
                raise error
            else:
                return Failure(error)
        except Exception as error:
            if not pcall:
                raise error
            else:
                return Failure(error)


parser = Argv("Some CLI application")
parser.on(
    "-i",
    "--input-file",
    processor=[(int, []), lambda x: x + 1, float, lambda x: x + 1.5],
    nargs="?",
)
parser.on("-f", "--flag", action="store_true")

__all__ = ["mkdefault", "Argv"]
