import re
import argparse
import sys
import shlex

from collections import defaultdict
from typing import Callable
from argparse import ArgumentParser
from src.common_utils.result import Success, Failure, Result

ArgvParsedValue = str | bool | int | float
ArgvParsedDict = dict[str, ArgvParsedValue | list[ArgvParsedValue]]
ArgvValidatorCallable = Callable[
    [ArgvParsedValue],
    Success[str] | Failure[AssertionError] | AssertionError | ArgvParsedValue,
]
ArgvProcessorCallable = Callable[
    [ArgvParsedValue, ...],
    Callable[[ArgvParsedValue], Success | Failure[ValueError]],
]


def mkdefault(x, y=None):
    if x is None:
        return y
    else:
        return x


class ArgvProcessor:
    def __init__(
        self,
        variable: str,
        f: ArgvProcessorCallable,
        *args,
        **kwargs,
    ) -> None:
        self.variable = variable
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def process(
        self, value: ArgvParsedValue
    ) -> Success[ArgvParsedValue] | Failure[ValueError]:
        f = self.f
        value = f(value, *self.args, **self.kwargs)

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


class ArgvValidator:
    def __init__(
        self,
        variable: str,
        f: ArgvValidatorCallable,
        *args,
        **kwargs,
    ) -> None:
        self.variable = variable
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def validate(self, value: ArgvParsedValue) -> Success | Failure[AssertionError]:
        f = self.f
        value = f(value, *self.args, **self.kwargs)

        match value:
            case Success() as success:
                return success
            case Failure() as failure:
                return failure
            case Exception() as error:
                return Failure(error)
            case True:
                return Success()
            case _ if not value:
                return Failure(AssertionError(f"Validation error: {value}"))
            case result:
                return Success(result)


class Argv(ArgumentParser):
    def __init__(self, prog: str, *args, **kwargs) -> None:
        super().__init__(prog, *args, **kwargs)

        self.options: dict[str, bool] = {'rest': True}
        self.parsed: ArgvParsedDict = {}
        self.validators: dict[str, list[ArgvValidator]] = defaultdict(lambda: [])
        self.processors: dict[str, list[ArgvProcessor]] = defaultdict(lambda: [])
        self.add_argument("rest", nargs="*", help="Rest of the arguments passed")

    def on(
        self,
        *option,
        validate: ArgvValidatorCallable | ArgvValidatorCallable | None = None,
        validate_args: list | None = None,
        validate_kwargs: dict | None = None,
        process: ArgvProcessorCallable | list[ArgvProcessorCallable] | None = None,
        process_args: list | None = None,
        process_kwargs: dict | None = None,
        **kwargs,
    ) -> argparse._StoreAction:
        name: str
        if len(option) == 1:
            name = option[0]
        else:
            name = option[-1]

        name = re.sub("^-?-?", "", name)
        name = name.replace("-", "_")

        if validate:
            validate = (
                [validate] if not isinstance(validate, (list, tuple)) else validate
            )
            validate_args = [] if not validate_args else validate_args
            validate_kwargs = {} if not validate_kwargs else validate_kwargs

            for f in validate:
                self.add_validator(name, f, *validate_args, **validate_kwargs)

        if process:
            process = [process] if not isinstance(process, (list, tuple)) else process
            process_args = [] if not process_args else process_args
            process_kwargs = {} if not process_kwargs else process_kwargs

            for f in process:
                self.add_processor(name, f, *process_args, **process_kwargs)

        self.options[name] = True
        return self.add_argument(*option, **kwargs)

    def __getitem__(self, var: str) -> ArgvParsedValue | None:
        return self.parsed.get(var)

    def add_processor(
        self,
        var: str,
        f: ArgvProcessorCallable,
        *default_args,
        **default_kwargs,
    ) -> None:
        self.processors[var].append(
            ArgvProcessor(var, f, *default_args, **default_kwargs)
        )

    def add_validator(
        self,
        var: str,
        f: ArgvValidatorCallable,
        *default_args,
        **default_kwargs,
    ) -> ArgvProcessorCallable:
        self.validators[var].append(
            ArgvValidator(var, f, *default_args, **default_kwargs)
        )

    def validate(
        self,
        parsed: ArgvParsedDict | None = None,
        pcall: bool = False,
    ) -> Success[ArgvParsedDict] | Failure[AssertionError]:
        parsed = parsed is None and self.parsed or parsed
        assert parsed is not None, (
            "No arguments were parsed. Have you run <obj>.parse(...)"
        )

        parsed: ArgvParsedDict
        new: ArgvParsedDict = {}

        for k, value in parsed.items():
            if validators := self.validators.get(k):
                for validator in validators:
                    match validator.validate(value):
                        case Success(str(v)):
                            new[k] = v
                        case Success():
                            new[k] = value 
                        case Failure() as failure:
                            if not pcall:
                                failure.unwrap()
                            else:
                                return failure
            else:
                new[k] = value

        return Success(new)

    def process(
        self,
        parsed: ArgvParsedDict | None = None,
        pcall: bool = False,
    ) -> Success[ArgvParsedDict] | Failure[AssertionError]:
        parsed = parsed is None and self.parsed or parsed
        assert parsed is not None, (
            "No arguments were parsed. Have you run <obj>.parse(...)"
        )

        new: ArgvParsedDict = {}
        for k, v in parsed.items():
            if fs := self.processors.get(k):
                for f in fs:
                    match f.process(v):
                        case Success(value) if type(value) is not bool:
                            new[k] = value
                        case Success(value) if value:
                            new[k] = v
                        case Failure() as failure:
                            if not pcall:
                                failure.unwrap()
                            else:
                                return failure
            else:
                new[k] = v

        return Success(new)

    def parse(
        self,
        args: str | list[str] | None = None,
        pcall: bool = True,
        unwrap: bool = True,
    ) -> Success[ArgvParsedDict] | Failure[AssertionError] | ArgvParsedDict:
        parsed = None

        try:
            # Parsing known_args will print the error message anyway. That is what we need
            if args:
                args = isinstance(args, str) and shlex.split(args) or args
                parsed, _ = self.parse_known_args(args)
            else:
                parsed, _ = self.parse_known_args()

            _parsed = {}
            for k in self.options.keys():
                _parsed[k] = getattr(parsed, k)

            parsed = _parsed
            match self.validate(parsed, pcall=pcall):
                case Success(new):
                    parsed = new
                case Failure() as failure:
                    if not pcall:
                        failure.unwrap()
                    else:
                        return failure

            match self.process(parsed, pcall=pcall):
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
            if not pcall:
                raise error
            else:
                return Failure(error)
        except Exception as error:
            if not pcall:
                raise error
            else:
                return Failure(error)
#
#
# parser = Argv("Some CLI application")
# parser.on(
#     "-i",
#     "--input-file",
#     validate=lambda x: Success(True),
#     process=lambda x: int(x) + 1,
#     nargs='?',
# )
# parser.on(
#     '-f',
#     '--flag',
#     action='store_true'
# )
# parser.parse(["a", "b", 'c', '-i', '1', '-f'])
# parser.print_help()
