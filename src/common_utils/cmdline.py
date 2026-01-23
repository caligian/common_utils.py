import re
import argparse
import sys
import shlex

from typing import Callable
from argparse import ArgumentParser

from src.common_utils.result import Ok, Err, Result

ArgvParsedValue = str | bool | int | float
ArgvParsedDict = dict[str, ArgvParsedValue | list[ArgvParsedValue]]
ArgvValidatorCallable = Callable[[ArgvParsedValue], Result]
ArgvProcessorCallable = Callable[[ArgvParsedValue, ...], any]


def mkdefault(x, y=None):
    if x is None:
        return y
    else:
        return x


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

    def validate(
        self,
        value: ArgvParsedValue,
        pcall: bool = True,
    ) -> Result:
        f = self.f
        value = f(value, *self.args, **self.kwargs)

        if type(value) is Result:
            if not value.ok:
                msg = ""
                if value.message:
                    msg = f"{self.variable}: Validation error: {value.message}"
                else:
                    msg = f"{self.variable}: Validation error.\nArgument supplied: {str(value)}"
                if not pcall:
                    raise AssertionError(msg)
                else:
                    return Result(False, AssertionError, msg)
            else:
                return value
        elif not value:
            msg = f"{self.variable}: Validation error"
            return Result(False, AssertionError, msg)
        else:
            return Result(True, value)


class Argv(ArgumentParser):
    def __init__(self, prog: str, *args, **kwargs) -> None:
        super().__init__(prog, *args, **kwargs)

        self.parsed: ArgvParsedDict = {}
        self.validators: dict[str, ArgvValidator] = {}
        self.processors: dict[str, ArgvProcessorCallable] = {}
        self.add_argument("rest", nargs="*", help="Rest of the arguments passed")
        self.rest_args: list[str] | None = []

    def on(
        self,
        *option,
        validate: ArgvValidatorCallable | ArgvValidatorCallable | None = None,
        process: ArgvProcessorCallable | list[ArgvProcessorCallable] | None = None,
        **kwargs,
    ) -> argparse._StoreAction:
        name: str
        if len(option) == 1:
            name = option[0]
        else:
            name = option[-1]

        name = re.sub("^-?-?", "", name)
        name = name.replace("-", "_")

        if process:
            process = [process] if not isinstance(process, (list, tuple)) else process
            for f in process:
                self.add_processor(name, f)

        if validate:
            validate = (
                [validate] if not isinstance(validate, (list, tuple)) else validate
            )
            for v in validate:
                self.add_validator(name, v)

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
        def function(value: ArgvParsedValue) -> any:
            return f(value, *default_args, **default_kwargs)

        exists = self.processors.get(var)
        if not exists:
            self.processors[var] = [function]
        else:
            self.processors[var].append(function)

    def add_validator(
        self,
        var: str,
        f: ArgvValidatorCallable,
        *default_args,
        **default_kwargs,
    ) -> ArgvProcessorCallable:
        exists = self.validators.get(var)
        f = ArgvValidator(var, f, *default_args, **default_kwargs)

        if not exists:
            self.validators[var] = [f]
        else:
            self.validators[var].append(f)

        return f

    def validate_parsed_args(
        self,
        parsed: ArgvParsedDict | None = None,
        pcall: bool = False,
    ) -> tuple[ArgvParsedDict, ArgvParsedDict]:
        parsed = parsed is None and self.parsed or parsed
        assert parsed is not None, (
            "No arguments were parsed. Have you run <obj>.parse(...)"
        )

        parsed: ArgvParsedDict
        failed = {}
        items = parsed.copy()

        def validate_arg(v, arg) -> Result:
            result = v.validate(arg, pcall=pcall)
            if pcall:
                if not result.ok:
                    return result
                else:
                    result.errorf()
            elif not result.ok:
                result.errorf()
            else:
                return result

        for k, value in parsed.items():
            if validators := self.validators.get(k):
                result = validate_arg(validators[0], value)

                for validator in validators[1:]:
                    result = validator.validate(result.value, pcall=pcall)
                    if not result.ok and pcall:
                        failed[k] = value
                        items.pop(k)
                    elif not result.ok:
                        result.errorf()

                    breakpoint()

        return (items, failed)

    def process_parsed_args(
        self, parsed: ArgvParsedDict | None = None
    ) -> ArgvParsedDict:
        parsed = parsed is None and self.parsed or parsed
        assert parsed is not None, (
            "No arguments were parsed. Have you run <obj>.parse(...)"
        )

        new: ArgvParsedDict = {}
        for k, v in parsed.items():
            if f := self.processors.get(k):
                new[k] = f(v)
            else:
                new[k] = v

        return new

    def parse(
        self,
        args: str | list[str] | None = None,
        pcall: bool = True,
        on_failure: Callable[[Exception, list[str]], any] = lambda error, args: (
            error,
            args,
        ),
    ) -> Result:
        if args:
            args = isinstance(args, str) and shlex.split(args) or args
            try:
                # this will print the error message anyway
                parsed, _ = self.parse_known_args(args)
                parsed = parsed.__dict__
                parsed, _ = self.validate_parsed_args(parsed, pcall=pcall)
                parsed = self.process_parsed_args(parsed)
                self.parsed = parsed

                return Result(True, parsed)
            except SystemExit as error:
                if not pcall:
                    raise error
                else:
                    return Result(False, error, error.args[0])
            except Exception as error:
                if not pcall:
                    raise error
                else:
                    return Result(False, error, error.args[0])
        else:
            try:
                # this will print the error message anyway
                parsed, _ = self.parse_known_args()
                parsed = parsed.__dict__
                parsed, _ = self.validate_parsed_args(parsed, pcall=pcall)
                parsed = self.process_parsed_args(parsed)
                self.parsed = parsed

                return Result(True, parsed)
            except SystemExit as error:
                if not pcall:
                    raise error
                else:
                    return Result(False, error, error.args[0])
            except Exception as error:
                if not pcall:
                    raise error
                else:
                    return Result(False, error, error.args[0])

            return Result(True, self.parsed)


parser = Argv("Some CLI application")
parser.on(
    "-i",
    "--input-file",
    validate=[
        lambda x: Result(True, x),
        lambda x: Result(False, None, "Laude madarchod"),
    ],
    nargs=1,
)
