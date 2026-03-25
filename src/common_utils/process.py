import subprocess

from subprocess import CompletedProcess, CalledProcessError
from collections import namedtuple
from typing import Literal, Callable, Any, overload

from .table import strip
from .result import Result, Ok, Err
from .error import raise_error

StringProcessOutput = tuple[CompletedProcess, str, str]
ListProcessOutput = tuple[CompletedProcess, list[str], list[str]]
ProcessOutput: StringProcessOutput | ListProcessOutput = namedtuple(
    "ProcessOutput",
    ("process", "stdout", "stderr"),
    defaults=(None, None, None),
)
ProcessError = CalledProcessError | FileNotFoundError


@overload
def system(
    cmd: list[str] | str,
    capture: Literal[True] = True,
    splitlines: Literal[True] = False,
    pcall: bool = False,
    chomp: bool = True,
    no_stdout: Literal[False] = False,
    no_stderr: Literal[False] = False,
    **kwargs,
) -> Result[StringProcessOutput, ProcessError]: ...


@overload
def system(
    cmd: list[str] | str,
    capture: Literal[True] = True,
    splitlines: Literal[True] = True,
    pcall: bool = False,
    chomp: bool = True,
    no_stdout: Literal[False] = False,
    no_stderr: Literal[False] = False,
    **kwargs,
) -> Result[ListProcessOutput, ProcessError]: ...


@overload
def system(
    cmd: list[str] | str,
    capture: bool = True,
    splitlines: bool = True,
    pcall: bool = False,
    chomp: bool = True,
    no_stdout: Literal[True] = False,
    no_stderr: Literal[True] = False,
    **kwargs,
) -> Result[bool, ProcessError]: ...


def system(
    cmd: list[str] | str,
    capture: bool = True,
    splitlines: bool = False,
    pcall: bool = False,
    chomp: bool = True,
    no_stdout: bool = False,
    no_stderr: bool = False,
    **kwargs,
) -> Result[ProcessOutput | bool, ProcessError]:
    kwargs = kwargs.copy()
    kwargs["check"] = True
    kwargs["capture_output"] = capture

    if type(cmd) is str:
        kwargs["shell"] = True

    if no_stdout:
        kwargs["stdout"] = subprocess.DEVNULL

    if no_stderr:
        kwargs["stderr"] = subprocess.DEVNULL

    try:
        process = subprocess.run(cmd, **kwargs)
        if capture:
            stdout = process.stdout.decode() if not no_stdout else None
            stderr = process.stderr.decode() if not no_stderr else None

            if stdout:
                stdout = strip(stdout, lhs=False) if chomp else stdout
                stdout = splitlines and stdout.split("\n") or stdout

            if stderr:
                stderr = strip(stderr, lhs=False) if chomp else stderr
                stderr = splitlines and stderr.split("\n") or stderr

            return Ok(ProcessOutput(process, stdout, stderr))
        else:
            return Ok(ProcessOutput(True))
    except Exception as error:
        if pcall:
            return Err(error, dict(cmd=cmd, kwargs=kwargs))
        else:
            raise_error(error, dict(cmd=cmd, kwargs=kwargs))


def systemlist(
    cmd: list[str] | str,
    pcall: bool = False,
    chomp: bool = True,
    **kwargs,
) -> Result[list[str], ProcessError]:
    return system(
        cmd,
        capture=True,
        splitlines=True,
        chomp=chomp,
        pcall=pcall,
        **kwargs,
    )


__all__ = [
    "system",
    "systemlist",
    "ProcessError",
    "CalledProcessError",
]
