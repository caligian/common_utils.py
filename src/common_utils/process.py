import subprocess

from .table import strip


def system(
    cmd: list[str] | str,
    capture: bool = True,
    splitlines: bool = False,
    pcall: bool = False,
    chomp: bool = True,
    no_stdout: bool = False,
    no_stderr: bool = False,
    **kwargs,
) -> (
    subprocess.CompletedProcess
    | subprocess.CalledProcessError
    | FileNotFoundError
    | tuple[str, str]
    | tuple[list[str], list[str]]
    | bool
):
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
        proc = subprocess.run(cmd, **kwargs)
        if capture:
            stdout = proc.stdout.decode()
            stderr = proc.stderr.decode()
            stdout = strip(stdout, lhs=False) if chomp else stdout
            stderr = strip(stderr, lhs=False) if chomp else stderr
            stdout = splitlines and stdout.split("\n") or stdout
            stderr = splitlines and stderr.split("\n") or stderr

            return (stdout, stderr)
        else:
            return True
    except Exception as error:
        if pcall:
            return error
        else:
            raise error


def systemlist(
    cmd: list[str] | str,
    pcall: bool = False,
    chomp: bool = True,
    **kwargs,
) -> list[str] | Exception:
    return system(
        cmd,
        capture=True,
        splitlines=True,
        chomp=chomp,
        pcall=pcall,
        **kwargs,
    )


__all__ = ["system", "systemlist"]
