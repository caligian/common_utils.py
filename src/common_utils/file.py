import csv
import subprocess
import shutil
import os
import re

from glob import glob
from typing import Callable, overload, Literal
from pickle import (
    load as fh_load_pkl,
    dump as fh_dump_pkl,
    loads as load_pkl,
    dumps as dump_pkl,
)
from csv import (
    reader as csv_reader,
    writer as csv_writer,
)
from json import (
    load as fh_load_json,
    dump as fh_dump_json,
    loads as load_json,
    dumps as dump_json,
)
from .table import split
from .process import system
from .result import Ok, Err, T, E

mkdir = os.makedirs
is_dir = os.path.isdir
is_file = os.path.isfile
is_mount = os.path.ismount
is_link = os.path.islink
is_junction = os.path.isjunction
exists = os.path.exists
rmtree = shutil.rmtree
basename = os.path.basename
dirname = os.path.dirname
abspath = os.path.abspath
stat = os.stat
cpstat = shutil.copystat


def file_extension(filename: str) -> str:
    return filename.rsplit(".", maxsplit=1)[-1]


def has_extension(filename: str, *pattern: str | re.Pattern) -> bool:
    extension = file_extension(filename)
    for pat in pattern:
        if re.search(pat, extension, flags=re.I):
            return True
    return False


def read_json(filename: str) -> Ok[list | dict] | Err[Exception]:
    with open(filename, "r") as fh:
        try:
            return Ok(fh_load_json(fh))
        except Exception as error:
            return Err(error)


def write_json(filename: str, obj: any) -> Ok[str] | Err[Exception]:
    with open(filename, "w") as fh:
        try:
            fh_dump_json(obj, fh)
            return Ok(filename)
        except Exception as error:
            return Err(error, {"file": filename})


def read_pkl(filename: str) -> Ok[any] | Err[Exception]:
    try:
        with open(filename, "rb") as fh:
            return Ok(fh_load_pkl(fh))
    except Exception as error:
        return Err(error)


def write_pkl(filename: str, obj: any) -> Ok[str] | Err[Exception]:
    try:
        with open(filename, "wb") as fh:
            fh_dump_pkl(obj, fh)
            return Ok(filename)
    except Exception as error:
        return Err(error, {"file": filename})


# Do the same for pkl, json and other stuff
def read_csv(
    filename: str,
    everything: bool = True,
    **kwargs,
) -> Ok[list[list[str]] | csv.reader] | Err[Exception]:
    try:
        with open(filename) as fh:
            if everything:
                return Ok([line for line in csv_reader(fh, **kwargs)])
            else:
                return Ok(csv_reader(fh))
    except Exception as error:
        return Err(error)


def write_csv(
    filename: str,
    lines: list[list[str]],
    **kwargs,
) -> Ok[str] | Err[Exception]:
    try:
        with open(filename, "w") as fh:
            writer = csv_writer(fh, **kwargs)
            writer.writerows(lines)
            return Ok(filename)
    except Exception as error:
        return Err(error, {"file": filename})


def slurp(
    filename: str,
    format: str = "text",
    reader: Callable = None,
    newlines: bool = False,
    chomp: bool = True,
    binary: bool = False,
) -> Ok[any] | Err[Exception]:
    if format in ("json", "j"):
        return read_json(filename)
    elif format in ("text", "txt", "t"):
        if newlines:
            return readlines(filename, binary=binary, chomp=chomp)
        else:
            return readtext(filename, binary=binary, chomp=chomp)
    elif format in ("pickle", "pkl", "p"):
        return read_pkl(filename)
    elif callable(reader):
        try:
            return Ok(reader(filename))
        except Exception as error:
            return Err(error, dict(file=filename, reader=reader))
    else:
        return Err(
            NotImplementedError(f"{format} reader is not implemented"),
            {
                "file": filename,
                "format": format,
                "reader": reader,
                "binary": binary,
            },
        )


def spit(
    filename: str,
    obj: any,
    format: str = "text",
    writer: Callable[[any], str] | None = None,
    binary: bool = False,
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]:
    if format in ("json", "j"):
        return write_json(filename, obj)
    elif format in ("text", "txt", "t"):
        if isinstance(obj, (str, bytes)):
            return writetext(
                filename,
                obj,
                binary=binary,
                append_newline=append_newline,
                encoding=encoding,
                errors=errors,
            )
        else:
            return writelines(
                filename,
                obj,
                binary=binary,
                append_newline=append_newline,
                encoding=encoding,
                errors=errors,
            )
    elif format in ("pickle", "pkl", "p"):
        return write_pkl(filename, obj)
    elif writer:
        try:
            return Ok(writer(filename, obj))
        except Exception as error:
            return Err(error, dict(file=filename))
    else:
        return Err(
            NotImplementedError(f"{format} writer is not implemented"),
            {"file": filename, "writer": writer, "format": format},
        )


def ls(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
    include: str = "dflmj",
    stat: bool = False,
    follow_symlinks: bool = False,
) -> None | list[str] | list[tuple[str, os.stat_result]]:
    if not os.path.isdir(d):
        return

    pattern = re.compile(pattern, flags=re.I | re.M)
    files: list[str] = glob(f"{d}/*") + glob(f"{d}/.*")
    exclude = exclude and re.compile(exclude, flags=re.I | re.M) or None
    files = [x for x in files if pattern.search(x)]

    if exclude:
        files = [x for x in files if not exclude.search(x)]

    res = []
    d = "d" in include
    f = "f" in include
    lnk = "l" in include
    m = "m" in include
    j = "j" in include

    def append_file(filename: str) -> None:
        if stat:
            res.append((filename, os.stat(filename)))
        else:
            res.append(filename)

    for file in files:
        if (
            (d and is_dir(file))
            or (f and is_file(file))
            or (lnk and is_link(file))
            or (m and is_mount(file))
            or (j and is_junction(file))
        ):
            append_file(file)

    return res


def mimetype(filename: str) -> str | None:
    try:
        out = subprocess.check_output(["file", "--mime-encoding", filename])
        out = out.decode()
        out = out.split(":")
        out = out[-1]
        out = out.strip()

        if "cannot open" in out[:12]:
            return

        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    except Exception:
        return


def cp(
    src: str,
    dest: str,
    makedirs: bool = False,
    overwrite: bool = True,
    **kwargs,
) -> Ok[str] | Err[PermissionError | FileNotFoundError | OSError | FileExistsError]:
    if not os.path.exists(src):
        return Err(FileNotFoundError(src), dict(src=src))

    if os.path.exists(dest) and not overwrite:
        return Err(FileExistsError(dest), dict(dest=dest))

    dirname = os.path.dirname(dest)
    if not makedirs:
        if not os.path.exists(dirname):
            return Err(FileNotFoundError(dirname), dict(dest_dir=dirname, dest=dest))
    else:
        try:
            os.makedirs(dirname)
        except Exception as error:
            return Err(error, {"dest_dir": dirname, "dest": dest})

    try:
        if os.path.isdir(src):
            shutil.copytree(src, dest, **kwargs)
        else:
            shutil.copy(src, dest, **kwargs)

        return Ok(dest)
    except Exception as error:
        return Err(error, dict(src=src, dest=dest))


def rm(path: str, **kwargs) -> Ok[str] | Err[Exception]:
    if not os.path.exists(path):
        return Err(FileNotFoundError(path), {"path": path})
    elif os.path.isdir(path):
        try:
            shutil.rmtree(path, **kwargs)
            return Ok(path)
        except Exception as error:
            return Err(error, {"path": path})

    try:
        os.unlink(path)
        return Ok(path)
    except Exception as error:
        return Err(error, {"path": path})


def readlines(
    filename: str,
    binary: bool = False,
    chomp: bool = False,
) -> Ok[list[str] | list[bytes]] | Err[Exception]:
    mode = "rb" if binary else "r"
    try:
        with open(filename, mode) as fh:
            if not chomp:
                return Ok(fh.readlines())
            else:
                return Ok([x.rstrip(b"\n\r") for x in fh.readlines()])
    except Exception as error:
        return Err(error)


def readtext(
    filename: str,
    binary: bool = False,
    chomp: bool = True,
) -> Ok[str | bytes] | Err[Exception]:
    mode = "rb" if binary else "r"
    try:
        with open(filename, mode) as fh:
            if not chomp:
                return Ok(fh.read())
            else:
                return Ok(fh.read().rstrip(b"\n\r"))
    except Exception as error:
        return Err(error)


@overload
def writetext(
    filename: str,
    text: str,
    binary: Literal[False] = False,
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]: ...


@overload
def writetext(
    filename: str,
    text: bytes,
    binary: Literal[True],
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]: ...


def writetext(
    filename: str,
    text: str | bytes,
    binary: bool = False,
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]:
    mode = "w"
    use = text

    if binary:
        mode = "wb"
        if not isinstance(text, bytes):
            try:
                use = text.encode(encoding, errors=errors)
            except Exception as error:
                return Err(
                    error,
                    {
                        "file": filename,
                        "binary": binary,
                        "encoding": encoding,
                        "errors": errors,
                    },
                )
    elif not isinstance(text, str):
        try:
            use = text.decode(encoding=encoding, errors=errors)
        except Exception as error:
            return Err(
                error,
                dict(file=filename, binary=binary, encoding=encoding, errors=errors),
            )

    try:
        with open(filename, mode) as fh:
            fh.write(use)
            if append_newline:
                if not binary:
                    fh.write("\n")
                else:
                    fh.write(b"\n")

            return Ok(filename)
    except Exception as error:
        return Err(
            error, dict(file=filename, binary=binary, encoding=encoding, errors=errors)
        )


@overload
def writelines(
    filename: str,
    text: list[str],
    binary: Literal[False] = False,
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]: ...


@overload
def writelines(
    filename: str,
    text: list[bytes],
    binary: Literal[True],
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]: ...


def writelines(
    filename: str,
    text: list[str] | list[bytes],
    binary: bool = False,
    append_newline: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Ok[str] | Err[Exception]:
    lines = text
    mode = "w"

    if binary:
        mode = "wb"
        if len(text) > 0 and isinstance(text[0], str):
            try:
                lines = [x.encode(encoding, errors) for x in text]
            except Exception as error:
                return Err(
                    error,
                    dict(
                        file=filename, binary=binary, encoding=encoding, errors=errors
                    ),
                )
    elif len(text) > 0 and isinstance(text[0], bytes):
        try:
            lines = [x.decode(encoding, errors) for x in text]
        except Exception as error:
            return Err(
                error,
                dict(file=filename, binary=binary, encoding=encoding, errors=errors),
            )

    try:
        with open(filename, mode) as fh:
            for line in lines:
                fh.write(line)
                if append_newline:
                    if binary:
                        fh.write(b"\n")
                    else:
                        fh.write("\n")
            return Ok(filename)
    except Exception as error:
        return Err(
            error,
            dict(
                file=filename,
                encoding=encoding,
                errors=errors,
                binary=binary,
            ),
        )


def whereis(binary: str) -> list[str] | None:
    stdout, _ = system(f"whereis {binary}")
    out = split(stdout, r"\s+")
    out[0] = out[0][:-1]
    out.pop(0)

    if len(out) == 0:
        return
    else:
        return [x for x in out if os.access(x, os.X_OK)]


__all__ = [
    "abspath",
    "basename",
    "cp",
    "cpstat",
    "csv_reader",
    "csv_writer",
    "dirname",
    "dump_json",
    "dump_pkl",
    "exists",
    "fh_dump_json",
    "fh_dump_pkl",
    "fh_load_json",
    "fh_load_pkl",
    "file_extension",
    "has_extension",
    "is_dir",
    "is_file",
    "is_junction",
    "is_link",
    "is_mount",
    "load_json",
    "load_json",
    "load_pkl",
    "load_pkl",
    "ls",
    "mimetype",
    "mkdir",
    "read_csv",
    "read_json",
    "read_pkl",
    "readlines",
    "rm",
    "rmtree",
    "slurp",
    "spit",
    "stat",
    "whereis",
    "write_csv",
    "write_json",
    "write_pkl",
    "writelines",
]
