import subprocess
import shutil
import os
import re

from glob import glob
from typing import Callable
from pickle import (
    load as fh_pkl_load,
    dump as fh_pkl_dump,
    loads as load_pkl,
    dumps as dump_pkl,
)
from csv import (
    reader as csv_reader,
    writer as csv_writer,
)
from json import (
    load as fh_json_load,
    dump as fh_json_dump,
    loads as load_json,
    dumps as dump_json,
)
from src.common_utils.table import split
from src.common_utils.process import system

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
isdir = is_dir
isfile = is_file
ismount = is_mount
isjunction = is_junction
ispath = exists


def file_extension(filename: str) -> str:
    return filename.rsplit(".", maxsplit=1)[-1]


def has_extension(filename: str, *pattern: str | re.Pattern) -> bool:
    extension = file_extension(filename)
    for pat in pattern:
        if re.search(pat, extension, flags=re.I):
            return True

    return False


def read_json(filename: str) -> any:
    with open(filename, "r") as fh:
        return fh_json_load(fh)


def write_json(filename: str, obj: any) -> None:
    with open(filename, "w") as fh:
        fh_json_dump(obj, fh)


def read_pkl(filename: str) -> any:
    with open(filename, "rb") as fh:
        return fh_pkl_load(fh)


def write_pkl(filename: str, obj: any) -> any:
    with open(filename, "wb") as fh:
        return fh_pkl_dump(obj, fh)


def read_csv(
    filename: str,
    read_all: bool = True,
    **kwargs,
) -> list[str]:
    with open(filename) as fh:
        if read_all:
            return [line for line in csv_reader(fh, **kwargs)]
        else:
            return csv_reader(fh)


def write_csv(
    filename: str,
    lines: str | list[str],
    sep: str = r"\n",
    **kwargs,
) -> int:
    with open(filename, "w") as fh:
        writer = csv_writer(fh, **kwargs)
        size = 0

        if type(lines) is str and r"\n" in lines:
            size = len(size)
            lines = lines.split(r"\n")
        elif type(lines) is list:
            size = sum(map(len, lines))

        writer.writerows(lines)
        return size


def slurp(
    filename: str,
    mode: str = "r",
    format: str = "text",
    reader: Callable[[str | bytes], any] | None = None,
    newlines: bool = False,
    chomp: bool = True,
) -> list[str] | str:
    match format:
        case ft if ft in ("json", "j"):
            return read_json(filename)
        case ft if ft in ("text", "txt", "t"):
            with open(filename, mode) as fh:
                text = fh.read()
                text = chomp and text.strip() or text

                if newlines:
                    return text.split("\n")
                else:
                    return text
        case ft if ft in ("pickle", "pkl", "p"):
            return read_pkl(filename)
        case reader if callable(reader):
            return reader(filename)
        case ft:
            raise NotImplementedError(f"{ft} reader is not implemented")


def spit(
    filename: str,
    obj: any,
    mode: str = "w",
    format: str = "text",
) -> int:
    match format:
        case ft if ft in ("json", "j"):
            return write_json(filename, obj)
        case ft if ft in ("text", "txt", "t"):
            with open(filename, mode) as fh:
                fh.write(str(obj))
        case ft if ft in ("pickle", "pkl", "p"):
            return write_pkl(filename, obj)
        case writer if callable(writer):
            return writer(filename, obj)
        case ft:
            raise NotImplementedError(f"{ft} writer is not implemented")


def ls(
    d: str,
    pattern: str = ".+",
    exclude: str | None = None,
    include: str = "dflmj",
    stat: bool = False,
    follow_symlinks: bool = False,
) -> list[str] | list[tuple[str, os.stat_result]]:
    pattern = re.compile(pattern, flags=re.I + re.M)
    files: list[str] = glob(f"{d}/*") + glob(f"{d}/.*")
    exclude = exclude and re.compile(exclude, flags=re.I + re.M) or None
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
    out = subprocess.check_output(["file", "--mime-encoding", filename])
    out = out.decode()
    out = out.split(":")
    out = out[-1]
    out = out.strip()

    if "cannot open" in out[:12]:
        return

    return out


def cp(src: str, dest: str, **kwargs) -> str:
    if os.path.isdir(src):
        shutil.copytree(src, dest, **kwargs)
    else:
        shutil.copy(src, dest, **kwargs)

    return dest


def rm(path: str, **kwargs) -> bool:
    if not os.path.exists(path):
        return False
    elif os.path.isdir(path):
        shutil.rmtree(path, **kwargs)
    else:
        os.remove(path, **kwargs)

    return True


def readlines(filename: str) -> list[str]:
    return slurp(filename, newlines=True)


def writelines(
    filename: str,
    *text: list[str],
    append_newline: bool = True,
) -> int:
    with open(filename, "w") as fh:
        size = 0
        for line in text:
            if append_newline:
                fh.write(line + "\n")
                size += len(line) + 1
            else:
                fh.write(line)
                size += len(line)

        return size


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
    "load_pkl",
    "dump_pkl",
    "load_json",
    "dump_json",
    "abspath",
    "basename",
    "cp",
    "cpstat",
    "dirname",
    "file_extension",
    "has_extension",
    "is_dir",
    "isdir",
    "is_file",
    "isfile",
    "is_junction",
    "isjunction",
    "is_link",
    "is_mount",
    "ismount",
    "ispath",
    "ls",
    "mimetype",
    "mkdir",
    "exists",
    "read_csv",
    "read_json",
    "readlines",
    "read_pkl",
    "rm",
    "rmtree",
    "slurp",
    "spit",
    "stat",
    "whereis",
    "write_csv",
    "write_json",
    "writelines",
    "write_pkl",
]
