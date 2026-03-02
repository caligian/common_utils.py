#!/bin/bash

file="${1:-*.py}"

/bin/grep -Po '^def ([^(]+)|^([a-zA-Z_]+)\s*=' $file |\
    cut -d ':' -f 2 |\
    /bin/grep -v __all__ |\
    sed 's/^def //' |\
    sed -r 's/\s*=\s*//' |\
    sed -r 's/(.+)/"\1",/' |\
    sort |\
    uniq |\
    tee  |\
    wl-copy
