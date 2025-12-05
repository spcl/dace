#!/usr/bin/env bash

# Folder name without slash, e.g. gcc_amd_epyc
folder_name="."

if [[ -d ".dacecache" ]]; then
    for dir in .dacecache/*/; do
        name=$(basename "$dir")
        so_file=".dacecache/$name/build/lib${name}.so"

        if [[ -f "$so_file" ]]; then
            # Output filename = folder + kernel
            out_name="${folder_name}_${name}.asm"

            echo "  Dumping $so_file → asm/$out_name"
            objdump -d "$so_file" > "asm/$out_name"
        else
            echo "  Skipping $name (no .so found)"
        fi
    done
else
    echo "  No .dacecache in $sub"
fi

