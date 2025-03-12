#!/bin/bash

# Check for uncommitted changes in the working tree
if [ -n "$(git status --porcelain)" ]; then
    echo "Linting tools found the following changes are needed to comply"
    echo "with our automatic styling."
    echo ""
    echo "Please run \"pre-commit run --all-files\" locally to fix these."
    echo "See also https://github.com/spcl/dace/blob/main/CONTRIBUTING.md"
    echo ""
    echo "git status"
    echo "----------"
    git status
    echo ""
    echo "git diff"
    echo "--------"
    git --no-pager diff
    echo ""

    exit 1
fi
