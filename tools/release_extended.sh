#!/usr/bin/env bash
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# Patch release of DaCe extended as the PyPI package dace-extended (import dace), from the ThrudPrimrose/dace fork.
# The fork's main mirrors spcl/dace main exactly. Its extended is spcl/dace@extended plus ONE release commit that
# changes only the package metadata: [project] name "dace-extended", dace.__version__ = X.Y.Z, dace.__commit__ = the
# released spcl/dace commit, Python >=3.12 and the alloy extra
# `dace-alloy>=0.1.0` (PyPI refuses direct URLs). The tag extended-vX.Y.Z names that commit, so a git pin of the tag
# and the PyPI release are the same distribution. Each run releases the spcl/dace@extended head as the next patch
# (extended-vX.Y.(Z+1); the first is extended-v1.0.0): fork sync, release commit, tag, wheel + sdist, GitHub
# release, PyPI upload. The fork runs no CI (its Actions are off): the gate reads spcl/dace's own CI runs on the head.
# Versions are dace-extended's own line: each release bumps the patch (1.0.0, 1.0.1, ...); --upstream-release bumps
# the minor and resets the patch (1.1.0) when spcl/dace has made a release since. Each release's notes name the
# upstream DaCe version (dace/version.py at the released commit) it builds on.
#   tools/release_extended.sh [--upstream-release] [--allow-running] [--outdir DIR] [--dry-run]
#     --upstream-release  the release follows a new spcl/dace release: bump the minor, not the patch
#     --allow-running  also take spcl/dace workflows still running on the head (never a failed one)
#     --outdir DIR     keep the wheel and sdist in DIR (default: a temporary folder)
#     --dry-run        gate, next tag, release commit, build + twine check, notes; push and upload nothing
# A head that already has a release is left alone. Dependents move their pins to the newest PyPI release themselves.
set -euo pipefail
ulimit -c 0
FORK="ThrudPrimrose/dace"
UPSTREAM="spcl/dace"
die() { echo "release_extended.sh: $*" >&2; exit 1; }
dry=0
running=0
upstream=0
outdir=""
while (($#)); do
    case "$1" in
        --dry-run) dry=1 ;;
        --allow-running) running=1 ;;
        --upstream-release) upstream=1 ;;
        --outdir) outdir="$2"; shift ;;
        *) echo "usage: $0 [--upstream-release] [--allow-running] [--outdir DIR] [--dry-run]" >&2; exit 2 ;;
    esac
    shift
done
((dry)) || [[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] || die "TWINE_USER and TWINE_TOKEN must be set"

sha="$(git ls-remote "https://github.com/${UPSTREAM}.git" refs/heads/extended | cut -f1)"
[[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || die "no extended head on ${UPSTREAM}"
echo "=== ${UPSTREAM}@extended head ${sha} ==="

work="$(mktemp -d "${TMPDIR:-/tmp}/dace-extended-release.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
src="${work}/src"
git init -q "${src}"
git -C "${src}" remote add fork "https://github.com/${FORK}.git"
if git -C "${src}" ls-remote --exit-code --tags fork 'refs/tags/extended-v*' >/dev/null; then
    git -C "${src}" fetch -q --depth 2 fork '+refs/tags/extended-v*:refs/tags/extended-v*'
fi
# The released spcl/dace commit of each tag is its release commit's parent.
last=""
since=""
current=""
while read -r name; do
    [[ -n "${name}" ]] || continue
    parent="$(git -C "${src}" rev-parse "${name}^{commit}^")"
    [[ "${parent}" != "${sha}" ]] || current="${name}"
    last="${name}"
    since="${parent}"
done < <(git -C "${src}" tag -l 'extended-v*' | sort -V)

[[ -z "${current}" ]] || { echo "${sha:0:9} is already ${FORK} ${current}, nothing to release"; exit 0; }

runs="$(gh run list -R "${UPSTREAM}" -c "${sha}" -L 100 --json workflowName,status,conclusion)"
[[ "$(jq length <<<"${runs}")" -gt 0 ]] || die "no ${UPSTREAM} CI run on ${sha:0:9}"
open="$(jq -r --argjson running "${running}" '.[]
    | select((.status != "completed" and $running == 0)
        or (.status == "completed" and .conclusion != "success" and .conclusion != "skipped"))
    | "  \(.workflowName): \(.status) \(.conclusion)"' <<<"${runs}")"
[[ -z "${open}" ]] || die "${UPSTREAM} CI on ${sha:0:9} is not green:"$'\n'"${open}"
unfinished="$(jq '[.[] | select(.status != "completed")] | length' <<<"${runs}")"
echo "CI: $(jq length <<<"${runs}") ${UPSTREAM} workflows, none failed, ${unfinished} still running"

if [[ -z "${last}" ]]; then
    version="1.0.0"
else
    IFS=. read -r major minor patch <<<"${last#extended-v}"
    if ((upstream)); then
        version="${major}.$((minor + 1)).0"
    else
        version="${major}.${minor}.$((patch + 1))"
    fi
fi
tag="extended-v${version}"
echo "tag: ${tag} (previous: ${last:-none})"

# The release commit: the head plus the PyPI metadata, nothing else.
git -C "${src}" fetch -q --depth 1 "https://github.com/${UPSTREAM}.git" "${sha}"
git -C "${src}" checkout -q --detach "${sha}"
base="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "${src}/dace/version.py")"
[[ -n "${base}" ]] || die "no __version__ in dace/version.py"
echo "upstream DaCe version: ${base}"
python3 - "${src}" "${version}" "${sha}" <<'EOF'
import pathlib
import re
import sys

root, version, commit = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = root / "pyproject.toml"
text = path.read_text()
edits = [
    ('\n[project]\nname = "dace"\n', '\n[project]\nname = "dace-extended"\n'),
    ('requires-python = ">=3.10, <3.15"\n', 'requires-python = ">=3.12"\n'),
]
for old, new in edits:
    if text.count(old) != 1:
        sys.exit(f"pyproject.toml: expected one {old.strip()!r}")
    text = text.replace(old, new)
text, count = re.subn(r'^alloy = \["alloy @ git\+[^"]*"\]$', 'alloy = ["dace-alloy>=0.1.0"]', text, flags=re.MULTILINE)
if count != 1:
    sys.exit("pyproject.toml: expected one alloy extra with a git URL")
if re.search(r'"[^"]* @ (git\+|https?://)', text):
    sys.exit("pyproject.toml: a direct URL is left, which PyPI refuses")
path.write_text(text)
# setuptools reads the package version from dace.__version__, so the two always agree; __commit__ names the spcl/dace
# commit released, which a PyPI install has no direct_url.json for (dependents key their caches on it).
module = root / "dace" / "version.py"
source, count = re.subn(r'^__version__ = "[^"]*"$', f'__version__ = "{version}"\n__commit__ = "{commit}"',
                        module.read_text(), flags=re.MULTILINE)
if count != 1 or source.count("__commit__") != 1:
    sys.exit("dace/version.py: expected one __version__ line and no __commit__")
module.write_text(source)
EOF
git -C "${src}" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -q -am \
    "dace-extended ${version}: PyPI metadata for ${UPSTREAM}@extended ${sha:0:9}" \
    -m "Name dace-extended (import dace), version ${version} (dace.__version__), dace.__commit__ ${sha}, Python >=3.12, the alloy extra as dace-alloy>=0.1.0. Nothing else changes."
release="$(git -C "${src}" rev-parse HEAD)"
git -C "${src}" submodule update -q --init --depth 1
dist="${outdir:-${work}/dist}"
mkdir -p "${dist}"
(cd "${src}" && uv build -q -o "${dist}")
uvx twine check --strict "${dist}"/dace_extended-"${version}"*

notes="${work}/notes.md"
{
    echo "dace-extended ${version}: DaCe's extended branch at [${UPSTREAM}@extended \`${sha:0:9}\`](https://github.com/${UPSTREAM}/commit/${sha}),"
    echo "built on upstream DaCe ${base} (dace-extended ${version%.*}.x follows DaCe ${base}),"
    echo "on PyPI as [\`dace-extended\`](https://pypi.org/project/dace-extended/${version}/). The import name stays \`dace\`."
    echo "The release commit \`${release:0:9}\` changes only the package metadata (name, version, the released commit as \`dace.__commit__\`, Python >=3.12, the alloy extra)."
    echo
    echo '```'
    echo "pip install dace-extended==${version}"
    echo '```'
    echo
    if ((unfinished == 0)); then
        echo "Every ${UPSTREAM} CI workflow passed on \`${sha:0:9}\`."
    else
        echo "No ${UPSTREAM} CI workflow failed on \`${sha:0:9}\`; workflows still running at release time: ${unfinished}."
    fi
    if [[ -n "${since}" ]]; then
        echo
        echo "## Changes since ${last}"
        echo
        gh api "repos/${UPSTREAM}/compare/${since}...${sha}" \
            -q '.commits[] | "- " + (.commit.message | split("\n")[0]) + " (" + .sha[0:9] + ")"'
    fi
} >"${notes}"
cat "${notes}"
if ((dry)); then
    echo "dry run: nothing pushed; rerun without --dry-run to release ${tag} (${release:0:9})"
    exit 0
fi

# Both branches first take spcl/dace's heads, so the fork holds the release commit's parent before the push.
for branch in main extended; do
    gh repo sync "${FORK}" --source "${UPSTREAM}" --branch "${branch}" --force
done
git -C "${src}" push -q --force fork "${release}:refs/heads/extended"
git -C "${src}" tag -a "${tag}" -m "dace-extended ${version}" "${release}"
git -C "${src}" push -q fork "refs/tags/${tag}"
gh release create "${tag}" -R "${FORK}" --verify-tag --title "dace-extended ${version}" --notes-file "${notes}" \
    "${dist}"/dace_extended-"${version}"*
echo "released https://github.com/${FORK}/releases/tag/${tag} (${release})"
TWINE_USERNAME="${TWINE_USER}" TWINE_PASSWORD="${TWINE_TOKEN}" \
    uvx twine upload --non-interactive "${dist}"/dace_extended-"${version}"*
echo "uploaded https://pypi.org/project/dace-extended/${version}/"
