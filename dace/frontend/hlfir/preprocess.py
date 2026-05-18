"""Fortran-source-level pre-processor.

Some transforms have to happen on the Fortran *text*, before
``flang-new -fc1 -emit-hlfir`` runs, because they either change what
flang accepts or what arithmetic each backend is free to pick.  This
module holds three independent text rewrites:

* ``rewrite_integer_powers`` -- expands an integer-valued REAL-literal
  power (``x**2.0`` -> ``(x*x)``).  Runs **unconditionally** in
  ``compile_to_hlfir``: the rewrite is algebraically exact and removes
  a backend-dependent ``pow(x, 2.0)`` vs ``x*x`` rounding difference
  against the gfortran reference.

* ``promote_real_literals_to_double`` -- rewrites single/default REAL
  literals to an explicit double form (``2.0`` -> ``2.0D0``).  A
  standalone utility, applied directly to kernel source on disk when a
  codebase must be globally double; **not** wired into the build path.

* ``preprocess_fortran`` -- rewrites ``IF (intvar)`` to
  ``IF (intvar /= 0)`` for INTEGER scalars.  flang-new-21 rejects bare
  INTEGER as an IF condition (only LOGICAL is legal); legacy ECRAD /
  CloudSC / ICON code ships this shape.  **Opt-in** per call site
  (``compile_to_hlfir(..., preprocess=True)``) -- off by default so we
  don't paper over real issues in clean source.

These are pragmatic SED-style transforms, NOT a Fortran parser; they
are deliberately narrow (single-identifier IF guards only; powers with
a primary base only) and brittle by construction.  Comment- and
string-awareness is shared via ``_scan_line`` so a ``!`` or ``**``
inside a character literal is never touched.
"""

import re

# A REAL-literal exponent whose value is a whole number (``**2.0``,
# ``**3.0``, ``**2.0_JPRB``, ``**2.0D0``, ``**2.``).  Only this form is
# expanded to repeated multiplication: it is the case where each backend
# is free to pick ``pow(x, 2.0)`` and round differently from gfortran's
# ``x*x``.  A bare-integer exponent (``x**2``) is deliberately left
# alone -- flang already lowers it to the integer-power (multiply) path
# bit-identically to gfortran -- and genuine fractional powers
# (``**0.5``, ``**0.333``) must stay as ``pow()``.  ``_REAL_EXP``
# requires a digit before the dot so a bare integer never matches.
_REAL_EXP = r"\d+\.\d*(?:[eEdD][+-]?\d+)?(?:_[A-Za-z]\w*|_\d+)?"
_INT_POW_RE = re.compile(r"\*\*\s*(?:\(\s*(" + _REAL_EXP + r")\s*\)|(" + _REAL_EXP + r")(?![\w.]))")

# An identifier immediately followed by ``(`` -- a function call or an
# array reference.  A power base containing one must not be duplicated.
_CALL_IN_BASE = re.compile(r"[A-Za-z_]\w*\s*\(")

# A Fortran REAL literal: needs a fractional point or an exponent (so a
# bare integer never matches).  ``mantissa`` is groups 1-3, ``kind`` the
# optional ``_KIND`` / ``_8`` suffix.  Lookbehind/ahead keep us off
# identifiers (``R2ES``) and kind selectors.
_REAL_LIT_RE = re.compile(r"(?<![\w.])"
                          r"(\d+\.\d*|\.\d+|\d+)"  # mantissa
                          r"([eEdD][+-]?\d+)?"  # optional exponent
                          r"(_[A-Za-z]\w*|_\d+)?"  # optional kind suffix
                          r"(?![\w.])")
# Kind suffixes that are already double precision -- leave those alone.
_DOUBLE_KINDS = {"jprb", "jprd", "dp", "8", "16", "r8", "qp"}

_INTEGER_DECL_RE = re.compile(
    r"\bINTEGER\b(?:\s*\([^)]*\))?(?:\s*,\s*[A-Z_]+(?:\s*\([^)]*\))?)*\s*::\s*([^\n!]+)",
    re.IGNORECASE,
)
_BARE_IF_RE = re.compile(r"\b(IF\s*\(\s*)([A-Za-z_]\w*)(\s*\))", re.IGNORECASE)


def _scan_line(body: str):
    """Locate the comment start and the character-string spans of one
    physical Fortran line.  Shared by every text rewrite so a ``!`` or
    ``**`` inside a character literal is never treated as code.

    :param body: the line without its newline.
    :returns: ``(comment_index, [(start, end), ...])`` -- ``comment_index``
        is ``len(body)`` when the line has no comment; the span list
        covers ``'...'`` / ``"..."`` literals (Fortran ``''`` / ``""``
        doubling stays inside one span).
    """
    spans, i, n = [], 0, len(body)
    while i < n:
        c = body[i]
        if c in "'\"":
            j = i + 1
            while j < n:
                if body[j] == c:
                    if j + 1 < n and body[j + 1] == c:
                        j += 2  # doubled quote -> escaped, stay in string
                        continue
                    break
                j += 1
            spans.append((i, min(j + 1, n)))
            i = j + 1
        elif c == "!":
            return i, spans
        else:
            i += 1
    return n, spans


def _collect_integer_scalar_names(source: str) -> set[str]:
    """Return the set of INTEGER scalar identifiers declared in
    ``source``.  Skip array declarations -- those can't be the bare
    operand of an ``IF`` anyway.  All names are lowercased for
    case-insensitive matching.

    :param source: full Fortran source text.
    :returns: lowercased INTEGER scalar names.
    """
    names: set[str] = set()
    for m in _INTEGER_DECL_RE.finditer(source):
        decl = m.group(1).split('!', 1)[0]
        for tok in decl.split(','):
            head = tok.strip().split('=', 1)[0].strip()
            # Skip array forms ("name(...)") and assumed-shape (":") --
            # an array can't be the bare argument of IF.
            if '(' in head:
                continue
            name = head.split()[0] if head else ''
            if name and name.replace('_', '').isalnum() and not name[0].isdigit():
                names.add(name.lower())
    return names


def _extract_power_base(code: str, star: int):
    """Find the base (left primary) of a ``**`` operator.

    Scans leftward from the ``**`` over a Fortran *primary*: a
    parenthesised group, an identifier, an array/function reference
    (``name(...)``) and ``%`` component chains (``a%b(i)%c``).

    :param code: the comment-stripped source line.
    :param star: index of the first ``*`` of the ``**`` token.
    :returns: ``(begin, end)`` slice of the base in ``code``, or
        ``None`` when no base is found or parens are unbalanced.
    """
    i = star
    while i > 0 and code[i - 1] in " \t":
        i -= 1
    end = i
    while True:
        if i > 0 and code[i - 1] == ")":
            depth, k = 0, i
            while k > 0:
                k -= 1
                if code[k] == ")":
                    depth += 1
                elif code[k] == "(":
                    depth -= 1
                    if depth == 0:
                        break
            if depth != 0:
                return None  # unbalanced -- refuse to rewrite
            i = k  # now at the matching '('
            while i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
                i -= 1  # consume the array/function name, if any
        elif i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
            while i > 0 and (code[i - 1].isalnum() or code[i - 1] == "_"):
                i -= 1
        else:
            break
        if i > 0 and code[i - 1] == "%":
            i -= 1  # designator chain -- keep walking the components
            continue
        break
    return None if i == end else (i, end)


def _real_exp_int_value(tok: str):
    """Integer value of a REAL-literal exponent token, or ``None``.

    :param tok: e.g. ``2.0``, ``3.0_JPRB``, ``2.0D0``, ``2.5``.
    :returns: the int ``n`` when ``tok`` is a whole number >= 1
        (``2.0`` -> 2), else ``None`` (``2.5`` / ``0.0``).
    """
    mant = re.sub(r"(_[A-Za-z]\w*|_\d+)$", "", tok)
    try:
        val = float(mant.replace("d", "e").replace("D", "e"))
    except ValueError:
        return None
    return int(val) if val >= 1 and val == int(val) else None


def rewrite_integer_powers(source: str) -> str:
    """Expand integer-valued REAL-literal powers to repeated multiply:
    ``base**2.0`` -> ``(base*base)``, ``base**3.0_JPRB`` ->
    ``(base*base*base)``.

    Only one outer pair of parentheses is added -- the minimal change
    that keeps the diff close to the source.  ``_extract_power_base``
    always returns a Fortran *primary* (identifier, ``a%b(i)`` chain,
    array/function reference, or an already-parenthesised group), so
    each copied factor is safe to juxtapose with ``*`` without its own
    wrapping.  The single outer pair preserves precedence in every
    surrounding context (``2.0*x**2.0`` -> ``2.0*(x*x)``, ``a/b**2.0``
    -> ``a/(b*b)``, ``-x**2.0`` -> ``-(x*x)``, ``(p-q)**3.0`` ->
    ``((p-q)*(p-q)*(p-q))``).  Only a whole-number REAL exponent is
    matched: bare-integer ``x**2`` is left for flang's (correct)
    integer-power lowering, and genuine fractional powers (``**0.5``,
    ``**0.333``) are never altered.  A base containing a function /
    array reference (``f(x)``, ``arr(i,j)``, ``a%b(i)%c``) is also
    left alone -- duplicating it would call twice (impure functions /
    shared inlined accumulators).  Comments and overlapping (stacked
    ``a**2.0**2.0``) matches are skipped.

    Idempotent: the output contains no ``**<real>`` left to match, so
    a second pass returns its input unchanged.

    :param source: full Fortran source text.
    :returns: source with integer-valued REAL powers expanded.
    """
    out = []
    for line in source.splitlines(keepends=True):
        nl = line[len(line.rstrip("\r\n")):]
        body = line[:len(line) - len(nl)]
        cut, strings = _scan_line(body)  # string-aware, shared
        code, tail = body[:cut], body[cut:]
        edits = []
        for m in _INT_POW_RE.finditer(code):
            if any(s <= m.start() < e for s, e in strings):
                continue  # ``**`` inside a character literal
            n = _real_exp_int_value(m.group(1) or m.group(2))
            if n is None:
                continue
            span = _extract_power_base(code, m.start())
            if span is None:
                continue
            begin, base_end = span
            if edits and begin < edits[-1][1]:
                continue  # overlaps a stacked power -- leave both
            base = code[begin:base_end]
            if _CALL_IN_BASE.search(base):
                # Base contains a function / array reference
                # (``f(x)``, ``arr(i,j)``, ``a%b(i)%c``).  Duplicating
                # it would invoke the call twice -- unsafe for impure
                # functions, and the bridge's call-inlining shares the
                # callee's accumulator across the copies (observed:
                # ``custom_sum(d)**2.0`` -> 2500 instead of 625).  Leave
                # such powers for flang's own lowering.
                continue
            repl = "(" + "*".join(base for _ in range(n)) + ")"
            edits.append((begin, m.end(), repl))
        for begin, fin, repl in reversed(edits):  # right-to-left: stable idx
            code = code[:begin] + repl + code[fin:]
        out.append(code + tail + nl)
    return "".join(out)


def _promote_one(m: re.Match):
    """Rewrite a single real-literal match to a double-precision form.

    :param m: a ``_REAL_LIT_RE`` match.
    :returns: the double literal text, or the original match when it is
        an integer or already double precision.
    """
    mant, expo, kind = m.group(1), m.group(2) or "", m.group(3) or ""
    if "." not in mant and not expo:
        return m.group(0)  # bare integer -- not a real literal
    if expo[:1] in ("d", "D"):
        return m.group(0)  # already double via D-exponent
    if kind and kind[1:].lower() in _DOUBLE_KINDS:
        return m.group(0)  # already double via kind suffix
    if expo:
        return f"{mant}D{expo[1:]}"  # E-exponent -> D-exponent
    return f"{mant}D0"  # bare/single -> append D0


def promote_real_literals_to_double(source: str) -> str:
    """Rewrite every single-precision / default REAL literal to an
    explicit double-precision form (``2.0`` -> ``2.0D0``, ``0.85E5`` ->
    ``0.85D5``, ``1.0_JPRM`` -> ``1.0D0``).

    Literals already double -- a ``D`` exponent or a double kind suffix
    (``_JPRB``, ``_8``, ...) -- and integer literals are left untouched.
    Comments and character strings are never modified.

    Idempotent: a promoted literal carries a ``D`` exponent, which the
    classifier treats as already-double on a second pass.

    :param source: full Fortran source text.
    :returns: source with single/default REAL literals doubled.
    """
    out = []
    for line in source.splitlines(keepends=True):
        nl = line[len(line.rstrip("\r\n")):]
        body = line[:len(line) - len(nl)]
        cut, strings = _scan_line(body)
        code, tail = body[:cut], body[cut:]

        def _repl(m: re.Match) -> str:
            if any(s <= m.start() < e for s, e in strings):
                return m.group(0)  # inside a character string
            return _promote_one(m)

        out.append(_REAL_LIT_RE.sub(_repl, code) + tail + nl)
    return "".join(out)


def preprocess_fortran(source: str) -> str:
    """Rewrite ``IF (intvar)`` to ``IF (intvar /= 0)`` for any INTEGER
    scalar declared in ``source``.

    Idempotent: a second invocation finds no bare-identifier IF guards
    left to rewrite and returns the input unchanged.

    :param source: full Fortran source text.
    :returns: source with bare-INTEGER IF guards made LOGICAL.
    """
    int_names = _collect_integer_scalar_names(source)
    if not int_names:
        return source

    def _rewrite(m: re.Match) -> str:
        ident = m.group(2)
        if ident.lower() in int_names:
            return f"{m.group(1)}{ident} /= 0{m.group(3)}"
        return m.group(0)

    return _BARE_IF_RE.sub(_rewrite, source)


# Intrinsic / compiler-provided modules: never resolved or merged --
# flang supplies them itself, so a ``USE`` of one is left untouched.
_INTRINSIC_MODULES = frozenset({
    "iso_c_binding",
    "iso_fortran_env",
    "ieee_arithmetic",
    "ieee_exceptions",
    "ieee_features",
    "omp_lib",
    "omp_lib_kinds",
    "openacc",
    "mpi",
    "mpi_f08",
})

# ``use [, intrinsic] [::] <name>`` -- captured from the code part of a
# line only (``_scan_line`` strips comments / character literals first).
_USE_RE = re.compile(r"^\s*use\b\s*(?:,\s*intrinsic\s*)?(?:::)?\s*([A-Za-z]\w*)", re.IGNORECASE)
# ``module <name>`` opening a module definition -- excludes
# ``module procedure`` / ``module subroutine`` / ``module function``
# and the ``submodule (...)`` form.
_MODULE_OPEN_RE = re.compile(r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)([A-Za-z]\w*)\s*$", re.IGNORECASE)
_MODULE_END_RE = re.compile(r"^\s*end\s*module\b", re.IGNORECASE)


def _code_of(line: str) -> str:
    """Return the code portion of one physical line with character
    literals blanked, so keyword scans never trip on a ``!`` / module
    name inside a string or comment.

    :param line: one physical Fortran line (no newline).
    :returns: the pre-comment text with ``'...'`` / ``"..."`` spans
        replaced by spaces.
    """
    cut, strings = _scan_line(line)
    code = list(line[:cut])
    for s, e in strings:
        for i in range(s, min(e, cut)):
            code[i] = " "
    return "".join(code)


def _module_blocks(text: str):
    """Yield ``(name_lower, block_text)`` for every top-level ``module``
    definition in ``text`` (modules do not nest; ``submodule`` and
    ``module procedure`` are not matched).

    :param text: Fortran source.
    :returns: generator of ``(lowercase module name, verbatim block)``.
    """
    lines = text.splitlines(keepends=True)
    i, n = 0, len(lines)
    while i < n:
        m = _MODULE_OPEN_RE.match(_code_of(lines[i].rstrip("\r\n")))
        if not m:
            i += 1
            continue
        start, name = i, m.group(1).lower()
        i += 1
        while i < n and not _MODULE_END_RE.match(_code_of(lines[i].rstrip("\r\n"))):
            i += 1
        end = min(i, n - 1)
        yield name, "".join(lines[start:end + 1])
        i = end + 1


def _used_modules(text: str) -> list:
    """Ordered, de-duplicated lowercase names of modules ``USE``-d in
    ``text`` (intrinsic modules excluded).

    :param text: Fortran source.
    :returns: list of module names in first-appearance order.
    """
    seen, out = set(), []
    for raw in text.splitlines():
        m = _USE_RE.match(_code_of(raw))
        if not m:
            continue
        nm = m.group(1).lower()
        if nm in _INTRINSIC_MODULES or nm in seen:
            continue
        seen.add(nm)
        out.append(nm)
    return out


def merge_used_modules(source: str, *, search_dirs=()) -> str:
    """Inline every ``USE``-d module's real source into ``source``,
    producing one self-contained translation unit.

    A minimal, fparser-free port of the f2dace single-TU concept: scan
    ``search_dirs`` for module definitions, resolve the ``USE`` graph
    transitively from ``source``, and prepend each needed module's
    verbatim block in dependency order (deps first), de-duplicated.

    Pass-through (returns ``source`` unchanged) when nothing external is
    resolvable -- a self-contained single-file input, every ``USE``
    being intrinsic or defined in ``source`` itself.  This makes the
    pass safe to run by default: only genuine multi-file projects are
    transformed.  Idempotent: re-running finds the modules already
    inlined and adds nothing.

    :param source: the entry Fortran source text.
    :param search_dirs: directories scanned (recursively, ``*.f90`` /
        ``*.F90`` / ``*.incf``) for module definitions.
    :returns: a single-TU source, or ``source`` unchanged.
    """
    from pathlib import Path

    in_source = {nm for nm, _ in _module_blocks(source)}
    index: dict = {}
    for d in search_dirs:
        d = Path(d)
        files = [
            d
        ] if d.is_file() else sorted(list(d.rglob("*.f90")) + list(d.rglob("*.F90")) + list(d.rglob("*.incf")))
        for f in files:
            try:
                txt = f.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for nm, blk in _module_blocks(txt):
                if nm not in in_source:
                    index.setdefault(nm, blk)

    order: list = []
    placed = set(in_source)
    stack = [(nm, False) for nm in reversed(_used_modules(source))]
    while stack:
        nm, expanded = stack.pop()
        if nm in placed or nm not in index:
            continue
        if expanded:
            placed.add(nm)
            order.append(index[nm])
            continue
        stack.append((nm, True))
        for dep in reversed(_used_modules(index[nm])):
            if dep not in placed and dep in index:
                stack.append((dep, False))

    if not order:
        return source
    return "".join(order) + "\n" + source
