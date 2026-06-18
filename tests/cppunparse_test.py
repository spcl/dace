# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.codegen import cppunparse


def _test_py2cpp(func, expected_string):
    result = cppunparse.py2cpp(func)
    if result != expected_string:
        print("ERROR in py2cpp, expected:\n%s\n\ngot:\n%s\n" % (expected_string, result))
        return False
    return True


def _test_pyexpr2cpp(func, expected_string):
    result = cppunparse.pyexpr2cpp(func)
    if result != expected_string:
        print("ERROR in pyexpr2cpp, expected:\n%s\n\ngot:\n%s\n" % (expected_string, result))
        return False
    return True


def gfunc(woo):
    i = 0
    result = 0
    while i < woo and i > 0:
        for j in range(i):
            result += (2 // 1)**j
    return result


def test():
    print('cppunparse unit test')
    success = True

    success &= _test_py2cpp(
        """def notype(a, b):
    a = a + 5
    c = a + b
    return c*b
""", """auto notype(auto a, auto b) {
    a = (a + 5);
    auto c = (a + b);
    return (c * b);
}""")

    success &= _test_py2cpp("""def typed(a: int, b: float) -> float:
    c = a + b
    return c*b
""", """float typed(int a, float b) {
    auto c = (a + b);
    return (c * b);
}""")

    # Ternary operators, strings
    success &= _test_py2cpp("""printf('%f\\n', a if b else c);""", """printf("%f\\n", (b ? a : c));""")

    # Global functions, operators
    success &= _test_py2cpp(
        gfunc, """auto gfunc(auto woo) {
    auto i = 0;
    auto result = 0;
    while (((i < woo) && (i > 0))) {
        for (auto j : range(i)) {
            result += dace::math::pow(dace::math::ifloor(2 / 1), j);
        }
    }
    return result;
}""")

    def lfunc():
        exit(1 >> 3)

    # Local functions
    success &= _test_py2cpp(lfunc, """auto lfunc() {
    exit((1 >> 3));
}""")

    # void return value
    success &= _test_py2cpp("""
def lfunc() -> None:
    exit(1 >> 3)
""", """void lfunc() {
    exit((1 >> 3));
}""")

    # Local variable tracking
    success &= _test_py2cpp('l = 1 + a; l = l + 8;', """auto l = (1 + a);
l = (l + 8);""")

    # Operations (augmented assignment)
    success &= _test_py2cpp('l *= 3; l //= 8', """l *= 3;
l = dace::math::ifloor(l / 8);""")

    success &= _test_pyexpr2cpp('a << 3', '(a << 3)')

    # Array assignment
    success &= _test_py2cpp('A[i] = b[j]', """A[i] = b[j];""")

    # Named constants
    success &= _test_py2cpp('''if x is not None:
    y = True if x else False
    ''', '''if ((x != nullptr)) {
    auto y = (x ? true : false);
}''')

    print('Result: %s' % ('PASSED' if success else 'FAILED'))
    assert success


def test_annotated_definition():
    success = _test_py2cpp('''a: dace.float32
if something:
    a = 5
    ''', '''dace::float32 a;
if (something) {
    a = 5;
}''')
    assert success


def test_typecast_function_namespacing():
    """A bare DaCe typeclass cast (e.g. ``float64(x)`` emitted by sympy
    lowering or the Fortran bridge) must be namespaced to its ``dace::``
    typedef so it resolves in generated C++, the same as ``dace.float64(x)``.
    The plain C++ builtins (``int`` / ``float`` / ``bool``) stay bare."""
    success = True
    # Sized typeclasses get the ``dace::`` prefix.
    success &= _test_pyexpr2cpp('float64(x)', 'dace::float64(x)')
    success &= _test_pyexpr2cpp('int32(a + b)', 'dace::int32((a + b))')
    success &= _test_pyexpr2cpp('sqrt(float64(i + k))', 'sqrt(dace::float64((i + k)))')
    success &= _test_pyexpr2cpp('uint64(n)', 'dace::uint64(n)')
    # The explicit attribute form is unchanged, and now matches the bare form.
    success &= _test_pyexpr2cpp('dace.float64(x)', 'dace::float64(x)')
    # Plain C++ builtin casts must NOT be namespaced (there is no dace::int).
    success &= _test_pyexpr2cpp('int(x)', 'int(x)')
    success &= _test_pyexpr2cpp('float(x)', 'float(x)')
    assert success


if __name__ == "__main__":
    test()
    test_annotated_definition()
    test_typecast_function_namespacing()
