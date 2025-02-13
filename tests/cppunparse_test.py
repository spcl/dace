# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function
from dace.codegen import cppunparse
import six


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

    if six.PY3:
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
    if six.PY3:
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
    if six.PY3:
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


if __name__ == "__main__":
    test()
    test_annotated_definition()
