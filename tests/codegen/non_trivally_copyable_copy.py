import dace
import numpy

struct_str = """
class interesting_type {
private:
    float *f = nullptr;

public:
    interesting_type() {
        f = new float[10];
    }

    interesting_type(const interesting_type& other) {
        f = new float[10];
        for (int i = 0; i < 10; ++i){
            f[i] = other.f[i];
        }
    }

    interesting_type& operator=(const interesting_type& other) {
        if (this != &other) {
            for (int i = 0; i < 10; ++i){
                f[i] = other.f[i];
            }
        }
        return *this;
    }

    interesting_type(interesting_type&& other) noexcept {
        f = other.f;
        other.f = nullptr;
    }

    interesting_type& operator=(interesting_type&& other) noexcept {
        if (this != &other) {
            delete[] f;
            f = other.f;
            other.f = nullptr;
        }
        return *this;
    }

    interesting_type(const double value){
        f = new float[10];
        for (int i = 0; i < 10; ++i){
            f[i] = static_cast<float>(i + value);
        }
    }

    ~interesting_type() {
        delete[] f;
    }

    operator double() const {
        return f[1];
    }
};
"""


def _gen_sdfg():
    sdfg = dace.SDFG("non_trivial_copy_test")
    s1 = sdfg.add_state("main")

    interesting_type = dace.opaque("interesting_type")

    sdfg.add_array(
        "A",
        dtype=interesting_type,
        shape=[
            10,
        ],
        strides=[
            1,
        ],
        transient=False,
    )
    sdfg.add_array(
        "B",
        dtype=interesting_type,
        shape=[
            10,
        ],
        strides=[
            1,
        ],
        transient=False,
    )

    a = s1.add_access("A")
    b = s1.add_access("B")

    s1.add_edge(a, None, b, None, dace.memlet.Memlet.from_array("A", sdfg.arrays["A"]))

    sdfg.append_global_code(struct_str)

    return sdfg


def _gen_sdfg_with_copy_in_and_out():
    sdfg = dace.SDFG("non_trivial_copy_test_execution")
    s1 = sdfg.add_state("s1")
    s0 = sdfg.add_state_before(s1, "s0")
    s2 = sdfg.add_state_after(s1, "s2")

    interesting_type = dace.opaque("interesting_type")

    for name, is_transient, dtype in [("A", False, dace.float64), ("iA", True, interesting_type),
                                      ("B", False, dace.float64), ("iB", True, interesting_type)]:

        sdfg.add_array(
            name,
            dtype=dtype,
            shape=[
                10,
            ],
            strides=[
                1,
            ],
            transient=is_transient,
        )

    ia = s1.add_access("iA")
    ib = s1.add_access("iB")

    s1.add_edge(ia, None, ib, None, dace.memlet.Memlet.from_array("iA", sdfg.arrays["iA"]))

    sdfg.append_global_code(struct_str)

    a = s0.add_access("A")
    ia = s0.add_access("iA")

    s0.add_mapped_tasklet(
        "copy_in_map",
        map_ranges={
            "i": "0:10",
        },
        inputs={"__in": dace.Memlet("A[i]")},
        code="__out = __in;",
        outputs={"__out": dace.Memlet("iA[i]")},
        language=dace.dtypes.Language.CPP,
        external_edges=True,
        output_nodes={a},
        input_nodes={ia},
    )

    ib = s2.add_access("iB")
    b = s2.add_access("B")

    s2.add_mapped_tasklet(
        "copy_in_map",
        map_ranges={
            "i": "0:10",
        },
        inputs={"__in": dace.Memlet("iB[i]")},
        code="__out = __in;",
        outputs={"__out": dace.Memlet("B[i]")},
        language=dace.dtypes.Language.CPP,
        external_edges=True,
        output_nodes={ib},
        input_nodes={b},
    )

    for s in [s0, s1, s2]:
        for n in s.nodes():
            if s.degree(n) == 0:
                s.remove_node(n)

    return sdfg


def test_non_trivial_copy_compilation():
    sdfg = _gen_sdfg()
    sdfg.validate()
    sdfg.compile()


def test_non_trivial_copy_execution():
    sdfg = _gen_sdfg_with_copy_in_and_out()
    sdfg.validate()
    csdfg = sdfg.compile()
    A = numpy.ndarray((10, ), dtype=float)
    B = numpy.ndarray((10, ), dtype=float)
    for i in range(10):
        A[i] = 1
        B[i] = -1
    csdfg(A=A, B=B)
    assert all(v == 2 for v in B)


if __name__ == "__main__":
    test_non_trivial_copy_compilation()
    test_non_trivial_copy_execution()
