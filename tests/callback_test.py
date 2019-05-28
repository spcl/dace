import dace
import numpy


# Performs square
def notypes():
    print("hello from callback")


def mysquarer(inp):
    return inp ** 2


def answertolifeuniverseandeverything():
    return 42


def consumer(inp):
    print(inp)


@dace.program(
    dace.uint32[2],
    dace.uint32[2],
    dace.callback(dace.uint32, dace.uint32),
    dace.callback(None, dace.uint32),
    dace.callback(dace.uint32, None),
    dace.callback(None)
)
def callback_test(A, B, giveandtake, take, give, donothing):
    @dace.map(_[0:2])
    def index(i):
        a << A[i]
        b >> B[i]
        b = giveandtake(a)
        take(a + 1)
        if give() != 42:
            printf("The answer to life, universe and everything is not 42!?")
        donothing()


if __name__ == "__main__":

    A = dace.ndarray((2,), dtype=dace.int32)
    B = dace.ndarray((2,), dtype=dace.int32)

    A[:] = 5
    B[:] = 0

    print(A)
    callback_test(A, B, mysquarer, consumer, answertolifeuniverseandeverything, notypes)
    print(B)
