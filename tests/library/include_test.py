import dace.library

def assert_exists(name):
    dace.library.get_library(name)

def assert_not_exists(name):
    raised = False
    try:
        dace.library.get_library(name)
    except:
        raised = True
        pass
    if not raised:
        raise RuntimeError("Library " + name + " exists.")

assert_not_exists("FooLib")
assert_not_exists("BarLib")

import foolib  # foolib depends on barlib

assert_exists("FooLib")
assert_exists("BarLib")
