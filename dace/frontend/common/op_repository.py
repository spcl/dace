from dace.dtypes import paramdec


class Replacements(object):
    """ A management singleton for functions that replace existing function calls with either an SDFG or a node.
        Used in the Python frontend to replace functions such as `numpy.ndarray` and operators such
        as `Array.__add__`. """

    _rep = {}

    @staticmethod
    def get(name, implementation='sdfg'):
        """ Returns an implementation of a function or an operator. """
        if (name, implementation) not in Replacements._rep:
            return None
        return Replacements._rep[(name, implementation)]


@paramdec
def replaces(func, name, implementation='sdfg'):
    Replacements._rep[(name, implementation)] = func
    return func
