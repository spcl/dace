# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from absl import app, flags
import numpy as np
import functools
import dace

flags.DEFINE_bool('simulate', False, 'Use the DaCe python simulator')
flags.DEFINE_bool('specialize', True, 'Compile problem with evaluated ' + '(specialized to constant value) symbols')
flags.DEFINE_bool('sequential', False, 'Automatically change all maps to sequential schedule')
flags.DEFINE_bool('compile', False, 'Only compile without running')
flags.DEFINE_bool('save', False, 'Save results to file')
flags.DEFINE_enum('size', 'large', ['mini', 'small', 'medium', 'large', 'extralarge'], 'Dataset/problem size')
_SIZE_TO_IND = {'mini': 0, 'small': 1, 'medium': 2, 'large': 3, 'extralarge': 4}

FLAGS = flags.FLAGS


def polybench_dump(filename, args, output_args):
    """ Dumps the outputs in a format that matches the Polybench dumper. """
    with open(filename, 'w') as fp:
        fp.write("==BEGIN DUMP_ARRAYS==\n")

        for i, name in output_args:
            fp.write("begin dump: %s\n" % name)
            np.savetxt(fp,
                       args[i].reshape(args[i].shape[0], functools.reduce(lambda a, b: a * b, args[i].shape[1:], 1)),
                       fmt="%0.7lf")
            fp.write("\nend   dump: %s\n" % name)

        fp.write("==END   DUMP_ARRAYS==\n")


def _main(sizes, args, output_args, init_array, func, argv, keywords=None):
    print('Polybench test %s, problem size: %s' % (func.name, FLAGS.size))

    # Initialize symbols with values from dataset size
    psize = sizes[_SIZE_TO_IND[FLAGS.size]]
    for k, v in psize.items():
        k.set(v)
    psize = {str(k): v for k, v in psize.items()}

    # Construct arrays from tuple arguments
    for i, arg in enumerate(args):
        if isinstance(arg, tuple):
            args[i] = dace.ndarray(*arg)

    if FLAGS.simulate == False:
        if isinstance(func, dace.SDFG):
            sdfg = func
        else:
            sdfg = func.to_sdfg(*args)
        if FLAGS.sequential:
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.sdfg.nodes.MapEntry):
                        node.map.schedule = dace.ScheduleType.Sequential
        if FLAGS.specialize:
            sdfg.specialize(psize)
        compiled_sdfg = sdfg.compile()

    if FLAGS.compile == False:
        print('Initializing arrays...')
        init_array(*args)
        print('Running %skernel...' % ('specialized ' if FLAGS.specialize else ''))

        if FLAGS.simulate:
            dace.simulate(func, *args)
        else:
            if isinstance(func, dace.SDFG):
                compiled_sdfg(**keywords, **psize)
            else:
                compiled_sdfg(**{n: arg for n, arg in zip(func.argnames, args)}, **psize)

        if FLAGS.save:
            if not isinstance(output_args, list):
                output_args(func.name + '.dace.out', *args)
            else:
                polybench_dump(func.name + '.dace.out', args, output_args)

    print('==== Done ====')


def main(sizes, args, outputs, init_array, func, keywords=None):
    # Pass application arguments and command-line arguments through abseil
    app.run(lambda argv: _main(sizes, args, outputs, init_array, func, argv, keywords))
