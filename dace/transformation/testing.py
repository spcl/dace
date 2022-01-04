# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from io import StringIO
import os
import sys
import traceback

from dace.sdfg import SDFG
from dace.transformation.optimizer import Optimizer


class TransformationTester(Optimizer):
    """ An SDFG optimizer that consecutively applies available transformations
        up to a fixed depth. """
    def __init__(self,
                 sdfg: SDFG,
                 depth=1,
                 validate=True,
                 generate_code=True,
                 compile=False,
                 print_exception=True,
                 halt_on_exception=False):
        """ Creates a new Transformation tester, which brute-forces applying the
            available transformations up to a certain level.
            :param sdfg: The SDFG to transform.
            :param depth: The number of levels to run transformations. For
                          instance, depth=1 means to only run immediate
                          transformations, whereas depth=2 would run
                          transformations resulting from those transformations.
            :param validate: If True, the SDFG is validated after applying.
            :param generate_code: If True, the SDFG will generate code after
                                  transformation.
            :param compile: If True, the SDFG will be compiled after applying.
            :param print_exception: If True, prints exception when it is raised.
            :param halt_on_exception: If True, stops when a transformation
                                      raises an exception.
        """
        super().__init__(sdfg)
        self.depth = depth
        self.validate = validate
        self.generate_code = generate_code
        self.compile = compile
        self.print_exception = print_exception
        self.halt_on_exception = halt_on_exception
        self.passed_tests = 0
        self.failed_tests = 0
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def _optimize_recursive(self, sdfg: SDFG, depth: int):
        if depth == self.depth:
            return

        matches = list(self.get_pattern_matches(sdfg=sdfg))

        # Apply each transformation
        for match in matches:
            # Copy the SDFG
            new_sdfg: SDFG = copy.deepcopy(sdfg)

            # Try to apply, handle any exception
            try:
                # Redirect outputs
                output = StringIO()
                sys.stdout = output
                sys.stderr = output

                print('    ' * depth, type(match).__name__, '- ', end='', file=self.stdout)

                tsdfg: SDFG = new_sdfg.sdfg_list[match.sdfg_id]
                tgraph = tsdfg.node(match.state_id) if match.state_id >= 0 else tsdfg
                match.apply(tgraph, tsdfg)

                sdfg.save(os.path.join('_dacegraphs', 'program.sdfg'))

                # Validate
                if self.validate:
                    new_sdfg.validate()

                # Expand library nodes
                new_sdfg.expand_library_nodes()

                # Generate code
                if self.generate_code:
                    new_sdfg.generate_code()

                if self.compile:
                    compiled = new_sdfg.compile()
                    del compiled

                print('PASS', file=self.stdout)
                self.passed_tests += 1

                # Recursively optimize as necessary
                self._optimize_recursive(sdfg, depth + 1)

            except:  # Literally anything can happen here
                print('FAIL', file=self.stdout)
                self.failed_tests += 1
                if self.halt_on_exception:
                    print(output.getvalue(), file=self.stderr)
                    raise
                if self.print_exception:
                    print(output.getvalue(), file=self.stderr)
                    traceback.print_exc(file=self.stderr)
                continue
            finally:
                # Restore redirected outputs
                sys.stdout = self.stdout
                sys.stderr = self.stderr

    def optimize(self):
        self._optimize_recursive(self.sdfg, 0)

        if self.failed_tests > 0:
            raise RuntimeError('%d / %d transformations passed' %
                               (self.passed_tests, self.passed_tests + self.failed_tests))

        return self.sdfg


if __name__ == '__main__':
    import dace

    @dace.program
    def example(A: dace.float32[2]):
        A *= 2

    sdfg = example.to_sdfg()

    tt = TransformationTester(sdfg, 2, halt_on_exception=True)
    tt.optimize()

    print('SUMMARY: %d / %d tests passed' % (tt.passed_tests, tt.passed_tests + tt.failed_tests))
