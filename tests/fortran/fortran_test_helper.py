import re
import subprocess
from dataclasses import dataclass, field
from os import path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Type, Union, List, Sequence, Collection, Iterable

import numpy as np
from fparser.two.Fortran2003 import Name

from dace.frontend.fortran.ast_internal_classes import Name_Node


@dataclass
class SourceCodeBuilder:
    """
    A helper class that helps to construct the source code structure for frontend tests.

    Example usage:
    ```python
    # Construct the builder, add files in the order you'd pass them to `gfortran`, (optional step) check if they all
    # compile together, then get a dictionary mapping file names (possibly auto-inferred) to their content.
    sources, main = SourceCodeBuilder().add_file('''
    module lib
    end module lib
    ''').add_file('''
    program main
    use lib
    implicit none
    end program main
    ''').check_with_gfortran().get()
    # Then construct the SDFG.
    sdfg = create_sdfg_from_string(main, "main", sources=sources)
    ```
    """
    sources: Dict[str, str] = field(default_factory=dict)

    def add_file(self, content: str, name: Optional[str] = None):
        """Add source file contents in the order you'd pass them to `gfortran`."""
        if not name:
            name = SourceCodeBuilder._identify_name(content)
        name, ext = name.rsplit('.', 1) if '.' in name else (name, 'f90')
        key, COUNTER = f"{name}.{ext}", 0
        while key in self.sources:
            key, COUNTER = f"{name}_{COUNTER}.{ext}", COUNTER + 1
        self.sources[key] = content
        return self

    def check_with_gfortran(self):
        """Assert that it all compiles with `gfortran`."""
        with TemporaryDirectory() as td:
            # Create temporary Fortran source-file structure.
            for fname, content in self.sources.items():
                with open(path.join(td, fname), 'w') as f:
                    f.write(content)
            # Run `gfortran -Wall` to verify that it compiles.
            # Note: we're relying on the fact that python dictionaries keeps the insertion order when calling `keys()`.
            cmd = ['gfortran', '-Wall', '-shared', '-fPIC', '-ffree-line-length-none', *self.sources.keys()]

            try:
                subprocess.run(cmd, cwd=td, capture_output=True).check_returncode()
                return self
            except subprocess.CalledProcessError as e:
                print("Fortran compilation failed!")
                print(e.stderr.decode())
                raise e

    def run_with_gfortran(self) -> str:
        """Compile and execute the program with `gfortran`."""
        with TemporaryDirectory() as td:
            # Create temporary Fortran source-file structure.
            for fname, content in self.sources.items():
                with open(path.join(td, fname), 'w') as f:
                    f.write(content)
            # Note: we're relying on the fact that python dictionaries keeps the insertion order when calling `keys()`.
            cmd = ['gfortran', '-Wall', '-fPIC', *self.sources.keys()]

            try:
                subprocess.run(cmd, cwd=td, capture_output=True).check_returncode()
                result = subprocess.run(path.join(td, 'a.out'), cwd=td, capture_output=True)
                result.check_returncode()
                return result.stdout.decode()
            except subprocess.CalledProcessError as e:
                print("Fortran running failed!")
                print(e.stderr.decode())
                raise e

    def run_with_gcc(self, extra_build_args: Optional[Iterable[str]] = None) -> str:
        """Compile and execute the program with `g++`."""
        extra_build_args = extra_build_args or tuple()
        with TemporaryDirectory() as td:
            # Create temporary Fortran source-file structure.
            for fname, content in self.sources.items():
                with open(path.join(td, fname), 'w') as f:
                    f.write(content)
            # Note: we're relying on the fact that python dictionaries keeps the insertion order when calling `keys()`.
            cmd = ['g++', '-Wall', '-fPIC', '-std=c++17', *extra_build_args, *self.sources.keys()]

            try:
                subprocess.run(cmd, cwd=td, capture_output=True).check_returncode()
                result = subprocess.run(path.join(td, 'a.out'), cwd=td, capture_output=True)
                result.check_returncode()
                return result.stdout.decode()
            except subprocess.CalledProcessError as e:
                print("C++ running failed!")
                print(e.stderr.decode())
                raise e

    def get(self) -> Tuple[Dict[str, str], Optional[str]]:
        """Get a dictionary mapping file names (possibly auto-inferred) to their content."""
        main = None
        if 'main.f90' in self.sources:
            main = self.sources['main.f90']
        return self.sources, main

    @staticmethod
    def _identify_name(content: str) -> str:
        PPAT = re.compile("^.*\\bprogram\\b\\s*\\b(?P<prog>[a-zA-Z0-9_]*)\\b.*$", re.I | re.M | re.S)
        if PPAT.match(content):
            match = PPAT.search(content)
            return match.group('prog') or 'main'
        MPAT = re.compile("^.*\\bmodule\\b\\s*\\b(?P<mod>[a-zA-Z0-9_]+)\\b.*$", re.I | re.M | re.S)
        if MPAT.match(content):
            match = MPAT.search(content)
            return match.group('mod')
        FPAT = re.compile("^.*\\bfunction\\b\\s*\\b(?P<fn>[a-zA-Z0-9_]+)\\b.*$", re.I | re.M | re.S)
        if FPAT.match(content):
            match = FPAT.search(content)
            return match.group('fn')
        SPAT = re.compile("^.*\\bsubroutine\\b\\s*\\b(?P<subr>[a-zA-Z0-9_]+)\\b.*$", re.I | re.M | re.S)
        if SPAT.match(content):
            match = SPAT.search(content)
            return match.group('subr')
        raise ValueError(f"Could not find any identifiable object in the content:\n{content}")


class FortranASTMatcher:
    """
    A "matcher" class that asserts if a given `node` has the right type, and its children, attributes etc. also matches
    the submatchers.

    Example usage:
    ```python
    # Construct a matcher that looks for specific patterns in the AST structure, while ignoring unnecessary details.
    m = M(Program, [
        M(Main_Program, [
            M.IGNORE(),  # program main
            M(Specification_Part),  # implicit none; double precision d(4)
            M(Execution_Part, [M(Call_Stmt)]),  # call fun(d)
            M.IGNORE(),  # end program main
        ]),
        M(Subroutine_Subprogram, [
            M(Subroutine_Stmt),  # subroutine fun(d)
            M(Specification_Part, [
                M(Implicit_Part),  # implicit none
                M(Type_Declaration_Stmt),  # double precision d(4)
            ]),
            M(Execution_Part, [M(Assignment_Stmt)]),  # d(2) = 5.5
            M(End_Subroutine_Stmt),  # end subroutine fun
        ]),
    ])
    # Check that a given Fortran AST matches that pattern.
    m.check(ast)
    ```
    """

    def __init__(self,
                 is_type: Union[None, Type, str] = None,
                 has_children: Union[None, list] = None,
                 has_attr: Optional[Dict[str, Union["FortranASTMatcher", List["FortranASTMatcher"]]]] = None,
                 has_value: Optional[str] = None):
        # TODO: Include Set[Self] to `has_children` type?
        assert not ((set() if has_attr is None else has_attr.keys()) & {'children'})
        self.is_type = is_type
        self.has_children = has_children
        self.has_attr = has_attr
        self.has_value = has_value

    def check(self, node):
        if self.is_type is not None:
            if isinstance(self.is_type, type):
                assert isinstance(node, self.is_type), \
                    f"type mismatch at {node}; want: {self.is_type}, got: {type(node)}"
            elif isinstance(self.is_type, str):
                assert node.__class__.__name__ == self.is_type, \
                    f"type mismatch at {node}; want: {self.is_type}, got: {type(node)}"
        if self.has_value is not None:
            assert node == self.has_value
        if self.has_children is not None and len(self.has_children) > 0:
            assert hasattr(node, 'children')
            all_children = getattr(node, 'children')
            assert len(self.has_children) == len(all_children), \
                f"#children mismatch at {node}; want: {len(self.has_children)}, got: {len(all_children)}"
            for (c, m) in zip(all_children, self.has_children):
                m.check(c)
        if self.has_attr is not None and len(self.has_attr.keys()) > 0:
            for key, subm in self.has_attr.items():
                assert hasattr(node, key)
                attr = getattr(node, key)

                if isinstance(subm, Sequence):
                    assert isinstance(attr, Sequence)
                    assert len(attr) == len(subm)
                    for (c, m) in zip(attr, subm):
                        m.check(c)
                else:
                    subm.check(attr)

    @classmethod
    def IGNORE(cls, times: Optional[int] = None) -> Union["FortranASTMatcher", List["FortranASTMatcher"]]:
        """
        A placeholder matcher to not check further down the tree.
        If `times` is `None` (which is the default), returns a single matcher.
        If `times` is an integer value, then returns a list of `IGNORE()` matchers of that size, indicating that many
        nodes on a row should be ignored.
        """
        if times is None:
            return cls()
        else:
            return [cls()] * times

    @classmethod
    def NAMED(cls, name: str):
        return cls(Name, has_attr={'string': cls(has_value=name)})


class InternalASTMatcher:
    """
    A "matcher" class that asserts if a given `node` has the right type, and its children, attributes etc. also matches
    the submatchers.

    Example usage:
    ```python
    # Construct a matcher that looks for specific patterns in the AST structure, while ignoring unnecessary details.
    m = M(Program_Node, {
        'main_program': M(Main_Program_Node, {
            'name': M(Program_Stmt_Node),
            'specification_part': M(Specification_Part_Node, {
                'specifications': [
                    M(Decl_Stmt_Node, {
                        'vardecl': [M(Var_Decl_Node)],
                    })
                ],
            }, {'interface_blocks', 'symbols', 'typedecls', 'uses'}),
            'execution_part': M(Execution_Part_Node, {
                'execution': [
                    M(Call_Expr_Node, {
                        'name': M(Name_Node),
                        'args': [M(Name_Node, {
                            'name': M(has_value='d'),
                            'type': M(has_value='DOUBLE'),
                        })],
                        'type': M(has_value='VOID'),
                    })
                ],
            }),
        }, {'parent'}),
        'structures': M(Structures, None, {'structures'}),
    }, {'function_definitions', 'module_declarations', 'modules'})
    # Check that a given internal AST matches that pattern.
    m.check(prog)
    ```
    """

    def __init__(self,
                 is_type: Optional[Type] = None,
                 has_attr: Optional[Dict[str, Union["InternalASTMatcher", List["InternalASTMatcher"],
                                                    Dict[str, "InternalASTMatcher"]]]] = None,
                 has_empty_attr: Optional[Collection[str]] = None,
                 has_value: Optional[str] = None):
        # TODO: Include Set[Self] to `has_children` type?
        assert not ((set() if has_attr is None else has_attr.keys())
                    & (set() if has_empty_attr is None else has_empty_attr))
        self.is_type: Type = is_type
        self.has_attr = has_attr
        self.has_empty_attr = has_empty_attr
        self.has_value = has_value

    def check(self, node):
        if self.is_type is not None:
            assert isinstance(node, self.is_type)
        if self.has_value is not None:
            assert node == self.has_value, f"{node} is not the same as {self.has_value}"
        if self.has_empty_attr is not None:
            for key in self.has_empty_attr:
                assert not hasattr(node, key) or not getattr(node, key), f"{node} is expected to not have key: {key}"
        if self.has_attr is not None and len(self.has_attr.keys()) > 0:
            for key, subm in self.has_attr.items():
                assert hasattr(node, key), f"{node} doesn't have key: {key}"
                attr = getattr(node, key)

                if isinstance(subm, Sequence):
                    assert isinstance(attr, Sequence), f"{attr} must be a sequence, since {subm} is."
                    assert len(attr) == len(subm), f"{attr} must have the same length as {subm}."
                    for (c, m) in zip(attr, subm):
                        m.check(c)
                elif isinstance(subm, Dict):
                    assert isinstance(attr, Dict)
                    assert len(attr) == len(subm)
                    assert subm.keys() <= attr.keys()
                    for k in subm.keys():
                        subm[k].check(attr[k])
                else:
                    subm.check(attr)

    @classmethod
    def IGNORE(cls, times: Optional[int] = None) -> Union["InternalASTMatcher", List["InternalASTMatcher"]]:
        """
        A placeholder matcher to not check further down the tree.
        If `times` is `None` (which is the default), returns a single matcher.
        If `times` is an integer value, then returns a list of `IGNORE()` matchers of that size, indicating that many
        nodes on a row should be ignored.
        """
        if times is None:
            return cls()
        else:
            return [cls()] * times

    @classmethod
    def NAMED(cls, name: str):
        return cls(Name_Node, {'name': cls(has_value=name)})


def deduce_f2dace_variables_for_array(arr: np.ndarray, arg: str,
                                      start_counting_from: int) -> Dict[str, Union[np.bool_, np.int32]]:
    """This needs to be kept in sync with the naming convention of "f2dace" variables."""
    out: Dict[str, Union[np.bool_, np.int32]] = {
        f"__f2dace_ALLOCATED_{arg}": np.bool_(arr.size > 0),
    }

    for idx, dim in enumerate(arr.shape):
        out[f"__f2dace_A_{arg}_d_{idx}_s_{start_counting_from}"] = np.int32(dim)
        # TODO: Do we ever pass non-1 offset here?
        out[f"__f2dace_OA_{arg}_d_{idx}_s_{start_counting_from}"] = np.int32(1)
        start_counting_from += 1

    return out
