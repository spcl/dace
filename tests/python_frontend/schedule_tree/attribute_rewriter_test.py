# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast

from dace.frontend.python import astutils
from dace.frontend.python.schedule_tree import AttributeRewriter


def _rewrite_expression(source: str, context):
    rewriter = AttributeRewriter(lambda: dict(context))
    expr = ast.parse(source, mode='eval').body
    return astutils.unparse(rewriter.rewrite_expression(expr))


def _rewrite_assignment(source: str, context):
    rewriter = AttributeRewriter(lambda: dict(context))
    assign = ast.parse(source).body[0]
    rewritten = rewriter.rewrite_assignment(assign.targets[0], assign.value)
    return None if rewritten is None else astutils.unparse(rewritten)


def test_attribute_rewriter_rewrites_descriptor_loads_and_stores():

    class ArrayDescriptor:

        def __set_name__(self, owner, name):
            self.name = '_' + name

        def __get__(self, obj, objtype=None):
            return getattr(obj, self.name)

        def __set__(self, obj, value):
            setattr(obj, self.name, value)

    class DescriptorHolder:
        arr = ArrayDescriptor()

        def __init__(self):
            self.arr = None

    descriptor_holder = DescriptorHolder()
    context = {'descriptor_holder': descriptor_holder}

    assert _rewrite_assignment('descriptor_holder.arr = A',
                               context) == ("type(descriptor_holder).__dict__['arr'].__set__(descriptor_holder, A)")
    assert _rewrite_expression(
        'descriptor_holder.arr',
        context) == ("type(descriptor_holder).__dict__['arr'].__get__(descriptor_holder, type(descriptor_holder))")


def test_attribute_rewriter_rewrites_custom_getattribute_and_setattr():

    class Proxy:

        def __getattribute__(self, name):
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    proxy = Proxy()
    context = {'proxy': proxy}

    assert _rewrite_expression('proxy.value', context) == "type(proxy).__getattribute__(proxy, 'value')"
    assert _rewrite_assignment('proxy.value = A', context) == "type(proxy).__setattr__(proxy, 'value', A)"


def test_attribute_rewriter_preserves_plain_attribute_syntax():

    class Holder:

        def __init__(self):
            self.value = None

    holder = Holder()
    context = {'holder': holder}

    assert _rewrite_expression('holder.value', context) == 'holder.value'
    assert _rewrite_assignment('holder.value = A', context) is None
