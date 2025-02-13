# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest
from aenum import Enum, auto
from dace import registry


@registry.make_registry
class ExtensibleClass(object):
    pass


class Extension(ExtensibleClass):
    pass


@registry.extensible_enum
class ExtensibleEnumeration(Enum):
    a = auto()
    b = auto()


class RegistryTests(unittest.TestCase):
    def test_class_registry(self):
        ExtensibleClass.register(Extension)
        self.assertTrue(Extension in ExtensibleClass.extensions())
        ExtensibleClass.unregister(Extension)
        self.assertTrue(Extension not in ExtensibleClass.extensions())

    def test_autoregister(self):
        @registry.autoregister
        class Extension2(ExtensibleClass):
            pass

        self.assertTrue(Extension2 in ExtensibleClass.extensions())

    def test_class_registry_args(self):
        ExtensibleClass.register(Extension, a=True, b=1, c=2)
        self.assertTrue(Extension in ExtensibleClass.extensions())
        self.assertEqual(ExtensibleClass.extensions()[Extension], dict(a=True, b=1, c=2))
        ExtensibleClass.unregister(Extension)
        self.assertTrue(Extension not in ExtensibleClass.extensions())

    def test_autoregister_args(self):
        @registry.autoregister_params(a=False, b=0)
        class Extension3(ExtensibleClass):
            pass

        self.assertTrue(Extension3 in ExtensibleClass.extensions())
        self.assertEqual(ExtensibleClass.extensions()[Extension3], dict(a=False, b=0))

    def test_autoregister_fail(self):
        with self.assertRaises(TypeError):

            @registry.autoregister
            class Extension4(object):
                pass

    def test_enum_registry(self):
        ExtensibleEnumeration.register('c')
        self.assertTrue(ExtensibleEnumeration.c in ExtensibleEnumeration)
        self.assertEqual(ExtensibleEnumeration.c.value, 3)

    def test_enum_registry_fail(self):
        with self.assertRaises(TypeError):

            @registry.extensible_enum
            class NotAnEnum(object):
                pass


if __name__ == '__main__':
    unittest.main()
