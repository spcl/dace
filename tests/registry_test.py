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

    def test_autoregister_fail(self):
        with self.assertRaises(TypeError):

            @registry.autoregister
            class Extension3(object):
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
