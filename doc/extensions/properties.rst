.. _properties:

Properties and Serialization
============================

Being an intermediate representation, SDFGs need to be serializable for storage and retaining modifications (such
as optimizations and transformations). The SDFG format is a JSON file, and as such, serialization needs to be able to
store every field in one of the JSON primitive data types.

To support serialization, deserialization, and data validation, we define and use :class:`~dace.properties.Property`
objects in all of the SDFG elements (e.g., memlets, passes, library nodes). Properties behave and function similarly to
annotations in ``dataclass`` objects (in fact, they predate dataclasses and support Python 2, though that is no longer
supported), and also offer *extra features* in addition to dataclass fields:

    * Description and metadata that integrate with the Visual Studio Code extension (descriptions become tooltips,
      enumerations become native combo-boxes, etc.)
    * Custom setters and getters for on-the-fly type conversion and validation
    * Customizable to/from JSON methods
    * Optional properties with conditions based on other properties (saves space when serializing)
    * Integration of Property type and description with automatic documentation
    * ...and more (see :class:`~dace.properties.Property`)

There are many property types in :mod:`dace.properties`, including :class:`~dace.properties.ListProperty`, 
:class:`~dace.properties.CodeProperty` (for code in arbitrary languages), :class:`~dace.properties.RangeProperty` (for
subsets), :class:`~dace.properties.DictProperty`, and even :class:`~dace.properties.DataclassProperty` (that can convert
arbitrary ``dataclass`` objects to properties).

Ensuring that an object is serializable requires two actions:

    * Decorating the class with :func:`dace.properties.make_properties`. This will wrap around the constructor and 
      field setters/getters, as well as register the object for deserialization.
    * Setting properties as field names

For example, if we intend to add a new type of :class:`~dace.transformation.transformation.SingleStateTransformation`
with some properties, it should be written as follows:

.. code-block:: python

    from dace.properties import Property, make_properties
    from dace.transformation import transformation as xf

    @make_properties
    class MyCustomXform(xf.SingleStateTransformation):
        express = Property(dtype=bool, default=False, desc='Evaluates the '
                           'transformation faster, at the expense of matching '
                           'fewer subgraphs')

        # ... PatternNodes go here ...
        # No need to create an ``__init__`` method, as defaults are defined

        def can_be_applied(self, state, expr_index, sdfg, permissive=False):
            # Properties can be used normally inside methods
            if self.express:
                ...
            ...
    
    # Properties can also be used normally once a class is instantiated
    x = MyCustomXform()
    x.express = True



.. rubric:: Custom serializable objects


If you wish to use serializable objects but do not want to use properties, you can use the :func:`dace.serialize.serializable`
decorator directly, and implement your own ``to_json`` and ``from_json`` methods. For example:

.. code-block:: python

    from dace.serialize import serializable
    
    @serializable
    class MyClass:
        # ...
        def to_json(self, parent_obj=None):
            # Converts this object to a JSON-compatible object (e.g., dict, string, float)
            # Optionally receives a parent object
            return json_compatible_object
        
        @classmethod
        def from_json(cls, json_obj, context=None):
            # Returns a class that matches the input JSON object. ``context`` contains the
            # decoding context, for example the current SDFG (see below exmaple).
            return MyClass(...)


.. note::
    Both ``@make_properties`` and ``@serializable`` will register the class for deserialization, so make sure the
    class is imported before loading an SDFG file that contains such custom properties!


.. rubric:: Custom Properties

When creating custom objects, sometimes it is preferable to also create new types of properties.
Similarly to serializable objects, custom properties need to implement the ``to_json`` and ``from_json`` methods,
as well as extend ``Property``.

Below we show an example of a custom property that uses Python's `pickle <https://docs.python.org/3/library/pickle.html>`_
built-in module to serialize arbitrary objects:

.. code-block:: python

    import base64
    from dace import properties
    import pickle
    from typing import Any, Dict, Optional

    class PickledProperty(properties.Property):
        """
        Custom Property type that uses ``pickle`` to save any data objects.
        
        :warning: Pickled objects may not be transferrable across Python
                  interpreter versions.
        """

        def to_json(self, obj: Any) -> Dict[str, Any]:
            """
            Converts the given property to a JSON-capable representation
            (e.g., string, list, dictionary). In this case, we save the pickle
            protocol and the pickled bytes as a base64-encoded string.
            """
            protocol = pickle.DEFAULT_PROTOCOL
            pbytes = pickle.dumps(obj, protocol=protocol)
            return {
                'pickle': base64.b64encode(pbytes).decode("utf-8"),
                'protocol': protocol,
            }

        @classmethod
        def from_json(cls, json_obj: Dict[str, Any], 
                      context: Optional[Dict[str, Any]] = None):
            """
            Converts the JSON object back to the original object. We can rely
            on the ``Property`` infrastructure to perform type checking before
            this method is called, so the object would either be ``None`` or
            a dictionary containing the keys we saved.

            :note: The context argument, if given, would include the top SDFG
                   currently being deserialized in ``context['sdfg']``
            """
            if json_obj is None:
                return None

            # Decode the base64-encoded string and unpickle
            byte_repr = base64.b64decode(json_obj['pickle'])
            return pickle.loads(byte_repr)
