.. _properties:

Properties and Serialization
============================

The SDFG format is a JSON file.

Property object
Function like type annotations in dataclasses, but with extra utilities

:class:`~dace.properties.DataclassProperty` can convert ``dataclass`` objects to properties.


How to use properties in serializable objects


Custom Properties
-----------------

How to make a custom property:

Example: Pickled Property
~~~~~~~~~~~~~~~~~~~~~~~~~

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
