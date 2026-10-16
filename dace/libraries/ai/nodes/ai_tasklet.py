# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A tasklet that records the library node and session it was generated from."""

import dace.serialize
from dace.properties import Property, make_properties
from dace.sdfg import nodes


@dace.serialize.serializable
@make_properties
class AITasklet(nodes.Tasklet):
    """
    A tasklet written by a language model. Code generation treats it as a
    :class:`~dace.sdfg.nodes.Tasklet`; :attr:`provenance` lets :mod:`dace.libraries.ai.iterate`
    restore the library node and continue its session.
    """

    provenance = Property(
        dtype=str,
        default="",
        desc="JSON with the session, the round and the serialized library node this tasklet was "
        "generated from. Empty if unknown.",
    )

    def to_json(self, parent):
        """
        Serializes the tasklet with its class path, so that :func:`dace.serialize.from_json` can
        import this module in a process that has not loaded the AI library (as for library nodes).

        :param parent: The state containing this node.
        :return: The serialized node.
        """
        jsonobj = super().to_json(parent)
        jsonobj["classpath"] = f"{type(self).__module__}.{type(self).__name__}"
        return jsonobj

    @staticmethod
    def from_json(json_obj, context=None):
        """
        Rebuilds an AI tasklet. :meth:`dace.sdfg.nodes.Tasklet.from_json` creates a plain
        ``Tasklet``, which would drop :attr:`provenance`.

        :param json_obj: The serialized node.
        :param context: Deserialization context, carrying the SDFG and state.
        :return: The reconstructed tasklet.
        """
        ret = AITasklet("dummylabel")
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret
