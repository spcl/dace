# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
A tasklet that remembers what it was generated from.

Expanding a library node destroys it: :meth:`dace.transformation.transformation.ExpandTransformation.apply`
re-points the edges onto the replacement and removes the original. For a hand-written expansion
that is fine -- the implementation is in the repository and can be read. For a generated one it is
not: the description, the prompt and the conversation that produced the code all go with it, and
without them the only way to ask for a better version is to rebuild the SDFG from scratch.

:class:`AITasklet` is an ordinary tasklet that additionally carries :attr:`provenance`, so that
:mod:`dace.libraries.ai.iterate` can restore the library node and continue the conversation --
including on an SDFG loaded from disk in a later session.
"""

import dace.serialize
from dace.properties import Property, make_properties
from dace.sdfg import nodes


@dace.serialize.serializable
@make_properties
class AITasklet(nodes.Tasklet):
    """
    A tasklet written by a language model, which records how it was produced.

    It is a :class:`~dace.sdfg.nodes.Tasklet` in every respect that matters to the rest of DaCe --
    the code generator emits it identically (see the ``_generate_AITasklet`` alias in
    :mod:`dace.codegen.targets.cpu`) -- and differs only in remembering its origin.
    """

    provenance = Property(dtype=str,
                          default='',
                          desc='JSON recording the session, the round, and the serialized library '
                          'node this tasklet was generated from. Empty if unknown.')

    def to_json(self, parent):
        """
        Serializes the tasklet, recording where its class lives.

        Nothing imports :mod:`dace.libraries.ai` when ``dace`` is imported -- libraries are loaded
        on demand -- so in a fresh process this class is not registered with the serializer and the
        node would come back as an unresolved stub, taking the whole SDFG down with it. The
        ``classpath`` lets :func:`dace.serialize.from_json` import the module and find it, which is
        the same mechanism :meth:`dace.sdfg.nodes.LibraryNode.to_json` uses.

        :param parent: The state containing this node.
        :return: The serialized node.
        """
        jsonobj = super().to_json(parent)
        jsonobj['classpath'] = f'{type(self).__module__}.{type(self).__name__}'
        return jsonobj

    @staticmethod
    def from_json(json_obj, context=None):
        """
        Rebuilds an AI tasklet from its serialized form.

        This override is required rather than cosmetic. :meth:`dace.sdfg.nodes.Tasklet.from_json`
        is a static method that hardcodes ``Tasklet("dummylabel")``, so inheriting it would quietly
        turn every AI tasklet back into a plain tasklet on load and drop :attr:`provenance` --
        which is exactly the failure :class:`~dace.sdfg.nodes.RTLTasklet` suffers by way of its
        ``__jsontype__`` override. Note that this class deliberately does *not* override
        ``__jsontype__``: it serializes under its own name so that it can be resolved back here.

        :param json_obj: The serialized node.
        :param context: Deserialization context, carrying the SDFG and state.
        :return: The reconstructed tasklet.
        """
        ret = AITasklet('dummylabel')
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret
