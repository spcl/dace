# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LastWriterDFSStreamScheduler`` -- DFS intra-state + LastWriter inter-block.

Sync points are materialised as the three sync libnodes
(``StreamEventRecord``, ``StreamWaitEvent``, ``StreamSynchronize``).
The strategy + libnodes + sync-hoister live together in this package
so the whole scheduling pipeline is co-located.
"""
from .libnodes import (StreamEventRecordLibraryNode, StreamSynchronizeLibraryNode, StreamWaitEventLibraryNode)
from .last_writer import LastWriter, StreamEventToken
from .strategy import LastWriterDFSStreamScheduler

__all__ = [
    "LastWriterDFSStreamScheduler",
    "StreamEventRecordLibraryNode",
    "StreamWaitEventLibraryNode",
    "StreamSynchronizeLibraryNode",
    "LastWriter",
    "StreamEventToken",
]
