"""Data-source adapters that emit point-in-time records.

An adapter's only job is to turn a vendor's raw response into
:class:`~vts.pit.schema.KnowledgeTimedRecord` objects with a correct
``knowledge_time``. Mapping is kept pure (JSON -> records) so it can be tested
offline against fixtures, with network I/O confined to the ``*Source`` classes.
"""

from __future__ import annotations

from vts.sources.base import DataSource
from vts.sources.fake import InMemorySource

__all__ = ["DataSource", "InMemorySource"]
