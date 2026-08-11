"""CosmosXRay360 predict2_5 package."""

import sys
from unittest.mock import MagicMock

# Safely stub optional missing third-party packages required by cosmos_predict2.easy_io backends
# so we don't need to modify submodule files directly.
if "multistorageclient" not in sys.modules:
    try:
        __import__("multistorageclient")
    except ImportError:
        sys.modules["multistorageclient"] = MagicMock()
