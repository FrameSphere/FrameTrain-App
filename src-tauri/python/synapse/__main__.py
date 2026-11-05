"""python -m synapse catalog  schreibt den Node-Katalog als JSON nach stdout."""

import json
import sys

from .spec import catalog

if len(sys.argv) > 1 and sys.argv[1] == "catalog":
    print(json.dumps(catalog(), indent=2, ensure_ascii=False))
else:
    sys.exit("Aufruf: python -m synapse catalog")
