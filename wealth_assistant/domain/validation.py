from __future__ import annotations

import re

# Client ids become file names and API path segments, so they are whitelisted at
# the entry point. This blocks path traversal like "../../secrets".
CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def is_valid_client_id(client_id: object) -> bool:
    return isinstance(client_id, str) and bool(CLIENT_ID_PATTERN.match(client_id))
