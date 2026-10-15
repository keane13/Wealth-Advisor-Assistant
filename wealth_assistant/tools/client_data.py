"""Client financial data source (JSON files) and the parser that validates it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.errors import DataValidationError, NotFoundError, ToolError
from ..core.tools import Tool
from ..domain.models import ClientSnapshot
from ..domain.parsing import parse_client_document
from ..domain.validation import is_valid_client_id


class ClientDataLoadTool(Tool):
    name = "client_data.load"
    description = "Load the raw client financial document (accounts, holdings, liabilities, transactions)."

    def __init__(self, clients_dir: Path) -> None:
        self.clients_dir = clients_dir

    def run(self, *, client_id: str) -> dict[str, Any]:
        if not is_valid_client_id(client_id):
            raise DataValidationError(f"invalid client id {client_id!r}", tool=self.name)
        path = self.clients_dir / f"{client_id}.json"
        if not path.is_file():
            raise NotFoundError(f"no financial data for client {client_id}", tool=self.name)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"client file is not valid JSON: {exc}", tool=self.name) from exc
        except OSError as exc:
            raise ToolError(f"could not read client file: {exc}", tool=self.name) from exc


class ClientDataParseTool(Tool):
    name = "client_data.parse"
    description = "Validate and normalise a raw client document into a ClientSnapshot, reporting data-quality issues."

    def run(self, *, raw: Any, client_id: str) -> ClientSnapshot:
        return parse_client_document(raw, client_id)


def list_client_ids(clients_dir: Path) -> list[str]:
    return sorted(p.stem for p in clients_dir.glob("*.json") if is_valid_client_id(p.stem))
