"""
Chemical‑formula validation helpers.

Dependencies
------------
pymatgen>=2025.5
mp-api>=0.42          # pip install mp-api
export MP_API_KEY='your‑key'  # or pass explicitly
"""
from __future__ import annotations

import functools
import logging
import os
import re
from typing import Optional

from mp_api.client import MPRester
from pymatgen.core import Composition

logger = logging.getLogger(__name__)

class ChemicalFormulaValidator:
    """Parse, normalise and optionally cross‑check a chemical formula."""

    _REDUCE_PATTERN = re.compile(r"\s+")

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.timeout = timeout

    # -------------------------- public helpers --------------------------- #

    def validate(self, raw: str) -> dict:
        """
        Validate *raw* formula text.

        Returns
        -------
        dict
            {
              "input": original string,
              "canonical": Hill‑ordered formula,  # None if unparsable
              "mp_hits": int,                    # ‑1 if MP lookup skipped/failed
              "status": "ok" | "corrected" | "invalid",
              "error": str | None
            }
        """
        cleaned = self._REDUCE_PATTERN.sub("", raw)
        result = {
            "input": raw,
            "canonical": None,
            "mp_hits": -1,
            "status": "invalid",
            "error": None,
        }

        # ---------- 1. fast local parse ---------- #
        try:
            comp = Composition(cleaned)
            # Hill system canonical representation
            result["canonical"] = comp.formula
            status = "ok" if cleaned == comp.formula else "corrected"
            result["status"] = status
        except Exception as exc:          # noqa: BLE001
            result["error"] = f"parse‑error: {exc}"
            return result                 # short‑circuit

        # ---------- 2. Materials Project cross‑check (optional) ---------- #
        if self.api_key:
            try:
                result["mp_hits"] = self._query_mp(result["canonical"])
                if result["mp_hits"] == 0 and status == "ok":
                    # syntactically valid but not in MP → still suspicious
                    result["status"] = "invalid"
            except Exception as exc:      # noqa: BLE001
                logger.warning("MP lookup failed: %s", exc)
                result["error"] = f"mp‑error: {exc}"

        return result

    # -------------------------- internals --------------------------- #

    @functools.lru_cache(maxsize=None)
    def _query_mp(self, formula: str) -> int:
        """Return number of MP materials matching *formula*."""
        with MPRester(self.api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id"],
                num_chunks=1,
                chunk_size=20,
            )  # fast & light
        return len(docs)
