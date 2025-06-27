#!/usr/bin/env python3
"""
enhanced_properties_conflict_resolution.py

Further improvements to PhysicalPropertyExtractor and PropertyNormalizer to avoid
spurious property‐value extractions when properties and material mentions are semantically 
disconnected. Implements:
  - Proximity‐based matching: require “<material>” and “<property phrase>” to appear within a few words
    of one another (rather than anywhere in the sentence).
  - Context‐pattern checks: look for “<property phrase> of <material>” or “<material> <property phrase>”
    before extracting numeric values.
  - Plausibility filtering for known property domains (e.g., efficiency ∈ [0,100], density > 0).
  - Retain “measured_property” relations as before, but do not attempt numeric extraction from them.

Usage:
    extractor = PhysicalPropertyExtractor()
    raw_entries = extractor.extract(text, ["EQE", "Na3Bi"])
    raw_from_rel = extractor.extract_from_relations(term_record)
    all_raw = raw_entries + raw_from_rel

    normalizer = PropertyNormalizer()
    normalized_entries = normalizer.normalize(all_raw)

Author: David Abramov (@dabramov)
"""

import re
from typing import List, Dict, Optional, Any


class PhysicalPropertyExtractor:
    """
    Extracts physical property mentions from text for known materials,
    using proximity‐based and pattern‐based matching to reduce false positives.

    Entry points:
      - extract(text, materials): scans free text for “<material> <property> <value><unit>”
      - extract_from_relations(term_record): reads “measured_property” relations in a term dict

    Returns a list of dicts with keys:
        - material: Material name (string)
        - property: Canonical property name (e.g., “band_gap”)
        - raw_value: Extracted numeric string or None
        - raw_unit: Extracted unit string or None
        - uncertainty: Extracted uncertainty string or None
        - context: Sentence or relation context (string)
        - verified: True (initial assumption; can be overridden)
    """

    # Map various textual forms to canonical property names
    PROPERTY_KEYWORDS: Dict[str, str] = {
        # Standard properties
        "band gap": "band_gap",
        "band-gap": "band_gap",
        "melting point": "melting_point",
        "melting_temperature": "melting_point",
        "glass transition temperature": "glass_transition_temperature",
        "t_g": "glass_transition_temperature",
        "conductivity": "conductivity",
        "electrical conductivity": "conductivity",
        "mobility": "mobility",
        "efficiency": "efficiency",
        "thermal conductivity": "thermal_conductivity",
        "dielectric constant": "dielectric_constant",
        # Materials Project properties
        "density": "density",
        "formation energy": "formation_energy",
        "formation energy per atom": "formation_energy",
        "bulk modulus": "bulk_modulus",
        "shear modulus": "shear_modulus",
        "elastic modulus": "bulk_modulus",
        "magnetic moment": "magnetic_moment",
        "energy above hull": "energy_above_hull",
    }

    # Regex to capture a numeric value (with optional uncertainty) and unit.
    # Matches: "1.9 ± 0.1 eV", "2.1eV", "2.1 eV", "250 K", "150 °C", "5.2 g/cm3", "200 GPa"
    VALUE_UNIT_PATTERN = re.compile(
        r'(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:\s*±\s*(?P<uncertainty>\d+(?:\.\d+)?))?\s*'
        r'(?P<unit>°[CF]|[A-Za-zµμ/°\^\-0-9]+)'
    )

    # Maximum number of characters between material and property phrase to be considered “proximal” 
    PROXIMITY_WINDOW = 30  # characters

    def __init__(self, property_keywords: Optional[Dict[str, str]] = None):
        """
        Initialize extractor with optional custom property keywords.

        Args:
            property_keywords: override for PROPERTY_KEYWORDS mapping.
        """
        self.keywords = property_keywords or self.PROPERTY_KEYWORDS

    def _is_proximal(self, material: str, prop_phrase: str, sentence: str) -> bool:
        """
        Returns True if material and prop_phrase appear within PROXIMITY_WINDOW characters of each other.
        """
        mat_low = material.lower()
        prop_low = prop_phrase.lower()
        sent_low = sentence.lower()

        for m in re.finditer(re.escape(mat_low), sent_low):
            for p in re.finditer(re.escape(prop_low), sent_low):
                if abs(m.start() - p.start()) <= self.PROXIMITY_WINDOW:
                    return True
        return False

    def _matches_pattern(self, material: str, prop_phrase: str, sentence: str) -> bool:
        """
        Enforce pattern: either “<prop_phrase> of <material>” or “<material> <prop_phrase>” 
        (with optional words in between). Uses regex to detect these common syntaxes.
        """
        mat = re.escape(material)
        prop = re.escape(prop_phrase)
        # pattern1: “<prop_phrase> of <material>”
        pattern1 = rf"\b{prop}\b\s+of\s+{mat}\b"
        # pattern2: “<material> <prop_phrase>”
        pattern2 = rf"\b{mat}\b\s+.*?\b{prop}\b"
        return bool(re.search(pattern1, sentence, flags=re.IGNORECASE) or
                    re.search(pattern2, sentence, flags=re.IGNORECASE))

    def extract(self, text: str, materials: List[str]) -> List[Dict[str, Any]]:
        """
        Extract property entries for given materials from a block of text.

        Args:
            text: The full paragraph or page text to scan.
            materials: List of material names (e.g., ["EQE", "Na3Bi"]).

        Returns:
            A list of dicts. Each dict contains:
                - material: Material name (string)
                - property: Canonical property name (e.g., "band_gap")
                - raw_value: Extracted numeric string (e.g. "1.9") or None
                - raw_unit: Extracted unit string (e.g. "eV", "°C") or None
                - uncertainty: Extracted uncertainty (e.g. "0.05") or None
                - context: The full sentence containing the mention
                - verified: True if numeric extracted and patterns matched; False otherwise
        """
        results: List[Dict[str, Any]] = []
        # Split text into sentences to preserve context
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        lower_text = text.lower()

        for material in materials:
            mat_low = material.lower()
            if mat_low not in lower_text:
                continue  # skip if material is not in text at all

            for prop_phrase, prop_name in self.keywords.items():
                prop_low = prop_phrase.lower()
                for sent in sentences:
                    sent_low = sent.lower()
                    # First check proximity and pattern before numeric extraction
                    if mat_low in sent_low and prop_low in sent_low:
                        if not (self._is_proximal(material, prop_phrase, sent) and
                                self._matches_pattern(material, prop_phrase, sent)):
                            continue  # skip if not truly about “prop_phrase of material”

                        # Attempt numeric extraction in this proximal sentence
                        found_value = False
                        for match in self.VALUE_UNIT_PATTERN.finditer(sent):
                            found_value = True
                            value = match.group("value")
                            unit = match.group("unit")
                            uncertainty = match.group("uncertainty")
                            entry = {
                                "material": material,
                                "property": prop_name,
                                "raw_value": value,
                                "raw_unit": unit,
                                "uncertainty": uncertainty if uncertainty else None,
                                "context": sent.strip(),
                                "verified": True,
                            }
                            results.append(entry)

                        # If property phrase & material match patterns but no numeric found,
                        # record placeholder with raw_value/unit = None and verified=False
                        if not found_value:
                            entry = {
                                "material": material,
                                "property": prop_name,
                                "raw_value": None,
                                "raw_unit": None,
                                "uncertainty": None,
                                "context": sent.strip(),
                                "verified": False,
                            }
                            results.append(entry)

        return results

    def extract_from_relations(self, term_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract “measured_property” relations from a term record.

        Args:
            term_record: A dict representing a term, containing a "relations" list,
                         where each relation is a dict with keys:
                           - relation: relation name (string)
                           - related_term: the term on the right side (string)
                           - verified: bool

        Returns:
            A list of dicts, each with keys:
                - material: term_record["term"]
                - property: canonical property name if matched, else raw related_term lowercased
                - raw_value: None (no numeric in relation itself)
                - raw_unit: None
                - uncertainty: None
                - context: "measured_property:<related_term>"
                - verified: copied from relation entry
        """
        results: List[Dict[str, Any]] = []
        material = term_record.get("term", "").strip()
        for rel in term_record.get("relations", []):
            if rel.get("relation") == "measured_property":
                raw_prop = rel.get("related_term", "").strip().lower()
                prop_name = self.keywords.get(raw_prop, raw_prop.replace(" ", "_"))
                entry = {
                    "material": material,
                    "property": prop_name,
                    "raw_value": None,
                    "raw_unit": None,
                    "uncertainty": None,
                    "context": f"measured_property:{rel.get('related_term')}",
                    "verified": bool(rel.get("verified", False)),
                }
                results.append(entry)
        return results


class PropertyNormalizer:
    """
    Normalize extracted property values into standard units and apply plausibility checks.

    Usage:
        normalizer = PropertyNormalizer()
        normalized = normalizer.normalize(extracted_entries)
    """

    # Desired target units for each property
    TARGET_UNITS: Dict[str, str] = {
        "band_gap": "eV",
        "melting_point": "K",
        "glass_transition_temperature": "K",
        "conductivity": "S/m",
        "mobility": "m2/(V*s)",
        "efficiency": "%",
        "thermal_conductivity": "W/(m*K)",
        "dielectric_constant": "",  # dimensionless
        "density": "kg/m3",
        "formation_energy": "eV/atom",
        "bulk_modulus": "GPa",
        "shear_modulus": "GPa",
        "magnetic_moment": "μ_B",
        "energy_above_hull": "eV/atom",
    }

    # Plausibility ranges for certain properties; if normalized_value falls outside,
    # flag unit_conversion_failed=True
    PLAUSIBILITY_RANGES: Dict[str, Any] = {
        "efficiency": (0.0, 100.0),       # percent
        "density": (0.0, float("inf")),   # density must be positive
        "band_gap": (0.0, 10.0),          # reasonable eV range
        "formation_energy": (-10.0, 10.0),# eV/atom range
        "bulk_modulus": (0.0, 1000.0),    # GPa range
        "shear_modulus": (0.0, 1000.0),   # GPa range
        "magnetic_moment": (0.0, 100.0),  # μ_B range
        "energy_above_hull": (0.0, 5.0),  # eV/atom
    }

    def normalize(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert raw values and units into canonical numeric + unit form,
        then apply plausibility filtering.

        Args:
            entries: List of dicts from PhysicalPropertyExtractor, each containing:
                     raw_value (str or None), raw_unit (str or None),
                     uncertainty (str or None), property (str), etc.

        Returns:
            List of dicts with added fields for each entry:
                - normalized_value: float or None
                - normalized_unit: str or None
                - uncertainty_value: float or None
                - unit_conversion_failed: bool
        """
        normalized_entries: List[Dict[str, Any]] = []

        for e in entries:
            prop = e["property"]
            raw_val = e.get("raw_value")
            raw_unit = e.get("raw_unit")
            unc = e.get("uncertainty")
            target_unit = self.TARGET_UNITS.get(prop)

            norm_entry = dict(e)  # copy original fields
            norm_entry.update({
                "normalized_value": None,
                "normalized_unit": None,
                "uncertainty_value": None,
                "unit_conversion_failed": False
            })

            # If there is no raw_value or raw_unit, skip numeric conversion
            if raw_val is None or raw_unit is None:
                normalized_entries.append(norm_entry)
                continue

            try:
                val = float(raw_val)
                unit_lower = raw_unit.strip().lower()

                # Temperature-based properties → Kelvin
                if prop in ("melting_point", "glass_transition_temperature"):
                    if unit_lower in {"°c", "degc", "c"}:
                        norm_val = val + 273.15
                        unit = "K"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"k", "kelvin"}:
                        norm_val = val
                        unit = "K"
                        unc_norm = float(unc) if unc else None
                    else:
                        raise ValueError(f"Unknown temperature unit '{raw_unit}'")

                # Band gap → eV
                elif prop == "band_gap":
                    if unit_lower in {"ev"}:
                        norm_val = val
                        unit = "eV"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"mev"}:
                        norm_val = val / 1000.0
                        unit = "eV"
                        unc_norm = (float(unc) / 1000.0) if unc else None
                    else:
                        raise ValueError(f"Unknown energy unit '{raw_unit}'")

                # Conductivity → S/m (1 S/cm = 100 S/m)
                elif prop == "conductivity":
                    if unit_lower in {"s/cm", "s·cm-1", "s·cm⁻¹"}:
                        norm_val = val * 100.0
                        unit = "S/m"
                        unc_norm = (float(unc) * 100.0) if unc else None
                    elif unit_lower in {"s/m", "s·m-1", "s·m⁻¹"}:
                        norm_val = val
                        unit = "S/m"
                        unc_norm = float(unc) if unc else None
                    else:
                        raise ValueError(f"Unknown conductivity unit '{raw_unit}'")

                # Mobility → m2/(V*s) (1 cm²/V·s = 1e-4 m²/V·s)
                elif prop == "mobility":
                    if unit_lower in {"cm2/vs", "cm^2/vs", "cm²/vs"}:
                        norm_val = val * 1e-4
                        unit = "m2/(V*s)"
                        unc_norm = (float(unc) * 1e-4) if unc else None
                    elif unit_lower in {"m2/vs", "m^2/vs", "m²/vs"}:
                        norm_val = val
                        unit = "m2/(V*s)"
                        unc_norm = float(unc) if unc else None
                    else:
                        raise ValueError(f"Unknown mobility unit '{raw_unit}'")

                # Efficiency → percent
                elif prop == "efficiency":
                    if raw_unit.strip() == "%":
                        norm_val = val
                        unit = "%"
                        unc_norm = float(unc) if unc else None
                    else:
                        # assume fraction (e.g. 0.85 → 85%)
                        norm_val = val * 100.0
                        unit = "%"
                        unc_norm = (float(unc) * 100.0) if unc else None

                # Thermal conductivity → W/(m*K)
                elif prop == "thermal_conductivity":
                    if unit_lower in {"w/mk", "w/(m*k)", "w/(m·k)"}:
                        norm_val = val
                        unit = "W/(m*K)"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"w/cmk", "w/(cm*k)"}:
                        # 1 W/(cm*K) = 100 W/(m*K)
                        norm_val = val * 100.0
                        unit = "W/(m*K)"
                        unc_norm = (float(unc) * 100.0) if unc else None
                    else:
                        raise ValueError(f"Unknown thermal conductivity unit '{raw_unit}'")

                # Dielectric constant → dimensionless
                elif prop == "dielectric_constant":
                    norm_val = val
                    unit = ""
                    unc_norm = float(unc) if unc else None

                # Density → kg/m3 (1 g/cm3 = 1000 kg/m3)
                elif prop == "density":
                    if unit_lower in {"g/cm3", "g/cm^3", "g·cm-3", "g·cm⁻³"}:
                        norm_val = val * 1000.0
                        unit = "kg/m3"
                        unc_norm = (float(unc) * 1000.0) if unc else None
                    elif unit_lower in {"kg/m3", "kg·m-3"}:
                        norm_val = val
                        unit = "kg/m3"
                        unc_norm = float(unc) if unc else None
                    else:
                        raise ValueError(f"Unknown density unit '{raw_unit}'")

                # Formation energy → eV/atom (1 meV/atom = 0.001 eV/atom)
                elif prop == "formation_energy":
                    if unit_lower in {"ev/atom"}:
                        norm_val = val
                        unit = "eV/atom"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"mev/atom"}:
                        norm_val = val / 1000.0
                        unit = "eV/atom"
                        unc_norm = (float(unc) / 1000.0) if unc else None
                    else:
                        raise ValueError(f"Unknown formation energy unit '{raw_unit}'")

                # Bulk modulus → GPa (1 MPa = 0.001 GPa)
                elif prop == "bulk_modulus":
                    if unit_lower in {"gpa"}:
                        norm_val = val
                        unit = "GPa"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"mpa"}:
                        norm_val = val / 1000.0
                        unit = "GPa"
                        unc_norm = (float(unc) / 1000.0) if unc else None
                    else:
                        raise ValueError(f"Unknown bulk modulus unit '{raw_unit}'")

                # Shear modulus → GPa (same conversion as bulk modulus)
                elif prop == "shear_modulus":
                    if unit_lower in {"gpa"}:
                        norm_val = val
                        unit = "GPa"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"mpa"}:
                        norm_val = val / 1000.0
                        unit = "GPa"
                        unc_norm = (float(unc) / 1000.0) if unc else None
                    else:
                        raise ValueError(f"Unknown shear modulus unit '{raw_unit}'")

                # Magnetic moment → μ_B (assuming raw unit "μB", "uB", "mu_B", etc.)
                elif prop == "magnetic_moment":
                    if unit_lower in {"μb", "ub", "mu_b", "mu-b", "mu b"}:
                        norm_val = val
                        unit = "μ_B"
                        unc_norm = float(unc) if unc else None
                    else:
                        raise ValueError(f"Unknown magnetic moment unit '{raw_unit}'")

                # Energy above hull → eV/atom (1 meV/atom = 0.001 eV/atom)
                elif prop == "energy_above_hull":
                    if unit_lower in {"ev/atom"}:
                        norm_val = val
                        unit = "eV/atom"
                        unc_norm = float(unc) if unc else None
                    elif unit_lower in {"mev/atom"}:
                        norm_val = val / 1000.0
                        unit = "eV/atom"
                        unc_norm = (float(unc) / 1000.0) if unc else None
                    else:
                        raise ValueError(f"Unknown energy above hull unit '{raw_unit}'")

                # Fallback: unknown property → pass through raw
                else:
                    norm_val = val
                    unit = raw_unit
                    unc_norm = float(unc) if unc else None

                # Apply plausibility checks
                if prop in self.PLAUSIBILITY_RANGES:
                    low, high = self.PLAUSIBILITY_RANGES[prop]
                    if not (low <= norm_val <= high):
                        raise ValueError(f"Plausibility check failed for {prop}: {norm_val} not in [{low},{high}]")

                norm_entry["normalized_value"] = round(norm_val, 6)
                norm_entry["normalized_unit"] = unit
                if unc_norm is not None:
                    norm_entry["uncertainty_value"] = round(unc_norm, 6)

            except Exception:
                norm_entry["unit_conversion_failed"] = True

            normalized_entries.append(norm_entry)

        return normalized_entries


# ----------------------------------------
# Unit Tests
# ----------------------------------------
def test_extractor_basic_bandgap():
    text = "The Na3Bi film exhibited a band gap of 1.9 eV at room temperature."
    extractor = PhysicalPropertyExtractor()
    entries = extractor.extract(text, ["Na3Bi"])
    assert len(entries) == 1
    e = entries[0]
    assert e["material"] == "Na3Bi"
    assert e["property"] == "band_gap"
    assert e["raw_value"] == "1.9"
    assert e["raw_unit"] == "eV"
    assert e["verified"] is True

def test_extractor_no_value_but_pattern():
    text = "Measured_property: band gap of Na3Bi."
    extractor = PhysicalPropertyExtractor()
    entries = extractor.extract(text, ["Na3Bi"])
    assert len(entries) == 1
    e = entries[0]
    assert e["raw_value"] is None
    assert e["verified"] is False

def test_extractor_out_of_proximity():
    text = (
        "The PL quenching efficiency of PM6: Y6-2O blends is 30.2%, which indicates that "
        "results in a low EQE of the Y6-2O-based devices."
    )
    extractor = PhysicalPropertyExtractor()
    # We only want "efficiency" of "EQE" if truly “EQE efficiency” is mentioned near each other.
    entries = extractor.extract(text, ["EQE"])
    # Should be empty, because “EQE” and “efficiency” are far apart (not proximal/pattern-matched).
    assert len(entries) == 0

def test_extractor_density():
    text = "The density of Na3Bi is 6.7 g/cm3 at 300 K."
    extractor = PhysicalPropertyExtractor()
    entries = extractor.extract(text, ["Na3Bi"])
    assert len(entries) == 1
    e = entries[0]
    assert e["property"] == "density"
    assert e["raw_value"] == "6.7"
    assert e["raw_unit"] == "g/cm3"
    assert e["verified"] is True

def test_extract_from_relations():
    term_record = {
        "term": "Na3Bi",
        "relations": [
            {"relation": "measured_property", "related_term": "band gap", "verified": True},
            {"relation": "measured_property", "related_term": "mobility", "verified": False},
            {"relation": "energy_level", "related_term": "Fermi energy", "verified": True},
        ]
    }
    extractor = PhysicalPropertyExtractor()
    raw = extractor.extract_from_relations(term_record)
    assert len(raw) == 2
    props = {r["property"] for r in raw}
    assert "band_gap" in props
    assert "mobility" in props

def test_normalizer_ev_to_ev():
    entries = [{
        "material": "Na3Bi",
        "property": "band_gap",
        "raw_value": "1.9",
        "raw_unit": "eV",
        "uncertainty": None,
        "context": "",
        "verified": True
    }]
    normalizer = PropertyNormalizer()
    ne = normalizer.normalize(entries)[0]
    assert ne["normalized_value"] == 1.9
    assert ne["normalized_unit"] == "eV"
    assert ne["unit_conversion_failed"] is False

def test_normalizer_density():
    entries = [{
        "material": "Na3Bi",
        "property": "density",
        "raw_value": "6.7",
        "raw_unit": "g/cm3",
        "uncertainty": "0.1",
        "context": "",
        "verified": True
    }]
    normalizer = PropertyNormalizer()
    ne = normalizer.normalize(entries)[0]
    # 6.7 g/cm3 = 6700 kg/m3
    assert abs(ne["normalized_value"] - 6700.0) < 1e-6
    assert ne["normalized_unit"] == "kg/m3"
    assert abs(ne["uncertainty_value"] - 100.0) < 1e-6

def test_normalizer_efficiency_plausibility():
    # Efficiency = -20% (invalid)
    entries = [{
        "material": "EQE",
        "property": "efficiency",
        "raw_value": "-20",
        "raw_unit": "%",
        "uncertainty": None,
        "context": "",
        "verified": True
    }]
    normalizer = PropertyNormalizer()
    ne = normalizer.normalize(entries)[0]
    assert ne["unit_conversion_failed"] is True

    # Efficiency = 600% (invalid)
    entries2 = [{
        "material": "EQE",
        "property": "efficiency",
        "raw_value": "600",
        "raw_unit": "%",
        "uncertainty": None,
        "context": "",
        "verified": True
    }]
    ne2 = normalizer.normalize(entries2)[0]
    assert ne2["unit_conversion_failed"] is True

def test_normalizer_bulk_modulus():
    entries = [{
        "material": "Na3Bi",
        "property": "bulk_modulus",
        "raw_value": "200",
        "raw_unit": "GPa",
        "uncertainty": "5",
        "context": "",
        "verified": True
    }, {
        "material": "Na3Bi",
        "property": "bulk_modulus",
        "raw_value": "200000",
        "raw_unit": "MPa",
        "uncertainty": "5000",
        "context": "",
        "verified": True
    }]
    normalizer = PropertyNormalizer()
    ne1, ne2 = normalizer.normalize(entries)
    assert ne1["normalized_unit"] == "GPa" and abs(ne1["normalized_value"] - 200.0) < 1e-6
    assert ne2["normalized_unit"] == "GPa" and abs(ne2["normalized_value"] - 200.0) < 1e-6

def test_normalizer_failed_unit():
    entries = [{
        "material": "Unknown",
        "property": "band_gap",
        "raw_value": "abc",
        "raw_unit": "xyz",
        "uncertainty": None,
        "context": "",
        "verified": True
    }]
    normalizer = PropertyNormalizer()
    ne = normalizer.normalize(entries)[0]
    assert ne["unit_conversion_failed"]

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
