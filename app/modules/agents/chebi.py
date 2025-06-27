#!/usr/bin/env python3
import obonet
import sys
from typing import Optional, Dict, Any, List, Tuple

def load_chebi_obo(obo_path: str):
    """
    Loads the ChEBI OBO into a networkx graph using obonet.
    Each node is keyed by its OBO ID (e.g. 'CHEBI:15377'),
    and node attributes include everything from the OBO (name, def, synonym, xref, property_value, etc.).
    """
    try:
        graph = obonet.read_obo(obo_path)
        return graph
    except Exception as e:
        print(f"Failed to load OBO file '{obo_path}': {e}", file=sys.stderr)
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
class ChebiOboLookup:
    def __init__(self, obo_path: str):
        self.graph = load_chebi_obo(obo_path)

        # Build a lookup index from lowercase name + synonyms → node ID
        self.name_index: Dict[str, str] = {}
        for node_id, attrs in self.graph.nodes(data=True):
            # 1) Index the primary label
            name = attrs.get("name")
            if name:
                self.name_index[name.strip().lower()] = node_id

            # 2) Index synonyms (format: '"synonym text" SCOPE [DB:ID,...]')
            for syn in attrs.get("synonym", []):
                parts = syn.strip().split('"')
                if len(parts) >= 2:
                    syn_text = parts[1].strip().lower()
                    self.name_index[syn_text] = node_id

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Lookup a chemical by exact label or synonym (case-insensitive).
        Returns all metadata if found, else None.
        """
        q = query.strip().lower()
        node_id = self.name_index.get(q)
        if not node_id:
            return None
        return self._extract(node_id)

    def _extract(self, node_id: str) -> Dict[str, Any]:
        """
        Given a ChEBI OBO ID (e.g. 'CHEBI:15377'), extract:
          - name
          - definition
          - synonyms
          - each property_value (formula, mass, charge, InChI, InChIKey, SMILES, etc.)
          - roles (has_role edges)
          - parents (is_a)
          - other relationships
          - xrefs
          - all raw attributes (for debugging)
        """
        attrs = self.graph.nodes[node_id]

        def first_or_none(lst: List[str]) -> Optional[str]:
            return lst[0] if lst else None

        data: Dict[str, Any] = {}
        data["chebi_id"] = node_id
        data["name"] = attrs.get("name", "—")

        # ── parse `def` field ──
        raw_def = attrs.get("def", "")
        if isinstance(raw_def, str) and '"' in raw_def:
            parts = raw_def.split('"')
            data["definition"] = parts[1] if len(parts) >= 2 else raw_def
        else:
            data["definition"] = raw_def or "—"

        # ── parse synonyms ──
        syns: List[str] = []
        for syn in attrs.get("synonym", []):
            parts = syn.split('"')
            if len(parts) >= 2:
                syns.append(parts[1])
        data["synonyms"] = syns

        # ── parse property_value triples ──
        # ChEBI uses PURLs like http://purl.obolibrary.org/obo/chebi/formula, /mass, /charge, /inchi, /inchikey, /smiles, /monoisotopicmass, etc.
        prop_map: Dict[str, str] = {
            "formula": None,
            "mass": None,
            "charge": None,
            "inchi": None,
            "inchikey": None,
            "smiles": None,
        }
        for pv in attrs.get("property_value", []):
            # Each pv is like: 
            #   'http://purl.obolibrary.org/obo/chebi/formula "C" xsd:string'
            # or 'http://purl.obolibrary.org/obo/chebi/charge "0"^^xsd:string', etc.
            parts = pv.split()
            # parts[0] = full PURL, parts[1] = literal in quotes (or literal^^type)
            if len(parts) < 2:
                continue
            purl = parts[0]
            # find which key it matches
            for key in list(prop_map.keys()):
                if purl.endswith(f"/chebi/{key}"):
                    # extract the quoted value
                    literal = parts[1]
                    if literal.startswith('"') and literal.endswith('"'):
                        literal = literal.strip('"')
                    else:
                        # handle cases like '"14.007"^^xsd:string'
                        literal = literal.strip('"').split('^^')[0]
                    prop_map[key] = literal
                    break

        data["formula"]  = prop_map["formula"]
        data["mass"]     = prop_map["mass"]
        data["charge"]   = prop_map["charge"]
        data["inchi"]    = prop_map["inchi"]
        data["inchikey"] = prop_map["inchikey"]
        data["smiles"]   = prop_map["smiles"]

        # ── roles (outgoing edge labeled 'has_role') ──
        roles: List[str] = []
        for _, v, key, _ in self.graph.out_edges(node_id, data=True, keys=True):
            if key == "has_role":
                roles.append(v)
        data["roles"] = roles or ["—"]

        # ── parents (is_a) ──
        parents = attrs.get("is_a", [])
        data["parents"] = parents or []

        # ── other relationships (edges excluding is_a and has_role) ──
        all_rels: Dict[str, List[str]] = {}
        for _, v, key, _ in self.graph.out_edges(node_id, data=True, keys=True):
            if key in ("is_a", "has_role"):
                continue
            all_rels.setdefault(key, []).append(v)
        data["relationships"] = all_rels

        # ── xrefs ──
        xref_dict: Dict[str, List[str]] = {}
        for x in attrs.get("xref", []):
            if ":" in x:
                db, db_id = x.split(":", 1)
                xref_dict.setdefault(db, []).append(db_id)
        data["xrefs"] = xref_dict

        # ── raw attributes dump for debugging ──
        data["all_attributes"] = attrs.copy()

        return data

# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print("Usage: python chebi.py <path/to/chebi.obo>", file=sys.stderr)
        sys.exit(1)

    obo_path = sys.argv[1]
    chebi_lookup = ChebiOboLookup(obo_path)

    print("Enter a chemical name (exact label or synonym) to look it up in ChEBI (Ctrl+C to exit):")
    try:
        while True:
            query = input(">>> ").strip()
            if not query:
                continue

            result = chebi_lookup.lookup(query)
            if not result:
                print(f"No match found for '{query}'.\n")
                continue

            print(f"\n✅ Match for '{query}':")
            for key, val in result.items():
                if key == "all_attributes":
                    print(f"  {key}:")
                    for attr_name, attr_val in val.items():
                        if isinstance(attr_val, list):
                            if attr_val:
                                joined = ", ".join(str(x) for x in attr_val)
                                print(f"    - {attr_name}: [{joined}]")
                            else:
                                print(f"    - {attr_name}: []")
                        else:
                            print(f"    - {attr_name}: {attr_val}")
                elif isinstance(val, dict):
                    print(f"  {key}:")
                    for subk, subv in val.items():
                        if isinstance(subv, list):
                            print(f"    - {subk}: {', '.join(subv)}")
                        else:
                            print(f"    - {subk}: {subv}")
                elif isinstance(val, list):
                    if val:
                        print(f"  {key}: {', '.join(val)}")
                    else:
                        print(f"  {key}: —")
                else:
                    print(f"  {key}: {val}")
            print()

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
