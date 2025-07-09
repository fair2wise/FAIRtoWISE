#!/usr/bin/env python3
"""
Convert Materials Science SKOS to OWL ontology using OAK and ROBOT,
taking inspiration from the Biolink model relationships.

This script:
  - Parses a SKOS file to build an OboGraph (a simplified graph representation)
  - Processes the graph to create a LinkML-style meta-ontology with Biolink-inspired
    relationships.
  - Attempts conversion to OWL using linkml-owl (gen-owl) and falls back to ROBOT if needed.
  - Post-processes the resulting ontology with ROBOT (reasoning, OBO conversion, reporting,
    and documentation).

Requirements:
  - Python packages: rdflib, PyYAML
  - External tools: ROBOT, gen-owl (linkml-owl)

Usage:
    python app/modules/build_onto1.py
"""

import os
import subprocess
import logging
from pathlib import Path
import yaml
import json
import rdflib
import dataclasses
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define a simplified OboGraph model
@dataclasses.dataclass
class Node:
    id: str
    lbl: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class Edge:
    sub: str
    pred: str
    obj: str
    meta: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class Graph:
    id: str
    nodes: List[Node] = dataclasses.field(default_factory=list)
    edges: List[Edge] = dataclasses.field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None


def graph_as_dict(graph: Graph) -> Dict[str, Any]:
    """Convert Graph object to dictionary representation"""
    return {
        "id": graph.id,
        "nodes": [dataclasses.asdict(node) for node in graph.nodes],
        "edges": [dataclasses.asdict(edge) for edge in graph.edges],
        "meta": graph.meta,
    }


# Configuration
INPUT_SKOS = "storage/terminology/extracted_terms_skos.ttl"
OUTPUT_DIR = "storage/ontology"
ONTOLOGY_ID = "MATONTO"
ONTOLOGY_TITLE = "Materials Science Ontology"
BASE_IRI = "http://example.org/materials/"
PREFIX = "matonto"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_with_robot(input_path: str, output_path: str) -> bool:
    """
    Use ROBOT to convert from SKOS to OWL.

    Parameters:
        input_path (str): Path to the input SKOS file.
        output_path (str): Path to save the converted OWL file.

    Returns:
        bool: True if conversion succeeds; otherwise False.
    """
    logger.info(f"Converting {input_path} to {output_path} using ROBOT")

    try:
        result = subprocess.run(
            ["robot", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        logger.info(result)
        robot_cmd = ["robot"]
    except FileNotFoundError:
        robot_script = str(Path.home() / "bin" / "robot")
        if os.path.exists(robot_script):
            robot_cmd = [robot_script]
        else:
            logger.error(
                "ROBOT not found. Please install ROBOT: http://robot.obolibrary.org/"
            )
            logger.info("Proceeding without ROBOT conversion step")
            return False

    try:
        cmd = robot_cmd + [
            "convert",
            "--input",
            input_path,
            "--output",
            output_path,
            "--format",
            "owl",
        ]
        logger.debug(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("ROBOT conversion completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ROBOT conversion failed: {e}")
        return False


def skos_to_obograph(input_path: str) -> Graph:
    """
    Convert SKOS to OboGraph directly using RDFLib.

    Parameters:
        input_path (str): Path to the SKOS (TTL) file.

    Returns:
        Graph: An OboGraph representation of the SKOS.
    """
    logger.info(f"Converting SKOS to OboGraph: {input_path}")
    g = rdflib.Graph()
    g.parse(input_path, format="turtle")
    obograph = Graph(id="http://example.org/materials/graph")

    # Define namespaces
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")

    # Process concept schemes as nodes
    for s, p, o in g.triples((None, RDF.type, SKOS.ConceptScheme)):
        label = None
        for _, _, label_val in g.triples((s, RDFS.label, None)):
            label = str(label_val)
            break
        if not label:
            for _, _, label_val in g.triples((s, SKOS.prefLabel, None)):
                label = str(label_val)
                break
        node_id = str(s)
        node = Node(
            id=node_id,
            lbl=label or os.path.basename(node_id),
            meta={
                "basicPropertyValues": [
                    {"pred": str(RDF.type), "val": str(SKOS.ConceptScheme)}
                ]
            },
        )
        obograph.nodes.append(node)

    # Process SKOS concepts as nodes
    for s, p, o in g.triples((None, RDF.type, SKOS.Concept)):
        label = None
        for _, _, label_val in g.triples((s, SKOS.prefLabel, None)):
            label = str(label_val)
            break
        if not label:
            for _, _, label_val in g.triples((s, RDFS.label, None)):
                label = str(label_val)
                break
        node_id = str(s)
        meta = {
            "basicPropertyValues": [{"pred": str(RDF.type), "val": str(SKOS.Concept)}]
        }
        for _, _, definition in g.triples((s, SKOS.definition, None)):
            meta["basicPropertyValues"].append(
                {"pred": str(SKOS.definition), "val": str(definition)}
            )
        for _, _, scheme in g.triples((s, SKOS.inScheme, None)):
            meta["basicPropertyValues"].append(
                {"pred": str(SKOS.inScheme), "val": str(scheme)}
            )
        node = Node(id=node_id, lbl=label or os.path.basename(node_id), meta=meta)
        obograph.nodes.append(node)

    # Process broader and narrower relationships
    for s, p, o in g.triples((None, SKOS.broader, None)):
        edge = Edge(sub=str(s), pred=str(SKOS.broader), obj=str(o))
        obograph.edges.append(edge)
    for s, p, o in g.triples((None, SKOS.narrower, None)):
        # Invert direction for narrower to broader
        edge = Edge(sub=str(o), pred=str(SKOS.broader), obj=str(s))
        obograph.edges.append(edge)

    logger.info(
        f"Created OboGraph with {len(obograph.nodes)} nodes and {len(obograph.edges)} edges"
    )
    return obograph


def process_ontology(graph: Graph, output_path: str) -> bool:
    """
    Process the ontology graph and convert to OWL.

    Parameters:
        graph (Graph): The OboGraph representation of the ontology.
        output_path (str): Path to save the final OWL ontology.

    Returns:
        bool: True if the process succeeds; otherwise False.
    """
    logger.info("Building ontology from graph")
    graph_dict = graph_as_dict(graph)
    logger.info(f"Building new ontology with ID: {ONTOLOGY_ID}")

    # Mappings for concept schemes and SKOS concepts to OWL classes
    concept_map = {}
    skos_concept_schemes = set()

    # Domain classes inspired by Biolink (with slight modifications)
    domain_classes = {
        "NamedThing": {
            "name": "NamedThing",
            "title": "Named Thing",
            "description": "A named entity in the materials science domain, a common superclass of all entities",
        },
        "Material": {
            "name": "Material",
            "title": "Material",
            "description": "Any material, substance, or compound in materials science",
            "is_a": "NamedThing",
            "id_prefixes": ["CHEBI", "PUBCHEM", "MESH", "MATERIAL"],
        },
        "Property": {
            "name": "Property",
            "title": "Property",
            "description": "Properties or characteristics of materials",
            "is_a": "NamedThing",
            "id_prefixes": ["PROP"],
        },
        "Structure": {
            "name": "Structure",
            "title": "Structure",
            "description": "Structural aspects of materials",
            "is_a": "NamedThing",
            "id_prefixes": ["STRUCT"],
        },
        "Method": {
            "name": "Method",
            "title": "Method",
            "description": "Processing methods, synthesis techniques, or characterization approaches",
            "is_a": "NamedThing",
            "id_prefixes": ["METHOD"],
        },
        "Application": {
            "name": "Application",
            "title": "Application",
            "description": "Applications or uses of materials",
            "is_a": "NamedThing",
            "id_prefixes": ["APP"],
        },
        "Instrument": {
            "name": "Instrument",
            "title": "Instrument",
            "description": "Devices or equipment used in material processing or characterization",
            "is_a": "NamedThing",
            "id_prefixes": ["INSTRUMENT"],
        },
    }

    # Material restrictions (with equivalent axioms)
    material_restrictions = {
        "Polymer": {
            "description": "A material composed of macromolecules with repeating structural units",
            "equivalent_to": {
                "description": "A Material with polymer structure",
                "items": [
                    {"class": f"{PREFIX}:Material"},
                    {
                        "property": f"{PREFIX}:has_structure",
                        "value": f"{PREFIX}:PolymerStructure",
                    },
                ],
            },
        },
        "Ceramic": {
            "description": "An inorganic, non-metallic solid material",
            "equivalent_to": {
                "description": "A Material with ceramic properties",
                "items": [
                    {"class": f"{PREFIX}:Material"},
                    {
                        "property": f"{PREFIX}:has_property",
                        "value": f"{PREFIX}:Hardness",
                    },
                    {
                        "property": f"{PREFIX}:has_property",
                        "value": f"{PREFIX}:BrittlenessToughness",
                    },
                ],
            },
        },
        "Semiconductor": {
            "description": "A material with electrical conductivity between conductors and insulators",
            "equivalent_to": {
                "description": "A Material with semiconductor properties",
                "items": [
                    {"class": f"{PREFIX}:Material"},
                    {
                        "property": f"{PREFIX}:has_property",
                        "value": f"{PREFIX}:ElectricalConductivity",
                    },
                    {
                        "property": f"{PREFIX}:has_property",
                        "value": f"{PREFIX}:BandGap",
                    },
                ],
            },
        },
    }

    # Process concept schemes
    logger.info("Processing concept schemes")
    for node in graph_dict.get("nodes", []):
        if "meta" in node and "basicPropertyValues" in node["meta"]:
            for prop_value in node["meta"]["basicPropertyValues"]:
                if (
                    prop_value.get("pred")
                    == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                    and prop_value.get("val")
                    == "http://www.w3.org/2004/02/skos/core#ConceptScheme"
                ):
                    skos_concept_schemes.add(node["id"])
                    scheme_label = node.get("lbl", os.path.basename(node["id"]))
                    class_id = scheme_label.title().replace(" ", "")
                    # Map to a domain class if possible
                    mapped = False
                    for domain_class in domain_classes.keys():
                        if domain_class.lower() in scheme_label.lower():
                            concept_map[node["id"]] = f"{PREFIX}:{domain_class}"
                            mapped = True
                            break
                    if not mapped:
                        concept_map[node["id"]] = f"{PREFIX}:{class_id}"
                    logger.debug(
                        f"Found concept scheme: {node['id']} -> {concept_map[node['id']]}"
                    )

    # Process SKOS concepts
    logger.info("Processing SKOS concepts")
    for node in graph_dict.get("nodes", []):
        if node["id"] in concept_map:
            continue
        is_concept = False
        in_scheme = None
        if "meta" in node and "basicPropertyValues" in node["meta"]:
            for prop_value in node["meta"]["basicPropertyValues"]:
                if (
                    prop_value.get("pred")
                    == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                    and prop_value.get("val")
                    == "http://www.w3.org/2004/02/skos/core#Concept"
                ):
                    is_concept = True
                if (
                    prop_value.get("pred")
                    == "http://www.w3.org/2004/02/skos/core#inScheme"
                ):
                    in_scheme = prop_value.get("val")
        if is_concept:
            label = node.get("lbl", os.path.basename(node["id"]))
            # Create a valid class ID from the label
            class_id = "".join(
                c if c.isalnum() or c == "_" else "_" for c in label.replace(" ", "_")
            )
            if not class_id or not class_id[0].isalpha():
                class_id = f"X{class_id}"
            concept_map[node["id"]] = f"{PREFIX}:{class_id}"
            logger.debug(
                f"Processed concept: {node['id']} -> {class_id} (in scheme: {in_scheme})"
            )

    # Build OWL output via a LinkML-style metaontology
    logger.info("Building OWL ontology")
    meta_onto = {
        "id": PREFIX,
        "name": ONTOLOGY_ID,  # Valid NCName
        "title": ONTOLOGY_TITLE,
        "description": "A materials science ontology converted from SKOS thesaurus",
        "prefixes": {
            PREFIX: BASE_IRI,
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "oio": "http://www.geneontology.org/formats/oboInOwl#",
            "dcterms": "http://purl.org/dc/terms/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        "default_prefix": PREFIX,
        "classes": {},
        "types": {
            "SymmetricProperty": {"base": "string", "uri": "owl:SymmetricProperty"},
            "TransitiveProperty": {"base": "string", "uri": "owl:TransitiveProperty"},
            "FunctionalProperty": {"base": "string", "uri": "owl:FunctionalProperty"},
            "InverseFunctionalProperty": {
                "base": "string",
                "uri": "owl:InverseFunctionalProperty",
            },
        },
        # Slots (properties) defined with Biolink-inspired relationships.
        "slots": {
            "related_to": {
                "description": "A relationship that exists between two entities",
                "slot_uri": "owl:ObjectProperty",
                "domain": f"{PREFIX}:NamedThing",
                "range": f"{PREFIX}:NamedThing",
                "symmetric": False,
            },
            "physically_related_to": {
                "is_a": "related_to",
                "description": "A relationship where the entities are physically related",
                "slot_uri": "owl:ObjectProperty",
            },
            "informationally_related_to": {
                "is_a": "related_to",
                "description": "A relationship where there is an informational connection",
                "slot_uri": "owl:ObjectProperty",
            },
            "derives_from": {
                "is_a": "related_to",
                "description": "Indicates that one entity is derived from another",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "has_derivative",
            },
            "interacts_with": {
                "is_a": "physically_related_to",
                "description": "Indicates that entities interact with each other",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": True,
            },
            "affects": {
                "is_a": "related_to",
                "description": "Indicates that one entity affects another",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "affected_by",
            },
            "has_property": {
                "is_a": "related_to",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Property",
                "description": "Relates a material to its properties",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "is_property_of",
            },
            "is_property_of": {
                "is_a": "related_to",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Material",
                "description": "Relates a property to the material it belongs to",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "has_property",
            },
            "has_structure": {
                "is_a": "physically_related_to",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Structure",
                "description": "Relates a material to its structural aspects",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "is_structure_of",
            },
            "is_structure_of": {
                "is_a": "physically_related_to",
                "domain": f"{PREFIX}:Structure",
                "range": f"{PREFIX}:Material",
                "description": "Relates a structure to the material it belongs to",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "has_structure",
            },
            "processed_by": {
                "is_a": "physically_related_to",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Method",
                "description": "Relates a material to a processing or synthesis method",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "processes",
            },
            "processes": {
                "is_a": "physically_related_to",
                "domain": f"{PREFIX}:Method",
                "range": f"{PREFIX}:Material",
                "description": "Relates a method to the material it processes",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "processed_by",
            },
            "used_in": {
                "is_a": "related_to",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Application",
                "description": "Relates a material to its applications",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "uses",
            },
            "uses": {
                "is_a": "related_to",
                "domain": f"{PREFIX}:Application",
                "range": f"{PREFIX}:Material",
                "description": "Relates an application to the materials it uses",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "used_in",
            },
            "characterizes": {
                "is_a": "informationally_related_to",
                "domain": f"{PREFIX}:Method",
                "range": f"{PREFIX}:Property",
                "description": "Relates a characterization method to the property it measures",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "characterized_by",
            },
            "characterized_by": {
                "is_a": "informationally_related_to",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Method",
                "description": "Relates a property to the methods used to characterize it",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "characterizes",
            },
            "modifies": {
                "is_a": "affects",
                "domain": f"{PREFIX}:Method",
                "range": f"{PREFIX}:Property",
                "description": "Relates a processing method to the properties it modifies",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "modified_by",
            },
            "modified_by": {
                "is_a": "affects",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Method",
                "description": "Relates a property to the processing methods that modify it",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "modifies",
            },
            "requires_instrument": {
                "is_a": "related_to",
                "domain": f"{PREFIX}:Method",
                "range": f"{PREFIX}:Instrument",
                "description": "Relates a method to the instruments required to perform it",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "used_by_method",
            },
            "used_by_method": {
                "is_a": "related_to",
                "domain": f"{PREFIX}:Instrument",
                "range": f"{PREFIX}:Method",
                "description": "Relates an instrument to the methods that use it",
                "slot_uri": "owl:ObjectProperty",
                "inverse": "requires_instrument",
            },
            "has_component": {
                "is_a": "physically_related_to",
                "domain": f"{PREFIX}:Structure",
                "range": f"{PREFIX}:Structure",
                "description": "Relates a structure to its component structures",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "transitive": True,
                "inverse": "part_of",
            },
            "part_of": {
                "is_a": "physically_related_to",
                "domain": f"{PREFIX}:Structure",
                "range": f"{PREFIX}:Structure",
                "description": "Relates a component structure to its parent structure",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "transitive": True,
                "inverse": "has_component",
            },
            "forms_composite_with": {
                "is_a": "interacts_with",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Material",
                "description": "Relates a material to other materials forming a composite",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": True,
            },
            "chemically_reacts_with": {
                "is_a": "interacts_with",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Material",
                "description": "Relates a material to others with which it reacts chemically",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": True,
            },
            "dissolves_in": {
                "is_a": "interacts_with",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Material",
                "description": "Relates a material to a solvent in which it dissolves",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "inverse": "dissolves",
            },
            "dissolves": {
                "is_a": "interacts_with",
                "domain": f"{PREFIX}:Material",
                "range": f"{PREFIX}:Material",
                "description": "Relates a solvent to materials it can dissolve",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "inverse": "dissolves_in",
            },
            "correlates_with": {
                "is_a": "informationally_related_to",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Property",
                "description": "Relates a property to another property with which it correlates",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": True,
            },
            "increases": {
                "is_a": "affects",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Property",
                "description": "Relates a property to another property which it increases",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "inverse": "increased_by",
            },
            "increased_by": {
                "is_a": "affects",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Property",
                "description": "Relates a property to another property which increases it",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "inverse": "increases",
            },
            "decreases": {
                "is_a": "affects",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Property",
                "description": "Relates a property to another property which it decreases",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "inverse": "decreased_by",
            },
            "decreased_by": {
                "is_a": "affects",
                "domain": f"{PREFIX}:Property",
                "range": f"{PREFIX}:Property",
                "description": "Relates a property to another property which decreases it",
                "slot_uri": "owl:ObjectProperty",
                "symmetric": False,
                "inverse": "decreases",
            },
            "has_value": {
                "domain": f"{PREFIX}:Property",
                "range": "xsd:float",
                "description": "Relates a property to its numerical value",
                "slot_uri": "owl:DatatypeProperty",
            },
            "has_unit": {
                "domain": f"{PREFIX}:Property",
                "range": "xsd:string",
                "description": "Relates a property to its measurement unit",
                "slot_uri": "owl:DatatypeProperty"
                # Removed unsupported "functional": True key
            },
            "has_description": {
                "domain": f"{PREFIX}:NamedThing",
                "range": "xsd:string",
                "description": "A textual description of an entity",
                "slot_uri": "owl:DatatypeProperty",
            },
            "has_formula": {
                "domain": f"{PREFIX}:Material",
                "range": "xsd:string",
                "description": "The chemical formula of a material",
                "slot_uri": "owl:DatatypeProperty",
            },
        },
        "rules": [
            {
                "description": "Every Material has at least one Property",
                "rule": "Material(?m) -> exists(?p) (Property(?p) && has_property(?m, ?p))",
            },
            {
                "description": "If a Method modifies a Property and a Material has that Property, then the Method can be used on the Material",
                "rule": "Method(?mth) && modifies(?mth, ?p) && Material(?m) && has_property(?m, ?p) -> can_be_used_on(?mth, ?m)",
            },
            {
                "description": "If a Material has certain Property values, it can be classified as a specific type",
                "rule": "Material(?m) && has_property(?m, ?p) && Property(?p) && has_value(?p, ?v) && greaterThan(?v, threshold) -> MaterialType(?m)",
            },
        ],
    }

    # Add domain classes
    for class_id, class_data in domain_classes.items():
        meta_onto["classes"][class_id] = class_data

    # Add disjointness axioms between top-level classes
    disjoint_classes = [
        ("Material", "Property"),
        ("Material", "Method"),
        ("Material", "Structure"),
        ("Material", "Application"),
        ("Property", "Method"),
        ("Property", "Structure"),
        ("Property", "Application"),
        ("Method", "Structure"),
        ("Method", "Application"),
        ("Structure", "Application"),
    ]
    for class1, class2 in disjoint_classes:
        if class1 in meta_onto["classes"] and class2 in meta_onto["classes"]:
            meta_onto["classes"][class1].setdefault("disjoint_with", []).append(
                f"{PREFIX}:{class2}"
            )

    # Add material restrictions (equivalent axioms)
    for material_id, material_data in material_restrictions.items():
        if material_id not in meta_onto["classes"]:
            meta_onto["classes"][material_id] = {
                "name": material_id,
                "title": material_id,
                "description": material_data["description"],
                "is_a": f"{PREFIX}:Material",
            }
            if "equivalent_to" in material_data:
                meta_onto["classes"][material_id]["equivalent_to"] = material_data[
                    "equivalent_to"
                ]

    # Process nodes (SKOS concepts) to add as classes
    for node in graph_dict.get("nodes", []):
        if node["id"] in concept_map:
            class_ref = concept_map[node["id"]]
            class_id = class_ref.split(":")[1]
            if class_id in domain_classes:
                continue
            class_data = {"name": class_id, "description": ""}
            if "lbl" in node:
                class_data["title"] = node["lbl"]
            if "meta" in node and "basicPropertyValues" in node["meta"]:
                for prop_value in node["meta"]["basicPropertyValues"]:
                    if (
                        prop_value.get("pred")
                        == "http://www.w3.org/2004/02/skos/core#definition"
                    ):
                        class_data["description"] = prop_value.get("val", "")
                    if (
                        prop_value.get("pred")
                        == "http://www.w3.org/2004/02/skos/core#inScheme"
                        and prop_value.get("val") in concept_map
                    ):
                        parent_class = concept_map[prop_value.get("val")]
                        if "is_a" not in class_data:
                            class_data["is_a"] = parent_class
            if "is_a" not in class_data:
                best_match = None
                for domain_class in domain_classes.keys():
                    if domain_class.lower() in class_data.get("title", "").lower():
                        best_match = domain_class
                        break
                class_data["is_a"] = (
                    f"{PREFIX}:{best_match}" if best_match else f"{PREFIX}:Material"
                )
            meta_onto["classes"][class_id] = class_data

    # Process edges to add subclass relationships from broader relationships
    for edge in graph_dict.get("edges", []):
        if (
            edge.get("pred") == "http://www.w3.org/2004/02/skos/core#broader"
            and edge.get("sub") in concept_map
            and edge.get("obj") in concept_map
        ):
            sub_class_id = concept_map[edge.get("sub")].split(":")[1]
            obj_class_id = concept_map[edge.get("obj")].split(":")[1]
            obj_class_id
            if sub_class_id in meta_onto["classes"]:
                if "is_a" not in meta_onto["classes"][sub_class_id]:
                    meta_onto["classes"][sub_class_id]["is_a"] = concept_map[
                        edge.get("obj")
                    ]

    # Save the metaontology as YAML
    yaml_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_meta.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(meta_onto, f, sort_keys=False)
    logger.info(f"Saved metaontology to {yaml_path}")

    # Convert to OWL using linkml-owl (gen-owl)
    try:
        logger.info("Converting to OWL using linkml-owl")
        subprocess.run(["gen-owl", "--output", output_path, yaml_path], check=True)
        logger.info(f"OWL ontology generated: {output_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to convert using linkml-owl: {e}")
        logger.info(
            "Saving as JSON instead - process with your preferred OWL generator"
        )
        json_path = os.path.join(OUTPUT_DIR, f"{PREFIX}.json")
        with open(json_path, "w") as f:
            json.dump(meta_onto, f, indent=2)

        # Fallback conversion using ROBOT with a basic OWL file built via RDFLib
        try:
            logger.info("Attempting fallback conversion with ROBOT")
            temp_owl = os.path.join(OUTPUT_DIR, "temp_basic.owl")
            g = rdflib.Graph()
            ontology_iri = rdflib.URIRef(BASE_IRI)
            RDF = rdflib.namespace.RDF
            OWL = rdflib.namespace.OWL
            RDFS = rdflib.namespace.RDFS
            XSD = rdflib.namespace.XSD
            NS = rdflib.Namespace(BASE_IRI)

            # Declare ontology
            g.add((ontology_iri, RDF.type, OWL.Ontology))
            g.add((ontology_iri, RDFS.label, rdflib.Literal(ONTOLOGY_TITLE)))
            g.add(
                (ontology_iri, RDFS.comment, rdflib.Literal(meta_onto["description"]))
            )

            # Add classes from metaontology
            for class_id, class_data in meta_onto["classes"].items():
                class_iri = NS[class_id]
                g.add((class_iri, RDF.type, OWL.Class))
                g.add(
                    (
                        class_iri,
                        RDFS.label,
                        rdflib.Literal(class_data.get("title", class_id)),
                    )
                )
                g.add(
                    (
                        class_iri,
                        RDFS.comment,
                        rdflib.Literal(class_data.get("description", "")),
                    )
                )
                if "is_a" in class_data:
                    parent_id = class_data["is_a"].split(":")[1]
                    parent_iri = NS[parent_id]
                    g.add((class_iri, RDFS.subClassOf, parent_iri))
                if "disjoint_with" in class_data:
                    for disjoint_class in class_data["disjoint_with"]:
                        disjoint_id = disjoint_class.split(":")[1]
                        disjoint_iri = NS[disjoint_id]
                        g.add((class_iri, OWL.disjointWith, disjoint_iri))
                if (
                    "equivalent_to" in class_data
                    and "items" in class_data["equivalent_to"]
                ):
                    intersection = rdflib.BNode()
                    g.add((intersection, RDF.type, OWL.Class))
                    g.add((class_iri, OWL.equivalentClass, intersection))
                    # Build RDF collection using rdflib.collection.Collection
                    from rdflib.collection import Collection

                    items = []
                    for item in class_data["equivalent_to"]["items"]:
                        if "class" in item:
                            class_ref_id = item["class"].split(":")[1]
                            items.append(NS[class_ref_id])
                        elif "property" in item and "value" in item:
                            restriction = rdflib.BNode()
                            g.add((restriction, RDF.type, OWL.Restriction))
                            prop_id = item["property"].split(":")[1]
                            prop_iri = NS[prop_id]
                            value_id = item["value"].split(":")[1]
                            value_iri = NS[value_id]
                            g.add((restriction, OWL.onProperty, prop_iri))
                            g.add((restriction, OWL.someValuesFrom, value_iri))
                            items.append(restriction)
                    if items:
                        list_bnode = rdflib.BNode()
                        Collection(g, list_bnode, items)
                        g.add((intersection, OWL.intersectionOf, list_bnode))

            # Add properties from metaontology
            for prop_id, prop_data in meta_onto["slots"].items():
                if (
                    "slot_uri" in prop_data
                    and prop_data["slot_uri"] == "owl:DatatypeProperty"
                ):
                    prop_iri = NS[prop_id]
                    g.add((prop_iri, RDF.type, OWL.DatatypeProperty))
                else:
                    prop_iri = NS[prop_id]
                    g.add((prop_iri, RDF.type, OWL.ObjectProperty))
                g.add((prop_iri, RDFS.label, rdflib.Literal(prop_id)))
                if "description" in prop_data:
                    g.add(
                        (
                            prop_iri,
                            RDFS.comment,
                            rdflib.Literal(prop_data["description"]),
                        )
                    )
                if "domain" in prop_data:
                    domain_id = prop_data["domain"].split(":")[1]
                    domain_iri = NS[domain_id]
                    g.add((prop_iri, RDFS.domain, domain_iri))
                if "range" in prop_data:
                    if ":" in prop_data["range"]:
                        range_id = prop_data["range"].split(":")[1]
                        range_iri = NS[range_id]
                        g.add((prop_iri, RDFS.range, range_iri))
                    elif prop_data["range"] == "xsd:float":
                        g.add((prop_iri, RDFS.range, XSD.float))
                    elif prop_data["range"] == "xsd:string":
                        g.add((prop_iri, RDFS.range, XSD.string))
                if prop_data.get("transitive"):
                    g.add((prop_iri, RDF.type, OWL.TransitiveProperty))
                if prop_data.get("symmetric"):
                    g.add((prop_iri, RDF.type, OWL.SymmetricProperty))
                if prop_data.get("functional"):
                    g.add((prop_iri, RDF.type, OWL.FunctionalProperty))
                if "inverse" in prop_data:
                    inverse_id = (
                        prop_data["inverse"].split(":")[1]
                        if ":" in prop_data["inverse"]
                        else prop_data["inverse"]
                    )
                    inverse_iri = NS[inverse_id]
                    g.add((prop_iri, OWL.inverseOf, inverse_iri))

            g.serialize(destination=temp_owl, format="xml")
            subprocess.run(
                ["robot", "convert", "--input", temp_owl, "--output", output_path],
                check=True,
            )
            logger.info(f"Basic OWL file generated with ROBOT: {output_path}")
            if os.path.exists(temp_owl):
                os.remove(temp_owl)
            return True

        except Exception as robot_err:
            logger.error(f"Fallback OWL generation with ROBOT failed: {robot_err}")
            logger.info(
                f"Consider running manual conversion from the JSON file: {json_path}"
            )
            return False


def postprocess_with_robot(owl_path: str) -> None:
    """
    Post-process the OWL ontology with ROBOT for reasoning, conversion, reporting, and documentation.

    Parameters:
        owl_path (str): Path to the OWL ontology file.
    """
    logger.info(f"Post-processing ontology with ROBOT: {owl_path}")
    processed_dir = os.path.join(OUTPUT_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(owl_path))[0]

    # Reasoning step
    reasoned_path = os.path.join(processed_dir, f"{base_name}_reasoned.owl")
    try:
        logger.info("Running ROBOT reason to check consistency")
        subprocess.run(
            [
                "robot",
                "reason",
                "--input",
                owl_path,
                "--reasoner",
                "ELK",
                "--output",
                reasoned_path,
            ],
            check=True,
        )
        logger.info(f"Reasoning completed: {reasoned_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Reasoning failed: {e}")

    # OBO conversion
    obo_path = os.path.join(processed_dir, f"{base_name}.obo")
    try:
        logger.info("Converting to OBO format")
        subprocess.run(
            [
                "robot",
                "convert",
                "--input",
                reasoned_path if os.path.exists(reasoned_path) else owl_path,
                "--output",
                obo_path,
                "--format",
                "obo",
            ],
            check=True,
        )
        logger.info(f"OBO format generated: {obo_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"OBO conversion failed: {e}")

    # Generate validation report
    report_path = os.path.join(processed_dir, f"{base_name}_report.tsv")
    try:
        logger.info("Generating validation report")
        subprocess.run(
            [
                "robot",
                "report",
                "--input",
                reasoned_path if os.path.exists(reasoned_path) else owl_path,
                "--output",
                report_path,
                "--format",
                "tsv",
            ],
            check=True,
        )
        logger.info(f"Validation report generated: {report_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Report generation failed: {e}")

    # Generate documentation (adjust header to remove unsupported "TYPE" column)
    docs_path = os.path.join(processed_dir, f"{base_name}_docs")
    os.makedirs(docs_path, exist_ok=True)
    try:
        logger.info("Generating ontology documentation")
        subprocess.run(
            [
                "robot",
                "export",
                "--input",
                reasoned_path if os.path.exists(reasoned_path) else owl_path,
                "--header",
                "ID|LABEL|DEFINITION",
                "--export",
                os.path.join(docs_path, "entities.csv"),
            ],
            check=True,
        )
        logger.info(f"Documentation generated: {docs_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Documentation generation failed: {e}")


def main():
    """Main execution function."""
    temp_owl = os.path.join(OUTPUT_DIR, "temp_robot_output.owl")
    final_owl = os.path.join(OUTPUT_DIR, f"{PREFIX}.owl")

    # Step 1: Attempt initial ROBOT conversion (optional)
    robot_success = convert_with_robot(INPUT_SKOS, temp_owl)

    # Step 2: Build OboGraph from SKOS and process the ontology
    try:
        logger.info("Building OboGraph directly from SKOS")
        graph = skos_to_obograph(INPUT_SKOS)
        process_success = process_ontology(graph, final_owl)
        logger.info(f"Processing completed: {process_success}")
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        return

    if robot_success and os.path.exists(temp_owl):
        try:
            os.remove(temp_owl)
        except Exception:
            pass

    # Step 3: Run post-processing with ROBOT if OWL was generated
    if os.path.exists(final_owl):
        logger.info(f"Conversion complete. Output saved to {final_owl}")
        try:
            postprocess_with_robot(final_owl)
        except Exception as e:
            logger.error(f"Error in post-processing: {e}", exc_info=True)
    else:
        json_path = os.path.join(OUTPUT_DIR, f"{PREFIX}.json")
        if os.path.exists(json_path):
            logger.info(f"OWL conversion failed, but JSON is available at: {json_path}")
            logger.info(
                "You can convert this to OWL manually using your preferred method"
            )
        else:
            logger.error("Conversion failed. No output generated.")


if __name__ == "__main__":
    main()
