"""
Phase 1: Initial Ontology Development with LLM Support
-----------------------------------------------------
This module handles the creation and management of the OPV ontology.
"""

import logging
from typing import Dict, Any
import rdflib
from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

# LlamaIndex imports
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)


class OntologyDevelopment:
    """Handles the development of the initial OPV ontology."""

    def __init__(
        self,
        llm_model: str = "mistral",
        temperature: float = 0.1,
        data_dir: str = "./data",
        persist_dir: str = "./storage",
        debug: bool = True,
        ollama_base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        """Initialize the ontology development phase."""
        self.persist_dir = persist_dir
        self.debug = debug

        # Initialize LLM
        self.llm = Ollama(
            model=llm_model,
            temperature=temperature,
            base_url=ollama_base_url,
        )

        # Set up callback manager for debugging
        self.callback_manager = (
            CallbackManager([LlamaDebugHandler()]) if debug else None
        )

        # Configure global settings with the LLM
        Settings.llm = self.llm
        Settings.callback_manager = self.callback_manager

        # Initialize ontology graph
        self.ontology = rdflib.Graph()
        self.opv_ns = Namespace("http://opv-kg.org/ontology/")
        self._initialize_base_ontology()

        # Setup ontology agent
        self.agent = self._create_ontology_agent()

        # Initial domain concepts for OPV
        self.initial_concepts = [
            {
                "name": "Polymer",
                "description": "A large molecule composed of repeating structural units",
                "parent": "Material",
            },
            {
                "name": "P3HT",
                "description": "Poly(3-hexylthiophene), a semiconducting polymer commonly used in OPVs",
                "parent": "Polymer",
            },
            {
                "name": "PCBM",
                "description": "Phenyl-C61-butyric acid methyl ester, "
                "a fullerene derivative commonly used as an electron acceptor in OPVs",
                "parent": "Material",
            },
            {
                "name": "BulkHeterojunction",
                "description": "A mixed layer of donor and acceptor materials with a high interfacial area",
                "parent": "DeviceArchitecture",
            },
            {
                "name": "DomainOrientation",
                "description": "The alignment direction of polymer domains within a film",
                "parent": "MorphologicalProperty",
            },
            {
                "name": "ProcessingMethod",
                "description": "A technique used to create or modify a material or device",
                "parent": None,
            },
        ]

        # Initial relationships between concepts
        self.initial_relationships = [
            {
                "source": "P3HT",
                "relation": "has_property",
                "target": "DomainOrientation",
                "description": "P3HT forms domains with specific orientations that affect device performance",
            },
            {
                "source": "ProcessingMethod",
                "relation": "affects",
                "target": "DomainOrientation",
                "description": "Processing methods can influence the orientation of P3HT domains",
            },
            {
                "source": "BulkHeterojunction",
                "relation": "contains_material",
                "target": "P3HT",
                "description": "P3HT is commonly used as the donor material in bulk heterojunction OPVs",
            },
            {
                "source": "BulkHeterojunction",
                "relation": "contains_material",
                "target": "PCBM",
                "description": "PCBM is commonly used as the acceptor material in bulk heterojunction OPVs",
            },
        ]

    def _initialize_base_ontology(self):
        """Initialize the base OPV ontology with fundamental concepts."""
        # Bind namespaces
        self.ontology.bind("opv", self.opv_ns)
        self.ontology.bind("rdf", RDF)
        self.ontology.bind("rdfs", RDFS)
        self.ontology.bind("owl", OWL)

        # Create base classes for OPV ontology
        material = URIRef(self.opv_ns + "Material")
        polymer = URIRef(self.opv_ns + "Polymer")
        device = URIRef(self.opv_ns + "Device")
        property = URIRef(self.opv_ns + "Property")
        experiment = URIRef(self.opv_ns + "Experiment")
        device_architecture = URIRef(self.opv_ns + "DeviceArchitecture")
        morphological_property = URIRef(self.opv_ns + "MorphologicalProperty")

        # Add classes to ontology
        self.ontology.add((material, RDF.type, OWL.Class))
        self.ontology.add((polymer, RDF.type, OWL.Class))
        self.ontology.add((device, RDF.type, OWL.Class))
        self.ontology.add((property, RDF.type, OWL.Class))
        self.ontology.add((experiment, RDF.type, OWL.Class))
        self.ontology.add((device_architecture, RDF.type, OWL.Class))
        self.ontology.add((morphological_property, RDF.type, OWL.Class))

        # Add subclass relationships
        self.ontology.add((polymer, RDFS.subClassOf, material))
        self.ontology.add((device_architecture, RDFS.subClassOf, device))
        self.ontology.add((morphological_property, RDFS.subClassOf, property))

        # Save base ontology
        self.save_ontology()

        logger.info("Base OPV ontology initialized.")

    def _create_ontology_agent(self):
        """Create an agent specialized in ontology development and refinement."""
        tools = []

        # Tool for suggesting new concepts
        suggest_concept_tool = FunctionTool.from_defaults(
            fn=self.suggest_ontology_concept,
            name="suggest_ontology_concept",
            description="Suggests new concepts to add to the OPV ontology based on domain knowledge.",
        )
        tools.append(suggest_concept_tool)

        # Tool for adding relationships
        add_relationship_tool = FunctionTool.from_defaults(
            fn=self.add_ontology_relationship,
            name="add_ontology_relationship",
            description="Adds a relationship between two existing concepts in the ontology.",
        )
        tools.append(add_relationship_tool)

        # Create the agent with the tools
        ontology_agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="""You are an expert in organic photovoltaics (OPVs) and ontology development.
            Your task is to build and refine an ontology for the OPV domain, focusing on materials,
            properties, device architectures, and experimental techniques. Think carefully about
            the hierarchical relationships between concepts and ensure the ontology is both
            comprehensive and logically consistent. Consider concepts like P3HT domains,
            polymer orientation, and processing effects.""",
            max_iterations=20,
        )

        return ontology_agent

    def suggest_ontology_concept(
        self, concept_name: str, description: str, parent_concept: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Suggests a new concept to add to the OPV ontology."""
        # Create a URI for the concept
        concept_uri = URIRef(self.opv_ns + concept_name.replace(" ", "_"))

        # Add the concept to the ontology
        self.ontology.add((concept_uri, RDF.type, OWL.Class))
        self.ontology.add((concept_uri, RDFS.label, Literal(concept_name)))
        self.ontology.add((concept_uri, RDFS.comment, Literal(description)))

        # Add parent relationship if provided
        if parent_concept:
            parent_uri = URIRef(self.opv_ns + parent_concept.replace(" ", "_"))
            self.ontology.add((concept_uri, RDFS.subClassOf, parent_uri))

        # Create a KGCL change description
        kgcl_change = f'CREATE CLASS {concept_name} DESCRIPTION "{description}"'
        if parent_concept:
            kgcl_change += f" SUBCLASS OF {parent_concept}"

        # Save the updated ontology
        self.save_ontology()

        logger.info(
            "Ontology concept '%s' added (parent: %s)", concept_name, parent_concept
        )

        return {
            "concept_added": concept_name,
            "uri": str(concept_uri),
            "parent": parent_concept,
            "kgcl_change": kgcl_change,
            "status": "success",
        }

    def add_ontology_relationship(
        self,
        source_concept: str,
        relation_type: str,
        target_concept: str,
        description: str = None,
    ) -> Dict[str, Any]:
        """Adds a relationship between two existing concepts in the ontology."""
        # Create URIs for the concepts and relation
        source_uri = URIRef(self.opv_ns + source_concept.replace(" ", "_"))
        target_uri = URIRef(self.opv_ns + target_concept.replace(" ", "_"))
        relation_uri = URIRef(self.opv_ns + relation_type.replace(" ", "_"))

        # Add the relation to the ontology
        self.ontology.add((relation_uri, RDF.type, OWL.ObjectProperty))
        if description:
            self.ontology.add((relation_uri, RDFS.comment, Literal(description)))
        self.ontology.add((source_uri, relation_uri, target_uri))

        # Create a KGCL change description
        kgcl_change = (
            f"ADD RELATIONSHIP {source_concept} {relation_type} {target_concept}"
        )

        # Save the updated ontology
        self.save_ontology()

        logger.info("Ontology relationship added: %s", kgcl_change)

        return {
            "relationship_added": f"{source_concept} -> {relation_type} -> {target_concept}",
            "kgcl_change": kgcl_change,
            "status": "success",
        }

    def save_ontology(self, version: str = "current"):
        """Save the ontology to disk in RDF/XML format."""
        file_path = f"{self.persist_dir}/ontology/opv_ontology_{version}.owl"
        self.ontology.serialize(destination=file_path, format="xml")
        logger.info("Ontology saved to %s", file_path)

        # Also save in Turtle format for human readability
        turtle_path = f"{self.persist_dir}/ontology/opv_ontology_{version}.ttl"
        self.ontology.serialize(destination=turtle_path, format="turtle")
        logger.info("Ontology also saved in Turtle format to %s", turtle_path)

        return file_path

    def get_ontology(self):
        """Return the current ontology object."""
        return self.ontology

    def run(self) -> Dict[str, Any]:
        """Run the ontology development phase."""
        logger.info("Running ontology development phase")

        # Add initial concepts to the ontology
        concept_results = []
        for concept in self.initial_concepts:
            result = self.suggest_ontology_concept(
                concept_name=concept["name"],
                description=concept["description"],
                parent_concept=concept.get("parent"),
            )
            concept_results.append(result)

        # Add initial relationships to the ontology
        relationship_results = []
        for rel in self.initial_relationships:
            result = self.add_ontology_relationship(
                source_concept=rel["source"],
                relation_type=rel["relation"],
                target_concept=rel["target"],
                description=rel.get("description"),
            )
            relationship_results.append(result)

        # Save the final ontology for this phase
        ontology_path = self.save_ontology("phase1")

        return {
            "status": "completed",
            "phase": "ontology_development",
            "concepts_added": len(concept_results),
            "relationships_added": len(relationship_results),
            "ontology_path": ontology_path,
        }
