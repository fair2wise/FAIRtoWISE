"""
Phase 2: Publication Analysis with LLM Support and Ontology Enrichment
--------------------------------------------------------------------
This module handles the extraction of information from scientific publications
and enriches the ontology with newly discovered concepts.
"""

import os
import json
import logging
import hashlib
import re
from typing import Dict, Any, List

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import KnowledgeGraphIndex

# RDFLib imports
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

logger = logging.getLogger(__name__)


class PublicationAnalysis:
    """Handles the analysis of scientific publications for knowledge extraction and ontology enrichment."""

    def __init__(
        self,
        llm_model: str = "llama3.2:latest",
        temperature: float = 0.1,
        data_dir: str = "./",
        persist_dir: str = "./storage",
        debug: bool = True,
        ollama_base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        """Initialize the publication analysis phase."""
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.debug = debug

        # Initialize LLM
        self.llm = Ollama(
            model=llm_model,
            temperature=temperature,
            base_url=ollama_base_url,
            request_timeout=240.0,
        )

        # Setup callback manager for debugging
        self.callback_manager = (
            CallbackManager([LlamaDebugHandler()]) if debug else None
        )

        # Configure global settings
        Settings.llm = self.llm
        Settings.callback_manager = self.callback_manager

        # Initialize node parser with scientific document settings
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=4096, chunk_overlap=200  # Smaller chunks for scientific content
        )

        # Initialize graph store for knowledge graph
        self.graph_store = SimpleGraphStore()
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )

        # Initialize publications list
        self.publications = []

        # Initialize indices
        self.vector_index = None
        self.kg_index = None

        # Initialize ontology (will be set from Phase 1)
        self.ontology = None

        # Cache of existing ontology concepts for quick lookup
        self.existing_concepts = set()

        # Setup publication agent
        self.agent = self._create_publication_agent()

        # Setup validation agent
        self.validation_agent = self._create_validation_agent()

        # Setup KG agent
        self.kg_agent = self._create_kg_agent()

        # Setup ontology enrichment agent
        self.ontology_agent = self._create_ontology_enrichment_agent()

    def _create_publication_agent(self):
        """Create an agent specialized in extracting information from publications."""
        tools = []

        # Tool for extracting material properties
        extract_materials_tool = FunctionTool.from_defaults(
            fn=self.extract_material_properties,
            name="extract_material_properties",
            description="Extracts material properties and descriptions from scientific papers.",
        )
        tools.append(extract_materials_tool)

        # Tool for extracting experimental methods
        extract_methods_tool = FunctionTool.from_defaults(
            fn=self.extract_experimental_methods,
            name="extract_experimental_methods",
            description="Extracts experimental methods and procedures from scientific papers.",
        )
        tools.append(extract_methods_tool)

        # Tool for extracting device architecture
        extract_device_tool = FunctionTool.from_defaults(
            fn=self.extract_device_architecture,
            name="extract_device_architecture",
            description="Extracts device architecture information from scientific papers.",
        )
        tools.append(extract_device_tool)

        # Create the agent
        publication_agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="""You are an expert in analyzing scientific publications in the field of
            organic photovoltaics (OPVs). Your task is to extract structured information from
            research papers, including material properties, experimental methods, and device
            architectures. Focus on identifying entities, relationships, and technical parameters
            that can be mapped to an OPV ontology. Be precise in your extraction and avoid hallucination.""",
        )

        return publication_agent

    def _create_validation_agent(self):
        """Create an agent specialized in human-in-the-loop validation."""
        tools = []

        # Tool for validating extractions
        validate_extraction_tool = FunctionTool.from_defaults(
            fn=self.validate_extraction,
            name="validate_extraction",
            description="Presents extracted information to a domain expert for validation.",
        )
        tools.append(validate_extraction_tool)

        # Create the agent
        validation_agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="""You are a facilitator for human-in-the-loop validation of an organic
            photovoltaics (OPVs) knowledge system. Your task is to present extracted information
            to domain experts in a clear and understandable format. You should incorporate their
            feedback to improve the system and ensure scientific accuracy.""",
        )

        return validation_agent

    def _create_kg_agent(self):
        """Create an agent specialized in knowledge graph operations."""
        tools = []

        # Tool for adding nodes to KG
        add_node_tool = FunctionTool.from_defaults(
            fn=self.add_kg_node,
            name="add_kg_node",
            description="Adds a new node to the knowledge graph.",
        )
        tools.append(add_node_tool)

        # Tool for adding relations to KG
        add_relation_tool = FunctionTool.from_defaults(
            fn=self.add_kg_relation,
            name="add_kg_relation",
            description="Adds a new relationship between nodes in the knowledge graph.",
        )
        tools.append(add_relation_tool)

        # Create the agent
        kg_agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="""You are an expert in knowledge graph construction and management for
            organic photovoltaics (OPVs). Your task is to build, maintain, and query a knowledge
            graph that represents the domain knowledge extracted from publications.""",
        )

        return kg_agent

    def _create_ontology_enrichment_agent(self):
        """Create an agent specialized in ontology enrichment."""
        tools = []

        # Tool for adding new concepts to ontology
        add_concept_tool = FunctionTool.from_defaults(
            fn=self.add_new_concept_to_ontology,
            name="add_new_concept_to_ontology",
            description="Adds a new concept to the OPV ontology based on extracted information.",
        )
        tools.append(add_concept_tool)

        # Tool for classifying a new concept
        classify_concept_tool = FunctionTool.from_defaults(
            fn=self.classify_concept,
            name="classify_concept",
            description="Classifies a new concept and identifies its parent concept in the ontology.",
        )
        tools.append(classify_concept_tool)

        # Create the agent
        ontology_agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="""You are an expert in organic photovoltaics (OPVs) and ontology development.
            Your task is to analyze extracted information from scientific papers and identify new concepts
            that should be added to the OPV ontology. Focus on identifying materials, properties, methods,
            and device architectures that are not yet in the ontology. When you find a new concept,
            determine its appropriate place in the ontology hierarchy.""",
        )

        return ontology_agent

    def call_llm(self, prompt: str) -> Any:
        """Helper method to call the LLM with retry logic."""
        if not prompt or prompt.strip() == "":
            logger.error("LLM call received an empty prompt.")
            raise ValueError("Prompt for LLM call cannot be empty.")

        logger.info("Calling LLM with prompt length %d", len(prompt))

        try:
            response = self.llm.complete(prompt)
            logger.info("LLM call succeeded")
            return response
        except Exception as e:
            logger.error("LLM call failed: %s", str(e))
            raise e

    def extract_material_properties(
        self, document_id: str, text_chunk: str
    ) -> Dict[str, Any]:
        """Extracts material properties from a scientific paper."""
        prompt = f"""
        Extract material properties from the following text chunk from an OPV research paper.
        Focus on:
        1. Material names (polymers, molecules, etc.)
        2. Physical properties (conductivity, mobility, etc.)
        3. Optical properties (absorption, emission, etc.)
        4. Morphological properties (crystallinity, orientation, etc.)

        Format the output as JSON with the following structure and ONLY return the JSON - no explanations,
        no text before or after, no markdown formatting:
        {{
            "materials": [{{
                "name": "material name",
                "type": "polymer/small molecule/etc",
                "properties": [{{
                    "property_name": "name of property",
                    "value": "numerical value if available",
                    "unit": "unit of measurement if available",
                    "description": "description of the property",
                    "confidence": "high/medium/low"
                }}]
            }}]
        }}

        Text chunk:
        {text_chunk}
        """

        response = self.call_llm(prompt)

        try:
            # Try to extract JSON from the response text
            json_text = self._extract_json_from_text(response.text)
            extraction_result = json.loads(json_text)
            extraction_result["document_id"] = document_id
            extraction_result["extraction_type"] = "material_properties"
            extraction_result["status"] = "needs_validation"

            # Log some details about what was extracted
            materials = extraction_result.get("materials", [])
            if materials:
                logger.info(
                    f"Extracted {len(materials)} materials from document {document_id}"
                )
                for i, material in enumerate(materials[:3]):  # Print first 3 materials
                    logger.info(
                        f"Material {i+1}: {material.get('name')} - {len(material.get('properties', []))} properties"
                    )
                    for j, prop in enumerate(
                        material.get("properties", [])[:2]
                    ):  # Print first 2 properties
                        logger.info(
                            f"  Property {j+1}: {prop.get('property_name')} = {prop.get('value')} {prop.get('unit', '')}"
                        )

            logger.info("Extracted material properties for document '%s'", document_id)
            return extraction_result
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response: %s", response.text)
            return {
                "document_id": document_id,
                "extraction_type": "material_properties",
                "status": "error",
                "error": "Failed to parse extraction result",
            }

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from a text that might contain explanations or markdown."""
        # Look for text between triple backticks with json
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_pattern, text)
        if matches:
            return matches[0].strip()

        # If not found, try to find text that looks like JSON
        # (starting with { and ending with })
        json_pattern = r"(\{[\s\S]*\})"
        matches = re.findall(json_pattern, text)
        if matches:
            return matches[0].strip()

        # If still not found, return the original text
        return text

    def extract_experimental_methods(
        self, document_id: str, text_chunk: str
    ) -> Dict[str, Any]:
        """Extracts experimental methods from a scientific paper."""
        prompt = f"""
        Extract experimental methods from the following text chunk from an OPV research paper.
        Focus on:
        1. Characterization techniques
        2. Fabrication methods
        3. Measurement procedures
        4. Analysis methods

        Format the output as JSON with the following structure and ONLY return the JSON - no explanations,
        no text before or after, no markdown formatting:
        {{
            "methods": [{{
                "name": "method name",
                "type": "characterization/fabrication/measurement/analysis",
                "parameters": [{{
                    "parameter_name": "name of parameter",
                    "value": "numerical value if available",
                    "unit": "unit of measurement if available",
                    "description": "description of the parameter"
                }}],
                "description": "detailed description of the method",
                "confidence": "high/medium/low"
            }}]
        }}

        Text chunk:
        {text_chunk}
        """

        response = self.call_llm(prompt)

        try:
            # Try to extract JSON from the response text
            json_text = self._extract_json_from_text(response.text)
            extraction_result = json.loads(json_text)
            extraction_result["document_id"] = document_id
            extraction_result["extraction_type"] = "experimental_methods"
            extraction_result["status"] = "needs_validation"

            logger.info("Extracted experimental methods for document '%s'", document_id)
            return extraction_result
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response: %s", response.text)
            return {
                "document_id": document_id,
                "extraction_type": "experimental_methods",
                "status": "error",
                "error": "Failed to parse extraction result",
            }

    def extract_device_architecture(
        self, document_id: str, text_chunk: str
    ) -> Dict[str, Any]:
        """Extracts device architecture information from a scientific paper."""
        prompt = f"""
        Extract device architecture information from the following text chunk from an OPV research paper.
        Focus on:
        1. Layer structure
        2. Materials used in each layer
        3. Layer thicknesses
        4. Device performance metrics

        Format the output as JSON with the following structure and ONLY return the JSON - no explanations,
        no text before or after, no markdown formatting:
        {{
            "device": {{
                "type": "device type",
                "architecture": "brief description of architecture",
                "layers": [{{
                    "name": "layer name",
                    "material": "material used",
                    "thickness": "thickness value",
                    "thickness_unit": "unit of measurement",
                    "function": "function of this layer",
                    "position": "position in stack (bottom to top)"
                }}],
                "performance": [{{
                    "metric": "performance metric",
                    "value": "numerical value",
                    "unit": "unit of measurement"
                }}],
                "confidence": "high/medium/low"
            }}
        }}

        Text chunk:
        {text_chunk}
        """

        response = self.call_llm(prompt)

        try:
            # Try to extract JSON from the response text
            json_text = self._extract_json_from_text(response.text)
            extraction_result = json.loads(json_text)
            extraction_result["document_id"] = document_id
            extraction_result["extraction_type"] = "device_architecture"
            extraction_result["status"] = "needs_validation"

            logger.info("Extracted device architecture for document '%s'", document_id)
            return extraction_result
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response: %s", response.text)
            return {
                "document_id": document_id,
                "extraction_type": "device_architecture",
                "status": "error",
                "error": "Failed to parse extraction result",
            }

    def validate_extraction(
        self, extraction_result: Dict[str, Any], domain_expert: str = "user"
    ) -> Dict[str, Any]:
        """Presents extracted information to a domain expert for validation."""
        # In a real implementation, this would present the extraction to a human expert
        # For now, we'll just simulate validation
        logger.info("Validation request for %s to review extraction", domain_expert)

        extraction_result["validated_by"] = domain_expert
        extraction_result["status"] = "validated"

        logger.info("Extraction validated by %s", domain_expert)
        return extraction_result

    def add_kg_node(
        self, node_id: str, node_type: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds a new node to the knowledge graph."""
        # Create metadata for the node
        metadata = {"node_type": node_type, **properties}

        # Generate a node ID if not provided
        if not node_id:
            node_id = (
                f"{node_type}_{hashlib.md5(str(properties).encode()).hexdigest()[:8]}"
            )

        # Create the node
        node = TextNode(
            text=f"{node_type}: {properties.get('name', node_id)}",
            id_=node_id,
            metadata=metadata,
        )

        # Add the node to the knowledge graph
        if self.kg_index:
            self.kg_index.insert_nodes([node])
        else:
            self.kg_index = KnowledgeGraphIndex(
                [node], storage_context=self.storage_context
            )

        logger.info("Added KG node '%s' of type '%s'", node_id, node_type)

        return {
            "node_id": node_id,
            "node_type": node_type,
            "properties": properties,
            "status": "added_to_kg",
        }

    def add_kg_relation(
        self,
        source_node_id: str,
        relation_type: str,
        target_node_id: str,
        properties: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Adds a relationship between nodes in the knowledge graph."""
        if not self.kg_index:
            logger.error("Knowledge graph not initialized when adding relation.")
            return {"status": "error", "error": "Knowledge graph not initialized"}

        try:
            # Add the relation to the knowledge graph
            self.kg_index.add_node_relation(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relation_type=relation_type,
            )

            logger.info(
                "Added KG relation: %s -> %s -> %s",
                source_node_id,
                relation_type,
                target_node_id,
            )

            return {
                "source_node_id": source_node_id,
                "relation_type": relation_type,
                "target_node_id": target_node_id,
                "properties": properties,
                "status": "added_to_kg",
            }
        except Exception as e:
            logger.error("Error adding relationship to KG: %s", str(e))
            return {"status": "error", "error": str(e)}

    def load_publications(self, directory: str = None) -> List[Document]:
        """Load scientific publications for processing."""
        if directory is None:
            directory = f"{self.data_dir}/polymer_papers"

        logger.info("Loading publications from %s", directory)

        try:
            # Check if directory exists
            if not os.path.exists(directory):
                logger.warning("Publication directory %s does not exist", directory)
                return []

            # Load documents from directory
            reader = SimpleDirectoryReader(directory)
            documents = reader.load_data()
            self.publications = documents

            logger.info("Loaded %d publications", len(documents))
            return documents
        except Exception as e:
            logger.error("Error loading publications: %s", str(e))
            return []

    def _load_existing_ontology_concepts(self):
        """Load existing concepts from the ontology for quick lookup."""
        if not self.ontology:
            logger.warning("No ontology available to load concepts from")
            return

        # Query all classes in the ontology
        query = """
        SELECT ?concept ?label
        WHERE {
            ?concept a owl:Class .
            OPTIONAL { ?concept rdfs:label ?label . }
        }
        """
        results = self.ontology.query(query)

        # Store concepts in a set for quick lookup
        self.existing_concepts = set()
        for row in results:
            concept_uri = str(row[0])
            label = str(row[1]) if row[1] else concept_uri.split("/")[-1]
            self.existing_concepts.add(label.lower())

        logger.info(
            f"Loaded {len(self.existing_concepts)} existing concepts from ontology"
        )

    def classify_concept(
        self, concept_name: str, concept_description: str
    ) -> Dict[str, Any]:
        """Classifies a new concept and identifies its parent concept in the ontology."""
        if not self.ontology:
            logger.warning("No ontology available for concept classification")
            return {"status": "error", "error": "No ontology available"}

        # Use LLM to classify the concept
        ontology_classes = list(self.existing_concepts)
        ontology_classes_str = ", ".join(sorted(ontology_classes))

        prompt = f"""
        I need to classify a new concept for an organic photovoltaics (OPV) ontology.

        The concept is: {concept_name}
        Description: {concept_description}

        Existing classes in the ontology include: {ontology_classes_str}

        Please determine:
        1. What type of concept this is (Material, Property, Device, Method, etc.)
        2. What existing class would be its most appropriate parent
        3. A short description for this concept

        Format the answer as a JSON with the following structure:
        {{
            "concept_type": "the most specific type for this concept",
            "parent_concept": "the most appropriate parent class",
            "description": "a clear, concise description of the concept"
        }}

        ONLY return the JSON with no other text or explanations.
        """

        response = self.call_llm(prompt)

        try:
            # Extract JSON from response
            json_text = self._extract_json_from_text(response.text)
            classification = json.loads(json_text)

            logger.info(
                f"Classified concept '{concept_name}' as type '{classification.get('concept_type')}' "
                "with parent '{classification.get('parent_concept')}'"
            )

            return {
                "concept_name": concept_name,
                "concept_type": classification.get("concept_type"),
                "parent_concept": classification.get("parent_concept"),
                "description": classification.get("description"),
                "status": "classified",
            }
        except Exception as e:
            logger.error(f"Error classifying concept '{concept_name}': {str(e)}")
            return {"concept_name": concept_name, "status": "error", "error": str(e)}

    def add_new_concept_to_ontology(
        self,
        concept_name: str,
        concept_type: str,
        parent_concept: str,
        description: str,
    ) -> Dict[str, Any]:
        """Adds a new concept to the OPV ontology."""
        if not self.ontology:
            logger.warning("No ontology available to add new concept")
            return {"status": "error", "error": "No ontology available"}

        try:
            # Check if concept already exists
            if concept_name.lower() in self.existing_concepts:
                logger.info(f"Concept '{concept_name}' already exists in ontology")
                return {"status": "exists", "concept_name": concept_name}

            # Create a URI for the concept
            concept_uri = URIRef(
                self.ontology.namespace_manager.store.namespace("opv")
                + concept_name.replace(" ", "_")
            )

            # Add the concept to the ontology
            self.ontology.add((concept_uri, RDF.type, OWL.Class))
            self.ontology.add((concept_uri, RDFS.label, Literal(concept_name)))
            self.ontology.add((concept_uri, RDFS.comment, Literal(description)))

            # Add parent relationship if provided and parent exists
            if parent_concept:
                parent_uri = URIRef(
                    self.ontology.namespace_manager.store.namespace("opv")
                    + parent_concept.replace(" ", "_")
                )
                self.ontology.add((concept_uri, RDFS.subClassOf, parent_uri))

            # Save the updated ontology
            self.ontology.serialize(
                destination=f"{self.persist_dir}/ontology/opv_ontology_updated.owl",
                format="xml",
            )
            self.ontology.serialize(
                destination=f"{self.persist_dir}/ontology/opv_ontology_updated.ttl",
                format="turtle",
            )

            # Add to existing concepts set
            self.existing_concepts.add(concept_name.lower())

            logger.info(
                f"Added new concept '{concept_name}' to ontology with parent '{parent_concept}'"
            )

            return {
                "status": "success",
                "concept_name": concept_name,
                "concept_type": concept_type,
                "parent_concept": parent_concept,
                "description": description,
            }
        except Exception as e:
            logger.error(f"Error adding concept '{concept_name}' to ontology: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _identify_new_concepts(
        self, extraction_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identifies new concepts from extracted information."""
        new_concepts = []

        extraction_type = extraction_result.get("extraction_type")

        if extraction_type == "material_properties":
            # Extract material names
            for material in extraction_result.get("materials", []):
                material_name = material.get("name")
                if (
                    material_name
                    and material_name.lower() not in self.existing_concepts
                ):
                    new_concepts.append(
                        {
                            "name": material_name,
                            "type": material.get("type", "Material"),
                            "description": f"A {material.get('type', 'material')} used in organic photovoltaics",
                        }
                    )

                # Extract property names
                for prop in material.get("properties", []):
                    prop_name = prop.get("property_name")
                    if prop_name and prop_name.lower() not in self.existing_concepts:
                        description = prop.get(
                            "description", f"A property of {material_name}"
                        )
                        new_concepts.append(
                            {
                                "name": prop_name,
                                "type": "Property",
                                "description": description,
                            }
                        )

        elif extraction_type == "experimental_methods":
            # Extract method names
            for method in extraction_result.get("methods", []):
                method_name = method.get("name")
                if method_name and method_name.lower() not in self.existing_concepts:
                    new_concepts.append(
                        {
                            "name": method_name,
                            "type": "Method",
                            "description": method.get(
                                "description", "A method used in organic photovoltaics"
                            ),
                        }
                    )

                # Extract parameter names
                for param in method.get("parameters", []):
                    param_name = param.get("parameter_name")
                    if param_name and param_name.lower() not in self.existing_concepts:
                        description = param.get(
                            "description", f"A parameter of {method_name}"
                        )
                        new_concepts.append(
                            {
                                "name": param_name,
                                "type": "Parameter",
                                "description": description,
                            }
                        )

        elif extraction_type == "device_architecture":
            # Extract device type
            device = extraction_result.get("device", {})
            device_type = device.get("type")
            if device_type and device_type.lower() not in self.existing_concepts:
                new_concepts.append(
                    {
                        "name": device_type,
                        "type": "DeviceArchitecture",
                        "description": device.get(
                            "architecture",
                            "A type of device architecture used in organic photovoltaics",
                        ),
                    }
                )

            # Extract layer names and materials
            for layer in device.get("layers", []):
                layer_name = layer.get("name")
                if layer_name and layer_name.lower() not in self.existing_concepts:
                    new_concepts.append(
                        {
                            "name": layer_name,
                            "type": "Layer",
                            "description": layer.get(
                                "function",
                                "A layer used in organic photovoltaic devices",
                            ),
                        }
                    )

                layer_material = layer.get("material")
                if (
                    layer_material
                    and layer_material.lower() not in self.existing_concepts
                ):
                    new_concepts.append(
                        {
                            "name": layer_material,
                            "type": "Material",
                            "description": f"A material used in the {layer_name} layer of organic photovoltaic devices",
                        }
                    )

        return new_concepts

    def _enrich_ontology_with_new_concepts(
        self, extraction_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enriches the ontology with new concepts from extracted information."""
        if not self.ontology:
            logger.warning("No ontology available for enrichment")
            return []

        logger.info("Enriching ontology with new concepts from extraction")

        # Identify new concepts
        new_concepts = self._identify_new_concepts(extraction_result)
        logger.info(f"Identified {len(new_concepts)} potential new concepts")

        added_concepts = []

        # Process each new concept
        for concept in new_concepts:
            concept_name = concept.get("name")

            # Skip if concept already exists in ontology
            if concept_name.lower() in self.existing_concepts:
                continue

            # Classify the concept to determine its parent
            classification = self.classify_concept(
                concept_name=concept_name,
                concept_description=concept.get("description", ""),
            )

            if classification.get("status") == "classified":
                # Add the concept to the ontology
                result = self.add_new_concept_to_ontology(
                    concept_name=concept_name,
                    concept_type=classification.get("concept_type"),
                    parent_concept=classification.get("parent_concept"),
                    description=classification.get(
                        "description", concept.get("description", "")
                    ),
                )

                if result.get("status") == "success":
                    added_concepts.append(result)
                    logger.info(f"Added new concept '{concept_name}' to ontology")

        return added_concepts

    def _update_kg_from_extraction(self, extraction_result: Dict[str, Any]) -> None:
        """Update knowledge graph based on validated extraction results."""
        extraction_type = extraction_result.get("extraction_type")

        # First, enrich the ontology with new concepts
        if self.ontology:
            try:
                added_concepts = self._enrich_ontology_with_new_concepts(
                    extraction_result
                )
                if added_concepts:
                    logger.info(
                        f"Added {len(added_concepts)} new concepts to the ontology from extraction"
                    )
                    # Save the updated ontology
                    ontology_path = (
                        f"{self.persist_dir}/ontology/opv_ontology_enriched.owl"
                    )
                    self.ontology.serialize(destination=ontology_path, format="xml")
                    turtle_path = (
                        f"{self.persist_dir}/ontology/opv_ontology_enriched.ttl"
                    )
                    self.ontology.serialize(destination=turtle_path, format="turtle")
                    logger.info(f"Saved enriched ontology to {ontology_path}")
            except Exception as e:
                logger.error(f"Error enriching ontology: {str(e)}")

        if extraction_type == "material_properties":
            # Process material properties
            for material in extraction_result.get("materials", []):
                # Add material node
                material_node = self.kg_agent.query(
                    f"add_kg_node('', 'Material', {json.dumps(material)})"
                )

                # Add property nodes and relationships
                for prop in material.get("properties", []):
                    # Add property node
                    prop_node = self.kg_agent.query(
                        f"add_kg_node('', 'Property', {json.dumps(prop)})"
                    )

                    # Add relationship between material and property
                    self.kg_agent.query(
                        f"add_kg_relation('{material_node.get('node_id')}', 'has_property', '{prop_node.get('node_id')}')"
                    )

        elif extraction_type == "experimental_methods":
            # Process experimental methods
            for method in extraction_result.get("methods", []):
                # Add method node
                method_node = self.kg_agent.query(
                    f"add_kg_node('', 'Method', {json.dumps(method)})"
                )

                # Add parameter nodes and relationships
                for param in method.get("parameters", []):
                    # Add parameter node
                    param_node = self.kg_agent.query(
                        f"add_kg_node('', 'Parameter', {json.dumps(param)})"
                    )

                    # Add relationship between method and parameter
                    self.kg_agent.query(
                        f"add_kg_relation('{method_node.get('node_id')}', 'has_parameter', '{param_node.get('node_id')}')"
                    )

        elif extraction_type == "device_architecture":
            # Process device architecture
            device = extraction_result.get("device", {})

            # Add device node
            device_node = self.kg_agent.query(
                f"add_kg_node('', 'Device', {json.dumps(device)})"
            )

            # Add layer nodes and relationships
            for layer in device.get("layers", []):
                # Add layer node
                layer_node = self.kg_agent.query(
                    f"add_kg_node('', 'Layer', {json.dumps(layer)})"
                )

                # Add relationship between device and layer
                self.kg_agent.query(
                    f"add_kg_relation('{device_node.get('node_id')}', 'has_layer', '{layer_node.get('node_id')}')"
                )

    def process_publications(self, documents: List[Document] = None) -> Dict[str, Any]:
        """Process publications to extract information and update KG."""
        if documents is None:
            documents = self.publications

        if not documents:
            logger.error("No documents available for processing.")
            return {"status": "error", "error": "No documents to process"}

        # Load existing ontology concepts before processing
        if self.ontology:
            self._load_existing_ontology_concepts()

        results = []
        # ontology_additions = []

        for doc in documents:
            # Generate a document ID
            doc_id = doc.metadata.get(
                "file_name", hashlib.md5(doc.text.encode()).hexdigest()[:8]
            )

            # Parse the document into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents([doc])

            # Process each chunk
            for i, node in enumerate(nodes):
                # Extract material properties
                material_result = self.agent.query(
                    f"extract_material_properties('{doc_id}', '''{node.text}''')"
                )

                # Extract experimental methods
                methods_result = self.agent.query(
                    f"extract_experimental_methods('{doc_id}', '''{node.text}''')"
                )

                # Extract device architecture
                device_result = self.agent.query(
                    f"extract_device_architecture('{doc_id}', '''{node.text}''')"
                )

                # Collect results for this chunk
                chunk_results = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "material_properties": material_result,
                    "experimental_methods": methods_result,
                    "device_architecture": device_result,
                }

                results.append(chunk_results)

                # Validate and update KG and ontology for each extraction type
                for extraction_type in [
                    "material_properties",
                    "experimental_methods",
                    "device_architecture",
                ]:
                    if (
                        chunk_results[extraction_type].get("status")
                        == "needs_validation"
                    ):
                        try:
                            # Validate the extraction
                            logger.info(f"Validating {extraction_type} extraction")
                            validated_result = self.validation_agent.query(
                                f"validate_extraction({json.dumps(chunk_results[extraction_type])})"
                            )

                            # Update the knowledge graph if validated
                            if validated_result.get("status") == "validated":
                                logger.info(
                                    f"Updating KG with validated {extraction_type}"
                                )
                                self._update_kg_from_extraction(validated_result)
                        except Exception as e:
                            logger.error(
                                f"Error in validation or update for {extraction_type}: {str(e)}"
                            )

        logger.info(
            "Processed %d documents and %d chunks", len(documents), len(results)
        )

        # Final ontology save
        if self.ontology:
            ontology_path = (
                f"{self.persist_dir}/ontology/opv_ontology_after_extraction.owl"
            )
            self.ontology.serialize(destination=ontology_path, format="xml")
            turtle_path = (
                f"{self.persist_dir}/ontology/opv_ontology_after_extraction.ttl"
            )
            self.ontology.serialize(destination=turtle_path, format="turtle")
            logger.info(f"Saved final ontology to {ontology_path}")

        return {
            "status": "completed",
            "documents_processed": len(documents),
            "chunks_processed": len(results),
            "results": results,
        }

    def set_ontology(self, ontology):
        """Set the ontology from Phase 1."""
        self.ontology = ontology
        logger.info("Ontology set from Phase 1")
        # Load existing concepts for quick lookup
        self._load_existing_ontology_concepts()

    def get_kg_index(self):
        """Return the knowledge graph index."""
        return self.kg_index

    def get_vector_index(self):
        """Return the vector index."""
        return self.vector_index

    def get_ontology(self):
        """Return the current ontology (possibly enriched)."""
        return self.ontology

    def run(self) -> Dict[str, Any]:
        """Run the publication analysis phase."""
        logger.info("Running publication analysis phase")

        # Check if ontology is available
        if not self.ontology:
            logger.warning("Ontology not set, proceeding without ontology integration")

        # Load publications
        documents = self.load_publications()

        if not documents:
            logger.error("No publications found to process in Phase 2.")
            return {
                "status": "error",
                "phase": "publication_analysis",
                "error": "No publications found to process",
            }

        # Process publications
        results = self.process_publications(documents)

        # Create vector index from document nodes
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.vector_index = VectorStoreIndex(
            nodes, storage_context=StorageContext.from_defaults()
        )

        # Persist indices
        if self.vector_index:
            self.vector_index.storage_context.persist(
                f"{self.persist_dir}/vector/phase2"
            )

        if self.kg_index:
            self.kg_index.storage_context.persist(f"{self.persist_dir}/kg/phase2")

        # Print statistics about ontology enrichment
        if self.ontology:
            # Count classes in the ontology
            query = """
            SELECT (COUNT(?class) AS ?count)
            WHERE { ?class a owl:Class . }
            """
            result = list(self.ontology.query(query))
            if result:
                class_count = result[0][0].value
                logger.info(f"Ontology now has {class_count} classes after enrichment")

        logger.info("Phase 2 completed: Processed %d documents", len(documents))

        return {
            "status": "completed",
            "phase": "publication_analysis",
            "documents_processed": len(documents),
            "results": results,
        }
