"""
OPV Knowledge Graph Multi-Agent Framework using LlamaIndex
--------------------------------------------------------
This implementation follows the phased approach described in the project document:
1. Initial Ontology Development with LLM Support
2. First AI Augmentation Cycle - Publications
3. Second AI Augmentation Cycle - Scientific Data
4. RAG Integration and User-Driven Evaluation
"""

import os
import json
import time
import logging
import re
from typing import List, Dict, Any
from datetime import datetime
import hashlib

# LlamaIndex Imports
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document, TextNode

# Replace OpenAI with Ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.evaluation import ResponseEvaluator

# For saving ontologies in standard formats (OWL, RDF)
import rdflib
from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Set up debug handler for monitoring
llama_debug = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug])

# ---------------- Custom Exceptions ----------------


class LLMError(Exception):
    pass


class ParsingError(Exception):
    pass


class TimeoutError(Exception):
    pass


# ---------------- Helper Functions ----------------


def _ensure_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return val.get("title", str(val))
    else:
        return str(val)


def robust_parse_json(
    response_text: str, opv_system, fallback_attempts: int = 1
) -> Dict[str, Any]:
    """Attempt to parse JSON with several fallback strategies."""
    try:
        return json.loads(response_text)
    except Exception:
        # Try to extract JSON from markdown block:
        m = re.search(r"```(.*?)```", response_text, re.DOTALL)
        if m:
            json_str = m.group(1)
            try:
                return json.loads(json_str)
            except Exception:
                pass
        # As a fallback, re-prompt the LLM for a strictly valid JSON output.
        if fallback_attempts > 0:
            fallback_prompt = "Please reformat the output strictly as valid JSON."
            fallback_response = opv_system.call_llm(fallback_prompt)
            try:
                return json.loads(fallback_response.text)
            except Exception as ex:
                raise ParsingError("Fallback parsing failed: " + str(ex))
        else:
            raise ParsingError("Could not parse JSON from response.")


def is_relevant_chunk(text: str) -> bool:
    """Return True if the text chunk is likely to contain scientific content."""
    # Always consider chunks containing key section titles
    key_sections = ["methods", "results", "discussion"]
    text_lower = text.lower()
    if any(kw in text_lower for kw in key_sections):
        return True
    irrelevant_keywords = [
        "accepted version",
        "copyright",
        "university",
        "email:",
        "http://",
        "https://",
        "self-archiving policy",
        "funding",
        "acknowledgment",
        "doi:",
        "abstract",
        "keywords",
    ]
    return not any(kw in text_lower for kw in irrelevant_keywords)


# ---------------- OPVKnowledgeGraphSystem Class ----------------


class OPVKnowledgeGraphSystem:
    """
    Multi-agent system for building and utilizing knowledge graphs
    for organic photovoltaics (OPVs) research.
    """

    def __init__(
        self,
        llm_model: str = "mistral",
        temperature: float = 0.1,
        data_dir: str = "./data",
        persist_dir: str = "./storage",
        debug: bool = True,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.debug = debug

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)
        os.makedirs(f"{persist_dir}/vector", exist_ok=True)
        os.makedirs(f"{persist_dir}/kg", exist_ok=True)
        os.makedirs(f"{persist_dir}/ontology", exist_ok=True)

        self.llm = Ollama(
            model=llm_model, temperature=temperature, base_url=ollama_base_url
        )
        self.callback_manager = (
            CallbackManager([LlamaDebugHandler()]) if debug else None
        )
        Settings.llm = self.llm
        Settings.callback_manager = self.callback_manager

        self.graph_store = SimpleGraphStore()
        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512, chunk_overlap=50
        )

        self.publications = []
        self.experimental_data = []
        self.vector_index = None
        self.kg_index = None

        self.auto_validate_extraction = True

        self.ontology = rdflib.Graph()
        self.ovp_ns = Namespace("http://opv-kg.org/ontology/")
        # Define provenance properties:
        self.ontology.bind("opv", self.ovp_ns)
        self.ovp_ns.extractionSource = URIRef(self.ovp_ns + "extractionSource")
        self.ovp_ns.extractionTimestamp = URIRef(self.ovp_ns + "extractionTimestamp")
        self.ovp_ns.extractionConfidence = URIRef(self.ovp_ns + "extractionConfidence")

        self.initialize_base_ontology()
        self.agents = {}
        self.setup_agents()
        logger.info("OPV Knowledge Graph System initialized.")

    def _normalize_input(self, inp: Any) -> Any:
        if hasattr(inp, "text"):
            try:
                parsed = json.loads(inp.text)
                if not isinstance(parsed, dict):
                    return {"text": parsed}
                return parsed
            except Exception:
                return {"text": inp.text}
        elif hasattr(inp, "to_dict"):
            return inp.to_dict()
        elif isinstance(inp, dict):
            return {k: self._normalize_input(v) for k, v in inp.items()}
        elif isinstance(inp, list):
            return [self._normalize_input(x) for x in inp]
        else:
            return inp

    def _serialize_response(self, response) -> str:
        if hasattr(response, "text"):
            return response.text
        return str(response)

    def call_llm(self, prompt: str) -> Any:
        max_retries = 5
        attempt = 0
        last_exception = None
        if not prompt or prompt.strip() == "":
            logger.error("LLM call received an empty prompt.")
            raise ValueError("Prompt for LLM call cannot be empty.")
        logger.info("Calling LLM with prompt length %d", len(prompt))
        while attempt < max_retries:
            try:
                response = self.llm.complete(prompt)
                logger.info("LLM call succeeded on attempt %d", attempt + 1)
                return response
            except Exception as e:
                attempt += 1
                # Differentiate timeouts from other errors:
                err_msg = str(e).lower()
                if "timed out" in err_msg:
                    last_exception = TimeoutError("LLM call timed out: " + str(e))
                else:
                    last_exception = LLMError("LLM call error: " + str(e))
                logger.warning("LLM call failed on attempt %d: %s", attempt, str(e))
                time.sleep(2**attempt)
        logger.error("LLM call failed after %d attempts", max_retries)
        raise last_exception

    def initialize_base_ontology(self):
        self.ontology.bind("opv", self.ovp_ns)
        self.ontology.bind("rdf", RDF)
        self.ontology.bind("rdfs", RDFS)
        self.ontology.bind("owl", OWL)
        material = URIRef(self.ovp_ns + "Material")
        polymer = URIRef(self.ovp_ns + "Polymer")
        device = URIRef(self.ovp_ns + "Device")
        property = URIRef(self.ovp_ns + "Property")
        experiment = URIRef(self.ovp_ns + "Experiment")
        self.ontology.add((material, RDF.type, OWL.Class))
        self.ontology.add((polymer, RDF.type, OWL.Class))
        self.ontology.add((device, RDF.type, OWL.Class))
        self.ontology.add((property, RDF.type, OWL.Class))
        self.ontology.add((experiment, RDF.type, OWL.Class))
        self.ontology.add((polymer, RDFS.subClassOf, material))
        self.save_ontology()
        logger.info("Base OPV ontology initialized.")

    def save_ontology(self, version: str = "current"):
        file_path = f"{self.persist_dir}/ontology/opv_ontology_{version}.owl"
        self.ontology.serialize(destination=file_path, format="xml")
        logger.info("Ontology saved to %s", file_path)
        turtle_path = f"{self.persist_dir}/ontology/opv_ontology_{version}.ttl"
        self.ontology.serialize(destination=turtle_path, format="turtle")
        logger.info("Ontology also saved in Turtle format to %s", turtle_path)
        return file_path

    def setup_agents(self):
        self.agents["ontology"] = self._create_ontology_agent()
        self.agents["publication"] = self._create_publication_agent()
        self.agents["experiment"] = self._create_experiment_agent()
        self.agents["kg"] = self._create_kg_agent()
        self.agents["validation"] = self._create_validation_agent()
        self.agents["orchestrator"] = self._create_orchestrator()
        logger.info("Agent system initialized with 6 specialized agents.")

    def _create_ontology_agent(self):
        tools = []
        tools.append(
            FunctionTool.from_defaults(
                fn=self.suggest_ontology_concept,
                name="suggest_ontology_concept",
                description="Suggests new concepts to add to the OPV ontology.",
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.add_ontology_relationship,
                name="add_ontology_relationship",
                description="Adds a relationship between two ontology concepts.",
            )
        )
        return ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt=(
                "You are an OPV ontology expert. Use domain-specific language "
                "(e.g., BHJ morphology, charge carrier mobility, interfacial mixing) "
                "and focus solely on material science, polymers, chemistry, and physics. "
                "Ignore metadata like author names and copyright notices."
            ),
            max_iterations=20,
        )

    def _create_publication_agent(self):
        tools = []
        tools.append(
            FunctionTool.from_defaults(
                fn=self.extract_material_properties,
                name="extract_material_properties",
                description=(
                    "Extracts material properties from scientific papers. "
                    "Focus on polymers, chemical composition, physical properties, "
                    "and include a confidence score if information is ambiguous or missing."
                ),
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.extract_experimental_methods,
                name="extract_experimental_methods",
                description=(
                    "Extracts experimental methods from scientific papers. "
                    "Focus on fabrication, measurement, and analysis techniques relevant to material science."
                ),
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.extract_device_architecture,
                name="extract_device_architecture",
                description=(
                    "Extracts device architecture information from scientific papers. "
                    "Focus on layer composition, materials used, performance metrics, "
                    "and include any missing details if ambiguous."
                ),
            )
        )
        return ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="You analyze scientific publications on OPVs. Focus on material and device details,"
            " ignoring non‑scientific metadata.",
        )

    def _create_experiment_agent(self):
        tools = []
        tools.append(
            FunctionTool.from_defaults(
                fn=self.extract_experimental_results,
                name="extract_experimental_results",
                description="Extracts experimental results from raw data.",
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.map_parameters_to_ontology,
                name="map_parameters_to_ontology",
                description="Maps experimental parameters to ontology concepts.",
            )
        )
        return ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="You analyze experimental data on OPVs and map parameters to the ontology.",
        )

    def _create_kg_agent(self):
        tools = []
        tools.append(
            FunctionTool.from_defaults(
                fn=self.add_kg_node,
                name="add_kg_node",
                description="Adds a new node to the knowledge graph.",
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.add_kg_relation,
                name="add_kg_relation",
                description="Adds a relationship between KG nodes.",
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.query_knowledge_graph,
                name="query_knowledge_graph",
                description="Queries the knowledge graph.",
            )
        )
        return ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="You build and manage a knowledge graph for OPVs, ensuring accurate domain relationships.",
        )

    def _create_validation_agent(self):
        tools = []
        tools.append(
            FunctionTool.from_defaults(
                fn=self.validate_extraction,
                name="validate_extraction",
                description="Validates extracted information with a domain expert.",
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.validate_ontology_change,
                name="validate_ontology_change",
                description="Validates proposed ontology changes.",
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=self.validate_kg_addition,
                name="validate_kg_addition",
                description="Validates KG additions with a domain expert.",
            )
        )
        return ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="You facilitate human validation for the OPV knowledge system. Provide clear, concise validation.",
        )

    def _create_orchestrator(self):
        all_tools = []
        for agent_name, agent in self.agents.items():
            if agent_name != "orchestrator":
                all_tools.append(
                    QueryEngineTool(
                        query_engine=agent,
                        metadata=ToolMetadata(
                            name=f"{agent_name}_agent",
                            description=f"Invokes the {agent_name} agent.",
                        ),
                    )
                )
        return ReActAgent.from_tools(
            all_tools,
            llm=self.llm,
            verbose=self.debug,
            system_prompt="You coordinate the multi-agent system for building the OPV knowledge graph.",
        )

    # ---------------- Ontology Functions ----------------

    def suggest_ontology_concept(
        self, concept_name: str, description: str, parent_concept: str = None, **kwargs
    ) -> Dict[str, Any]:
        concept_name = _ensure_str(concept_name)
        description = _ensure_str(description)
        if parent_concept is not None:
            parent_concept = _ensure_str(parent_concept)
        concept_uri = URIRef(self.ovp_ns + concept_name.replace(" ", "_"))
        # Duplicate check:
        if list(self.ontology.triples((concept_uri, None, None))):
            logger.info("Concept '%s' already exists. Skipping addition.", concept_name)
            return {
                "concept_added": concept_name,
                "uri": str(concept_uri),
                "status": "exists",
            }
        self.ontology.add((concept_uri, RDF.type, OWL.Class))
        self.ontology.add((concept_uri, RDFS.label, Literal(concept_name)))
        self.ontology.add((concept_uri, RDFS.comment, Literal(description)))
        if parent_concept:
            parent_uri = URIRef(self.ovp_ns + parent_concept.replace(" ", "_"))
            self.ontology.add((concept_uri, RDFS.subClassOf, parent_uri))
        kgcl_change = f'CREATE CLASS {concept_name} DESCRIPTION "{description}"'
        if parent_concept:
            kgcl_change += f" SUBCLASS OF {parent_concept}"
        # Add provenance metadata (if available in kwargs)
        provenance = kwargs.get("provenance", {})
        if provenance.get("document_id"):
            self.ontology.add(
                (
                    concept_uri,
                    self.ovp_ns.extractionSource,
                    Literal(provenance["document_id"]),
                )
            )
        self.ontology.add(
            (
                concept_uri,
                self.ovp_ns.extractionTimestamp,
                Literal(datetime.now().isoformat()),
            )
        )
        self.ontology.add(
            (
                concept_uri,
                self.ovp_ns.extractionConfidence,
                Literal(provenance.get("confidence", "high")),
            )
        )
        self.save_ontology()
        logger.info(
            "Ontology concept '%s' added (parent: %s)", concept_name, parent_concept
        )
        return {
            "concept_added": concept_name,
            "uri": str(concept_uri),
            "parent": parent_concept,
            "kgcl_change": kgcl_change,
            "status": "needs_validation",
        }

    def add_ontology_relationship(
        self,
        source_concept: str,
        relation_type: str,
        target_concept: str,
        description: str = None,
    ) -> Dict[str, Any]:
        source_concept = _ensure_str(source_concept)
        relation_type = _ensure_str(relation_type)
        target_concept = _ensure_str(target_concept)
        if description:
            description = _ensure_str(description)
        source_uri = URIRef(self.ovp_ns + source_concept.replace(" ", "_"))
        target_uri = URIRef(self.ovp_ns + target_concept.replace(" ", "_"))
        relation_uri = URIRef(self.ovp_ns + relation_type.replace(" ", "_"))
        # Duplicate check: if such a relationship already exists
        if list(self.ontology.triples((source_uri, relation_uri, target_uri))):
            logger.info(
                "Relationship %s %s %s already exists.",
                source_concept,
                relation_type,
                target_concept,
            )
            return {
                "relationship_added": f"{source_concept} -> {relation_type} -> {target_concept}",
                "status": "exists",
            }
        self.ontology.add((relation_uri, RDF.type, OWL.ObjectProperty))
        if description:
            self.ontology.add((relation_uri, RDFS.comment, Literal(description)))
        self.ontology.add((source_uri, relation_uri, target_uri))
        kgcl_change = (
            f"ADD RELATIONSHIP {source_concept} {relation_type} {target_concept}"
        )
        self.save_ontology()
        logger.info("Ontology relationship added: %s", kgcl_change)
        return {
            "relationship_added": f"{source_concept} -> {relation_type} -> {target_concept}",
            "kgcl_change": kgcl_change,
            "status": "needs_validation",
        }

    # ---------------- Updated Extraction Functions ----------------

    def extract_material_properties(
        self, document_id: str, text_chunk: str
    ) -> Dict[str, Any]:
        prompt = f"""
[IMPORTANT: Focus on material science details. Use terms such as 'BHJ morphology', 'charge carrier mobility', "
"and 'interfacial mixing'. Skip metadata such as author names and affiliations. If any information is ambiguous or missing, "
"indicate so with a confidence score.]

Extract material properties from the following text chunk:
{text_chunk}

Format the output strictly as JSON:
{{
    "materials": [{{
        "name": "material name",
        "type": "polymer/small molecule/etc",
        "description": "brief description of the material",
        "confidence": "high/medium/low",
        "properties": [{{
            "property_name": "property name",
            "value": "value if available or 'N/A'",
            "unit": "unit if available",
            "description": "detailed description or 'information missing'",
            "confidence": "high/medium/low"
        }}]
    }}]
}}
"""
        response = self.call_llm(prompt)
        try:
            extraction_result = robust_parse_json(response.text, self)
            extraction_result["document_id"] = document_id
            extraction_result["extraction_type"] = "material_properties"
            # Use tiered validation: if confidence is high in all items, mark as validated
            extraction_result["status"] = (
                "validated"
                if all(
                    mat.get("confidence", "high").lower() == "high"
                    for mat in extraction_result.get("materials", [])
                )
                else "needs_validation"
            )
            logger.info("Extracted material properties for document '%s'", document_id)
            # Update ontology with provenance metadata.
            for material in extraction_result.get("materials", []):
                name = material.get("name")
                description = material.get("description", "No description provided")
                confidence = material.get("confidence", "high")
                if name:
                    provenance = {"document_id": document_id, "confidence": confidence}
                    concept_uri = URIRef(self.ovp_ns + name.replace(" ", "_"))
                    if not list(self.ontology.triples((concept_uri, None, None))):
                        self.suggest_ontology_concept(
                            name, description, "Material", provenance=provenance
                        )
                        logger.info("Material concept '%s' added to ontology.", name)
                    else:
                        logger.info(
                            "Material concept '%s' already exists in the ontology.",
                            name,
                        )
            return extraction_result
        except Exception as e:
            logger.error(
                "Failed to parse JSON for document '%s'. Problematic text: %.100s; Error: %s",
                document_id,
                text_chunk,
                str(e),
            )
            return {
                "document_id": document_id,
                "extraction_type": "material_properties",
                "status": "error",
                "error": "Parsing failed",
            }

    def extract_experimental_methods(
        self, document_id: str, text_chunk: str
    ) -> Dict[str, Any]:
        prompt = f"""
[IMPORTANT: Focus on fabrication, measurement, and analysis methods relevant to material science and polymers."
" Use domain-specific terms and include confidence scores. If details are missing, indicate so.]

Extract experimental methods from the following text chunk:
{text_chunk}

Format the output strictly as JSON:
{{
    "methods": [{{
        "name": "method name",
        "type": "characterization/fabrication/measurement/analysis",
        "parameters": [{{
            "parameter_name": "parameter name",
            "value": "value if available or 'N/A'",
            "unit": "unit if available",
            "description": "detailed parameter description or 'missing'",
            "confidence": "high/medium/low"
        }}],
        "description": "detailed description of the method",
        "confidence": "high/medium/low"
    }}]
}}
"""
        response = self.call_llm(prompt)
        try:
            extraction_result = robust_parse_json(response.text, self)
            extraction_result["document_id"] = document_id
            extraction_result["extraction_type"] = "experimental_methods"
            extraction_result["status"] = (
                "validated"
                if all(
                    method.get("confidence", "high").lower() == "high"
                    for method in extraction_result.get("methods", [])
                )
                else "needs_validation"
            )
            logger.info("Extracted experimental methods for document '%s'", document_id)
            for method in extraction_result.get("methods", []):
                name = method.get("name")
                description = method.get("description", "No description provided")
                confidence = method.get("confidence", "high")
                if name:
                    provenance = {"document_id": document_id, "confidence": confidence}
                    concept_uri = URIRef(self.ovp_ns + name.replace(" ", "_"))
                    if not list(self.ontology.triples((concept_uri, None, None))):
                        self.suggest_ontology_concept(
                            name, description, "Experiment", provenance=provenance
                        )
                        logger.info("Experimental method '%s' added to ontology.", name)
                    else:
                        logger.info(
                            "Experimental method '%s' already exists in the ontology.",
                            name,
                        )
            return extraction_result
        except Exception as e:
            logger.error(
                "Failed to parse JSON for experimental methods in document '%s'. Error: %s",
                document_id,
                str(e),
            )
            return {
                "document_id": document_id,
                "extraction_type": "experimental_methods",
                "status": "error",
                "error": "Parsing failed",
            }

    def extract_device_architecture(
        self, document_id: str, text_chunk: str
    ) -> Dict[str, Any]:
        prompt = f"""
[IMPORTANT: Focus on detailed device structure: layer composition, material usage, and performance metrics. "
"Use technical terms (e.g., 'BHJ architecture', 'active layer', 'anode') and include confidence scores."
"If details are missing, indicate so.]

Extract device architecture information from the following text chunk:
{text_chunk}

Format the output strictly as JSON:
{{
    "device": {{
        "type": "device type",
        "architecture": "brief description of architecture",
        "layers": [{{
            "name": "layer name",
            "material": "material used",
            "thickness": "thickness value or 'N/A'",
            "thickness_unit": "unit if available",
            "function": "function of the layer",
            "position": "position in stack"
        }}],
        "performance": [{{
            "metric": "performance metric",
            "value": "value or 'N/A'",
            "unit": "unit if available"
        }}],
        "confidence": "high/medium/low"
    }}
}}
"""
        response = self.call_llm(prompt)
        try:
            extraction_result = robust_parse_json(response.text, self)
            extraction_result["document_id"] = document_id
            extraction_result["extraction_type"] = "device_architecture"
            extraction_result["status"] = (
                "validated"
                if extraction_result.get("device", {}).get("confidence", "high").lower()
                == "high"
                else "needs_validation"
            )
            logger.info("Extracted device architecture for document '%s'", document_id)
            device = extraction_result.get("device", {})
            name = device.get("type")
            description = device.get("architecture", "No description provided")
            if name:
                provenance = {
                    "document_id": document_id,
                    "confidence": device.get("confidence", "high"),
                }
                concept_uri = URIRef(self.ovp_ns + name.replace(" ", "_"))
                if not list(self.ontology.triples((concept_uri, None, None))):
                    self.suggest_ontology_concept(
                        name, description, "Device", provenance=provenance
                    )
                    logger.info("Device concept '%s' added to ontology.", name)
                else:
                    logger.info(
                        "Device concept '%s' already exists in the ontology.", name
                    )
            return extraction_result
        except Exception as e:
            logger.error(
                "Failed to parse JSON for device architecture in document '%s'. Error: %s",
                document_id,
                str(e),
            )
            return {
                "document_id": document_id,
                "extraction_type": "device_architecture",
                "status": "error",
                "error": "Parsing failed",
            }

    # Experimental Data Functions remain unchanged...
    def extract_experimental_results(
        self, experiment_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        results = {
            "experiment_id": experiment_id,
            "extraction_type": "experimental_results",
            "results": data.get("results", {}),
            "conditions": data.get("conditions", {}),
            "status": "needs_validation",
        }
        logger.info("Extracted experimental results for experiment '%s'", experiment_id)
        return results

    def map_parameters_to_ontology(
        self, parameter_name: str, parameter_value: Any, parameter_unit: str = None
    ) -> Dict[str, Any]:
        query = f"""
SELECT ?concept ?label ?comment
WHERE {{
    ?concept a owl:Class .
    ?concept rdfs:label ?label .
    OPTIONAL {{ ?concept rdfs:comment ?comment }}
    FILTER (CONTAINS(LCASE(STR(?label)), LCASE("{parameter_name}")))
}}
"""
        results = list(self.ontology.query(query))
        if results:
            matches = []
            for result in results:
                concept, label, comment = result
                matches.append(
                    {
                        "concept_uri": str(concept),
                        "label": str(label),
                        "comment": str(comment) if comment else None,
                        "match_confidence": (
                            "high"
                            if parameter_name.lower() == str(label).lower()
                            else "medium"
                        ),
                    }
                )
            logger.info("Found ontology matches for parameter '%s'", parameter_name)
            return {
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "parameter_unit": parameter_unit,
                "ontology_matches": matches,
                "status": "needs_validation",
            }
        else:
            description = f"Parameter representing {parameter_name}"
            if parameter_unit:
                description += f" measured in {parameter_unit}"
            return {
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "parameter_unit": parameter_unit,
                "suggested_concept": {
                    "name": parameter_name.replace(" ", "_"),
                    "description": description,
                    "parent_concept": None,
                },
                "status": "needs_ontology_addition",
            }

    # ---------------- Knowledge Graph and Validation Functions ----------------

    def add_kg_node(
        self, node_id: str, node_type: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        metadata = {"node_type": node_type, **properties}
        if not node_id:
            node_id = (
                f"{node_type}_{hashlib.md5(str(properties).encode()).hexdigest()[:8]}"
            )
        node = TextNode(
            text=f"{node_type}: {properties.get('name', node_id)}",
            id_=node_id,
            metadata=metadata,
        )
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
        if not self.kg_index:
            logger.error("Knowledge graph not initialized when adding relation.")
            return {"status": "error", "error": "KG not initialized"}
        try:
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
            logger.error("Error adding KG relation: %s", str(e))
            return {"status": "error", "error": str(e)}

    def query_knowledge_graph(self, query_text: str) -> Dict[str, Any]:
        if not self.kg_index:
            logger.error("KG not initialized when querying.")
            return {"status": "error", "error": "KG not initialized"}
        kg_query_engine = self.kg_index.as_query_engine(
            response_mode=ResponseMode.COMPACT
        )
        try:
            response = kg_query_engine.query(query_text)
            logger.info("Queried KG with query: '%s'", query_text)
            return {
                "query": query_text,
                "response": response.response,
                "source_nodes": [node.node.metadata for node in response.source_nodes],
            }
        except Exception as e:
            logger.error("Error querying KG: %s", str(e))
            return {"status": "error", "error": str(e)}

    def validate_extraction(
        self, extraction_result: Dict[str, Any], domain_expert: str = "user"
    ) -> Dict[str, Any]:
        extraction_result = self._normalize_input(extraction_result)
        logger.info(
            "Validation: Extraction reviewed by %s: %s",
            domain_expert,
            extraction_result,
        )
        if not isinstance(extraction_result, dict):
            extraction_result = {"result": extraction_result}
        if extraction_result.get("status") != "error":
            extraction_result["status"] = extraction_result.get(
                "status", "needs_validation"
            )
        extraction_result["validated_by"] = domain_expert
        extraction_result["validation_timestamp"] = datetime.now().isoformat()
        return extraction_result

    def validate_ontology_change(
        self, change_proposal: Dict[str, Any], domain_expert: str = "user"
    ) -> Dict[str, Any]:
        change_proposal = self._normalize_input(change_proposal)
        logger.info(
            "Validation: Ontology change reviewed by %s: %s",
            domain_expert,
            change_proposal,
        )
        if not isinstance(change_proposal, dict):
            change_proposal = {"change": change_proposal}
        change_proposal["validated_by"] = domain_expert
        change_proposal["validation_timestamp"] = datetime.now().isoformat()
        change_proposal["status"] = "validated"
        return change_proposal

    def validate_kg_addition(
        self, kg_addition: Dict[str, Any], domain_expert: str = "user"
    ) -> Dict[str, Any]:
        kg_addition = self._normalize_input(kg_addition)
        logger.info(
            "Validation: KG addition reviewed by %s: %s", domain_expert, kg_addition
        )
        if not isinstance(kg_addition, dict):
            kg_addition = {"kg_addition": kg_addition}
        kg_addition["validated_by"] = domain_expert
        kg_addition["validation_timestamp"] = datetime.now().isoformat()
        kg_addition["status"] = "validated"
        return kg_addition

    # ---------------- Ontology Update After Extraction ----------------

    def update_ontology_from_pdf_extraction(
        self, extraction_result: Dict[str, Any]
    ) -> None:
        if extraction_result.get("status") == "error":
            logger.warning("Extraction result has error, skipping ontology update.")
            return
        # For auto-validation, check confidence (if available)
        if self.auto_validate_extraction:
            # Use "high" as default if not specified
            if extraction_result.get("confidence", "high").lower() == "high":
                extraction_result["status"] = "validated"
            else:
                extraction_result["status"] = "needs_validation"
        etype = extraction_result.get("extraction_type")
        if etype == "material_properties":
            for material in extraction_result.get("materials", []):
                name = material.get("name")
                description = material.get("description", "No description provided")
                confidence = material.get("confidence", "high")
                if name:
                    provenance = {
                        "document_id": extraction_result.get("document_id"),
                        "confidence": confidence,
                    }
                    concept_uri = URIRef(self.ovp_ns + name.replace(" ", "_"))
                    if not list(self.ontology.triples((concept_uri, None, None))):
                        self.suggest_ontology_concept(
                            name, description, "Material", provenance=provenance
                        )
                        logger.info(
                            "Material concept '%s' added to ontology with provenance.",
                            name,
                        )
                    else:
                        logger.info(
                            "Material concept '%s' already exists in the ontology.",
                            name,
                        )
        elif etype == "experimental_methods":
            for method in extraction_result.get("methods", []):
                name = method.get("name")
                description = method.get("description", "No description provided")
                confidence = method.get("confidence", "high")
                if name:
                    provenance = {
                        "document_id": extraction_result.get("document_id"),
                        "confidence": confidence,
                    }
                    concept_uri = URIRef(self.ovp_ns + name.replace(" ", "_"))
                    if not list(self.ontology.triples((concept_uri, None, None))):
                        self.suggest_ontology_concept(
                            name, description, "Experiment", provenance=provenance
                        )
                        logger.info("Experimental method '%s' added to ontology.", name)
                    else:
                        logger.info(
                            "Experimental method '%s' already exists in the ontology.",
                            name,
                        )
        elif etype == "device_architecture":
            device = extraction_result.get("device", {})
            name = device.get("type")
            description = device.get("architecture", "No description provided")
            confidence = device.get("confidence", "high")
            if name:
                provenance = {
                    "document_id": extraction_result.get("document_id"),
                    "confidence": confidence,
                }
                concept_uri = URIRef(self.ovp_ns + name.replace(" ", "_"))
                if not list(self.ontology.triples((concept_uri, None, None))):
                    self.suggest_ontology_concept(
                        name, description, "Device", provenance=provenance
                    )
                    logger.info("Device concept '%s' added to ontology.", name)
                else:
                    logger.info(
                        "Device concept '%s' already exists in the ontology.", name
                    )

    # ---------------- Document Processing Functions ----------------

    def load_publications(self, directory: str = None) -> List[Document]:
        if directory is None:
            directory = f"{self.data_dir}/publications"
        logger.info("Loading publications from %s", directory)
        try:
            reader = SimpleDirectoryReader(directory)
            documents = reader.load_data()
            self.publications = documents
            logger.info("Loaded %d publications", len(documents))
            return documents
        except Exception as e:
            logger.error("Error loading publications: %s", str(e))
            return []

    def process_publications(self, documents: List[Document] = None) -> Dict[str, Any]:
        if documents is None:
            documents = self.publications
        if not documents:
            logger.error("No publications available for processing.")
            return {"status": "error", "error": "No documents to process"}
        results = []
        for doc in documents:
            doc_id = doc.metadata.get(
                "file_name", hashlib.md5(doc.text.encode()).hexdigest()[:8]
            )
            nodes = self.node_parser.get_nodes_from_documents([doc])
            for i, node in enumerate(nodes):
                chunk_text = node.text
                if not is_relevant_chunk(chunk_text):
                    logger.info("Skipping irrelevant chunk: %.100s", chunk_text)
                    continue
                try:
                    extraction_result = self.agents["publication"].query(
                        f"extract_material_properties({json.dumps(doc_id)}, {json.dumps(chunk_text)})"
                    )
                except ValueError as e:
                    logger.error("Ontology agent query error: %s", e)
                    extraction_result = {"error": str(e)}
                normalized = self._normalize_input(extraction_result)
                validated_result = self.validate_extraction(
                    normalized, domain_expert="user"
                )
                results.append(validated_result)
        logger.info(
            "Processed %d documents and %d chunks", len(documents), len(results)
        )
        return {
            "status": "completed",
            "documents_processed": len(documents),
            "chunks_processed": len(results),
            "results": results,
        }

    def load_experimental_data(self, directory: str = None) -> List[Dict[str, Any]]:
        if directory is None:
            directory = f"{self.data_dir}/experimental_data"
        logger.info("Loading experimental data from %s", directory)
        self.experimental_data = []  # Replace with actual loading code
        logger.info("Loaded %d experimental datasets", len(self.experimental_data))
        return self.experimental_data

    def process_experimental_data(
        self, datasets: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if datasets is None:
            datasets = self.experimental_data
        if not datasets:
            logger.error("No experimental data to process.")
            return {"status": "error", "error": "No experimental data to process"}
        results = []
        for dataset in datasets:
            dataset_id = dataset.get(
                "id", hashlib.md5(str(dataset).encode()).hexdigest()[:8]
            )
            results_extraction = self.agents["experiment"].query(
                f"extract_experimental_results({json.dumps(dataset_id)}, {json.dumps(dataset)})"
            )
            validated_result = self.validate_extraction(
                self._normalize_input(results_extraction), domain_expert="user"
            )
            if validated_result.get("status") != "error":
                self._update_kg_from_experiment(validated_result)
            results.append(validated_result)
        logger.info("Processed %d experimental datasets", len(datasets))
        return {
            "status": "completed",
            "datasets_processed": len(datasets),
            "results": results,
        }

    def _update_kg_from_experiment(self, experiment_result: Dict[str, Any]) -> None:
        id = experiment_result.get("experiment_id")
        results = experiment_result.get("results", {})
        experiment_node = self.agents["kg"].query(
            f"add_kg_node({json.dumps(id)}, 'Experiment', {json.dumps(results)})"
        )
        for condition_name, condition_value in experiment_result.get(
            "conditions", {}
        ).items():
            param_mapping = self.agents["experiment"].query(
                f"map_parameters_to_ontology({json.dumps(condition_name)}, {json.dumps(condition_value)})"
            )
            if param_mapping.get("status") == "needs_ontology_addition":
                suggestion = param_mapping.get("suggested_concept", {})
                new_concept = self.suggest_ontology_concept(
                    suggestion.get("name"),
                    suggestion.get("description"),
                    suggestion.get("parent_concept"),
                )
                self.validate_ontology_change(new_concept, domain_expert="user")
            condition_node = self.agents["kg"].query(
                f"add_kg_node('', 'Condition', {json.dumps({'name': condition_name, 'value': condition_value})})"
            )
            exp_node_id = experiment_node.get('node_id')
            cond_node_id = condition_node.get('node_id')
            self.agents["kg"].query(
                f"add_kg_relation({json.dumps(exp_node_id)}, 'has_condition', {json.dumps(cond_node_id)})"
            )

    def setup_rag_system(self) -> Dict[str, Any]:
        if not self.kg_index:
            logger.error("Cannot set up RAG system: KG not initialized.")
            return {"status": "error", "error": "KG not initialized"}
        kg_retriever = KnowledgeGraphQueryEngine(
            graph_store=self.graph_store, llm=self.llm, retrieval_mode="keyword"
        )
        vector_retriever = (
            self.vector_index.as_retriever() if self.vector_index else None
        )
        if vector_retriever:
            from llama_index.core.retrievers import QueryFusionRetriever

            hybrid_retriever = QueryFusionRetriever(
                [kg_retriever, vector_retriever],
                similarity_top_k=3,
                num_queries=1,
                mode="simple",
            )
            from llama_index.core.query_engine import RetrieverQueryEngine

            rag_query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever, service_context=self.storage_context
            )
        else:
            rag_query_engine = self.kg_index.as_query_engine()
        self.rag_query_engine = rag_query_engine
        logger.info("RAG system setup completed with KG integration.")
        return {
            "status": "success",
            "message": "RAG system initialized with KG integration",
        }

    def evaluate_rag_system(
        self, evaluation_queries: List[str], compare_with_baseline: bool = True
    ) -> Dict[str, Any]:
        if not hasattr(self, "rag_query_engine"):
            setup_result = self.setup_rag_system()
            if setup_result.get("status") != "success":
                return {
                    "status": "error",
                    "phase": "rag_evaluation",
                    "error": setup_result.get("error"),
                }
        evaluator = ResponseEvaluator()
        results = []
        for query in evaluation_queries:
            kg_rag_response = self.rag_query_engine.query(query)
            baseline_response = (
                self.vector_index.as_query_engine().query(query)
                if (compare_with_baseline and self.vector_index)
                else None
            )
            kg_rag_eval = evaluator.evaluate(
                query=query, response=kg_rag_response.response, reference=None
            )
            baseline_eval = (
                evaluator.evaluate(
                    query=query, response=baseline_response.response, reference=None
                )
                if baseline_response
                else None
            )
            query_result = {
                "query": query,
                "kg_rag_response": kg_rag_response.response,
                "kg_rag_evaluation": {
                    "score": kg_rag_eval.score,
                    "feedback": kg_rag_eval.feedback,
                },
            }
            if baseline_eval:
                query_result["baseline_response"] = baseline_response.response
                query_result["baseline_evaluation"] = {
                    "score": baseline_eval.score,
                    "feedback": baseline_eval.feedback,
                }
                query_result["improvement"] = kg_rag_eval.score - baseline_eval.score
            results.append(query_result)
        avg_kg_rag_score = sum(r["kg_rag_evaluation"]["score"] for r in results) / len(
            results
        )
        evaluation_result = {
            "status": "completed",
            "queries_evaluated": len(evaluation_queries),
            "average_kg_rag_score": avg_kg_rag_score,
            "detailed_results": results,
        }
        if compare_with_baseline and self.vector_index:
            avg_baseline_score = sum(
                r["baseline_evaluation"]["score"] for r in results
            ) / len(results)
            avg_improvement = sum(r["improvement"] for r in results) / len(results)
            evaluation_result["average_baseline_score"] = avg_baseline_score
            evaluation_result["average_improvement"] = avg_improvement
            evaluation_result["improvement_percentage"] = (
                (avg_improvement / avg_baseline_score) * 100
                if avg_baseline_score > 0
                else 0
            )
        logger.info("RAG system evaluated on %d queries", len(evaluation_queries))
        return evaluation_result

    def run_phase_1_ontology_development(self) -> Dict[str, Any]:
        logger.info("Starting Phase 1: Initial Ontology Development")
        concepts = [
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
                "description": "Phenyl-C61-butyric acid methyl ester, a fullerene derivative used as an electron acceptor",
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
        concept_results = []
        for concept in concepts:
            try:
                result = self.agents["ontology"].query(
                    f"suggest_ontology_concept({json.dumps(concept['name'])},"
                    f"{json.dumps(concept['description'])},"
                    f"{json.dumps(concept.get('parent'))})"
                )
            except ValueError as e:
                logger.error("Ontology agent query error: %s", e)
                result = {"error": str(e)}
            normalized = self._normalize_input(result)
            validated_result = self.validate_ontology_change(
                normalized, domain_expert="user"
            )
            logger.info("Concept validation result: %s", validated_result)
            concept_results.append(validated_result)
        relationships = [
            {
                "source": "P3HT",
                "relation": "has_property",
                "target": "DomainOrientation",
                "description": "P3HT forms domains with specific orientations",
            },
            {
                "source": "ProcessingMethod",
                "relation": "affects",
                "target": "DomainOrientation",
                "description": "Processing methods influence polymer orientation",
            },
            {
                "source": "BulkHeterojunction",
                "relation": "contains_material",
                "target": "P3HT",
                "description": "P3HT is used as donor material",
            },
            {
                "source": "BulkHeterojunction",
                "relation": "contains_material",
                "target": "PCBM",
                "description": "PCBM is used as acceptor material",
            },
        ]
        relationship_results = []
        for rel in relationships:
            try:
                result = self.agents["ontology"].query(
                    f"add_ontology_relationship({json.dumps(rel['source'])},"
                    f"{json.dumps(rel['relation'])},"
                    f"{json.dumps(rel['target'])},"
                    f"{json.dumps(rel['description'])})"
                )
            except ValueError as e:
                logger.error("Ontology agent query error: %s", e)
                result = {"error": str(e)}
            normalized = self._normalize_input(result)
            validated_result = self.validate_ontology_change(
                normalized, domain_expert="user"
            )
            relationship_results.append(validated_result)
        ontology_path = self.save_ontology("phase1")
        logger.info("Ontology development phase completed")
        return {
            "status": "completed",
            "phase": "ontology_development",
            "concepts_added": len(concept_results),
            "relationships_added": len(relationship_results),
            "ontology_path": ontology_path,
        }

    def run_phase_2_publication_analysis(self) -> Dict[str, Any]:
        logger.info("Starting Phase 2: Publication Analysis")
        documents = self.load_publications()
        if not documents:
            logger.error("No publications found in Phase 2.")
            return {
                "status": "error",
                "phase": "publication_analysis",
                "error": "No publications",
            }
        results = self.process_publications(documents)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        self.vector_index = VectorStoreIndex(
            nodes, storage_context=StorageContext.from_defaults()
        )
        if self.vector_index:
            self.vector_index.storage_context.persist(
                f"{self.persist_dir}/vector/phase2"
            )
        if self.kg_index:
            self.kg_index.storage_context.persist(f"{self.persist_dir}/kg/phase2")
        ontology_path = self.save_ontology("phase2")
        logger.info("Phase 2 completed: Processed %d documents", len(documents))
        return {
            "status": "completed",
            "phase": "publication_analysis",
            "documents_processed": len(documents),
            "ontology_path": ontology_path,
            "results": results,
        }

    def run_phase_3_experimental_data(self) -> Dict[str, Any]:
        logger.info("Starting Phase 3: Experimental Data Integration")
        datasets = self.load_experimental_data()
        if not datasets:
            logger.error("No experimental data found in Phase 3.")
            return {
                "status": "error",
                "phase": "experimental_data",
                "error": "No experimental data",
            }
        results = self.process_experimental_data(datasets)
        if self.kg_index:
            self.kg_index.storage_context.persist(f"{self.persist_dir}/kg/phase3")
        ontology_path = self.save_ontology("phase3")
        logger.info("Phase 3 completed: Processed %d datasets", len(datasets))
        return {
            "status": "completed",
            "phase": "experimental_data",
            "datasets_processed": len(datasets),
            "ontology_path": ontology_path,
            "results": results,
        }

    def run_phase_4_rag_evaluation(
        self, evaluation_queries: List[str] = None
    ) -> Dict[str, Any]:
        logger.info("Starting Phase 4: RAG Evaluation")
        if evaluation_queries is None:
            evaluation_queries = [
                "How does processing affect polymer orientation?",
                "What are key factors for organic solar cell performance?",
                "What is the relationship between polymer crystallinity and charge transport?",
                "Which methods are used for polymer thin films?",
                "How do different polymers compare in efficiency?",
            ]
        setup_result = self.setup_rag_system()
        if setup_result.get("status") != "success":
            logger.error("RAG system setup failed: %s", setup_result.get("error"))
            return {
                "status": "error",
                "phase": "rag_evaluation",
                "error": setup_result.get("error"),
            }
        evaluation_result = self.evaluate_rag_system(
            evaluation_queries, compare_with_baseline=True
        )
        logger.info("Phase 4 completed: RAG evaluation done")
        return {
            "status": "completed",
            "phase": "rag_evaluation",
            "evaluation": evaluation_result,
        }

    def run_complete_workflow(self) -> Dict[str, Any]:
        results = {}
        results["phase1"] = self.run_phase_1_ontology_development()
        results["phase2"] = (
            self.run_phase_2_publication_analysis()
            if results["phase1"].get("status") == "completed"
            else {"status": "skipped", "reason": "Phase 1 failed"}
        )
        results["phase3"] = (
            self.run_phase_3_experimental_data()
            if results["phase2"].get("status") == "completed"
            else {"status": "skipped", "reason": "Phase 2 failed"}
        )
        results["phase4"] = (
            self.run_phase_4_rag_evaluation()
            if results["phase3"].get("status") == "completed"
            else {"status": "skipped", "reason": "Phase 3 failed"}
        )
        overall_status = (
            "completed"
            if all(r.get("status") == "completed" for r in results.values())
            else "partial"
        )
        logger.info("Complete workflow finished with status: %s", overall_status)
        return {"status": overall_status, "workflow_results": results}


# ---------------- Example Usage ----------------

if __name__ == "__main__":
    opv_system = OPVKnowledgeGraphSystem(
        llm_model="llama3.2:latest",
        data_dir=".",
        persist_dir="./storage",
        debug=True,
        ollama_base_url="http://localhost:11434",
    )

    ontology_result = opv_system.run_phase_1_ontology_development()
    print(f"Ontology initialization: {ontology_result['status']}")

    papers = opv_system.load_publications("./polymer_papers/")
    print(f"Loaded {len(papers)} polymer research papers")

    processing_result = opv_system.process_publications(papers)
    print(f"Paper processing: {processing_result['status']}")
    print(f"Processed {processing_result.get('documents_processed', 0)} documents")

    if opv_system.kg_index:
        opv_system.kg_index.storage_context.persist("./storage/kg/polymer_papers")
        print("Knowledge graph saved")
    ontology_path = opv_system.save_ontology("polymer_papers_update")
    print(f"Updated ontology saved to {ontology_path}")

    rag_setup = opv_system.setup_rag_system()
    print(f"RAG system setup: {rag_setup['status']}")

    test_queries = [
        "How does processing affect polymer orientation?",
        "What are key factors for organic solar cell performance?",
        "What is the relationship between polymer crystallinity and charge transport?",
        "Which methods are used for polymer thin films?",
        "How do different polymers compare in efficiency?",
    ]
    evaluation = opv_system.evaluate_rag_system(
        test_queries, compare_with_baseline=True
    )
    print("\nRAG System Evaluation:")
    print(f"Average KG-enhanced score: {evaluation['average_kg_rag_score']:.2f}")
    if "average_baseline_score" in evaluation:
        print(f"Average baseline score: {evaluation['average_baseline_score']:.2f}")
        print(f"Improvement: {evaluation['improvement_percentage']:.2f}%")

    def ask_question():
        query = input(
            "\nEnter your question about polymer materials (or 'exit' to quit): "
        )
        if query.lower() == "exit":
            return False
        response = opv_system.query_knowledge_graph(query)
        print(f"\nResponse: {response['response']}")
        print("\nSources:")
        for i, source in enumerate(response.get("source_nodes", []), 1):
            print(
                f"{i}. {source.get('node_type', 'Unknown')}: {source.get('name', 'Unnamed')}"
            )
        return True

    print("\n--- Interactive Query Mode ---")
    print("Ask questions about the polymer papers that were processed")
    while ask_question():
        pass
