"""
CBORG Implementation of PublicationAnalysis
------------------------------------------
Implements the BasePublicationAnalysis class with CBORG LLM provider.
"""

import os
import json
import logging
import hashlib
from types import SimpleNamespace
from typing import Any, Dict, List, Callable

import openai
from modules.extractor_base import BasePublicationAnalysis

logger = logging.getLogger(__name__)


class DirectFunctionCaller:
    """
    A simple function caller that directly executes functions
    without the complexity of an agent framework.
    """

    def __init__(
        self,
        function: Callable,
        name: str = "",
        description: str = "",
    ):
        """Initialize with a specific function to call."""
        self.function = function
        self.name = name
        self.description = description

    def query(self, *args, **kwargs):
        """Directly execute the function with provided arguments."""
        logger.info(f"Calling function {self.name} directly")
        try:
            result = self.function(*args, **kwargs)
            logger.info(f"Function {self.name} execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing function {self.name}: {str(e)}")
            return {"status": "error", "error": str(e)}


class CBORGPublicationAnalysis(BasePublicationAnalysis):
    """Implements PublicationAnalysis using CBORG LLM provider with direct function calling."""

    def __init__(
        self,
        llm_model: str = None,
        temperature: float = 0.1,
        data_dir: str = "./",
        persist_dir: str = "./storage",
        debug: bool = True,
        cborg_api_key: str = None,
        cborg_base_url: str = "https://api.cborg.lbl.gov",
        **kwargs,
    ):
        """Initialize the CBORG publication analysis."""
        # Store CBORG-specific settings
        self.cborg_base_url = cborg_base_url
        self.cborg_api_key = cborg_api_key or os.environ.get("CBORG_API_KEY")

        if not self.cborg_api_key:
            raise ValueError(
                "CBORG API key must be provided or set as CBORG_API_KEY environment variable"
            )

        # Initialize OpenAI client for direct API calls - using the Client class as in the example
        self.client = openai.Client(api_key=self.cborg_api_key, base_url=cborg_base_url)

        # Set the model with correct format
        if llm_model is None:
            # Use the default model format as shown in the example
            self.llm_model = "lbl/cborg-deepthought:latest"
        elif not llm_model.startswith("lbl/"):
            # Add the prefix if it's missing
            self.llm_model = f"lbl/{llm_model}"
        else:
            # Use as provided
            self.llm_model = llm_model

        # Store basic parameters
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.debug = debug
        self.temperature = temperature

        # Log the model we're using
        logger.info(f"Using CBORG model: {self.llm_model}")

        # Call parent constructor to set up the base functionality
        super().__init__(
            llm_model=self.llm_model,
            temperature=temperature,
            data_dir=data_dir,
            persist_dir=persist_dir,
            debug=debug,
            **kwargs,
        )

        # Replace the agents with our direct function callers
        self.agent = DirectFunctionCaller(
            function=self.extract_material_properties,
            name="extract_material_properties",
            description="Extracts material properties from a scientific paper.",
        )

        self.extract_methods_caller = DirectFunctionCaller(
            function=self.extract_experimental_methods,
            name="extract_experimental_methods",
            description="Extracts experimental methods from a scientific paper.",
        )

        self.extract_device_caller = DirectFunctionCaller(
            function=self.extract_device_architecture,
            name="extract_device_architecture",
            description="Extracts device architecture from a scientific paper.",
        )

        self.validation_agent = DirectFunctionCaller(
            function=self.validate_extraction,
            name="validate_extraction",
            description="Validates extracted information.",
        )

        self.kg_agent = DirectFunctionCaller(
            function=self.add_kg_node,
            name="add_kg_node",
            description="Adds a node to the knowledge graph.",
        )

        self.kg_relation_agent = DirectFunctionCaller(
            function=self.add_kg_relation,
            name="add_kg_relation",
            description="Adds a relation to the knowledge graph.",
        )

        self.ontology_agent = DirectFunctionCaller(
            function=self.add_new_concept_to_ontology,
            name="add_new_concept_to_ontology",
            description="Adds a new concept to the ontology.",
        )

        self.concept_classifier = DirectFunctionCaller(
            function=self.classify_concept,
            name="classify_concept",
            description="Classifies a concept and identifies its parent.",
        )

        logger.info(
            f"Initialized CBORG publication analysis with model={self.llm_model}"
        )

    def _initialize_llm(self):
        """
        Initialize the CBORG LLM.
        For CBORG, we don't actually need to return a LlamaIndex LLM instance.
        """
        return None

    def call_llm(self, prompt: str) -> Any:
        """Call the CBORG LLM with a prompt using the OpenAI API."""
        if not prompt or prompt.strip() == "":
            logger.error("LLM call received an empty prompt.")
            raise ValueError("Prompt for LLM call cannot be empty.")

        logger.info("Calling CBORG LLM with prompt length %d", len(prompt))

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            logger.info("CBORG LLM call succeeded")

            # Create a simple object that mimics the structure expected by the rest of the code
            return SimpleNamespace(text=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"CBORG LLM call failed: {str(e)}")
            # Create a simple response with error message
            return SimpleNamespace(text=json.dumps({"error": str(e)}))

    def process_publications(self, documents: List = None) -> Dict[str, Any]:
        """
        Override the process_publications method to use our direct function callers
        instead of the ReAct agents.
        """
        if documents is None:
            documents = self.publications

        if not documents:
            logger.error("No documents available for processing.")
            return {"status": "error", "error": "No documents to process"}

        # Load existing ontology concepts before processing
        if self.ontology:
            self._load_existing_ontology_concepts()

        results = []
        new_concepts_added = 0

        # Test the model first with a simple query
        test_prompt = "Extract key concepts from this text: materials science is the study of properties of materials."
        try:
            test_response = self.call_llm(test_prompt)
            logger.info(
                f"Model test successful. Response: {test_response.text[:100]}..."
            )
            model_working = True
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            logger.warning("Skipping LLM-based extraction due to model error")
            model_working = False

        # Process each document
        for doc_index, doc in enumerate(documents):
            # Generate a document ID
            doc_id = doc.metadata.get(
                "file_name", hashlib.md5(doc.text.encode()).hexdigest()[:8]
            )

            logger.info(f"Processing document {doc_index+1}/{len(documents)}: {doc_id}")

            # Parse the document into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents([doc])
            logger.info(f"Document parsed into {len(nodes)} chunks")

            # Process each chunk
            for i, node in enumerate(nodes):
                logger.info(f"Processing chunk {i+1}/{len(nodes)} of document {doc_id}")

                # Check if this is a substantive chunk (we'll skip metadata chunks)
                if len(node.text.strip()) < 100:
                    logger.info("Skipping small chunk (likely metadata)")
                    continue

                # Create empty results for the chunk
                chunk_results = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "material_properties": {"status": "skipped", "materials": []},
                    "experimental_methods": {"status": "skipped", "methods": []},
                    "device_architecture": {"status": "skipped", "device": {}},
                }

                # Only attempt extraction if model is working
                if model_working:
                    # Print a small sample of the text to help debug
                    text_sample = node.text[:200].replace("\n", " ").strip()
                    logger.info(f"Chunk sample: {text_sample}...")

                    # Extract material properties
                    try:
                        material_result = self.agent.query(doc_id, node.text)
                        if material_result.get(
                            "status"
                        ) != "error" and material_result.get("materials", []):
                            logger.info(
                                f"Extracted {len(material_result.get('materials', []))} materials from chunk"
                            )

                            # Validate the extraction
                            validated_material = self.validation_agent.query(
                                material_result
                            )
                            if validated_material.get("status") == "validated":
                                logger.info("Material properties validated")

                                # Update KG and ontology
                                new_concepts = self._enrich_ontology_with_new_concepts(
                                    validated_material
                                )
                                new_concepts_added += len(new_concepts)
                                logger.info(
                                    f"Added {len(new_concepts)} new concepts from material properties"
                                )

                                # Add to KG
                                self._update_kg_from_extraction(validated_material)

                                # Update chunk results
                                chunk_results["material_properties"] = (
                                    validated_material
                                )
                        else:
                            logger.info("No materials extracted from this chunk")
                    except Exception as e:
                        logger.error(f"Error extracting material properties: {str(e)}")

                    # Extract experimental methods
                    try:
                        methods_result = self.extract_methods_caller.query(
                            doc_id, node.text
                        )
                        if methods_result.get(
                            "status"
                        ) != "error" and methods_result.get("methods", []):
                            logger.info(
                                f"Extracted {len(methods_result.get('methods', []))} methods from chunk"
                            )

                            # Validate the extraction
                            validated_methods = self.validation_agent.query(
                                methods_result
                            )
                            if validated_methods.get("status") == "validated":
                                logger.info("Experimental methods validated")

                                # Update KG and ontology
                                new_concepts = self._enrich_ontology_with_new_concepts(
                                    validated_methods
                                )
                                new_concepts_added += len(new_concepts)
                                logger.info(
                                    f"Added {len(new_concepts)} new concepts from experimental methods"
                                )

                                # Add to KG
                                self._update_kg_from_extraction(validated_methods)

                                # Update chunk results
                                chunk_results["experimental_methods"] = (
                                    validated_methods
                                )
                        else:
                            logger.info("No methods extracted from this chunk")
                    except Exception as e:
                        logger.error(f"Error extracting experimental methods: {str(e)}")

                    # Extract device architecture
                    try:
                        device_result = self.extract_device_caller.query(
                            doc_id, node.text
                        )
                        if device_result.get("status") != "error" and device_result.get(
                            "device", {}
                        ):
                            logger.info("Device architecture extracted")

                            # Validate the extraction
                            validated_device = self.validation_agent.query(
                                device_result
                            )
                            if validated_device.get("status") == "validated":
                                logger.info("Device architecture validated")

                                # Update KG and ontology
                                new_concepts = self._enrich_ontology_with_new_concepts(
                                    validated_device
                                )
                                new_concepts_added += len(new_concepts)
                                logger.info(
                                    f"Added {len(new_concepts)} new concepts from device architecture"
                                )

                                # Add to KG
                                self._update_kg_from_extraction(validated_device)

                                # Update chunk results
                                chunk_results["device_architecture"] = validated_device
                        else:
                            logger.info(
                                "No device architecture extracted from this chunk"
                            )
                    except Exception as e:
                        logger.error(f"Error extracting device architecture: {str(e)}")
                else:
                    logger.info("Skipping extraction due to model error")

                # Add the results for this chunk
                results.append(chunk_results)

        logger.info(f"Processed {len(documents)} documents and {len(results)} chunks")
        logger.info(f"Added {new_concepts_added} new concepts to the ontology")

        # Save the updated ontology
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
            "new_concepts_added": new_concepts_added,
            "results": results,
        }

    def run(self) -> Dict[str, Any]:
        """
        Override the run method to skip vector index creation for CBORG.
        """
        logger.info("Running publication analysis phase with CBORG")

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

        # Process publications with our custom implementation
        results = self.process_publications(documents)

        # Skip vector index creation for CBORG implementation
        logger.info("Skipping vector index creation for CBORG implementation")
        self.vector_index = None

        # If KG index exists, persist it
        if self.kg_index:
            try:
                self.kg_index.storage_context.persist(f"{self.persist_dir}/kg/phase2")
            except Exception as e:
                logger.warning(f"Could not persist KG index: {str(e)}")

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
            "new_concepts_added": results.get("new_concepts_added", 0),
        }

    def _update_kg_from_extraction(self, extraction_result: Dict[str, Any]) -> None:
        """Update knowledge graph based on validated extraction results."""
        extraction_type = extraction_result.get("extraction_type")

        # First, enrich the ontology with new concepts
        if self.ontology:
            try:
                added_concepts = self._enrich_ontology_with_new_concepts(
                    extraction_result
                )
                new_concepts_added = len(added_concepts)
                if new_concepts_added > 0:
                    logger.info(
                        f"Added {new_concepts_added} new concepts to the ontology from extraction"
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
                try:
                    # Add material node
                    material_node = self.add_kg_node("", "Material", material)
                    logger.info(
                        f"Added material node: {material.get('name', 'Unknown')}"
                    )

                    # Add property nodes and relationships
                    for prop in material.get("properties", []):
                        try:
                            # Add property node
                            prop_node = self.add_kg_node("", "Property", prop)
                            logger.info(
                                f"Added property node: {prop.get('property_name', 'Unknown')}"
                            )

                            # Add relationship between material and property
                            if material_node and prop_node:
                                self.add_kg_relation(
                                    material_node.get("node_id", ""),
                                    "has_property",
                                    prop_node.get("node_id", ""),
                                )
                                logger.info(
                                    f"Added relation: {material.get('name', 'Unknown')} -> has_property ->"
                                    f" {prop.get('property_name', 'Unknown')}"
                                )
                        except Exception as e:
                            logger.error(f"Error adding property node: {str(e)}")
                except Exception as e:
                    logger.error(f"Error adding material node: {str(e)}")

        elif extraction_type == "experimental_methods":
            # Process experimental methods
            for method in extraction_result.get("methods", []):
                try:
                    # Add method node
                    method_node = self.add_kg_node("", "Method", method)
                    logger.info(f"Added method node: {method.get('name', 'Unknown')}")

                    # Add parameter nodes and relationships
                    for param in method.get("parameters", []):
                        try:
                            # Add parameter node
                            param_node = self.add_kg_node("", "Parameter", param)
                            logger.info(
                                f"Added parameter node: {param.get('parameter_name', 'Unknown')}"
                            )

                            # Add relationship between method and parameter
                            if method_node and param_node:
                                self.add_kg_relation(
                                    method_node.get("node_id", ""),
                                    "has_parameter",
                                    param_node.get("node_id", ""),
                                )
                                logger.info(
                                    f"Added relation: {method.get('name', 'Unknown')} -> has_parameter -> "
                                    f"{param.get('parameter_name', 'Unknown')}"
                                )
                        except Exception as e:
                            logger.error(f"Error adding parameter node: {str(e)}")
                except Exception as e:
                    logger.error(f"Error adding method node: {str(e)}")

        elif extraction_type == "device_architecture":
            # Process device architecture
            device = extraction_result.get("device", {})
            if device:
                try:
                    # Add device node
                    device_node = self.add_kg_node("", "Device", device)
                    logger.info(f"Added device node: {device.get('type', 'Unknown')}")

                    # Add layer nodes and relationships
                    for layer in device.get("layers", []):
                        try:
                            # Add layer node
                            layer_node = self.add_kg_node("", "Layer", layer)
                            logger.info(
                                f"Added layer node: {layer.get('name', 'Unknown')}"
                            )

                            # Add relationship between device and layer
                            if device_node and layer_node:
                                self.add_kg_relation(
                                    device_node.get("node_id", ""),
                                    "has_layer",
                                    layer_node.get("node_id", ""),
                                )
                                logger.info(
                                    f"Added relation: {device.get('type', 'Unknown')} -> has_layer -> "
                                    f"{layer.get('name', 'Unknown')}"
                                )
                        except Exception as e:
                            logger.error(f"Error adding layer node: {str(e)}")
                except Exception as e:
                    logger.error(f"Error adding device node: {str(e)}")
