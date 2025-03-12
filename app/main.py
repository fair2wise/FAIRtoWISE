"""
OPV Knowledge Graph Multi-Agent Framework
----------------------------------------
A modular framework for building knowledge graphs for organic photovoltaics research.
Supports multiple LLM providers (Ollama and CBORG).
"""

import dotenv
import os
import logging
from typing import Dict, Any

# Import factory modules
from modules.extractor_factory import create_publication_analysis
from llm_factory import LLMProvider
from modules.ontology import OntologyDevelopment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


class OPVKnowledgeGraphSystem:
    """
    Multi-agent system for building and utilizing knowledge graphs
    for organic photovoltaics (OPVs) research.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2:latest",
        temperature: float = 0.1,
        data_dir: str = ".",
        persist_dir: str = "./storage",
        debug: bool = True,
        ollama_base_url: str = "http://localhost:11434",
        cborg_api_key: str = None,
        cborg_base_url: str = "https://api.cborg.lbl.gov",
    ):
        """Initialize the OPV Knowledge Graph System."""
        # Create necessary directories
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.debug = debug
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model

        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)
        os.makedirs(f"{persist_dir}/vector", exist_ok=True)
        os.makedirs(f"{persist_dir}/kg", exist_ok=True)
        os.makedirs(f"{persist_dir}/ontology", exist_ok=True)

        # Common settings for all phases
        self.config = {
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "temperature": temperature,
            "data_dir": data_dir,
            "persist_dir": persist_dir,
            "debug": debug,
            "ollama_base_url": ollama_base_url,
            "cborg_api_key": cborg_api_key or os.environ.get("CBORG_API_KEY"),
            "cborg_base_url": cborg_base_url,
        }

        # Validate provider-specific configurations
        if (
            self.llm_provider == LLMProvider.CBORG.value
            and not self.config["cborg_api_key"]
        ):
            raise ValueError(
                "CBORG API key must be provided or set as CBORG_API_KEY environment variable"
            )

        # Initialize phase handlers
        # Note: For now we're assuming OntologyDevelopment hasn't been adapted yet
        # In a real implementation, you'd create a factory for this too
        if self.llm_provider == LLMProvider.OLLAMA.value:
            self.phase1 = OntologyDevelopment(
                llm_model=llm_model,
                temperature=temperature,
                data_dir=data_dir,
                persist_dir=persist_dir,
                debug=debug,
                ollama_base_url=ollama_base_url,
            )
        else:
            # This assumes OntologyDevelopment has been adapted for CBORG
            # You may need to implement this
            self.phase1 = OntologyDevelopment(
                llm_model=llm_model,
                temperature=temperature,
                data_dir=data_dir,
                persist_dir=persist_dir,
                debug=debug,
                cborg_api_key=self.config["cborg_api_key"],
                cborg_base_url=cborg_base_url,
            )

        # Use factory to create the appropriate PublicationAnalysis implementation
        self.phase2 = create_publication_analysis(
            provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            data_dir=data_dir,
            persist_dir=persist_dir,
            debug=debug,
            ollama_base_url=ollama_base_url,
            cborg_api_key=self.config["cborg_api_key"],
            cborg_base_url=cborg_base_url,
        )

        # Future phases would be implemented similarly
        # self.phase3 = ...
        # self.phase4 = ...

        # Shared resources
        self.ontology = self.phase1.get_ontology()

        logger.info(
            f"OPV Knowledge Graph System initialized with {llm_provider} provider."
        )

    def run_phase_1_ontology_development(self) -> Dict[str, Any]:
        """Run Phase 1: Initial Ontology Development with LLM Support."""
        logger.info("Starting Phase 1: Initial Ontology Development")
        result = self.phase1.run()

        # Share ontology with other phases
        if result["status"] == "completed":
            self.phase2.set_ontology(self.phase1.get_ontology())
            # self.phase3.set_ontology(self.phase1.get_ontology())
            # self.phase4.set_ontology(self.phase1.get_ontology())

        return result

    def run_phase_2_publication_analysis(self) -> Dict[str, Any]:
        """Run Phase 2: First AI Augmentation Cycle - Publications."""
        logger.info("Starting Phase 2: Publication Analysis")
        result = self.phase2.run()

        # Share knowledge graph and vector index with other phases
        if result["status"] == "completed":
            # Uncomment when phase 3 and 4 are implemented
            # self.phase3.set_kg_index(self.phase2.get_kg_index())
            # self.phase3.set_vector_index(self.phase2.get_vector_index())
            # self.phase4.set_kg_index(self.phase2.get_kg_index())
            # self.phase4.set_vector_index(self.phase2.get_vector_index())
            pass

        return result

    # Future phases would be implemented here
    # def run_phase_3_experimental_data(self) -> Dict[str, Any]:
    #     ...

    # def run_phase_4_rag_evaluation(self, evaluation_queries: List[str] = None) -> Dict[str, Any]:
    #     ...

    def run_complete_workflow(self) -> Dict[str, Any]:
        """Run the complete workflow."""
        results = {}

        # Phase 1: Ontology Development
        results["phase1"] = self.run_phase_1_ontology_development()

        # Phase 2: Publication Analysis
        if results["phase1"].get("status") == "completed":
            results["phase2"] = self.run_phase_2_publication_analysis()
        else:
            results["phase2"] = {
                "status": "skipped",
                "reason": "Phase 1 did not complete successfully",
            }

        # Future phases would be run here
        # if results["phase2"].get("status") == "completed":
        #     results["phase3"] = self.run_phase_3_experimental_data()
        # ...

        # Determine overall status
        overall_status = (
            "completed"
            if all(r.get("status") == "completed" for r in results.values())
            else "partial"
        )
        logger.info(f"Complete workflow finished with status: {overall_status}")

        return {"status": overall_status, "workflow_results": results}

    def query_knowledge_graph(self, query_text: str) -> Dict[str, Any]:
        """Interface to query the knowledge graph."""
        # This would normally call phase4's query method
        # return self.phase4.query_knowledge_graph(query_text)
        return {"status": "error", "error": "Query functionality not yet implemented"}


if __name__ == "__main__":
    OPVKnowledgeGraphSystem(
        llm_model="cborg-deepthought:latest",
        llm_provider="cborg",
        cborg_api_key=os.environ.get("CBORG_API_KEY"),
    ).run_complete_workflow()

    # OPVKnowledgeGraphSystem().run_complete_workflow()
