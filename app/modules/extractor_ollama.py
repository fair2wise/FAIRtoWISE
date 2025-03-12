"""
Ollama Implementation of PublicationAnalysis
-------------------------------------------
Implements the BasePublicationAnalysis class with Ollama LLM provider.
"""

import logging

# from types import SimpleNamespace
from typing import Any

from llama_index.llms.ollama import Ollama
from modules.extractor_base import BasePublicationAnalysis

logger = logging.getLogger(__name__)


class OllamaPublicationAnalysis(BasePublicationAnalysis):
    """Implements PublicationAnalysis using Ollama LLM provider."""

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
        """Initialize the Ollama publication analysis."""
        # Store Ollama-specific settings
        self.ollama_base_url = ollama_base_url

        # Call parent constructor
        super().__init__(
            llm_model=llm_model,
            temperature=temperature,
            data_dir=data_dir,
            persist_dir=persist_dir,
            debug=debug,
            **kwargs,
        )

        logger.info(f"Initialized Ollama publication analysis with model={llm_model}")

    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        return Ollama(
            model=self.llm_model,
            temperature=self.temperature,
            base_url=self.ollama_base_url,
            request_timeout=240.0,
        )

    def call_llm(self, prompt: str) -> Any:
        """Call the Ollama LLM with a prompt."""
        if not prompt or prompt.strip() == "":
            logger.error("LLM call received an empty prompt.")
            raise ValueError("Prompt for LLM call cannot be empty.")

        logger.info("Calling Ollama LLM with prompt length %d", len(prompt))

        try:
            response = self.llm.complete(prompt)
            logger.info("Ollama LLM call succeeded")
            return response
        except Exception as e:
            logger.error("Ollama LLM call failed: %s", str(e))
            raise e
