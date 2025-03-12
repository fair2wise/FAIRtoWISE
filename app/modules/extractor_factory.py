"""
Publication Analysis Factory Module
---------------------------------
Factory for creating PublicationAnalysis instances based on LLM provider.
"""

import logging
from typing import Any

from modules.extractor_ollama import OllamaPublicationAnalysis
from modules.extractor_cborg import CBORGPublicationAnalysis
from llm_factory import LLMProvider

logger = logging.getLogger(__name__)


def create_publication_analysis(
    provider: str,
    llm_model: str,
    temperature: float = 0.1,
    data_dir: str = "./",
    persist_dir: str = "./storage",
    debug: bool = True,
    **kwargs,
) -> Any:
    """
    Factory function to create a PublicationAnalysis instance based on specified provider

    Args:
        provider: LLM provider (ollama or cborg)
        llm_model: Model name to use
        temperature: Temperature setting for generation
        data_dir: Directory for input data
        persist_dir: Directory for persistent storage
        debug: Enable debug mode
        **kwargs: Additional provider-specific arguments

    Returns:
        PublicationAnalysis instance for the specified provider
    """
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        logger.error(f"Unsupported LLM provider: {provider}. Defaulting to Ollama.")
        provider_enum = LLMProvider.OLLAMA

    if provider_enum == LLMProvider.OLLAMA:
        logger.info(f"Creating Ollama PublicationAnalysis with model={llm_model}")
        return OllamaPublicationAnalysis(
            llm_model=llm_model,
            temperature=temperature,
            data_dir=data_dir,
            persist_dir=persist_dir,
            debug=debug,
            # ollama_base_url=kwargs.get("ollama_base_url", "http://localhost:11434"),
            **kwargs,
        )

    elif provider_enum == LLMProvider.CBORG:
        logger.info(f"Creating CBORG PublicationAnalysis with model={llm_model}")
        return CBORGPublicationAnalysis(
            llm_model=llm_model,
            temperature=temperature,
            data_dir=data_dir,
            persist_dir=persist_dir,
            debug=debug,
            # cborg_api_key=kwargs.get("cborg_api_key"),
            # cborg_base_url=kwargs.get("cborg_base_url", "https://api.cborg.lbl.gov"),
            **kwargs,
        )
