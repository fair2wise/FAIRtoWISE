"""
LLM Factory Module
-----------------
Factory module to create LLM providers based on configuration.
Supports Ollama and CBORG backends.
"""

import os
import logging
from enum import Enum
from typing import Optional

import openai
from llama_index.llms.ollama import Ollama

# from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enum for supported LLM providers"""

    OLLAMA = "ollama"
    CBORG = "cborg"


class LLMFactory:
    """Factory class to create LLM instances based on configuration"""

    @staticmethod
    def create_llm(
        provider: str,
        model: str,
        temperature: float = 0.1,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Create and return an LLM instance based on specified provider

        Args:
            provider: The LLM provider (ollama or cborg)
            model: Model name to use
            temperature: Temperature setting for generation
            base_url: Base URL for the API
            api_key: API key (for CBORG)
            **kwargs: Additional provider-specific arguments

        Returns:
            LLM instance compatible with LlamaIndex
        """
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            logger.error(f"Unsupported LLM provider: {provider}. Defaulting to Ollama.")
            provider_enum = LLMProvider.OLLAMA

        if provider_enum == LLMProvider.OLLAMA:
            # Set defaults for Ollama
            ollama_url = base_url or "http://localhost:11434"

            logger.info(f"Creating Ollama LLM with model={model}, url={ollama_url}")
            return Ollama(
                model=model,
                temperature=temperature,
                base_url=ollama_url,
                request_timeout=240.0,
                **kwargs,
            )

        elif provider_enum == LLMProvider.CBORG:
            # Set defaults for CBORG
            cborg_url = base_url or "https://api.cborg.lbl.gov"
            cborg_api_key = api_key or os.environ.get("CBORG_API_KEY")

            if not cborg_api_key:
                raise ValueError(
                    "CBORG API key must be provided or set as CBORG_API_KEY environment variable"
                )

            logger.info(f"Creating CBORG LLM with model={model}, url={cborg_url}")
            return openai.OpenAI(
                model=model,
                temperature=temperature,
                api_key=cborg_api_key,
                base_url=cborg_url,
                **kwargs,
            )

    @staticmethod
    def create_direct_client(
        provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None
    ):
        """
        Create a direct client for the LLM API (for use outside of LlamaIndex)

        Args:
            provider: The LLM provider (ollama or cborg)
            api_key: API key (for CBORG)
            base_url: Base URL for the API

        Returns:
            Direct client for the LLM API
        """
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            logger.error(
                f"Unsupported LLM provider: {provider} for direct client. Defaulting to CBORG."
            )
            provider_enum = LLMProvider.CBORG

        if provider_enum == LLMProvider.CBORG:
            # Create OpenAI client for CBORG
            cborg_url = base_url or "https://api.cborg.lbl.gov"
            cborg_api_key = api_key or os.environ.get("CBORG_API_KEY")

            if not cborg_api_key:
                raise ValueError(
                    "CBORG API key must be provided or set as CBORG_API_KEY environment variable"
                )

            logger.info(f"Creating direct CBORG client with url={cborg_url}")
            return openai.OpenAI(api_key=cborg_api_key, base_url=cborg_url)

        # For Ollama, you might implement a direct client if needed
        # For now, return None
        logger.warning("Direct client for Ollama not implemented")
        return None
