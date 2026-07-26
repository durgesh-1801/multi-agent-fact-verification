"""
Resilient LLM Factory service for provider-agnostic Chat Model clients (Groq, OpenAI, Gemini, Claude).
Configured via app.core.config.settings with lazy initialization, startup model validation,
404 fallback switching, 429 exponential backoff, and centralized get_llm(provider).
"""

import asyncio
from enum import Enum
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings

logger = logging.getLogger(__name__)

_llm_factory_lock = threading.Lock()


class LLMProvider(str, Enum):
    """Supported LLM Providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


PRIORITY_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

PRIORITY_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

PRIORITY_GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-flash-lite-latest",
]


class LLMFactory:
    """
    Singleton LLM Factory supporting dynamic lazy instantiation of Groq, OpenAI, Gemini, and Anthropic chat models.
    """

    _instance: Optional["LLMFactory"] = None
    _active_groq_model: Optional[str] = None
    _active_openai_model: Optional[str] = None
    _active_gemini_model: Optional[str] = None
    _validated_gemini_models: List[str] = []

    def __new__(cls) -> "LLMFactory":
        if cls._instance is None:
            with _llm_factory_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._active_groq_model = None
                    cls._instance._active_openai_model = None
                    cls._instance._active_gemini_model = None
                    cls._instance._validated_gemini_models = []
        return cls._instance

    def get_active_groq_model(self) -> str:
        """Returns active Groq model name."""
        if not self._active_groq_model:
            self._active_groq_model = settings.effective_groq_model
        return self._active_groq_model

    def switch_to_next_groq_model(self, failed_model: str) -> str:
        """
        Dynamically switches to next fallback Groq model on 404 or missing model error.
        """
        candidates = [m for m in PRIORITY_GROQ_MODELS if m != failed_model]
        new_model = candidates[0] if candidates else "llama-3.1-8b-instant"
        logger.warning(f"Switched active Groq model from '{failed_model}' -> '{new_model}'")
        print(f"DEBUG: Switched active Groq model from '{failed_model}' -> '{new_model}'")
        self._active_groq_model = new_model
        return new_model

    def get_active_openai_model(self) -> str:
        """Returns active OpenAI model name."""
        if not self._active_openai_model:
            self._active_openai_model = settings.effective_openai_model
        return self._active_openai_model

    def switch_to_next_openai_model(self, failed_model: str) -> str:
        """
        Dynamically switches to next fallback OpenAI model on 404 or missing model error.
        """
        candidates = [m for m in PRIORITY_OPENAI_MODELS if m != failed_model]
        new_model = candidates[0] if candidates else "gpt-4o-mini"
        logger.warning(f"Switched active OpenAI model from '{failed_model}' -> '{new_model}'")
        print(f"DEBUG: Switched active OpenAI model from '{failed_model}' -> '{new_model}'")
        self._active_openai_model = new_model
        return new_model

    def _discover_and_validate_gemini_models(self) -> str:
        """
        Discovers available Gemini models via Google GenAI SDK and selects the highest-priority working model.
        """
        if self._active_gemini_model:
            return self._active_gemini_model

        api_key = settings.google_api_key_str
        configured_model = settings.effective_gemini_model

        if not api_key:
            logger.warning("GOOGLE_API_KEY is not configured. Defaulting to 'gemini-3.1-flash-lite'.")
            self._active_gemini_model = "gemini-3.1-flash-lite"
            return self._active_gemini_model

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            raw_models = list(client.models.list())
            available_names = [m.name.replace("models/", "") for m in raw_models]
            self._validated_gemini_models = available_names

            logger.info(f"Discovered {len(available_names)} available Gemini models via Google GenAI SDK.")

            candidates = []
            if configured_model in available_names:
                candidates.append(configured_model)

            for priority_model in PRIORITY_GEMINI_MODELS:
                if priority_model in available_names and priority_model not in candidates:
                    candidates.append(priority_model)

            for m in available_names:
                if "flash" in m and "preview" not in m and m not in candidates:
                    candidates.append(m)

            best_model = candidates[0] if candidates else "gemini-3.1-flash-lite"
            logger.info(f"Selected validated Gemini model: '{best_model}'")
            self._active_gemini_model = best_model
            return self._active_gemini_model

        except Exception as e:
            logger.warning(f"Error during Gemini startup model validation: {e}. Defaulting to 'gemini-3.1-flash-lite'.")
            self._active_gemini_model = "gemini-3.1-flash-lite"
            return self._active_gemini_model

    def get_active_gemini_model(self) -> str:
        """Returns the currently active, validated Gemini model name."""
        if not self._active_gemini_model:
            return self._discover_and_validate_gemini_models()
        return self._active_gemini_model

    def switch_to_next_gemini_model(self, failed_model: str) -> str:
        """
        Dynamically switches to the next fallback model when current model returns 404 NOT_FOUND.
        """
        failed_clean = failed_model.replace("models/", "")
        logger.warning(f"Model '{failed_clean}' received 404 NOT_FOUND. Triggering automatic model fallback...")

        candidates = [m for m in PRIORITY_GEMINI_MODELS if m != failed_clean]
        if self._validated_gemini_models:
            for m in self._validated_gemini_models:
                if "flash" in m and m != failed_clean and m not in candidates:
                    candidates.append(m)

        new_model = candidates[0] if candidates else "gemini-3.1-flash-lite"
        logger.info(f"Switched active Gemini model from '{failed_clean}' -> '{new_model}'")
        self._active_gemini_model = new_model
        return new_model

    def get_groq(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = False,
    ) -> Any:
        """
        Instantiates ChatGroq using GROQ_MODEL and GROQ_API_KEY.
        """
        from langchain_groq import ChatGroq

        selected_model = model_name or self.get_active_groq_model()
        api_key = settings.groq_api_key_str
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured in environment variables or .env file.")

        logger.info(f"Instantiating ChatGroq model='{selected_model}', temp={temperature}")
        print(f"DEBUG: Instantiating ChatGroq model='{selected_model}'")

        return ChatGroq(
            api_key=api_key,
            model=selected_model,
            temperature=temperature,
            streaming=streaming,
            timeout=30.0,
        )

    def get_openai(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = False,
    ) -> Any:
        """
        Instantiates ChatOpenAI using OPENAI_MODEL and OPENAI_API_KEY.
        """
        if settings.effective_provider == "groq":
            raise RuntimeError(
                "ASSERTION FAILURE: ChatOpenAI instantiation was attempted while LLM_PROVIDER is configured to 'groq'!"
            )

        from langchain_openai import ChatOpenAI

        selected_model = model_name or self.get_active_openai_model()
        api_key = settings.openai_api_key_str
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured in environment variables or .env file.")

        logger.info(f"Instantiating ChatOpenAI model='{selected_model}', temp={temperature}")
        print(f"DEBUG: Instantiating ChatOpenAI model='{selected_model}'")

        return ChatOpenAI(
            model=selected_model,
            api_key=api_key,
            temperature=temperature,
            streaming=streaming,
            request_timeout=30.0,
        )

    def get_anthropic(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = False,
    ) -> Any:
        from langchain_anthropic import ChatAnthropic

        selected_model = model_name or settings.CLAUDE_MODEL_NAME
        api_key = settings.anthropic_api_key_str
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")
        return ChatAnthropic(
            model=selected_model,
            api_key=api_key,
            temperature=temperature,
            streaming=streaming,
            timeout=30.0,
        )

    def get_gemini(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = False,
    ) -> Any:
        if settings.effective_provider in ("groq", "openai"):
            raise RuntimeError(
                f"ASSERTION FAILURE: ChatGoogleGenerativeAI instantiation was attempted while LLM_PROVIDER is configured to '{settings.effective_provider}'!"
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.google_api_key_str
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")

        selected_model = (model_name or self.get_active_gemini_model()).replace("models/", "")
        logger.info(f"Instantiating ChatGoogleGenerativeAI model='{selected_model}', temp={temperature}")
        print(f"DEBUG: Instantiating ChatGoogleGenerativeAI model='{selected_model}'")

        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
            temperature=temperature,
            streaming=streaming,
            timeout=30.0,
        )

    def get_raw_llm(
        self,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = False,
    ) -> Any:
        """
        Returns raw base ChatModel without retry wrapper.
        """
        configured_provider = settings.effective_provider

        # Enforce global provider if system is set to groq or openai
        if configured_provider == LLMProvider.GROQ.value:
            return self.get_groq(temperature=temperature, streaming=streaming)
        elif configured_provider == LLMProvider.OPENAI.value:
            return self.get_openai(temperature=temperature, streaming=streaming)

        provider_clean = (provider or configured_provider).lower().strip()

        if provider_clean == LLMProvider.GROQ.value:
            return self.get_groq(temperature=temperature, streaming=streaming)
        elif provider_clean == LLMProvider.OPENAI.value:
            return self.get_openai(temperature=temperature, streaming=streaming)
        elif provider_clean == LLMProvider.CLAUDE.value:
            return self.get_anthropic(temperature=temperature, streaming=streaming)
        elif provider_clean == LLMProvider.GEMINI.value:
            return self.get_gemini(temperature=temperature, streaming=streaming)
        else:
            return self.get_groq(temperature=temperature, streaming=streaming)


def estimate_tokens(text: str) -> int:
    """Estimates token count for input text (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int, provider: str) -> float:
    """Estimates USD cost based on provider and token counts."""
    p_lower = provider.lower()
    if "groq" in p_lower:
        cost = (prompt_tokens * 0.00059 / 1000) + (completion_tokens * 0.00079 / 1000)
    elif "openai" in p_lower:
        cost = (prompt_tokens * 0.0025 / 1000) + (completion_tokens * 0.010 / 1000)
    else:
        cost = (prompt_tokens * 0.00015 / 1000) + (completion_tokens * 0.00060 / 1000)
    return round(cost, 6)


def log_token_usage(prompt_text: str, estimated_completion_tokens: int, provider: str):
    """Prints token usage and estimated cost before every LLM call."""
    prompt_tokens = estimate_tokens(prompt_text)
    cost = estimate_cost(prompt_tokens, estimated_completion_tokens, provider)

    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {estimated_completion_tokens}")
    print(f"Estimated Cost: ${cost:.6f}")
    logger.info(f"LLM Call ({provider}) - Prompt Tokens: {prompt_tokens}, Completion Tokens: {estimated_completion_tokens}, Est Cost: ${cost:.6f}")


MAX_PROMPT_TOKENS = 6000


def enforce_prompt_token_limit(input_messages: Any) -> Any:
    """
    Ensures input prompt tokens do not exceed 6000 tokens limit.
    Automatically trims messages if necessary.
    """
    prompt_str = str(input_messages)
    tokens = estimate_tokens(prompt_str)
    if tokens <= MAX_PROMPT_TOKENS:
        return input_messages

    logger.warning(f"Prompt size ({tokens} tokens) exceeds 6000 limit. Automatically trimming evidence context...")
    print(f"DEBUG: Prompt size ({tokens} tokens) exceeds 6000 limit. Automatically trimming evidence context...")

    if isinstance(input_messages, list):
        for msg in reversed(input_messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 1000:
                allowed_chars = (MAX_PROMPT_TOKENS - 600) * 4
                msg.content = msg.content[:allowed_chars] + "\n\n[...Context trimmed to fit under 6000 token limit...]"
                break
    return input_messages


class ResilientLLM:
    """
    Resilient Wrapper around LangChain Chat Models.
    Intercepts 404 NOT_FOUND (model unavailable) and 429 RESOURCE_EXHAUSTED errors.
    Automatically switches models on 404 and applies exponential backoff on 429.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = False,
        factory: Optional[LLMFactory] = None,
    ):
        self.provider = (provider or settings.effective_provider).lower().strip()
        self.temperature = temperature
        self.streaming = streaming
        self.factory = factory or llm_factory

    def _get_model_instance(self) -> Any:
        return self.factory.get_raw_llm(
            provider=self.provider,
            temperature=self.temperature,
            streaming=self.streaming,
        )

    def with_structured_output(self, schema: Type[BaseModel], **kwargs) -> "ResilientRunnable":
        return ResilientRunnable(
            parent_resilient_llm=self,
            schema=schema,
            kwargs=kwargs,
        )

    async def ainvoke(self, input_messages: Any, **kwargs) -> Any:
        input_messages = enforce_prompt_token_limit(input_messages)
        log_token_usage(str(input_messages), estimated_completion_tokens=400, provider=self.provider)
        return await self._execute_with_retry(
            func=lambda llm: llm.ainvoke(input_messages, **kwargs)
        )

    async def _execute_with_retry(self, func) -> Any:
        max_429_retries = 3

        for attempt in range(max_429_retries + 1):
            try:
                llm = self._get_model_instance()
                return await func(llm)
            except Exception as e:
                err_str = str(e)
                logger.warning(f"LLM execution error (attempt {attempt}): {err_str[:160]}")

                # Case 1: 404 Model Unavailable / Deprecated / Does Not Exist
                if "404" in err_str or "NOT_FOUND" in err_str or "no longer available" in err_str or "does not exist" in err_str:
                    if self.provider == LLMProvider.GROQ.value:
                        current_name = self.factory.get_active_groq_model()
                        logger.warning(f"Groq model '{current_name}' failed with 404. Retrying with fallback model...")
                        self.factory.switch_to_next_groq_model(current_name)
                        new_llm = self._get_model_instance()
                        return await func(new_llm)
                    elif self.provider == LLMProvider.OPENAI.value:
                        current_name = self.factory.get_active_openai_model()
                        logger.warning(f"OpenAI model '{current_name}' failed with 404. Retrying with fallback model...")
                        self.factory.switch_to_next_openai_model(current_name)
                        new_llm = self._get_model_instance()
                        return await func(new_llm)
                    elif self.provider == LLMProvider.GEMINI.value:
                        current_name = self.factory.get_active_gemini_model()
                        logger.warning(f"Gemini model '{current_name}' failed with 404 NOT_FOUND. Retrying with fallback model...")
                        self.factory.switch_to_next_gemini_model(current_name)
                        new_llm = self._get_model_instance()
                        return await func(new_llm)
                    raise e

                # Case 2: 429 Rate Limit / Quota Exhausted / Timeouts
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str or "rate_limit_exceeded" in err_str or "timeout" in err_str.lower():
                    if attempt < max_429_retries:
                        backoff = 2 ** (attempt + 1)
                        logger.warning(
                            f"LLM call encountered transient error/429. Retrying attempt {attempt + 1}/{max_429_retries} in {backoff}s..."
                        )
                        print(f"DEBUG: Transient error/429 encountered. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error("Max retries exceeded for transient error/429.")
                        raise e
                else:
                    raise e


class ResilientRunnable:
    """
    Resilient Runnable wrapping structured output LLM invocations.
    """

    def __init__(self, parent_resilient_llm: ResilientLLM, schema: Type[BaseModel], kwargs: Dict[str, Any]):
        self.parent = parent_resilient_llm
        self.schema = schema
        self.kwargs = kwargs

    async def ainvoke(self, input_messages: Any, **kwargs) -> Any:
        input_messages = enforce_prompt_token_limit(input_messages)
        log_token_usage(str(input_messages), estimated_completion_tokens=500, provider=self.parent.provider)

        async def _run(llm_instance: BaseChatModel):
            structured_runnable = llm_instance.with_structured_output(self.schema, **self.kwargs)
            return await structured_runnable.ainvoke(input_messages, **kwargs)

        return await self.parent._execute_with_retry(_run)


# Singleton instance export
llm_factory = LLMFactory()


def get_llm_factory() -> LLMFactory:
    """
    Returns the singleton LLMFactory instance.
    """
    return llm_factory


def get_llm(
    provider: Optional[str] = None,
    temperature: float = 0.0,
    streaming: bool = False,
) -> ResilientLLM:
    """
    Primary LLM factory getter for multi-agent graph nodes.
    Returns a ResilientLLM instance with automatic 404 model switching and 429 backoff retries.
    """
    return ResilientLLM(provider=provider, temperature=temperature, streaming=streaming, factory=llm_factory)
