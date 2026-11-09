"""Small replaceable structured-output LLM boundary."""
import random
import time
from typing import TypeVar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from pydantic import BaseModel

from promoscout.config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMConfigurationError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 client=None, max_retries: int | None = None, sleep=time.sleep,
                 random_uniform=random.uniform):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model if model is not None else settings.gemini_model
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        self._client = client
        self._sleep = sleep
        self._random_uniform = random_uniform

    def _get_client(self):
        settings = get_settings()
        issues = []
        if settings.llm_provider != "gemini":
            issues.append("LLM_PROVIDER must be 'gemini'")
        if not self.api_key or self.api_key.lower() in {"your_api_key_here", "your_gemini_api_key", "changeme", "replace_me"}:
            issues.append("set GEMINI_API_KEY in .env, the environment, or Streamlit app Secrets")
        if not self.model or self.model.lower() in {
            "your_selected_free_tier_model", "your_model_name", "model_name", "your_gemini_model",
        }:
            issues.append("set GEMINI_MODEL to a supported model identifier")
        if issues:
            raise LLMConfigurationError("Gemini configuration is incomplete: " + "; ".join(issues))
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMConfigurationError("Install google-genai to use Gemini") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, response_model: type[T]) -> T:
        """Generate using the SDK's Pydantic/protobuf response_schema path."""
        return self._generate(prompt, response_model, use_json_schema=False)

    def generate_json_schema(self, prompt: str, response_model: type[T]) -> T:
        """Generate using a JSON Schema dictionary for providers that require it."""
        return self._generate(prompt, response_model, use_json_schema=True)

    @staticmethod
    def _provider_json_schema(response_model: type[BaseModel]) -> dict:
        """Project Pydantic JSON Schema onto Gemini's supported JSON Schema subset.

        Pydantic remains the authoritative local validator. Gemini's supported
        schema subset does not include JSON Schema ``default``, ``minLength``
        or ``maxLength`` keywords, so those provider-side hints are omitted;
        all nested structure, references, required fields, enums, bounds,
        additionalProperties, and array constraints are retained.
        """
        schema = response_model.model_json_schema()
        unsupported = {"default", "minLength", "maxLength"}

        def project(node):
            if isinstance(node, dict):
                return {
                    key: project(value)
                    for key, value in node.items()
                    if key not in unsupported
                }
            if isinstance(node, list):
                return [project(value) for value in node]
            return node

        return project(schema)

    @staticmethod
    def _http_status(exc: Exception) -> int | None:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_error_detail(exc: Exception, *, api_key: str, prompt: str) -> str:
        detail = str(exc)
        if api_key:
            detail = detail.replace(api_key, "[redacted]")
        if prompt:
            detail = detail.replace(prompt, "[redacted prompt]")
        return detail

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
        if not headers:
            return None
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, min(float(value), 120.0))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, min((retry_at - datetime.now(timezone.utc)).total_seconds(), 120.0))
            except (TypeError, ValueError, OverflowError):
                return None

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        ceiling = min(2 ** attempt, 8)
        jittered = self._random_uniform(0.0, float(ceiling))
        server_delay = self._retry_after_seconds(exc)
        return max(jittered, server_delay) if server_delay is not None else jittered

    def _generate(self, prompt: str, response_model: type[T], *, use_json_schema: bool) -> T:
        client = self._get_client()
        last_error = None
        request_config = {
            "response_mime_type": "application/json",
            "response_json_schema": self._provider_json_schema(response_model),
        } if use_json_schema else {
            "response_mime_type": "application/json",
            "response_schema": response_model,
        }
        # Build/validate the SDK config once, outside retries. A locally
        # invalid schema/config is permanent and must not be retried.
        from google.genai import types
        try:
            config = types.GenerateContentConfig(**request_config)
        except Exception as exc:
            raise LLMRequestError(
                "Gemini rejected the structured-output configuration locally: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for attempt in range(max(1, self.max_retries)):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            except LLMConfigurationError:
                raise
            except Exception as exc:
                last_error = exc
                status = self._http_status(exc)
                provider_status = getattr(exc, "status", None)
                detail = self._safe_error_detail(exc, api_key=self.api_key, prompt=prompt)

                if status == 401 or provider_status == "UNAUTHENTICATED":
                    raise LLMRequestError(
                        "Gemini authentication failed. Check GEMINI_API_KEY and the API project's access. "
                        f"Provider detail: {detail}"
                    ) from exc
                if status == 403 or provider_status == "PERMISSION_DENIED":
                    raise LLMRequestError(
                        "Gemini denied access. Check API permissions, project access, and model availability. "
                        f"Provider detail: {detail}"
                    ) from exc
                # Invalid arguments and other non-transient 4xx responses are
                # permanent. Retrying them repeats the same invalid request.
                if status is not None and 400 <= status < 500 and status not in {408, 429}:
                    status_label = f" HTTP {status} {provider_status or ''}".rstrip()
                    raise LLMRequestError(
                        f"Gemini rejected the request permanently ({status_label}). {detail}"
                    ) from exc
                if attempt + 1 >= max(1, self.max_retries):
                    break
                self._sleep(self._retry_delay(exc, attempt))
            else:
                body = getattr(response, "text", None)
                if not body:
                    raise LLMRequestError("Gemini returned an empty structured response")
                try:
                    return response_model.model_validate_json(body)
                except Exception as exc:
                    # Do not include response contents in the error; they may
                    # echo confidential screenplay material.
                    errors = getattr(exc, "errors", None)
                    details = errors(include_input=False) if callable(errors) else None
                    raise LLMRequestError(
                        "Gemini returned JSON that failed local Pydantic validation. "
                        f"Validation details: {details or type(exc).__name__}"
                    ) from exc

        detail = self._safe_error_detail(last_error, api_key=self.api_key, prompt=prompt)
        raise LLMRequestError(
            f"Structured Gemini request failed after {max(1, self.max_retries)} attempt(s). "
            "Check GEMINI_API_KEY, GEMINI_MODEL availability, rate limits, and service status. "
            f"Provider detail: {detail}"
        ) from last_error


