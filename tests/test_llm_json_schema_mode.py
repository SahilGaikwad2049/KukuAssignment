import json

import pytest
from pydantic import BaseModel, ValidationError

from promoscout.evaluation.blind_viewer import BlindViewerAssessment
from promoscout.utils.llm_client import GeminiClient, LLMRequestError


def valid_assessment_json():
    evidence = [{
        "event_id": "E1",
        "excerpt": "A line from the clip.",
        "why_it_matters": "It grounds this judgment.",
    }]
    score = {"score": 3, "rationale": "Some details remain unclear.", "evidence": evidence}
    return json.dumps({
        "standalone_story_reconstruction": "One person challenges another over a secret.",
        "characters_the_viewer_believes_are_involved": [],
        "directly_shown_or_stated": ["A challenge is spoken."],
        "reasonable_inferences": ["There may be hidden history."],
        "unclear_or_unestablished": ["The secret is not identified."],
        "main_conflict_as_understood": "A challenge about an unspecified secret.",
        "perceived_stakes": "The relationship could be affected, but the extent is unclear.",
        "main_unanswered_question": "What is the secret?",
        "confusing_elements": [],
        "missing_context_the_viewer_wishes_for": ["What led to the challenge?"],
        "standalone_clarity": score,
        "hook_effectiveness": score,
        "conflict_comprehension": score,
        "stakes_clarity": score,
        "cliffhanger_effectiveness": score,
        "desire_to_continue": score,
        "overall_viewer_verdict": "mixed",
        "confidence": 0.72,
        "verdict_explanation": "The conflict is intriguing but lacks context.",
    })


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


class MockAPIError(Exception):
    def __init__(self, code, status, message, headers=None):
        super().__init__(message)
        self.code = code
        self.status = status
        if headers is not None:
            self.response = type("Response", (), {"headers": headers})()


def make_client(outcomes, *, api_key="mock-key", retries=3, sleeps=None):
    fake = FakeClient(outcomes)
    sleeper = (sleeps.append if sleeps is not None else lambda _: None)
    client = GeminiClient(api_key=api_key, model="mock-model", client=fake,
                          max_retries=retries, sleep=sleeper,
                          random_uniform=lambda _low, high: high)
    return client, fake


def _walk_schema(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_schema(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_schema(value)


def test_gemini_request_uses_json_schema_and_not_response_schema():
    client, fake = make_client([FakeResponse(valid_assessment_json())])
    result = client.generate_json_schema("clip-only prompt", BlindViewerAssessment)
    config = fake.models.calls[0]["config"]
    expected = GeminiClient._provider_json_schema(BlindViewerAssessment)
    assert result == BlindViewerAssessment.model_validate_json(valid_assessment_json())
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == expected
    assert config.response_schema is None
    dumped = config.model_dump(by_alias=True, exclude_none=True)
    assert dumped["responseJsonSchema"] == expected
    assert "responseSchema" not in dumped


def test_gemini_schema_keeps_nested_types_refs_and_strict_objects():
    provider_schema = GeminiClient._provider_json_schema(BlindViewerAssessment)
    assert "$defs" in provider_schema
    assert provider_schema["properties"]["standalone_clarity"]["$ref"]
    assert provider_schema["$defs"]["DimensionScore"]["additionalProperties"] is False
    assert provider_schema["$defs"]["DimensionScore"]["properties"]["evidence"]["items"]["$ref"]
    assert provider_schema["$defs"]["DimensionScore"]["properties"]["score"]["minimum"] == 1
    assert provider_schema["$defs"]["DimensionScore"]["properties"]["score"]["maximum"] == 5
    # These constraints remain enforced by Pydantic locally, but Gemini's
    # documented JSON Schema subset omits these keywords.
    assert not any({"default", "minLength", "maxLength"} & node.keys()
                   for node in _walk_schema(provider_schema))


def test_invalid_json_is_rejected_without_retry():
    client, fake = make_client([FakeResponse("{not json")], retries=3)
    with pytest.raises(LLMRequestError, match="failed local Pydantic validation"):
        client.generate_json_schema("prompt", BlindViewerAssessment)
    assert len(fake.models.calls) == 1


def test_invalid_score_is_rejected_by_local_pydantic_validation():
    payload = json.loads(valid_assessment_json())
    payload["hook_effectiveness"]["score"] = 6
    client, fake = make_client([FakeResponse(json.dumps(payload))])
    with pytest.raises(LLMRequestError, match="failed local Pydantic validation"):
        client.generate_json_schema("prompt", BlindViewerAssessment)
    assert len(fake.models.calls) == 1


def test_pydantic_schema_still_rejects_out_of_range_scores_directly():
    payload = json.loads(valid_assessment_json())
    payload["hook_effectiveness"]["score"] = 0
    with pytest.raises(ValidationError):
        BlindViewerAssessment.model_validate(payload)


def test_http_400_invalid_argument_is_permanent_and_redacts_secrets():
    api_key = "mock-secret-key"
    prompt = "private screenplay prompt text"
    error = MockAPIError(400, "INVALID_ARGUMENT", f"schema rejected; {api_key}; {prompt}")
    client, fake = make_client([error, FakeResponse(valid_assessment_json())], api_key=api_key)
    sleeps = []
    client._sleep = sleeps.append
    with pytest.raises(LLMRequestError, match="permanently.*400 INVALID_ARGUMENT") as captured:
        client.generate_json_schema(prompt, BlindViewerAssessment)
    assert len(fake.models.calls) == 1
    assert sleeps == []
    assert api_key not in str(captured.value)
    assert prompt not in str(captured.value)


def test_transient_server_failure_retries_then_returns_valid_model():
    error = MockAPIError(503, "UNAVAILABLE", "temporary server issue")
    client, fake = make_client([error, FakeResponse(valid_assessment_json())], retries=3)
    sleeps = []
    client._sleep = sleeps.append
    result = client.generate_json_schema("prompt", BlindViewerAssessment)
    assert isinstance(result, BlindViewerAssessment)
    assert len(fake.models.calls) == 2
    assert sleeps == [1]


def test_repeated_transient_errors_stop_at_configured_retry_limit():
    failures = [MockAPIError(503, "UNAVAILABLE", "still unavailable") for _ in range(4)]
    client, fake = make_client(failures, retries=2)
    sleeps = []
    client._sleep = sleeps.append
    with pytest.raises(LLMRequestError, match=r"after 2 attempt\(s\)"):
        client.generate_json_schema("prompt", BlindViewerAssessment)
    assert len(fake.models.calls) == 2
    assert sleeps == [1]


def test_retry_after_header_is_respected_for_rate_limit():
    error = MockAPIError(429, "RESOURCE_EXHAUSTED", "try later", {"Retry-After": "4"})
    client, fake = make_client([error, FakeResponse(valid_assessment_json())], retries=2)
    sleeps = []
    client._sleep = sleeps.append
    result = client.generate_json_schema("prompt", BlindViewerAssessment)
    assert isinstance(result, BlindViewerAssessment)
    assert len(fake.models.calls) == 2
    assert sleeps == [4.0]


def test_transient_retry_delay_uses_jitter_within_exponential_bound():
    client, fake = make_client([
        MockAPIError(503, "UNAVAILABLE", "temporary"),
        FakeResponse(valid_assessment_json()),
    ], retries=2)
    client._random_uniform = lambda _low, _high: 0.37
    sleeps = []
    client._sleep = sleeps.append
    client.generate_json_schema("prompt", BlindViewerAssessment)
    assert len(fake.models.calls) == 2
    assert sleeps == [0.37]


@pytest.mark.parametrize(("code", "status", "message"), [
    (401, "UNAUTHENTICATED", "invalid credential"),
    (403, "PERMISSION_DENIED", "not authorized"),
])
def test_authentication_and_permission_failures_are_clear_and_not_retried(code, status, message):
    client, fake = make_client([MockAPIError(code, status, message)], retries=3)
    with pytest.raises(LLMRequestError, match="authentication failed|denied access"):
        client.generate_json_schema("prompt", BlindViewerAssessment)
    assert len(fake.models.calls) == 1


def test_legacy_response_schema_mode_remains_available():
    class SimpleModel(BaseModel):
        ok: bool

    client, fake = make_client([FakeResponse('{"ok": true}')])
    assert client.generate("prompt", SimpleModel).ok is True
    config = fake.models.calls[0]["config"]
    assert config.response_schema is SimpleModel
    assert config.response_json_schema is None
