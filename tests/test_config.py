from promoscout.config import Settings


def test_settings_load_from_streamlit_secrets_when_environment_is_empty(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    secrets = {"GEMINI_API_KEY": "private-test-value", "GEMINI_MODEL": "test-model"}
    monkeypatch.setattr("promoscout.config._streamlit_secret", lambda name: secrets.get(name, ""))

    settings = Settings()

    assert settings.gemini_api_key == "private-test-value"
    assert settings.gemini_model == "test-model"
    assert settings.gemini_configuration_issues() == []


def test_environment_settings_take_precedence_over_streamlit_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "local-test-value")
    monkeypatch.setenv("GEMINI_MODEL", "local-model")
    monkeypatch.setattr("promoscout.config._streamlit_secret", lambda _name: "hosted-secret-value")

    settings = Settings()

    assert settings.gemini_api_key == "local-test-value"
    assert settings.gemini_model == "local-model"


def test_missing_key_error_explains_local_and_hosted_configuration(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr("promoscout.config._streamlit_secret", lambda _name: "")

    settings = Settings()

    assert any("Streamlit app's Secrets settings" in issue for issue in settings.gemini_configuration_issues())
