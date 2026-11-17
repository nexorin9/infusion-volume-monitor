"""配置面测试：7 组凭据位缺失即报错、环境变量注入生效、密钥不明文外泄。"""

from __future__ import annotations

import pytest

from infusion_monitor.config import (
    CREDENTIAL_GROUPS,
    MissingCredentialError,
    Settings,
    get_settings,
    missing_credentials,
    require_settings,
)

from .conftest import CREDENTIAL_ENV


def test_credential_groups_cover_seven_slots():
    """凭据位按 7 组登记，出站 → 遥测 → 模型网关。"""
    assert len(CREDENTIAL_GROUPS) == 7
    keys = [group.key for group in CREDENTIAL_GROUPS]
    assert keys[0] == "nurse_exec_base_url"
    assert keys[1] == "nurse_exec_api_token"
    assert "mqtt" in keys
    assert keys[-1] == "llm_timeout"
    # 出站与遥测凭据不得漏项
    mqtt_group = next(g for g in CREDENTIAL_GROUPS if g.key == "mqtt")
    assert set(mqtt_group.env_names) == {
        "MQTT_HOST",
        "MQTT_PORT",
        "MQTT_USERNAME",
        "MQTT_PASSWORD",
    }


def test_missing_credentials_reports_every_absent_group(clean_env):
    """一个凭据位都不给时，7 组全部报缺失。"""
    missing, blank = missing_credentials()
    assert len(missing) == 7
    assert blank == []


def test_get_settings_raises_and_names_the_missing_variable(clean_env, monkeypatch):
    """缺 MQTT_HOST 时启动失败，并点名缺哪一个、给出配置示例。"""
    monkeypatch.setenv("NURSE_EXEC_BASE_URL", CREDENTIAL_ENV["NURSE_EXEC_BASE_URL"])
    monkeypatch.setenv("NURSE_EXEC_API_TOKEN", CREDENTIAL_ENV["NURSE_EXEC_API_TOKEN"])

    with pytest.raises(MissingCredentialError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    assert "MQTT_HOST" in message
    assert "mqtt" in message
    assert "MQTT_HOST=…" in message  # 带示例，运维能照着填
    assert "NURSE_EXEC_BASE_URL" not in message  # 已给齐的不再点名
    assert excinfo.value.missing_groups


def test_blank_value_counts_as_missing(clean_env, monkeypatch):
    """设了但只有空白，同样算缺凭据。"""
    for name, value in CREDENTIAL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_MODEL", "   ")

    missing, blank = missing_credentials()
    assert any(g.key == "llm_model" for g in missing)
    assert "LLM_MODEL" in blank

    with pytest.raises(MissingCredentialError):
        get_settings()


def test_require_settings_returns_settings_when_complete(credential_env):
    """凭据齐备时自检返回可用的 Settings 实例。"""
    settings, missing, blank = require_settings()
    assert settings is not None
    assert missing == [] and blank == []
    assert isinstance(settings, Settings)


def test_require_settings_reports_missing_without_raising(clean_env):
    """自检入口不抛异常，返回缺失清单供启动前打印。"""
    settings, missing, blank = require_settings()
    assert settings is None
    assert len(missing) == 7


def test_env_injection_takes_effect(credential_env, monkeypatch):
    """环境变量注入后派生字段按实际值计算。"""
    monkeypatch.setenv("NURSE_EXEC_BASE_URL", "https://nurse-exec.test-hospital.invalid/api/")
    monkeypatch.setenv("MQTT_TOPIC_PREFIX", "ward-a/infusion")
    settings = get_settings()

    assert settings.nurse_exec_base_url == "https://nurse-exec.test-hospital.invalid/api"
    assert settings.nurse_exec_events_url.endswith("/api/v1/infusion/events")
    assert settings.mqtt_telemetry_topic == "ward-a/infusion/+/telemetry"
    assert settings.llm_timeout == 8.0
    assert settings.mqtt_port == 1883
    assert settings.sqlite_path == "./var/infusion_monitor.db"


def test_optional_params_have_defaults(credential_env):
    """非凭据运行参数带默认值，不要求运维逐项设置。"""
    settings = get_settings()
    assert settings.model_impl == "geom_kernel"
    assert settings.change_fill_ratio_threshold == 0.12
    assert settings.change_consecutive_frames == 5
    assert settings.change_min_interval_minutes == 15
    assert settings.change_max_picks_per_round == 3
    assert settings.webhook_signing_secret is None


def test_invalid_base_url_is_rejected(credential_env, monkeypatch):
    """非 http(s) 端点直接判错，避免把错地址带到出站调用。"""
    monkeypatch.setenv("LLM_BASE_URL", "llm-gateway.internal/v1")
    with pytest.raises(ValueError):
        Settings()


def test_out_of_range_thresholds_are_rejected(credential_env, monkeypatch):
    """面积比阈值必须在 (0, 1]，否则换液判定失去意义。"""
    monkeypatch.setenv("CHANGE_FILL_RATIO_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        Settings()


def test_secrets_are_masked_in_describe(credential_env, monkeypatch):
    """describe() 只给掩码，凭据明文不得出现在日志或摘要里。"""
    monkeypatch.setenv("NURSE_EXEC_API_TOKEN", "supersecrettoken")
    monkeypatch.setenv("LLM_API_KEY", "supersecretkey")
    settings = get_settings()

    rendered = str(settings.describe())
    assert "supersecrettoken" not in rendered
    assert "supersecretkey" not in rendered
    assert "su***en" in settings.masked_credentials()["nurse_exec_api_token"]
    # 描述里仍能看出连的是哪个对端，便于排障
    assert "nurse-exec.test-hospital.invalid" in rendered