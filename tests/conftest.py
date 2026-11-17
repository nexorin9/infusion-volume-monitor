"""测试共用夹具：凭据位注入与隔离、临时 SQLite 库。"""

from __future__ import annotations

import sqlite3

import pytest

from infusion_monitor.config import reset_settings_cache

#: 一份"占位但非真实"的凭据位（.env.example 的替身，不含任何院内真实值）。
CREDENTIAL_ENV = {
    "NURSE_EXEC_BASE_URL": "https://nurse-exec.test-hospital.invalid",
    "NURSE_EXEC_API_TOKEN": "test-token-not-real",
    "MQTT_HOST": "mqtt.test-hospital.invalid",
    "MQTT_PORT": "1883",
    "MQTT_USERNAME": "test-user",
    "MQTT_PASSWORD": "test-password-not-real",
    "LLM_BASE_URL": "https://llm.test-hospital.invalid/v1",
    "LLM_API_KEY": "test-key-not-real",
    "LLM_MODEL": "test-model",
    "LLM_TIMEOUT": "8.0",
}

_ALL_CREDENTIAL_NAMES = tuple(CREDENTIAL_ENV)


@pytest.fixture
def credential_env(monkeypatch):
    """注入完整的 7 组凭据位，并在用例结束后清空缓存。"""
    for name, value in CREDENTIAL_ENV.items():
        monkeypatch.setenv(name, value)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def clean_env(monkeypatch):
    """清空全部凭据位（一个都不给），用于验证「缺项即失败」。"""
    for name in _ALL_CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def conn() -> sqlite3.Connection:
    """已迁移的内存 SQLite 连接。"""
    from infusion_monitor.store import apply_migrations, connect

    connection = connect("sqlite:///:memory:")
    apply_migrations(connection)
    yield connection
    connection.close()


@pytest.fixture
def client(credential_env, tmp_path):
    """带临时库与临时帧目录的 TestClient（凭据位由 ``credential_env`` 注入）。"""
    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings

    settings = Settings()
    app = create_app(
        settings,
        database_url="sqlite:///:memory:",
        frames_root=tmp_path / "frames",
    )
    with TestClient(app) as test_client:
        test_client.app_settings = settings  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def client_factory(credential_env, monkeypatch, tmp_path):
    """按需覆盖环境变量后建一个新的 TestClient。

    用来验证「配置项改了真的生效」——``client`` 夹具在用例开始前就把应用建好了，
    用例体内再改环境变量已经来不及。
    """

    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings

    def _factory(**env_overrides: str):
        for name, value in env_overrides.items():
            monkeypatch.setenv(name, value)
        reset_settings_cache()
        app = create_app(
            Settings(),
            database_url="sqlite:///:memory:",
            frames_root=tmp_path / "frames",
        )
        return TestClient(app)

    yield _factory


@pytest.fixture
def frames_dir(tmp_path):
    return tmp_path / "frames"


@pytest.fixture
def store_client(credential_env, tmp_path):
    """TestClient + 一个指向**同一个库文件**的独立连接。

    TestClient 的 lifespan 跑在门户线程里，``app.state.store`` 的 SQLite 连接只能在那条
    线程上用（SQLite 连接带线程亲和性）。测试线程要断言落库结果与 append-only 审计行，
    就得自己开一个连接指到同一个文件库：应用提交后，这里能读到同一份数据。
    """
    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings
    from infusion_monitor.store import apply_migrations, connect

    dsn = f"sqlite:///{tmp_path / 'store.db'}"
    app = create_app(Settings(), database_url=dsn, frames_root=tmp_path / "frames")
    observer = connect(dsn)
    apply_migrations(observer)
    with TestClient(app) as test_client:
        yield test_client, observer
    observer.close()