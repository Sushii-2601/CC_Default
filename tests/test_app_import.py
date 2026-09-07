"""Make sure app.py's module-level code (imports, page config) is import-safe."""
import runpy
from pathlib import Path


def test_app_module_imports_without_error(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    # Streamlit's set_page_config / caching decorators are safe to import
    # outside of a running Streamlit server; only st.stop()/widgets need one.
    module = runpy.run_path(str(repo_root / "app.py"), run_name="app_under_test")
    assert "main" in module
    assert "load_artifacts" in module
