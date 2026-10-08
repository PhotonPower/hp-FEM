"""The hpfem-gui entry point (M15 F15): app resolution, the streamlit command, dry runs and
the error paths; Streamlit itself is not needed."""

import sys

from hpfem import gui


def test_find_app_order(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.py"
    explicit.write_text("", encoding="utf-8")
    configured = tmp_path / "configured.py"
    configured.write_text("", encoding="utf-8")
    assert gui.find_app(str(explicit), env={gui.ENV_VAR: str(configured)}) == explicit.resolve()
    assert gui.find_app(None, env={gui.ENV_VAR: str(configured)}) == configured.resolve()
    monkeypatch.delenv(gui.ENV_VAR, raising=False)
    monkeypatch.setattr(gui.importlib.util, "find_spec", lambda name: None)
    assert gui.find_app(None, env={}) is None


def test_dry_run_prints_the_streamlit_command(tmp_path, monkeypatch, capsys):
    app = tmp_path / "fem_app.py"
    app.write_text("", encoding="utf-8")
    monkeypatch.setattr(gui, "streamlit_available", lambda: True)
    assert gui.main([str(app), "--dry-run", "--server.port", "8600"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(sys.executable)
    assert "-m streamlit run" in line and str(app.resolve()) in line and "8600" in line
    # the app from the environment
    monkeypatch.setenv(gui.ENV_VAR, str(app))
    assert gui.main(["--dry-run"]) == 0
    assert str(app.resolve()) in capsys.readouterr().out


def test_errors_and_version(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(gui.ENV_VAR, raising=False)
    monkeypatch.setattr(gui, "find_app", lambda explicit=None, env=None: None)
    assert gui.main(["--dry-run"]) == 2
    assert "no app configured" in capsys.readouterr().err
    missing = tmp_path / "missing.py"
    monkeypatch.setattr(gui, "find_app", lambda explicit=None, env=None: missing)
    assert gui.main(["--dry-run"]) == 2
    assert "does not exist" in capsys.readouterr().err
    app = tmp_path / "fem_app.py"
    app.write_text("", encoding="utf-8")
    monkeypatch.setattr(gui, "find_app", lambda explicit=None, env=None: app)
    monkeypatch.setattr(gui, "streamlit_available", lambda: False)
    assert gui.main(["--dry-run"]) == 2
    assert "hpfem[gui]" in capsys.readouterr().err
    assert gui.main(["--version"]) == 0
    assert '"hpfem"' in capsys.readouterr().out
