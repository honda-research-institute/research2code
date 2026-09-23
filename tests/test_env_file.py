"""Repo-root .env loading (opt-in config without shell-profile edits).

Contract: the shell always wins — a variable already in the environment
is never overridden. The same file must be `source`-able by bash, so
"export KEY=VALUE" lines parse identically to bare KEY=VALUE.
"""

from __future__ import annotations

import os

import env_file


def _load(tmp_path, text, monkeypatch=None):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return env_file.load_env_file(p)


def test_missing_file_is_a_noop(tmp_path):
    assert env_file.load_env_file(tmp_path / ".env") == {}


def test_sets_only_variables_the_shell_did_not(tmp_path, monkeypatch):
    monkeypatch.setenv("R2C_TEST_SHELL_WINS", "from-shell")
    monkeypatch.delenv("R2C_TEST_FROM_FILE", raising=False)
    applied = _load(tmp_path,
                    "R2C_TEST_SHELL_WINS=from-file\n"
                    "R2C_TEST_FROM_FILE=hello\n")
    assert applied == {"R2C_TEST_FROM_FILE": "hello"}
    assert os.environ["R2C_TEST_SHELL_WINS"] == "from-shell"
    assert os.environ["R2C_TEST_FROM_FILE"] == "hello"
    monkeypatch.delenv("R2C_TEST_FROM_FILE")


def test_accepts_bash_export_lines_comments_and_quotes(tmp_path, monkeypatch):
    for k in ("R2C_TEST_A", "R2C_TEST_B", "R2C_TEST_C"):
        monkeypatch.delenv(k, raising=False)
    applied = _load(tmp_path,
                    "# comment\n"
                    "\n"
                    'export R2C_TEST_A="http://marker:3004/"\n'
                    "R2C_TEST_B='single'\n"
                    "R2C_TEST_C=bare=value=with=equals\n")
    assert applied == {
        "R2C_TEST_A": "http://marker:3004/",
        "R2C_TEST_B": "single",
        "R2C_TEST_C": "bare=value=with=equals",
    }
    for k in applied:
        monkeypatch.delenv(k)


def test_malformed_lines_are_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_TEST_OK", raising=False)
    applied = _load(tmp_path,
                    "no equals sign here\n"
                    "=value-without-key\n"
                    "KEY WITH SPACE=x\n"
                    "R2C_TEST_OK=yes\n")
    assert applied == {"R2C_TEST_OK": "yes"}
    monkeypatch.delenv("R2C_TEST_OK")


def test_default_path_is_the_repo_root_env(tmp_path):
    # The zero-argument call reads <repo-root>/.env, which this test must
    # not depend on existing; it only pins that the resolution is the
    # repo root, one directory above scripts/.
    from pathlib import Path
    import inspect
    src_dir = Path(inspect.getfile(env_file)).resolve().parent
    assert env_file.default_env_path() == src_dir.parent / ".env"
