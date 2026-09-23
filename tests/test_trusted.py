import os

import pytest

from trusted import trusted_path, trusted_text


def test_text_roundtrips_allowed_chars():
    s = "scripts/run_pipeline.py --dir /tmp/a b_c-1.2:x=y@z,+ Résumé"
    assert trusted_text(s) == s
    code = "import time; time.sleep(1) & x | y `z` $v <a> \"q\" 'r'"
    assert trusted_text(code) == code  # list-form argv: metachars are content


@pytest.mark.parametrize("bad", ["a\nb", "a\rb", "a\x00b", "a\x0bb", "a\x0cb", "a\x1bb"])
def test_text_rejects_control_chars(bad):
    with pytest.raises(ValueError):
        trusted_text(bad)


def test_path_existing_file_equals_absolute(tmp_path):
    f = tmp_path / "sub" / "file.txt"
    f.parent.mkdir()
    f.write_text("x")
    assert trusted_path(f) == f.absolute()
    assert trusted_path(str(f)) == f.absolute()


def test_path_missing_tail_preserved(tmp_path):
    p = tmp_path / "new_dir" / "out.json"
    assert trusted_path(p) == p.absolute()


def test_path_normalizes_dotdot_and_trailing_slash(tmp_path):
    (tmp_path / "a").mkdir()
    raw = str(tmp_path / "a" / ".." / "a") + os.sep
    assert trusted_path(raw) == (tmp_path / "a").absolute()


def test_path_relative_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r").mkdir()
    assert trusted_path("r") == (tmp_path / "r").absolute()


def test_path_follows_directory_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "f").write_text("x")
    link = tmp_path / "link"
    link.symlink_to(real)
    assert trusted_path(link / "f") == (link / "f").absolute()


def test_path_case_insensitive_fs_matches_actual_name(tmp_path):
    (tmp_path / "Mixed").mkdir()
    swapped = tmp_path / "mIXED"
    result = trusted_path(swapped)
    if swapped.exists():  # case-insensitive filesystem (macOS default)
        assert result == (tmp_path / "Mixed").absolute()
    else:
        assert result == swapped.absolute()


def test_path_missing_tail_with_control_char_raises(tmp_path):
    with pytest.raises(ValueError):
        trusted_path(tmp_path / "bad\nname")
