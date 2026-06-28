from pathlib import Path


def test_audit_reliability_targets_script_exists():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_reliability_targets.py"
    assert script.is_file()


def test_audit_reliability_targets_script_supports_ckpt_path():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_reliability_targets.py"
    content = script.read_text(encoding="utf-8")
    assert "--ckpt_path" in content
    assert "strict=False" in content
