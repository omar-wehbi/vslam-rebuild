"""Stage 02 change B: main.write_resolved_settings writes resolved_settings.json
(argparse namespace -> JSON, versions, invocation)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import write_resolved_settings  # noqa: E402


def test_write_resolved_settings(tmp_path):
    args = argparse.Namespace(
        dataset=Path("/some/dataset"),
        use_relative_scale=True,
        default_k=False,
        keyframe_strategy=["interval", "ratio"],
        _invocation="main.py --dataset foo",
        _private_helper=object(),
    )
    out_path = tmp_path / "resolved_settings.json"

    write_resolved_settings(args, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())

    assert list(data.keys()) == sorted(data.keys())
    assert not any(k.startswith("_private") for k in data.keys())
    assert data["dataset"] == "/some/dataset"
    assert isinstance(data["dataset"], str)
    assert data["use_relative_scale"] is True
    assert data["default_k"] is False
    assert data["keyframe_strategy"] == ["interval", "ratio"]
    assert "versions" in data
    for key in ("python", "numpy", "cv2", "scipy"):
        assert key in data["versions"]
        assert isinstance(data["versions"][key], str)
    assert data["invocation"] == "main.py --dataset foo"


def test_write_resolved_settings_overwrites(tmp_path):
    out_path = tmp_path / "resolved_settings.json"
    out_path.write_text("stale content")

    args = argparse.Namespace(dataset="rgb", _invocation=None)
    write_resolved_settings(args, out_path)

    data = json.loads(out_path.read_text())
    assert data["dataset"] == "rgb"


def test_run_pipeline_calls_write_resolved_settings(monkeypatch, tmp_path):
    import main as main_mod

    calls = []
    monkeypatch.setattr(
        main_mod, "write_resolved_settings",
        lambda args, out_path: calls.append(out_path),
    )

    def _fake_iter_frames(*a, **kw):
        return None, 0, iter([])

    monkeypatch.setattr(main_mod, "iter_frames", _fake_iter_frames)
    monkeypatch.setattr(main_mod, "show_trajectory", lambda *a, **kw: None)

    save_path = tmp_path / "estimated.txt"
    parser = main_mod.get_parser()
    args = parser.parse_args([
        "--dataset", "rgb", "--no_display", "--save_estimated", str(save_path),
    ])
    main_mod.run_pipeline(args)

    assert len(calls) == 1
    assert Path(calls[0]) == save_path.parent / "resolved_settings.json"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
