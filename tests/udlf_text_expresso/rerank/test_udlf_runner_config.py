from __future__ import annotations

from pathlib import Path
import types

import pytest

from udlf_text_expresso.rerank.udlf_runner import UDLFReranker


class _DummyInputType:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.params: dict[str, object] = {}

    def set_method_name(self, name: str) -> None:
        self.params["method_name"] = name

    def set_input_files(self, path: str) -> None:
        self.params["input_files"] = path

    def set_lists_file(self, path: str) -> None:
        self.params["lists_file"] = path

    def set_param(self, key: str, value: object) -> None:
        self.params[key] = value

    def get_param(self, key: str) -> object:
        return self.params[key]


class _DummyUDLF:
    def __init__(self) -> None:
        self.binary_path: str | None = None

    def setBinaryPath(self, path: str) -> None:  # noqa: N802 - mirrors pyUDLF API
        self.binary_path = path

    def run(self, input_data: _DummyInputType, get_output: bool = True) -> None:  # noqa: D401
        raise AssertionError("run should not be reached when configuration is invalid")


def test_missing_required_config_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from udlf_text_expresso.rerank import udlf_runner

    monkeypatch.setattr(udlf_runner, "PYUDLF_AVAILABLE", True)
    monkeypatch.setattr(udlf_runner, "udlf", _DummyUDLF())
    monkeypatch.setattr(udlf_runner, "inputType", types.SimpleNamespace(InputType=_DummyInputType))

    reranker = UDLFReranker(
        method_name="CPRR",
        config={
            "binary_path": str(tmp_path / "bin" / "udlf"),
            "config_path": str(tmp_path / "config.ini"),
            "output_path": str(tmp_path / "output"),
            "task": "UDL",
            "input_format": "RK",
            "input_rk_format": "STR",
            "k": 2,
            "t": 1,
        },
    )

    rows = [{"query_id": "q1", "doc_id": "doc::d1", "score": 1.0}]

    with pytest.raises(ValueError, match="window_size"):
        reranker._rerank_with_pyudlf(rows)
