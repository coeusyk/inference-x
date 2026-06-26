"""Unit tests for ModelRegistry and ModelEntry."""
import pytest

from inference_x.schemas.model import ModelEntry, ModelList, ModelObject
from inference_x.services.model_service import ModelRegistry


def _entry(name: str, path: str = "test/stub") -> ModelEntry:
    return ModelEntry(name=name, model_path=path)


class TestModelEntry:
    def test_defaults(self):
        e = _entry("m")
        assert e.engine == "vllm"
        assert e.gpu_memory_utilization == "auto"
        assert e.max_model_len is None
        assert e.quantization is None

    def test_explicit_fields(self):
        e = ModelEntry(
            name="llm",
            model_path="org/model",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            quantization="awq",
        )
        assert e.gpu_memory_utilization == 0.9
        assert e.max_model_len == 4096
        assert e.quantization == "awq"

    def test_gpu_memory_utilization_bounds(self):
        with pytest.raises(Exception):
            ModelEntry(name="x", model_path="p", gpu_memory_utilization=1.5)
        with pytest.raises(Exception):
            ModelEntry(name="x", model_path="p", gpu_memory_utilization=-0.1)

    def test_gpu_memory_utilization_auto(self):
        e = ModelEntry(name="x", model_path="p", gpu_memory_utilization="auto")
        assert e.gpu_memory_utilization == "auto"

    def test_max_model_len_must_be_positive(self):
        with pytest.raises(Exception):
            ModelEntry(name="x", model_path="p", max_model_len=0)


class TestModelRegistry:
    def _registry(self, *names: str) -> ModelRegistry:
        return ModelRegistry([_entry(n) for n in names])

    def test_get_existing(self):
        reg = self._registry("alpha", "beta")
        e = reg.get("alpha")
        assert e.name == "alpha"

    def test_get_missing_raises(self):
        reg = self._registry("alpha")
        with pytest.raises(ValueError, match="not registered"):
            reg.get("missing")

    def test_all_preserves_order(self):
        reg = self._registry("a", "b", "c")
        names = [e.name for e in reg.all()]
        assert names == ["a", "b", "c"]

    def test_names(self):
        reg = self._registry("x", "y")
        assert reg.names() == ["x", "y"]

    def test_contains(self):
        reg = self._registry("a", "b")
        assert "a" in reg
        assert "z" not in reg

    def test_from_config_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelRegistry.from_config(str(tmp_path / "nonexistent"))

    def test_from_config_loads_entries(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "models.yaml").write_text(
            "models:\n"
            "  - name: m1\n"
            "    engine: vllm\n"
            "    model_path: org/m1\n"
            "  - name: m2\n"
            "    engine: vllm\n"
            "    model_path: org/m2\n"
        )
        reg = ModelRegistry.from_config(str(config_dir))
        assert "m1" in reg
        assert "m2" in reg
        assert len(reg.all()) == 2


class TestModelSchemas:
    def test_model_list_object_field(self):
        ml = ModelList(data=[ModelObject(id="foo")])
        assert ml.object == "list"
        assert ml.data[0].id == "foo"
        assert ml.data[0].object == "model"
        assert ml.data[0].owned_by == "inferencex"
