from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_example(filename: str = "nemotron_lightning_lora.py"):
    path = Path(__file__).parents[1] / "examples" / filename
    module_name = f"{path.stem}_example"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lightning_model_path(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "n_routed_experts": 128,
                "num_nextn_predict_layers": 1,
                "mtp_layers_block_type": ["attention", "moe"],
            }
        )
    )
    return tmp_path


def test_lightning_has_a_separate_pinned_model_contract() -> None:
    nano = load_example("nemotron_lora.py")
    lightning = load_example()

    assert nano.MODEL_NAME == "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    assert lightning.MODEL_NAME == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert lightning.MODEL_REVISION == "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"


def test_lightning_training_memory_defaults_cover_the_verified_pilot() -> None:
    example = load_example()

    assert example.training_memory_bytes(1, None) == 256 * 1024**3
    assert example.training_memory_bytes(2, None) == 256 * 1024**3
    assert example.training_memory_bytes(4, None) == 256 * 1024**3
    assert example.training_memory_bytes(8, None) == 512 * 1024**3
    assert example.training_memory_bytes(1, 192) == 192 * 1024**3


def test_lightning_recipe_uses_automodel_mtp(lightning_model_path: Path) -> None:
    example = load_example()

    recipe = example.make_recipe(
        model_path=lightning_model_path,
        training_data_path=Path("/artifacts/train.jsonl"),
        checkpoint_path=Path("/outputs/checkpoints"),
        max_steps=20,
        sequence_length=8192,
        maximum_samples=8,
        gpu_count=4,
        global_batch_size=8,
        checkpoint_every_steps=5,
        use_triton=False,
        distributed_strategy="fsdp2",
        shuffle=True,
        seed=1111,
    )

    model = recipe["model"]
    assert model["pretrained_model_name_or_path"] == str(lightning_model_path)
    assert "force_hf" not in model
    assert "attn_implementation" not in model
    assert model["num_nextn_predict_layers"] == 2
    assert model["mtp_use_repeated_layer"] is True
    assert model["mtp_loss_scaling_factor"] == 0.1
    assert recipe["step_scheduler"]["global_batch_size"] == 8
    assert recipe["peft"] == {
        "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
        "target_modules": [
            "*in_proj",
            "*q_proj",
            "*k_proj",
            "*v_proj",
            "*o_proj",
            "*up_proj",
            "*down_proj",
            "*eh_proj",
            "*.experts",
        ],
        "dim": 8,
        "alpha": 32,
        "dropout_position": "pre",
        "use_triton": False,
    }
    assert recipe["distributed"] == {
        "strategy": "fsdp2",
        "dp_size": None,
        "tp_size": 1,
        "cp_size": 1,
        "ep_size": 4,
        "sequence_parallel": False,
    }
    assert recipe["optimizer"] == {
        "_target_": "torch.optim.AdamW",
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "lr": 1e-4,
        "weight_decay": 0.1,
    }
    assert recipe["lr_scheduler"] == {
        "lr_decay_style": "cosine",
        "lr_warmup_steps": 10,
        "min_lr": 1e-5,
    }


@pytest.mark.parametrize(
    ("max_steps", "expected_warmup_steps"),
    [(1, 0), (2, 1), (10, 9), (11, 10)],
)
def test_lightning_recipe_shortens_warmup_for_bounded_runs(
    lightning_model_path: Path,
    max_steps: int,
    expected_warmup_steps: int,
) -> None:
    example = load_example()

    recipe = example.make_recipe(
        model_path=lightning_model_path,
        training_data_path=Path("/artifacts/train.jsonl"),
        checkpoint_path=Path("/outputs/checkpoints"),
        max_steps=max_steps,
        sequence_length=8192,
        maximum_samples=8,
        gpu_count=1,
        global_batch_size=8,
        checkpoint_every_steps=1,
        use_triton=False,
        distributed_strategy="ddp",
        shuffle=True,
        seed=1111,
    )

    assert recipe["lr_scheduler"]["lr_warmup_steps"] == expected_warmup_steps


def test_lightning_resolves_one_weighted_corpus_epoch() -> None:
    example = load_example()

    global_batch_size = example.resolve_global_batch_size(16_671, 4, 8)

    assert global_batch_size == 8
    assert example.resolve_training_steps(16_671, global_batch_size, None, 1) == 2084
    assert example.resolve_training_steps(16_671, global_batch_size, 100, None) == 100


def test_lightning_shrinks_the_pilot_batch_to_the_available_samples() -> None:
    example = load_example()

    assert example.resolve_global_batch_size(8, 4, 8) == 8
    with pytest.raises(ValueError, match="fewer emitted samples"):
        example.resolve_global_batch_size(3, 4, 8)


def test_lightning_requires_one_training_duration() -> None:
    example = load_example()

    with pytest.raises(ValueError, match="exactly one"):
        example.resolve_training_steps(32, 32, None, None)
    with pytest.raises(ValueError, match="exactly one"):
        example.resolve_training_steps(32, 32, 1, 1)


def test_lightning_validates_the_saved_expert_parallel_adapter_targets(tmp_path: Path) -> None:
    example = load_example()
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": [
                    "model.layers.0.mixer.in_proj",
                    "model.layers.1.mixer.q_proj",
                    "model.layers.1.mixer.k_proj",
                    "model.layers.1.mixer.v_proj",
                    "model.layers.1.mixer.o_proj",
                    "model.layers.2.mixer.shared_experts.up_proj",
                    "model.layers.2.mixer.shared_experts.down_proj",
                    "model.layers.2.mixer.experts.0.gate_proj",
                    "model.layers.2.mixer.experts.0.up_proj",
                    "model.layers.2.mixer.experts.0.down_proj",
                    "mtp.layers.0.eh_proj",
                ],
            }
        )
    )

    result = example.validate_adapter_manifest(tmp_path, require_routed_experts=True)

    assert result == {
        "rank": 8,
        "alpha": 32,
        "scale": 4.0,
        "target_module_count": 11,
        "routed_expert_target_count": 3,
    }


def test_lightning_rejects_a_ddp_adapter_with_routed_experts(tmp_path: Path) -> None:
    example = load_example()
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": [
                    "model.layers.0.mixer.in_proj",
                    "model.layers.1.mixer.q_proj",
                    "model.layers.1.mixer.k_proj",
                    "model.layers.1.mixer.v_proj",
                    "model.layers.1.mixer.o_proj",
                    "model.layers.2.mixer.shared_experts.up_proj",
                    "model.layers.2.mixer.shared_experts.down_proj",
                    "mtp.layers.0.eh_proj",
                    "model.layers.2.mixer.experts.0.gate_proj",
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="must not include routed experts"):
        example.validate_adapter_manifest(tmp_path, require_routed_experts=False)


def test_lightning_rejects_an_expert_parallel_adapter_without_routed_experts(
    tmp_path: Path,
) -> None:
    example = load_example()
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": [
                    "model.layers.0.mixer.in_proj",
                    "model.layers.1.mixer.q_proj",
                    "model.layers.1.mixer.k_proj",
                    "model.layers.1.mixer.v_proj",
                    "model.layers.1.mixer.o_proj",
                    "model.layers.2.mixer.shared_experts.up_proj",
                    "model.layers.2.mixer.shared_experts.down_proj",
                    "mtp.layers.0.eh_proj",
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="has no routed-expert targets"):
        example.validate_adapter_manifest(tmp_path, require_routed_experts=True)


def test_lightning_rejects_an_adapter_that_trains_lm_head(tmp_path: Path) -> None:
    example = load_example()
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["lm_head"],
            }
        )
    )

    with pytest.raises(RuntimeError, match="must not include lm_head"):
        example.validate_adapter_manifest(tmp_path, require_routed_experts=True)


def test_checkpoint_publication_ignores_extra_epoch_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    example = load_example()
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    checkpoint = example.CHECKPOINT_OUTPUT.path
    checkpoint.mkdir(parents=True)
    published: dict[str, int] = {}

    for step in (7, 15, 23):
        directory = checkpoint / f"epoch_{step // 8}_step_{step}"
        directory.mkdir()
        (directory / "state.pt").write_bytes(str(step).encode())
        latest = checkpoint / "LATEST"
        latest.unlink(missing_ok=True)
        latest.symlink_to(directory.name)
        example.publish_latest_checkpoint(checkpoint, published, 25)

    assert published == {}

    directory = checkpoint / "epoch_3_step_31"
    directory.mkdir()
    (directory / "state.pt").write_bytes(b"31")
    (checkpoint / "LATEST").unlink()
    (checkpoint / "LATEST").symlink_to(directory.name)
    example.publish_latest_checkpoint(checkpoint, published, 25)
    example.publish_latest_checkpoint(checkpoint, published, 25)

    assert published == {directory.name: 1}


def test_checkpoint_publication_ignores_restored_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    example = load_example()
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    checkpoint = example.CHECKPOINT_OUTPUT.path
    resume = checkpoint / "resume"
    resume.mkdir(parents=True)
    (resume / "state.pt").write_bytes(b"restored")
    (checkpoint / "LATEST").symlink_to(resume.name)
    published: dict[str, int] = {}

    example.publish_latest_checkpoint(checkpoint, published, 25)

    assert published == {}


def test_lightning_rejects_a_nemotron_model_without_mtp(tmp_path: Path) -> None:
    example = load_example()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "n_routed_experts": 128,
                "num_nextn_predict_layers": 0,
            }
        )
    )

    with pytest.raises(ValueError, match="physical MTP layer"):
        example.lightning_model_recipe(tmp_path)


def test_lightning_rejects_a_different_mtp_block_shape(tmp_path: Path) -> None:
    example = load_example()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "n_routed_experts": 128,
                "num_nextn_predict_layers": 1,
                "mtp_layers_block_type": ["attention"],
            }
        )
    )

    with pytest.raises(ValueError, match="attention and MoE MTP blocks"):
        example.lightning_model_recipe(tmp_path)
