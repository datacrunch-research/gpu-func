from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import types
from argparse import ArgumentTypeError, Namespace
from pathlib import Path

import pytest

from gfaas.errors import GfaasError


def load_example():
    path = Path(__file__).parents[1] / "examples" / "nemotron_lora.py"
    spec = importlib.util.spec_from_file_location("nemotron_lora_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recipe_uses_local_artifacts_and_one_gpu_configuration() -> None:
    example = load_example()

    recipe = example.make_recipe(
        model_path=Path("/artifacts/model"),
        training_data_path=Path("/artifacts/train.jsonl"),
        checkpoint_path=Path("/outputs/checkpoints"),
        max_steps=3,
        sequence_length=2048,
        maximum_samples=16,
        gpu_count=1,
        checkpoint_every_steps=1,
        use_triton=False,
        distributed_strategy="fsdp2",
        shuffle=True,
        seed=1111,
    )

    assert recipe["model"]["pretrained_model_name_or_path"] == "/artifacts/model"
    assert recipe["dataset"]["path"] == "/artifacts/train.jsonl"
    assert recipe["dataset"]["_target_"] == ("nemotron_lora.make_nemotron_agent_chat_dataset")
    assert recipe["dataset"]["tokenizer"]["pretrained_model_name_or_path"] == ("/artifacts/model")
    assert recipe["dataset"]["truncation"] is False
    assert recipe["dataset"]["tokenizer"]["pad_token_id"] == 0
    assert recipe["dataset"]["train_on_last_turn_only"] is True
    assert recipe["dataset"]["limit_dataset_samples"] == 16
    assert recipe["step_scheduler"]["global_batch_size"] == 1
    assert recipe["step_scheduler"]["max_steps"] == 3
    assert recipe["distributed"]["tp_size"] == 1
    assert recipe["distributed"]["cp_size"] == 1
    assert recipe["checkpoint"]["checkpoint_dir"] == "/outputs/checkpoints"
    assert recipe["peft"]["exclude_modules"] == ["*.out_proj"]
    assert recipe["peft"]["use_triton"] is False
    assert recipe["step_scheduler"]["ckpt_every_steps"] == 1
    assert recipe["dataloader"]["shuffle"] is True
    assert recipe["rng"]["seed"] == 1111


def test_recipe_can_disable_shuffle_and_set_the_seed() -> None:
    example = load_example()

    recipe = example.make_recipe(
        model_path=Path("/artifacts/model"),
        training_data_path=Path("/artifacts/train.jsonl"),
        checkpoint_path=Path("/outputs/checkpoints"),
        max_steps=3,
        sequence_length=2048,
        maximum_samples=None,
        gpu_count=4,
        checkpoint_every_steps=2,
        use_triton=True,
        distributed_strategy="fsdp2",
        shuffle=False,
        seed=42,
    )

    assert recipe["dataloader"]["shuffle"] is False
    assert recipe["rng"]["seed"] == 42
    assert recipe["step_scheduler"]["global_batch_size"] == 4
    assert recipe["step_scheduler"]["local_batch_size"] == 1
    assert recipe["distributed"]["dp_size"] is None


def test_recipe_rejects_a_negative_seed() -> None:
    example = load_example()

    with pytest.raises(ValueError, match="seed must not be negative"):
        example.make_recipe(
            model_path=Path("/artifacts/model"),
            training_data_path=Path("/artifacts/train.jsonl"),
            checkpoint_path=Path("/outputs/checkpoints"),
            max_steps=3,
            sequence_length=2048,
            maximum_samples=None,
            gpu_count=1,
            checkpoint_every_steps=1,
            use_triton=True,
            distributed_strategy="fsdp2",
            shuffle=True,
            seed=-1,
        )


def test_recipe_rejects_a_non_positive_gpu_count() -> None:
    example = load_example()

    with pytest.raises(ValueError, match="gpu_count must be positive"):
        example.make_recipe(
            model_path=Path("/artifacts/model"),
            training_data_path=Path("/artifacts/train.jsonl"),
            checkpoint_path=Path("/outputs/checkpoints"),
            max_steps=3,
            sequence_length=2048,
            maximum_samples=None,
            gpu_count=0,
            checkpoint_every_steps=1,
            use_triton=True,
            distributed_strategy="fsdp2",
            shuffle=True,
            seed=1111,
        )


@pytest.mark.parametrize(
    ("max_steps", "sequence_length", "maximum_samples", "message"),
    [
        (0, 2048, None, "max_steps"),
        (1, 64, None, "sequence_length"),
        (1, 2048, 0, "maximum_samples"),
    ],
)
def test_recipe_rejects_invalid_bounds(
    max_steps: int,
    sequence_length: int,
    maximum_samples: int | None,
    message: str,
) -> None:
    example = load_example()

    with pytest.raises(ValueError, match=message):
        example.make_recipe(
            model_path=Path("/artifacts/model"),
            training_data_path=Path("/artifacts/train.jsonl"),
            checkpoint_path=Path("/outputs/checkpoints"),
            max_steps=max_steps,
            sequence_length=sequence_length,
            maximum_samples=maximum_samples,
            gpu_count=1,
            checkpoint_every_steps=1,
            use_triton=True,
            distributed_strategy="fsdp2",
            shuffle=True,
            seed=1111,
        )


def test_model_directory_validation_requires_every_weight_shard(tmp_path: Path) -> None:
    example = load_example()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001.safetensors"}})
    )

    with pytest.raises(ValueError, match="model-00001.safetensors"):
        example.validate_model_directory(tmp_path)

    (tmp_path / "model-00001.safetensors").write_bytes(b"weights")
    example.validate_model_directory(tmp_path)


def test_cache_paths_use_the_sdk_scratch_directory(tmp_path: Path, monkeypatch) -> None:
    example = load_example()
    for name in (
        "HF_HOME",
        "HF_MODULES_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
    ):
        monkeypatch.delenv(name, raising=False)

    configured = example.configure_cache_paths(tmp_path)

    assert configured["HF_HOME"] == str(tmp_path / "cache" / "huggingface")
    assert configured["HF_MODULES_CACHE"].startswith(str(tmp_path))
    assert configured["TMPDIR"] == str(tmp_path / "tmp")
    assert all(Path(path).is_dir() for path in configured.values())


def test_training_thread_pools_match_the_requested_cpu_count(monkeypatch) -> None:
    example = load_example()
    names = (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OMP_MAX_ACTIVE_LEVELS",
        "OMP_DYNAMIC",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    configured = example.configure_training_threads(example.TRAINING_CPU_THREADS)

    assert example.TRAINING_CPU_MILLICORES == 16_000
    assert configured["OMP_NUM_THREADS"] == "16"
    assert configured["OMP_THREAD_LIMIT"] == "16"
    assert configured["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert configured["OMP_DYNAMIC"] == "false"
    assert all(os.environ[name] == value for name, value in configured.items())


def test_training_threads_are_divided_across_gpu_ranks() -> None:
    example = load_example()

    assert example.training_threads_per_rank(16, 1) == 16
    assert example.training_threads_per_rank(16, 4) == 4
    assert example.training_threads_per_rank(2, 4) == 1

    with pytest.raises(ValueError, match="gpu_count"):
        example.training_threads_per_rank(16, 0)


def test_automodel_command_launches_one_process_per_gpu() -> None:
    example = load_example()

    assert example.automodel_command(Path("/scratch/recipe.yaml"), 4) == [
        "automodel",
        "/scratch/recipe.yaml",
        "--nproc-per-node",
        "4",
    ]


def test_multigpu_automodel_uses_an_ip_literal_rendezvous() -> None:
    example = load_example()
    environment = {"PRESERVED": "value"}

    example.configure_local_rendezvous(environment, 4)

    assert environment == {
        "PRESERVED": "value",
        "PET_LOCAL_ADDR": "127.0.0.1",
        "PET_RDZV_BACKEND": "c10d",
        "PET_RDZV_ENDPOINT": "127.0.0.1:0",
    }

    single_gpu_environment: dict[str, str] = {}
    example.configure_local_rendezvous(single_gpu_environment, 1)
    assert single_gpu_environment == {}

    with pytest.raises(ValueError, match="gpu_count"):
        example.configure_local_rendezvous({}, 0)


def test_positive_integer_rejects_invalid_cli_values() -> None:
    example = load_example()

    assert example.positive_integer("4") == 4
    with pytest.raises(ArgumentTypeError, match="positive"):
        example.positive_integer("0")


def test_training_memory_scales_for_multi_gpu_startup() -> None:
    example = load_example()

    assert example.training_memory_bytes(1, None) == 128 * 1024**3
    assert example.training_memory_bytes(2, None) == 128 * 1024**3
    assert example.training_memory_bytes(4, None) == 256 * 1024**3
    assert example.training_memory_bytes(4, 192) == 192 * 1024**3

    with pytest.raises(ValueError, match="gpu_count"):
        example.training_memory_bytes(0, None)
    with pytest.raises(ValueError, match="memory_gib"):
        example.training_memory_bytes(1, 0)


def test_training_scratch_has_a_finite_configurable_limit() -> None:
    example = load_example()

    assert example.training_scratch_bytes(64) == 64 * 1024**3
    with pytest.raises(ValueError, match="scratch_gib must be positive"):
        example.training_scratch_bytes(0)


def test_ddp_strategy_replicates_the_model_and_tracks_unused_parameters() -> None:
    example = load_example()

    assert example.distributed_config("ddp") == {
        "strategy": "ddp",
        "dp_size": None,
        "find_unused_parameters": True,
    }
    with pytest.raises(ValueError, match="unsupported distributed strategy"):
        example.distributed_config("unknown")


def test_checkpoint_interval_bounds_durable_versions() -> None:
    example = load_example()

    assert example.checkpoint_interval(2_000, None) == 500
    assert example.checkpoint_interval(5_000, None) == 1_250
    assert example.checkpoint_interval(2_000, 500) == 500

    with pytest.raises(ValueError, match="too many checkpoint versions"):
        example.checkpoint_interval(2_000, 250)
    with pytest.raises(ValueError, match="must not exceed"):
        example.checkpoint_interval(2_000, 2_001)
    with pytest.raises(ValueError, match="positive"):
        example.checkpoint_interval(0, None)


def test_distributed_diagnostics_exclude_collective_logging() -> None:
    example = load_example()
    environment = {"PRESERVED": "value"}

    example.configure_distributed_diagnostics(environment)

    assert environment == {
        "PRESERVED": "value",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH",
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
    }


@pytest.mark.parametrize(
    ("line", "world_size"),
    [
        ("> initializing torch distributed with 4 workers.", 4),
        ("2026-08-24 | INFO | root | World size: 4", 4),
        ("Training: 10%", None),
    ],
)
def test_world_size_parser_accepts_automodel_logs(line: str, world_size: int | None) -> None:
    example = load_example()

    assert example.parse_world_size(line) == world_size


def test_training_runtime_metadata_does_not_return_library_string_subclasses(
    monkeypatch,
) -> None:
    example = load_example()

    class LibraryString(str):
        pass

    torch = types.SimpleNamespace(
        __version__=LibraryString("2.12.0+cu130"),
        cuda=types.SimpleNamespace(
            get_device_name=lambda _index: LibraryString("NVIDIA GB300"),
            device_count=lambda: 4,
        ),
    )
    monkeypatch.setattr(example, "version", lambda name: LibraryString(f"{name}-version"))

    metadata = example.training_runtime_metadata(torch)

    assert metadata == {
        "device": "NVIDIA GB300",
        "visible_gpu_count": 4,
        "torch_version": "2.12.0+cu130",
        "nemo_automodel_version": "nemo-automodel-version",
        "transformers_version": "transformers-version",
    }
    assert all(
        type(metadata[name]) is str
        for name in (
            "device",
            "torch_version",
            "nemo_automodel_version",
            "transformers_version",
        )
    )


def test_training_record_uses_nemotron_reasoning_and_only_called_tools() -> None:
    example = load_example()
    record = {
        "tools": [
            {"type": "function", "function": {"name": "used", "parameters": {}}},
            {"type": "function", "function": {"name": "unused", "parameters": {}}},
        ],
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reason",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "used", "arguments": "{}"},
                    }
                ],
            },
        ],
    }

    normalized = example.normalize_training_record(
        record,
        reasoning_mode="include",
        tool_selection="used",
    )

    assert normalized["messages"][-1]["content"] == "reason</think>\nanswer"
    assert "reasoning_content" not in normalized["messages"][-1]
    assert [tool["function"]["name"] for tool in normalized["tools"]] == ["used"]
    assert normalized["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {}


def test_training_record_restores_plan_tools_and_rejects_unknown_tools() -> None:
    example = load_example()
    record = {
        "tools": [],
        "messages": [
            {"role": "user", "content": "plan this"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_plan",
                        "type": "function",
                        "function": {
                            "name": "write_plan",
                            "arguments": {"steps": '["one"]'},
                        },
                    }
                ],
            },
        ],
    }

    normalized = example.normalize_training_record(
        record,
        reasoning_mode="omit",
        tool_selection="used",
    )
    assert [tool["function"]["name"] for tool in normalized["tools"]] == ["write_plan"]
    assert normalized["messages"][-1]["tool_calls"][0]["function"]["arguments"]["steps"] == ["one"]

    record["messages"][-1]["tool_calls"][0]["function"]["name"] = "wiki_search"
    with pytest.raises(ValueError, match="undeclared tool wiki_search"):
        example.normalize_training_record(
            record,
            reasoning_mode="omit",
            tool_selection="used",
        )


def test_training_record_repairs_escaped_apostrophe_in_plan_steps() -> None:
    example = load_example()
    repairs = {
        "decoded_tool_arguments": 0,
        "decoded_plan_steps": 0,
        "repaired_plan_escapes": 0,
        "restored_tool_schemas": 0,
    }
    record = {
        "tools": [],
        "messages": [
            {"role": "user", "content": "plan this"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_plan",
                        "type": "function",
                        "function": {
                            "name": "write_plan",
                            "arguments": {"steps": '["inspect who\\\'s task"]'},
                        },
                    }
                ],
            },
        ],
    }

    normalized = example.normalize_training_record(
        record,
        reasoning_mode="omit",
        tool_selection="used",
        repairs=repairs,
    )

    arguments = normalized["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert arguments["steps"] == ["inspect who's task"]
    assert repairs["decoded_plan_steps"] == 1
    assert repairs["repaired_plan_escapes"] == 1


def test_token_lengths_accept_tokenizer_mappings() -> None:
    example = load_example()

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            tokens = list(range(10 + sum(len(message["content"]) for message in messages)))
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    record = {
        "tools": [],
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }

    assert example._token_lengths(Tokenizer(), record) == (24, 6)


def test_token_lengths_do_not_use_a_generation_prompt_for_the_prefix() -> None:
    example = load_example()

    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
            if len(messages) == 2:
                return [1, 2, 3, 4, 5]
            if add_generation_prompt:
                return [1, 9]
            return [1, 2]

    record = {
        "tools": [],
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }

    assert example._token_lengths(Tokenizer(), record) == (5, 3)


def test_training_record_uses_automodel_agent_chat_schema() -> None:
    example = load_example()
    record = {
        "id": "sample",
        "tools": [],
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "original-call",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": {"target": "object"}},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "original-call"},
            {"role": "assistant", "content": "answer", "weight": 1},
        ],
    }

    encoded = example.encode_automodel_training_record(record)
    rendered = example.render_automodel_training_messages(encoded)

    assert [message["role"] for message in encoded["messages"]] == [
        "user",
        "assistant",
        "tool_call",
        "tool_response",
        "assistant",
    ]
    assert encoded["gfaas_training_format"] == example.AUTOMODEL_TRAINING_FORMAT
    assert json.loads(encoded["tools"]) == []
    assert rendered[1]["tool_calls"][0] == {
        "id": "call_sample_0",
        "type": "function",
        "function": {"name": "inspect", "arguments": {"target": "object"}},
    }
    assert rendered[2]["tool_call_id"] == "call_sample_0"


def test_training_dataset_patches_only_automodel_message_conversion(monkeypatch) -> None:
    example = load_example()
    captured = {}

    def make_agent_chat_dataset(tokenizer, **kwargs):
        captured["tokenizer"] = tokenizer
        captured["kwargs"] = kwargs
        return "dataset"

    agent_chat = types.SimpleNamespace(
        _convert_messages=lambda _messages: None,
        make_agent_chat_dataset=make_agent_chat_dataset,
    )
    monkeypatch.setattr(example, "version", lambda _name: "0.5.0")
    monkeypatch.setattr(example.importlib, "import_module", lambda _name: agent_chat)

    result = example.make_nemotron_agent_chat_dataset("tokenizer", path="training.jsonl")

    assert result == "dataset"
    assert captured == {
        "tokenizer": "tokenizer",
        "kwargs": {"path": "training.jsonl"},
    }
    converted = agent_chat._convert_messages(
        [
            {
                "role": "tool_call",
                "content": '{"name":"inspect","arguments":{"target":"object"}}',
            }
        ],
        example_id="sample",
    )
    assert converted[0]["tool_calls"][0]["function"]["arguments"] == {"target": "object"}


def test_automodel_tools_are_a_scalar_for_heterogeneous_schemas() -> None:
    example = load_example()
    first = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        "messages": [{"role": "assistant", "content": "answer"}],
    }
    second = copy.deepcopy(first)
    second["tools"][0]["function"]["parameters"]["properties"] = {"object": {"type": "string"}}

    encoded = [example.encode_automodel_training_record(record) for record in (first, second)]

    assert all(isinstance(record["tools"], str) for record in encoded)
    assert set(json.loads(encoded[0]["tools"])[0]["function"]["parameters"]["properties"]) == {
        "query"
    }
    assert set(json.loads(encoded[1]["tools"])[0]["function"]["parameters"]["properties"]) == {
        "object"
    }


def test_training_record_rejects_reordered_tool_responses() -> None:
    example = load_example()
    record = {
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "first",
                        "type": "function",
                        "function": {"name": "one", "arguments": {}},
                    },
                    {
                        "id": "second",
                        "type": "function",
                        "function": {"name": "two", "arguments": {}},
                    },
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "second"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    with pytest.raises(ValueError, match="response order"):
        example.encode_automodel_training_record(record)


def test_qualification_rejects_long_records_and_keeps_labels(tmp_path: Path, monkeypatch) -> None:
    example = load_example()

    class Tokenizer:
        pad_token_id = None

        def apply_chat_template(self, messages, **_kwargs):
            return list(range(10 + sum(len(message["content"]) for message in messages)))

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "tools": [],
                    "messages": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": answer},
                    ],
                }
            )
            for answer in ("x" * 100, "answer")
        )
        + "\n"
    )
    destination = tmp_path / "qualified.jsonl"

    counts = example.qualify_training_data(
        source,
        destination,
        tokenizer_path=tmp_path,
        sequence_length=30,
        maximum_samples=1,
        reasoning_mode="include",
        tool_selection="used",
    )

    assert counts == {
        "scanned": 2,
        "selected": 1,
        "emitted_samples": 1,
        "invalid": 0,
        "too_long": 1,
        "without_labels": 0,
        "decoded_tool_arguments": 0,
        "decoded_plan_steps": 0,
        "repaired_plan_escapes": 0,
        "restored_tool_schemas": 0,
        "weight_scale": 1,
        "record_weights": {"1": 1},
        "invalid_reasons": {},
    }
    assert json.loads(destination.read_text())["messages"][-1]["content"] == "answer"


def test_qualification_preserves_rational_record_weights(tmp_path: Path, monkeypatch) -> None:
    example = load_example()

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **_kwargs):
            return list(range(10 + sum(len(message["content"]) for message in messages)))

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
    )
    source = tmp_path / "source.jsonl"
    records = [
        {
            "id": f"sample-{index}",
            "weight": weight,
            "tools": [],
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        for index, weight in enumerate((0.5, 1, 1.5))
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    destination = tmp_path / "qualified.jsonl"

    counts = example.qualify_training_data(
        source,
        destination,
        tokenizer_path=tmp_path,
        sequence_length=64,
        maximum_samples=None,
        reasoning_mode="include",
        tool_selection="used",
    )

    emitted = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [record["id"] for record in emitted] == [
        "sample-0",
        "sample-1",
        "sample-1",
        "sample-2",
        "sample-2",
        "sample-2",
    ]
    assert all(record["weight"] == 1 for record in emitted)
    assert counts["selected"] == 3
    assert counts["emitted_samples"] == 6
    assert counts["weight_scale"] == 2
    assert counts["record_weights"] == {"1": 1, "1/2": 1, "3/2": 1}

    repeated_destination = tmp_path / "qualified-again.jsonl"
    repeated_counts = example.qualify_training_data(
        destination,
        repeated_destination,
        tokenizer_path=tmp_path,
        sequence_length=64,
        maximum_samples=None,
        reasoning_mode="include",
        tool_selection="used",
    )
    assert repeated_destination.read_bytes() == destination.read_bytes()
    assert repeated_counts["selected"] == 6
    assert repeated_counts["emitted_samples"] == 6
    assert repeated_counts["weight_scale"] == 1


def test_file_sha256_reads_the_complete_file(tmp_path: Path) -> None:
    example = load_example()
    path = tmp_path / "data.bin"
    path.write_bytes(b"agent training data")

    assert example.file_sha256(path) == (
        "06d05799ecd7f2c8dc91a3edeef3724f2af2be61c372a1ecd61f0be5f45025b8"
    )


def test_checkpoint_output_replaces_links_and_packages_adapter(tmp_path: Path) -> None:
    example = load_example()
    checkpoint = tmp_path / "checkpoint"
    first_model = checkpoint / "epoch_0" / "model"
    first_model.mkdir(parents=True)
    (first_model / "adapter_model.safetensors").write_bytes(b"first")
    latest_model = checkpoint / "epoch_1" / "model"
    latest_model.mkdir(parents=True)
    (latest_model / "adapter_model.safetensors").write_bytes(b"latest")
    (latest_model / "adapter_config.json").write_text("{}")
    (checkpoint / "LATEST").symlink_to("epoch_1")
    adapter = tmp_path / "adapter"

    files = example.prepare_checkpoint_outputs(checkpoint, adapter)

    assert not (checkpoint / "LATEST").is_symlink()
    assert (checkpoint / "LATEST").read_text() == "epoch_1\n"
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"latest"
    assert (adapter / "adapter_config.json").read_text() == "{}"
    assert any(item["path"] == "LATEST" for item in files)


def test_publishes_each_completed_checkpoint_once(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    example = load_example()
    monkeypatch.setenv("GFAAS_OUTPUT_ROOT", str(tmp_path))
    checkpoint = example.CHECKPOINT_OUTPUT.path
    first = checkpoint / "epoch_0_step_499"
    first.mkdir(parents=True)
    (first / "state.pt").write_bytes(b"first")
    (checkpoint / "LATEST").symlink_to(first.name)
    published = {}

    example.publish_latest_checkpoint(checkpoint, published)
    example.publish_latest_checkpoint(checkpoint, published)

    assert published == {first.name: 1}
    assert "generation=1" in capsys.readouterr().out
    (checkpoint / "LATEST").unlink()
    second = checkpoint / "epoch_0_step_999"
    second.mkdir()
    (second / "state.pt").write_bytes(b"second")
    (checkpoint / "LATEST").symlink_to(second.name)
    example.publish_latest_checkpoint(checkpoint, published)

    assert published == {first.name: 1, second.name: 2}


def test_restores_a_checkpoint_artifact_for_automodel(tmp_path: Path) -> None:
    example = load_example()
    source = tmp_path / "artifact"
    source.mkdir()
    (source / "step_scheduler.pt").write_bytes(b"state")
    checkpoint = tmp_path / "checkpoint"

    example.restore_checkpoint(source, checkpoint)

    assert (checkpoint / "resume/step_scheduler.pt").read_bytes() == b"state"
    assert (checkpoint / "LATEST").readlink() == Path("resume")
    with pytest.raises(RuntimeError, match="not empty"):
        example.restore_checkpoint(source, checkpoint)


@pytest.mark.parametrize(
    "metrics",
    [
        {"num_label_tokens": 0, "loss": 1.0, "grad_norm": 1.0},
        {"num_label_tokens": 1, "loss": 0.0, "grad_norm": 1.0},
        {"num_label_tokens": 1, "loss": 1.0, "grad_norm": float("nan")},
    ],
)
def test_training_metrics_require_real_optimizer_activity(metrics) -> None:
    example = load_example()

    with pytest.raises(RuntimeError):
        example.validate_training_metrics(metrics)


def test_training_metrics_accept_automodel_log_format() -> None:
    example = load_example()

    line = (
        "step 1 | epoch 0 | loss 2.1250 | grad_norm 0.3750 | lr 1.00e-05 | "
        "mem 32.00 GiB | tps 42.00(42.00/gpu) | num_label_tokens 247"
    )

    assert example.parse_training_metrics(line) == {
        "loss": 2.125,
        "grad_norm": 0.375,
        "num_label_tokens": 247,
    }


def test_failed_call_writes_a_local_report(tmp_path: Path, monkeypatch, capsys) -> None:
    example = load_example()
    report_path = tmp_path / "run.json"

    class Job:
        call_id = "call_test"

        def iter_events(self, **_kwargs):
            raise GfaasError("staging failed\n\njourney:\n- detail")
            yield

        def wait(self, **_kwargs):
            raise AssertionError("wait must not run after log iteration fails")

        def status(self):
            return {"id": self.call_id, "state": "failed"}

        def artifacts(self):
            return {"items": []}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_artifact(self, artifact_id):
            return {"id": artifact_id}

        def list_attempts(self, call_id):
            assert call_id == "call_test"
            return {"items": []}

        def submit(self, **kwargs):
            assert kwargs["capacity_wait_s"] == 3600
            assert kwargs["memory_bytes"] == 256 * 1024**3
            assert kwargs["ephemeral_storage_bytes"] == 64 * 1024**3
            assert kwargs["args"][2] is None
            assert kwargs["args"][3:5] == (
                example.MODEL_NAME,
                example.MODEL_REVISION,
            )
            assert kwargs["args"][5] == 4
            assert kwargs["args"][11:15] == (1, True, False, "fsdp2")
            assert kwargs["args"][-2:] == (True, 1111)
            assert kwargs["gpu_count"] == 4
            assert "gpu" not in kwargs
            return Job()

    monkeypatch.setattr(example.gfaas, "Client", Client)
    monkeypatch.setattr(
        example,
        "parse_args",
        lambda: Namespace(
            model=None,
            model_artifact="art_model",
            model_name=example.MODEL_NAME,
            model_revision=example.MODEL_REVISION,
            tokenizer_model=None,
            training_data=None,
            training_data_artifact="art_data",
            resume_checkpoint_artifact=None,
            gpu_type="gb300",
            gpu_count=4,
            memory_gib=None,
            scratch_gib=64,
            image=example.IMAGE_NAME,
            max_steps=1,
            checkpoint_every_steps=None,
            triton=True,
            startup_diagnostics=False,
            distributed_strategy="fsdp2",
            sequence_length=2048,
            shuffle=True,
            seed=1111,
            maximum_samples=8,
            reasoning_mode="include",
            tool_selection="used",
            timeout=3600,
            capacity_wait=3600,
            detach_on_interrupt=False,
            report=report_path,
        ),
    )

    assert example.main() == 1

    report = json.loads(report_path.read_text())
    assert report["call_id"] == "call_test"
    assert report["model"] == example.MODEL_NAME
    assert report["model_revision"] == example.MODEL_REVISION
    assert report["resources"]["memory_bytes"] == 256 * 1024**3
    assert report["resources"]["ephemeral_storage_bytes"] == 64 * 1024**3
    assert report["distributed_strategy"] == "fsdp2"
    assert report["status"]["state"] == "failed"
    assert report["attempts"] == {"items": []}
    assert report["error"]["type"] == "GfaasError"
    assert report["error"]["message"].startswith("staging failed")
    captured = capsys.readouterr()
    assert "ERROR: staging failed" in captured.err
    assert "journey:" not in captured.err


def test_call_follower_prints_preparation_progress_and_logs(capsys) -> None:
    example = load_example()

    class Job:
        def iter_events(self, **_kwargs):
            yield {
                "type": "preparation",
                "attributes": {
                    "phase": "artifact_staged",
                    "completed_files": 4,
                    "completed_bytes": 4096,
                    "worker_id": "worker-a",
                },
            }
            yield {"type": "stdout", "stream_data": "training\n"}
            yield {"type": "stderr", "stream_data": "warning\n"}
            yield {
                "type": "artifact",
                "attributes": {
                    "artifact_id": "art_checkpoint",
                    "generation": 1,
                },
            }

    example.follow_call(Job(), timeout_s=30, capacity_wait_s=20)

    captured = capsys.readouterr()
    assert "phase=artifact_staged files=4 bytes=4096 worker=worker-a" in captured.out
    assert "training" in captured.out
    assert "generation=1 artifact=art_checkpoint durable=true" in captured.out
    assert "warning" in captured.err


def test_call_follower_reports_capacity_rejections(capsys, monkeypatch) -> None:
    example = load_example()
    monotonic = iter((100.0, 100.0, 105.9))
    monkeypatch.setattr(example.time, "monotonic", lambda: next(monotonic))

    class Job:
        def iter_events(self, **_kwargs):
            yield {
                "type": "diagnostic",
                "attributes": {
                    "type": "placement_rejection",
                    "reason": "gpu_occupancy",
                    "worker_id": "worker-b",
                    "placement_generation": 4,
                },
            }

    example.follow_call(Job(), timeout_s=30, capacity_wait_s=120)

    captured = capsys.readouterr()
    assert (
        "waiting for capacity reason=gpu_occupancy worker=worker-b generation=4 "
        "elapsed=5s remaining=115s"
    ) in captured.out


def test_call_follower_reports_periodic_liveness(capsys, monkeypatch) -> None:
    example = load_example()
    monotonic = iter((100.0, 100.0, 130.0, 130.0, 131.0))
    monkeypatch.setattr(example.time, "monotonic", lambda: next(monotonic))

    class Job:
        call_id = "call_test"
        iterations = 0

        def iter_events(self, **kwargs):
            self.iterations += 1
            if self.iterations == 1:
                assert kwargs["after"] is None
                assert kwargs["timeout_s"] == 30
                raise TimeoutError("poll interval elapsed")
            yield {
                "cursor": "5",
                "type": "state",
                "state": "succeeded",
            }

        def status(self):
            return {"state": "queued"}

    example.follow_call(Job(), timeout_s=120, capacity_wait_s=120)

    captured = capsys.readouterr()
    assert "still waiting state=queued last=submission idle=30s elapsed=30s" in captured.out


def test_interrupt_requests_remote_cancellation() -> None:
    example = load_example()

    class Job:
        def cancel(self, *, reason):
            assert reason == "client interrupted while waiting"
            return {"state": "cancelling"}

    assert example.interrupt_message(Job(), detach=False) == (
        "local wait stopped; remote Call cancellation requested (state=cancelling)"
    )


def test_interrupt_can_leave_remote_call_active() -> None:
    example = load_example()

    class Job:
        def cancel(self, **_kwargs):
            raise AssertionError("detachment must not request cancellation")

    assert example.interrupt_message(Job(), detach=True) == (
        "local wait stopped; the remote Call remains active"
    )
