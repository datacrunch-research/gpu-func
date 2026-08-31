# Fine-tune Nemotron with LoRA

This chapter runs supervised fine-tuning (SFT) with LoRA through the public `gfaas` API.

gfaas stages the model and training data as immutable Artifacts. A prepared image runs NVIDIA NeMo
AutoModel on one worker. The Call publishes the adapter, training report, and recovery checkpoints
as new Artifacts.

This workflow supports one or more GPUs on one worker. It does not cover pretraining, reinforcement
learning, or training across multiple workers.

## How this workflow relates to the NVIDIA guides

NVIDIA documents complete Nemotron pipelines for data preparation, SFT, reinforcement learning,
evaluation, and deployment. This chapter focuses on the SFT stage with a user-provided corpus.

| NVIDIA guide stage          | gfaas equivalent                                                         |
| --------------------------- | ------------------------------------------------------------------------ |
| Select a model checkpoint   | Supply a model Artifact or upload a local model directory.               |
| Prepare chat data           | Supply JSONL. The example validates and converts each record.            |
| Configure a training recipe | Select command options. The example creates the AutoModel recipe.        |
| Launch SFT on a cluster     | Submit one durable gfaas Call.                                           |
| Track artifact lineage      | Keep the local report and its model, data, checkpoint, and adapter IDs.  |
| Evaluate and deploy         | Download the adapter and evaluate it with the exact base-model revision. |

The structure follows NVIDIA's
[Nemotron 3 Nano training guide](https://docs.nvidia.com/nemotron/latest/nemotron/nano3/README.html)
and
[Nemotron 3.5 Lightning training guide](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/README.html).
The gfaas examples use NeMo AutoModel instead of a direct Slurm or container launch.

## Choose a model

Start with the BF16 instruct checkpoint. LoRA keeps the base weights frozen and trains a much
smaller adapter.

| Model                          | gfaas example                         | Notes                                                                       |
| ------------------------------ | ------------------------------------- | --------------------------------------------------------------------------- |
| Nemotron 3 Nano 30B-A3B        | `examples/nemotron_lora.py`           | 31.6B total parameters and 3.6B active parameters.                          |
| Nemotron 3.5 Lightning 30B-A3B | `examples/nemotron_lightning_lora.py` | 30B total parameters, 3B active parameters, and model-specific MTP support. |

The general Nano example defaults to the earlier `NVIDIA-Nemotron-Nano-9B-v2` pilot model. For
Nemotron 3 Nano, always supply the model name and revision shown in this chapter.

The Lightning example is separate because its LoRA target list and multi-token prediction (MTP) path
are model-specific. Do not use the Nano recipe for Lightning.

Read the NVIDIA model cards before you select either checkpoint:

- [Nemotron 3 Nano 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)

## Prerequisites

Ask your gfaas operator for these public values:

- The API base URL and API key.
- A GPU-pool name, such as `gb300` or `b200`.
- The prepared image name. The examples default to `nemo-automodel-lora-cu130`.
- A readable model Artifact ID, if the service already stores the model.
- The resource ceilings for GPU count, memory, scratch space, and execution time.

Install the SDK and the client-side tokenizer dependency:

```bash
uv sync --extra training-example
```

Configure the client:

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
export GFAAS_API_KEY=replace-with-your-key
export GFAAS_GPU_TYPE=gb300
```

Keep the API key out of scripts, reports, shell history, and training data.

### Supply the model

An existing model Artifact avoids a large upload for each user. Set its identity when your operator
provides one:

```bash
export NANO_MODEL_ARTIFACT=art_replace_with_nano_model
export LIGHTNING_MODEL_ARTIFACT=art_replace_with_lightning_model
```

You can also download a pinned model to your client and use `--model`. The example uploads the
directory through the public API and prints the new Artifact ID.

```bash
export MODEL_DIR="$(hf download \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --revision 2d59de1cbd51c0adf384eb906b766d1aee0e0517)"
```

Replace `--model-artifact "$NANO_MODEL_ARTIFACT"` with `--model "$MODEL_DIR"` in later commands.
Large directory uploads can take time. Reuse the printed Artifact ID after the first upload.

## Prepare the training data

Write one JSON object on each line of the training file. Each record needs a `messages` array with
at least one user message. The final message must be an assistant response.

```json
{
  "messages": [
    { "role": "system", "content": "You write concise release notes." },
    { "role": "user", "content": "Describe a fix for a connection retry bug." },
    {
      "role": "assistant",
      "content": "Retry the connection after a bounded delay, and stop after the configured attempt limit."
    }
  ]
}
```

The example applies the model's pinned chat template. It trains on the final assistant response and
masks the system and user tokens.

The input can also contain OpenAI-format tool definitions, tool calls, and tool responses. Every
tool call must refer to a declared tool. Every tool response must match a preceding call.

The qualifier rejects these records:

- A record without a user message.
- A record that does not end with an assistant message.
- A record without supervised assistant tokens.
- A record that exceeds `--sequence-length`.
- A tool call that has no valid declaration or response.

The qualifier does not truncate long conversations. This rule prevents a partial assistant answer or
tool exchange from entering the training set.

Create separate training and evaluation files before submission. Split related sessions together to
prevent adjacent turns from entering both sets.

Keep the evaluation file outside the training Artifact. gfaas does not create a held-out split for
this example.

## Run a one-step pilot

A one-step pilot validates authentication, model staging, data conversion, GPU execution, and
adapter publication. It does not establish model quality.

### Nemotron 3 Nano pilot

```bash
uv run --extra training-example python examples/nemotron_lora.py \
  --model-artifact "$NANO_MODEL_ARTIFACT" \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --model-revision 2d59de1cbd51c0adf384eb906b766d1aee0e0517 \
  --training-data training.jsonl \
  --gpu-type "$GFAAS_GPU_TYPE" \
  --gpu-count 1 \
  --sequence-length 8192 \
  --maximum-samples 8 \
  --max-steps 1 \
  --report reports/nemotron-3-nano-pilot.json
```

### Nemotron 3.5 Lightning pilot

```bash
uv run --extra training-example python examples/nemotron_lightning_lora.py \
  --model-artifact "$LIGHTNING_MODEL_ARTIFACT" \
  --training-data training.jsonl \
  --gpu-type "$GFAAS_GPU_TYPE" \
  --gpu-count 1 \
  --sequence-length 8192 \
  --maximum-samples 8 \
  --max-steps 1 \
  --report reports/nemotron-3.5-lightning-pilot.json
```

The Lightning example pins revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`. It records this
revision in the local and remote reports.

## Read the pilot results

The command prints qualification counts before AutoModel starts. A healthy pilot has these
properties:

- `selected` and `emitted_samples` are more than zero.
- `num_label_tokens` is more than zero.
- The loss and gradient norm are finite and positive.
- The Call publishes an `adapter` Artifact.
- The Call publishes a `training-report` Artifact.
- The Call publishes one or more `checkpoint` versions.

The first optimizer step can take much longer than later steps. Model loading, kernel setup, and
distributed initialization occur before that step.

Read the local report after the Call stops:

```bash
jq '{call_id, model, model_revision, resources, status, error, artifacts}' \
  reports/nemotron-3.5-lightning-pilot.json
```

The local report records all input and output Artifact identities. Keep this report with your
experiment notes.

## Run a longer Nemotron 3 Nano experiment

Use four GPUs only when one worker has four idle GPUs. gfaas does not combine GPUs from different
workers for one Call.

```bash
uv run --extra training-example python examples/nemotron_lora.py \
  --model-artifact "$NANO_MODEL_ARTIFACT" \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --model-revision 2d59de1cbd51c0adf384eb906b766d1aee0e0517 \
  --training-data training.jsonl \
  --gpu-type "$GFAAS_GPU_TYPE" \
  --gpu-count 4 \
  --sequence-length 8192 \
  --full-corpus \
  --max-steps 500 \
  --checkpoint-every-steps 125 \
  --capacity-wait 7200 \
  --timeout 21600 \
  --startup-diagnostics \
  --report reports/nemotron-3-nano-500.json
```

For this recipe, the global batch size equals the GPU count. Divide `emitted_samples` by the GPU
count to estimate one corpus pass.

Treat `--max-steps` as an experiment limit, not a quality target. Compare checkpoints on held-out
data before you increase this value.

## Run one Lightning corpus pass

The Lightning example accepts `--epochs`. The worker calculates the step count after qualification
and record weighting.

NVIDIA's Lightning SFT guide uses role-based assistant loss and a model-specific data pipeline. The
gfaas example preserves assistant-only supervision and uses the checkpoint's chat template.

```bash
uv run --extra training-example python examples/nemotron_lightning_lora.py \
  --model-artifact "$LIGHTNING_MODEL_ARTIFACT" \
  --training-data training.jsonl \
  --gpu-type "$GFAAS_GPU_TYPE" \
  --gpu-count 4 \
  --global-batch-size 8 \
  --distributed-strategy ddp \
  --sequence-length 8192 \
  --full-corpus \
  --epochs 1 \
  --memory-gib 384 \
  --capacity-wait 7200 \
  --timeout 21600 \
  --startup-diagnostics \
  --report reports/nemotron-3.5-lightning-epoch-1.json
```

The memory request is deployment-specific. If the operator provides a different ceiling, change
`--memory-gib` before submission.

The Lightning example uses DDP for multi-GPU Calls by default. Each rank loads the model, and DDP
synchronizes the LoRA gradients through NCCL.

The example uses LoRA rank 8 and alpha 8. It targets the supported attention, Mamba input,
shared-expert, and MTP projections. It excludes the language-model head and routed experts.

The settings are not the complete NVIDIA production SFT recipe. NVIDIA's reference recipe uses a
larger cluster, packed sequences, and different parallelism. Use the gfaas recipe for bounded
same-worker LoRA experiments.

## Checkpoints and resume

Each Call can publish as many as four checkpoint versions. The default interval divides the run into
four parts.

Set an explicit interval when you need predictable recovery points:

```bash
--max-steps 500 --checkpoint-every-steps 125
```

Resume from a published checkpoint Artifact:

```bash
uv run --extra training-example python examples/nemotron_lightning_lora.py \
  --model-artifact "$LIGHTNING_MODEL_ARTIFACT" \
  --training-data-artifact "$TRAINING_DATA_ARTIFACT" \
  --resume-checkpoint-artifact art_replace_with_checkpoint \
  --gpu-type "$GFAAS_GPU_TYPE" \
  --gpu-count 4 \
  --global-batch-size 8 \
  --distributed-strategy ddp \
  --sequence-length 8192 \
  --full-corpus \
  --max-steps 500 \
  --checkpoint-every-steps 125 \
  --memory-gib 384 \
  --timeout 21600 \
  --report reports/nemotron-3.5-lightning-resumed.json
```

`--max-steps` is the total target step count. It is not the number of additional steps.

Use the same model revision, data Artifact, sequence length, seed, shuffle policy, batch size, and
LoRA recipe after a resume.

## Download the adapter

Copy the `adapter` Artifact ID from the command output or local report. Then download the directory:

```bash
export ADAPTER_ARTIFACT=art_replace_with_adapter
uv run python -c 'import os, gfaas; gfaas.Client().download_artifact_directory(os.environ["ADAPTER_ARTIFACT"], "adapter")'
```

The destination directory must not exist. The SDK validates the tree manifest and every file before
it moves the temporary directory into place.

The adapter is not a complete model. Load it with the exact BF16 checkpoint and revision that
created it.

## Evaluate before a full run

Evaluate the base model and each adapter on the same held-out prompts. Keep generation settings and
the prompt template constant.

Measure the behavior that your corpus intends to change:

- Task correctness.
- Output format.
- Tool selection and argument structure.
- Refusal and safety behavior.
- General capabilities that must remain stable.

Training loss alone does not measure these behaviors. A decreasing loss can occur while the adapter
memorizes a narrow corpus.

For local vLLM evaluation, use a version that supports the selected model and LoRA:

```bash
vllm serve "$MODEL_DIR" \
  --enable-lora \
  --lora-modules tuned="$PWD/adapter"
```

NVIDIA's model cards contain current serving options for each model. NVIDIA also provides a
[Nemotron 3 Nano LoRA walkthrough](https://docs.nvidia.com/nemotron/latest/use-case-examples/sql-lora-finetuning-and-deployment/README.html)
that continues from training to vLLM or NIM deployment.

CAUTION: Do not assume that a BF16-trained adapter works with a quantized checkpoint. Qualify the
adapter and runtime combination before deployment.

## Long Calls and interruption

By default, `Ctrl-C` requests remote cancellation. The worker then stops the owned training process
and publishes outputs that permit publication on failure.

Use `--detach-on-interrupt` when the Call must continue after the local command stops:

```bash
--detach-on-interrupt
```

Keep the printed Call ID. The Call remains a durable API resource after the client disconnects.

The `--capacity-wait` value covers queue time, image preparation, and Artifact staging. The
`--timeout` value starts after worker acceptance and covers execution.

## Common problems

### The Call waits for capacity

The selected pool has no worker with the complete GPU request available. A four-GPU Call needs four
idle GPUs on one worker.

Wait for capacity, request fewer GPUs, or select another operator-provided pool.

### Artifact staging appears slow

Large model trees contain many weight files. The example prints staged and reused bytes while the
service prepares the worker.

Later Calls on the same worker can reuse cached files. A Call on another worker needs a separate
cache population.

### No records pass qualification

Read `invalid_reasons`, `too_long`, and `without_labels` in the training report. Then correct the
source records or choose an appropriate sequence length.

Do not increase the sequence length without checking GPU memory and training cost.

### The worker rejects memory or storage

The requested resources and current reservations exceed the worker policy. Ask the operator for the
pool ceilings and available capacity.

Reduce the request only when the model still fits. Removing a valid reservation does not reduce the
model's real resource use.

### The first step takes a long time

The first step includes model load, process-group setup, kernel setup, and memory allocation. Use
`--startup-diagnostics` to print bounded process and NCCL details.

Use `--no-triton` only to diagnose the AutoModel Triton LoRA path. Compare throughput after the
first step before you select either mode.

## NVIDIA references

- [Nemotron 3 Nano training recipe](https://docs.nvidia.com/nemotron/latest/nemotron/nano3/README.html)
- [Nemotron 3 Nano LoRA and deployment example](https://docs.nvidia.com/nemotron/latest/use-case-examples/sql-lora-finetuning-and-deployment/README.html)
- [NeMo AutoModel model coverage](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/overview)
- [Nemotron 3.5 Lightning training recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/README.html)
- [Nemotron 3.5 Lightning SFT stage](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/sft.html)
- [Nemotron 3.5 Lightning evaluation](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/evaluate.html)
