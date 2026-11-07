---
license: apache-2.0
base_model:
  - prism-ml/Bonsai-27B-gguf
tags:
  - gguf
  - qwenpu
  - risc-v
  - stories260k
---

# QwenPU on Bonsai-27B Q1_0

This v0.1 experimental checkpoint adds QwenPU's RISC-V and vector circuit to Bonsai-27B Q1_0. The patched Bonsai decoder runs the sparse circuit inside its CUDA forward pass. A clean Windows build on a GTX 1080 validated the exact retained checkpoint. The three-token stories260K guest produced `Once upon a` from the fixed prefix `Run<tool_call></tool_call>`.

**Runtime requirement.** Build the patched Windows CUDA version of Prism llama.cpp at revision `9fcaed763ccda38ea81068ad9d7f991aaddca451`. The validated build targets Pascal `sm_61`. Stock llama.cpp, LM Studio, and Ollama are incompatible with this format. [Windows build and launch commands](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/docs/REPRODUCE.md) · [Runtime identity](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/results/forward/release-v0.1/runtime-identity.json)

## Model files

| File | Bytes | SHA-256 |
|---|---:|---|
| `Bonsai-27B-Q1_0-QwenPU-stories260k.gguf` | 3,825,213,248 | `688571d20993ccd9080fb4dc5348205c99180d03c54e922bf83b13e6377a3ef1` |
| `Bonsai-27B-Q1_0-QwenPU-stories260k.manifest.json` | 2,180 | `b2f9e5692f54f672ce3ab49cdd9d99b5d834efe8de11a8a802a593a63d384d14` |

Keep the manifest beside the GGUF when running the validator. The release manifest identifies the bundled ELFs with build paths redacted in debug data. Those ELFs reproduce the full suite's reference traces and export the identical GGUF. [Model repository](https://huggingface.co/millrace-ai/qwenpu-bonsai) · [Guest provenance](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/guests/README.md)

## Checkpoint contents

The exporter adds sparse circuit coefficients and named guest snapshots to the stock Bonsai GGUF. The snapshots provide initial program and machine state, including the guest's int4 stories260K weights. The export verified byte equality of all 851 stock tensor payloads and equality of the original GGUF metadata entries.

The patched decoder executes 23 active circuit sublayers as custom CUDA-backed GGML nodes. The pinned upstream standard-layer emitter groups them into 13 blocks. Retained circuit coordinates contribute to the language model's residual normalization, so equal stock tensor bytes can still produce different logits.

## Validation on the retained GGUF

| Check | GTX 1080 result |
|---|---|
| Through first matrix-vector multiply | All 3,252 execution-trace tokens matched |
| 1 stories260K token | `Once` with all 51,187 execution-trace tokens matched |
| 3 stories260K tokens | `Once upon a` with all 150,405 execution-trace tokens matched, 618 execution-trace tokens/s in the native loop, and 246.9 s probe wall time |
| Memory, 3-token run | 5,403 MiB peak sampled device-wide VRAM (1,192 MiB baseline) and 4.80 GiB peak sampled process-tree RAM working set |
| Host state | Bonsai's cache and text position remained unchanged while the circuit ran in all three tests |

An execution-trace token is one element of the circuit's encoded record stream. An instruction can produce several such tokens. The validator compared these records and the full bridge token stream with an independent reference interpreter. [Story validation report](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/results/forward/release-v0.1/validation.json)

## Separate vector fixture checks

| Check | Recorded result |
|---|---|
| Coefficient mutation | Output `8` became `0`, and restoring the coefficient restored the exact trace |
| Stock comparison | Matching greedy IDs with maximum absolute logit differences of about 0.106 during idle and 0.107 after handoff |

The clean Windows run repeated the historical vector checks on a separate fixture checkpoint with SHA-256 `0368485626d314216eacd33155976118568f57e386535385c36884205cec5a54`. [v0.1 vector fixture report](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/results/forward/release-v0.1/vector-validation.json) · [Historical vector fixture report](https://github.com/tim-osterhus/qwenpu-bonsai/blob/v0.1.0/results/forward/decoder-validation/validation.json)

## Scope

Coverage is limited to a fully offloaded, single-GPU decoder, one token per call in sequence zero, and the fixed tool-call prefix. A 60-token guest, autonomous dispatch on the CUDA path, broader language quality, and circuit-state serialization are outside the validation scope.

## Attribution and modifications

Created using Bonsai by Prism ML. The underlying Bonsai-27B GGUF and Qwen3.6-27B use Apache-2.0. This independent port builds on QwenPU, Qwenputer, and stories260K. The GGUF export adds circuit and snapshot tensors and metadata to the source file. [Model distribution license](https://huggingface.co/millrace-ai/qwenpu-bonsai/blob/main/LICENSE) · [Component licenses](https://github.com/tim-osterhus/qwenpu-bonsai/tree/v0.1.0/licenses) · [Notices](https://huggingface.co/millrace-ai/qwenpu-bonsai/blob/main/NOTICE)
