# QwenPU in packed Bonsai and Qwen3.5

A RISC-V processor and vector unit execute through model weights in the same GGUF as a packed language model. The launcher loads the stories260K guest and relays the circuit’s selected actions through stock Prism’s multi-token prediction APIs. A pinned CPU reference verifies the resulting execution trace offline.

## Full stories260K on four packed hosts

On an 8 GiB GTX 1080 with unmodified Windows Prism **b12128**, Bonsai-27B, Bonsai 2 9B and 4B, and Qwen3.5-4B UD-IQ2_M each printed `Once upon a` followed by a newline and halted. All **37,080 architectural records** and **726 VMAT operations** matched the pinned CPU reference. Each host's parked state, result insertion, and next decode matched the original packed model at the checked text histories.

| Host | GGUF (GB) | Capture (h:mm:ss) | Post-seed decodes/s | Process RAM (GiB) | Dedicated VRAM (GiB) | Device VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bonsai-27B b12128](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism) | 6.047 | 14:41:39 | 128.7 | 6.64 | 6.59 | 7.26 |
| [Bonsai 2 9B](https://huggingface.co/millrace-ai/qwenpu-bonsai-9b-stock-prism) | 4.500 | 10:10:31 | 174.0 | 4.64 | 4.94 | 6.27 |
| [Bonsai 2 4B](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism) | 4.031 | 8:48:59 | 159.3 | 4.46 | 4.73 | 6.84 |
| [Qwen3.5-4B UD-IQ2_M](https://huggingface.co/millrace-ai/qwenpu-qwen35-4b-iq2-m-stock-prism) | 5.121 | 8:13:40 | 170.7 | 5.72 | 5.73 | 6.72 |

Capture time includes setup and seeding. Decode speed uses the loop after seeding. Device VRAM includes other GPU activity. The 9B and Qwen3.5 runs complete the earlier prefix validation. [Full-run evidence](results/forward/standard-layer/evidence-snapshot.json) · [Portability report](docs/BONSAI-PORTABILITY.md)

## Downloads

- **Bonsai 2 4B on Prism b12128.** The model card includes the GGUF, Windows launcher, smoke test, and full-story verifier. [4B package](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism) · [Reconstruct the 4B GGUF from source](reconstruction/bonsai-4b/README.md)
- **Bonsai 2 9B and Qwen3.5-4B UD-IQ2_M on Prism b12128.** Both combined GGUFs have full Windows verification. Their model cards include setup and commands for the bounded capture. [9B package](https://huggingface.co/millrace-ai/qwenpu-bonsai-9b-stock-prism) · [Qwen3.5-4B package](https://huggingface.co/millrace-ai/qwenpu-qwen35-4b-iq2-m-stock-prism)
- **Bonsai-27B on Prism b12128.** The **6.047 GB** GGUF matches the full Windows run. [Checkpoint and checksums](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/tree/764da1a0a9f6f5cd68670a7a28eb2410a773c491/stock-prism-b12128) · [Windows run evidence](results/forward/standard-layer/windows-stories260k/README.md)
- **Bonsai-27B on Prism b9596.** This separate **6.920 GB** GGUF passed the full story on Linux and the vector smoke test and story prefix on Windows. [27B package](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism) · [Windows reproduction guide](docs/REPRODUCE-STOCK-PRISM.md)
- **Original patched v0.1.0.** The earlier **3.825 GB** 27B GGUF uses a patched Windows CUDA runtime. [Historical release](https://github.com/tim-osterhus/qwenpu-bonsai/releases/tag/v0.1.0) · [Patched build guide](docs/REPRODUCE.md)

[Qwenputer + Bonsai model collection](https://huggingface.co/collections/millrace-ai/qwenputer-bonsai-6aeee4defe3d3d356acb4b6c)

## Research update on stock Prism

### The Bonsai-27B b9596 baseline

The downloadable **6.920 GB stock checkpoint** runs the RV32 and vector circuit using standard model operations with packed Bonsai-27B weights. We tested it on an 8 GiB GTX 1080 using unmodified Windows Prism b9596. A research launcher loads the guest ELF and relays the model's selections through Prism's built-in MTP APIs.

- The checksum, dot product, and division suite matched **2,520 architectural states** across nine runs, including A → B → A reset. Changing one active circuit coefficient changed the vector result from 25 to 16. Restoring the coefficient recovered the exact trace and result.
- The original stories260K guest matched all **1,830 architectural states** through its first matrix-vector multiply in **21m 27s**, with Bonsai's parked text state preserved. Device-wide VRAM peaked at **7.56 GiB**.
- On an **8 GiB RTX 3070** running unmodified Linux Prism b9596, the same checkpoint ran the original three-token stories260K guest through halt. It printed **“Once upon a”** and a newline. All **37,080 architectural retirements** matched the independent reference, and Bonsai's parked state and matched-history handoff matched the original model. The run took **18h 29m** and peaked at **7.06 GiB device VRAM** and **6.74 GiB process RSS**. [Three-token report](results/forward/standard-layer/stock-mtp-vx-stories-three.json)

The **6.92 GB stock checkpoint** is available with a Windows launcher, setup commands, and the reference verifier. A fresh download passed all **40 vector fixture retirements**, Bonsai's parked state, and its matched-history handoff on a GTX 1080. [Stock checkpoint package](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism/tree/stock-prism-b9596/stock-prism) · [Windows smoke-test report](results/forward/standard-layer/stock-mtp-vx-package-smoke.json)

### stories260K prefix portability

Bonsai 27B, Bonsai 2 9B and 4B, and Qwen3.5-4B UD-IQ2_M all matched the same 2,578-record prefix. The normalized architectural trace and final vector state hashes agree across all four captures. In each run, the parked text state matched the corresponding original model. [Prefix comparison](docs/BONSAI-PORTABILITY.md)

## How the v0.1 patched build works

The exporter retargets the upstream circuit for Bonsai's geometry and adds its sparse coefficients and guest snapshots to the same GGUF as Bonsai's tensors. The circuit has 23 active sublayers, which the pinned QwenPU standard-layer emitter groups into 13 blocks. This port runs those sublayers as custom CUDA-backed GGML nodes. The retained circuit coordinates contribute to Bonsai's residual RMS normalization. During chip-only steps, the circuit advances while the language model's position and cache are held. [Decoder implementation](overlay/experimental/bonsai-qwenpu/FORWARD.md) · [Pinned upstream emitter](https://github.com/divagr18/QwenPU/blob/6e2767aaa59b69ea03300c0a5199d7b36debbe50/chipformer/q35/emit.py)

Upstream QwenPU documents a 26.7 GB dense BF16 checkpoint based on Qwen3.5-4B, with a 40 GB GPU for the whole model or 24 GB for the isolated circuit. The figures describe different base models and representations. The port's packed CUDA path and upstream's standard-layer checkpoint have different runtime requirements. [Pinned upstream QwenPU README](https://github.com/divagr18/QwenPU/blob/6e2767aaa59b69ea03300c0a5199d7b36debbe50/README.md)

## v0.1 Windows validation

| Check | Output | Exact execution-trace tokens | Native elapsed time | Probe wall time |
|---|---|---:|---:|---:|
| Through first matrix-vector multiply | — | 3,252 | 5.2 s | 32.7 s |
| 1 stories260K token | `Once` | 51,187 | 85.9 s | 93.6 s |
| 3 stories260K tokens | `Once upon a` | 150,405 | 243.2 s | 246.9 s |

The three-token run averaged 618 execution-trace tokens/s in the native loop. The sampled peak was 5,403 MiB of device-wide VRAM against a 1,192 MiB baseline, and the peak sampled process-tree working set was 4.80 GiB. The probe supplied the fixed prefix `Run<tool_call></tool_call>` and fed back the decoder's selections. The validator checked the bridge, CUDA layer execution, and preservation of Bonsai's cache and text position during circuit-only steps. An execution-trace token is one element of the circuit's encoded record stream. An instruction can produce several such tokens. [Story validation report](results/forward/release-v0.1/validation.json) · [Runtime identity](results/forward/release-v0.1/runtime-identity.json)

The retained GGUF is 3,825,213,248 bytes, compared with 3,803,452,480 bytes for stock Bonsai. The added tensors hold an 18,212,552-byte circuit graph and two program snapshots totaling 3,547,960 bytes. The exporter verified byte equality of all 851 original tensor payloads and equality of the original metadata entries. The checkpoint SHA-256 stayed the same across validation and re-export with the bundled ELFs. [Release manifest](results/forward/release-v0.1/Bonsai-27B-Q1_0-QwenPU-stories260k.manifest.json) · [Guest provenance](guests/README.md)

A **separate vector fixture** tested whether the decoder used the stored coefficients. Changing one coefficient changed the output from `8` to `0`. Restoring it restored the exact trace. The fresh run repeated the historical stock Bonsai comparison and found the same greedy ID both during idle and after handoff, with maximum absolute logit differences of about 0.106 and 0.107, respectively. The mutation itself preserved the fixture's idle logits. [v0.1 vector fixture report](results/forward/release-v0.1/vector-validation.json) · [Historical vector fixture report](results/forward/decoder-validation/validation.json)

An earlier CPU circuit backend in a shared llama context recorded a first-token story run and a small autonomous dispatch test with 6/6 expected calls and 7/7 expected no-calls. Those results concern the earlier backend. The CUDA story demo uses the fixed prefix above. [Historical shared-context reports](results/in-graph/)

## v0.1 scope

- The CUDA demo uses a single fully offloaded NVIDIA GPU, sequence zero, and one input token per decode call. The kernel build targets Pascal `sm_61`.
- The v0.1 GGUF requires this patched Prism runtime. Stock llama.cpp, LM Studio, and Ollama are incompatible with this experimental format.
- Story coverage ends at three guest tokens. General text quality, autonomous dispatch on the CUDA path, circuit-state serialization and batched decoding are outside the validation scope.

## Attribution

This independent port builds on the circuit and reference machine by Divyansh Agrawal, the Bonsai model and runtime by Prism ML, the Qwen model bases by Alibaba Cloud, the UD-IQ2_M quantization by Unsloth, and the stories260K guest weights by Andrej Karpathy. Created using Bonsai by Prism ML. [Qwenputer](https://github.com/divagr18/qwenputer) · [QwenPU](https://github.com/divagr18/QwenPU) · [Bonsai-27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) · [Prism llama.cpp](https://github.com/PrismML-Eng/llama.cpp) · [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) · [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K) · [Component licenses](LICENSE) · [Notices](NOTICE)
