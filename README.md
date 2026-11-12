# QwenPU on Bonsai-27B

This experimental port runs QwenPU's RISC-V and vector circuit as sparse CUDA operations inside the patched Bonsai-27B Q1_0 forward pass. A clean Windows build on an 8 GB GeForce GTX 1080 validated the retained 3.825 GB GGUF. The three-token stories260K guest returned `Once upon a`, and all 150,405 execution-trace tokens matched an independent reference interpreter.

This source release contains the overlay, guest programs, build scripts, and validation reports. Build the patched Prism llama.cpp runtime on Windows, then download the retained GGUF or export it from the pinned stock Bonsai GGUF. [Windows reproduction guide](docs/REPRODUCE.md) · [Model files on Hugging Face](https://huggingface.co/millrace-ai/qwenpu-bonsai) · [Validation evidence](results/README.md)

## Research update on stock Prism

A separate **6.92 GB research checkpoint** runs the RV32 and vector circuit using standard model operations with packed Bonsai-27B weights. We tested it on an 8 GB GTX 1080 using unmodified Windows Prism b9596. A research launcher loads the guest ELF and relays the model's selections through Prism's built-in MTP APIs.

- The checksum, dot product, and division suite matched **2,520 architectural states** across nine runs, including A → B → A reset. Changing one active circuit coefficient changed the vector result from 25 to 16. Restoring the coefficient recovered the exact trace and result.
- The original stories260K guest matched all **1,830 architectural states** through its first matrix-vector multiply in **21m 27s**, with Bonsai's parked text state preserved. Device-wide VRAM peaked at **7.56 GiB**.
- On an **8 GiB RTX 3070** running unmodified Linux Prism b9596, the same checkpoint ran the original three-token stories260K guest through halt. It printed **“Once upon a”** and a newline. All **37,080 architectural retirements** matched the independent reference, and Bonsai's parked state and matched-history handoff matched the original model. The run took **18h 29m** and peaked at **7.06 GiB device VRAM** and **6.74 GiB process RSS**. [Three-token report](results/forward/standard-layer/stock-mtp-vx-stories-three.json)

The full three-token Windows run is pending. The published **3.825 GB v0.1** checkpoint remains the release download for patched Prism. [Windows stock-Prism validation summary](results/forward/standard-layer/stock-mtp-vx-research.json)

## How it works

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

## Scope

- The CUDA demo uses a single fully offloaded NVIDIA GPU, sequence zero, and one input token per decode call. The kernel build targets Pascal `sm_61`.
- The GGUF requires this patched Prism runtime. Stock llama.cpp, LM Studio, and Ollama are incompatible with this experimental format.
- Story coverage ends at three guest tokens. General text quality, autonomous dispatch on the CUDA path, circuit-state serialization and batched decoding are outside the validation scope.

## Attribution

This independent port builds on the circuit and reference machine by Divyansh Agrawal, the Bonsai model and runtime by Prism ML, the Qwen3.6-27B base by Alibaba Cloud, and the stories260K guest weights by Andrej Karpathy. Created using Bonsai by Prism ML. [Qwenputer](https://github.com/divagr18/qwenputer) · [QwenPU](https://github.com/divagr18/QwenPU) · [Bonsai-27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) · [Prism llama.cpp](https://github.com/PrismML-Eng/llama.cpp) · [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) · [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K) · [Component licenses](LICENSE) · [Notices](NOTICE)
