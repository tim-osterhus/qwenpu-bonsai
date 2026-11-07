# Validation results

## v0.1 clean Windows validation

The full short suite passed on the retained stories260K GGUF using a clean patched Prism build on Windows 10 x64 and one GTX 1080 (8 GB, driver 582.66). The checkpoint's SHA-256 is `688571d20993ccd9080fb4dc5348205c99180d03c54e922bf83b13e6377a3ef1` before and after validation. The tests compared both the decoder's selected token stream and the circuit's encoded execution records with an independent interpreter. The full suite used the original guest ELFs. The source release redacts build paths in the ELF debug data and preserves all loadable bytes. [Guest hashes and provenance](../guests/README.md)

| File | What it records |
|---|---|
| [Story validation](forward/release-v0.1/validation.json) | All 3,252 execution-trace tokens through the first VMAT matched. The one- and three-token guests matched all 51,187 and 150,405, respectively, and returned `Once` and `Once upon a`. The probe also checked the handoff and preservation of Bonsai's cache and text position. |
| [Clean-build export manifest](forward/release-v0.1/clean-build-original-guests.manifest.json) | The manifest used by the full story suite, with the exact GGUF, stock input, tokenizer and original guest hashes. |
| [Release export manifest](forward/release-v0.1/Bonsai-27B-Q1_0-QwenPU-stories260k.manifest.json) | A re-export with the bundled ELFs produced the same GGUF SHA-256. This is the manifest distributed with the checkpoint. |
| [Hugging Face download verification](forward/release-v0.1/hf-download-verification.json) | The downloaded GGUF matched the release hash before and after the first-VMAT CUDA smoke test. All 3,252 execution-trace tokens matched. |
| [Runtime identity](forward/release-v0.1/runtime-identity.json) | Windows and GPU environment, build inputs and versions, and hashes for the tested `llama.dll`, probe, CUDA cubin and supporting DLLs. |
| [Separate vector fixture](forward/release-v0.1/vector-validation.json) | All checks passed for fixture checkpoint `03684856…`. The coefficient mutation changed `8` to `0` and restoring it restored the exact trace. The stock logit comparisons repeated the historical result. |

The clean-build story report and its manifest are byte-for-byte copies. The separate vector report replaces the absolute checkpoint path with its filename. The original vector report SHA-256 is `9b20e04ce6b775c7f54a1a895e8c34006ae83ad8e2caf9edc1d29a3773daa25f`. [Distributed report checksums](SHA256SUMS.txt)

## Historical validation

These reports come from runs on 2026-11-05 and 2026-11-06 on one GTX 1080 (8 GB, driver 582.66). Some reports replace absolute local paths with `<qwenputer>` or `<local>`. `SHA256SUMS.txt` records the distributed file hashes. The original hashes of the historical reports that contained paths appear below.

| Report | Original SHA-256 |
|---|---|
| `forward/decoder-validation/validation.json` | `311e931e05937ef8633836fda4c09a271f00081ff7b7e8af9119afe1b69e2985` |
| `forward/stories260k/validation.json` | `1d849b31c5c4b9f1304332c77b4c2df64b98639a11bacff55493348aebd62589` |
| `in-graph/checkpoint/two-program/validation.json` | `5fd171c4c4b06c0a94d9cf71790387434187f3ea629f74b52a0c98bd8c90dd22` |
| `in-graph/checkpoint/autonomous/validation.json` | `b374c0681419bae7262484654bf34282b4056a02ad06224ca94f3b6429a5d732` |
| `in-graph/checkpoint/stories260k/validation.json` | `85662a70a6a9cbb4e1d0f2d12ae240882dc6abd260b9843ad00ef6c34d48a3e7` |

### Forward decoder with the circuit inside Bonsai's CUDA forward pass

| File | What it shows |
|---|---|
| `forward/stories260k/validation.json` | stories260K runs exactly for 1 guest token (51,187 execution-trace tokens) and 3 guest tokens (150,405), with resources and runtime hashes. This run used a temporary copy of the story checkpoint (`688571d2…`, 3,825,213,248 bytes). |
| `forward/decoder-validation/validation.json` | Vector fixture: exact traces, the mutation test (`8` becomes `0`, and restoring the coefficient is exact), and Bonsai's logits compared with stock. |
| `forward/decoder-validation/provenance.json` | Runtime revision, hardware, and hashes of the DLL, cubin, probe and checkpoint (`03684856…`). |
| `forward/decoder-validation/upstream-kernel-correspondence.json` | How the CUDA kernels map to the upstream executor. |
| `forward/checkpoint-verification.json` | Original tensors and metadata are byte-equal; circuit payload sizes and layout. |
| `forward/mutation.json`, `reference.json`, `inputs.json` | Inputs for the mutation test and the independent reference. |

### Shared-context build with the circuit on the CPU backend

| File | What it shows |
|---|---|
| `in-graph/two-program/validation.json` | Two programs in one context. State carries over correctly, and a stale-state check is included. |
| `in-graph/autonomous/validation.json` | Autonomous tool dispatch: 6/6 expected calls and 7/7 expected no-calls. |
| `in-graph/stories260k/validation.json` | First stories260K token, exact. |
| `in-graph/checkpoint/*` | The same checks repeated on the combined single-file checkpoint. |
