# stories260K validation across packed Bonsai and Qwen3.5

## Complete three-token workload on stock Windows Prism

All four hosts matched **37,080 architectural records** and **726 VMAT operations** through output and halt on the GTX 1080. The normalized trace and final vector state hashes agree. Parked state, result insertion, and the next decode matched the original packed host at the checked histories for each checkpoint. An ordered-write audit checked **55,375 writes** on each 4B host. Those runs also checked the signed F16 weight cache.

| Host | GGUF (GB) | Capture (h:mm:ss) | Post-seed decodes/s | Process RAM (GiB) | Dedicated VRAM (GiB) | Device VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bonsai-27B b12128](https://huggingface.co/millrace-ai/qwenpu-bonsai-27b-stock-prism) | 6.047 | 14:41:39 | 128.7 | 6.64 | 6.59 | 7.26 |
| [Bonsai 2 9B](https://huggingface.co/millrace-ai/qwenpu-bonsai-9b-stock-prism) | 4.500 | 10:10:31 | 174.0 | 4.64 | 4.94 | 6.27 |
| [Bonsai 2 4B](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism) | 4.031 | 8:48:59 | 159.3 | 4.46 | 4.73 | 6.84 |
| [Qwen3.5-4B UD-IQ2_M](https://huggingface.co/millrace-ai/qwenpu-qwen35-4b-iq2-m-stock-prism) | 5.121 | 8:13:40 | 170.7 | 5.72 | 5.73 | 6.72 |

The wall times are sampled process capture times. Shared GPU peaks were 86 MiB for 27B and 9B and 84 MiB for the two 4B hosts. Different schedules, kernels and sessions affect the comparison. [Pinned checkpoint identities and full-run receipts](../results/forward/standard-layer/evidence-snapshot.json) · [9B full-run receipt](https://huggingface.co/millrace-ai/qwenpu-bonsai-9b-stock-prism/blob/a410a122a9005acb999cb595fb0cc96ddbc68b52/evidence/story-three-token-windows.json)

## Full three-token stories260K on Bonsai 2 4B

The **4.03 GB** native-width Bonsai 2 4B checkpoint on stock Windows Prism **b12128** printed `Once upon a` and a newline, then halted. The pinned CPU reference matched **37,080 architectural trace records, 55,375 ordered writes, and 726 VMAT operations**. Its normalized trace and final vector state matched the full 27B Windows result. Bonsai's parked text state, result insertion, and next decode matched the original packed 4B control at the recorded histories.

Capture took **8 h 49 min** on the GTX 1080. Sampled peaks were **4.73 GiB** dedicated VRAM, **84 MiB** shared GPU memory, **6.84 GiB** device-wide VRAM, and **4.46 GiB** process RAM. The public 4B package passed a fresh-download smoke test. The packaged verifier reproduced the full verdict from the saved capture. [4B download and setup](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism) · [Full-run measurement](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism/blob/c0b9de4dd5ad7057ad677fb01f7e99e4ce350e00/evidence/4b-full-windows.json) · [Packaged verifier replay](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism/blob/c0b9de4dd5ad7057ad677fb01f7e99e4ce350e00/evidence/4b-full-windows-portable-replay.json)

## Shared stories260K prefix

The same stories260K guest matched **2,578 architectural trace records** on packed Bonsai 27B, 9B, and 4B and Qwen3.5-4B UD-IQ2_M on an **8 GiB GTX 1080** with stock Windows Prism **b12128**. All four normalized traces and final vector states share the same hashes. Each parked text context matched its original packed model. These captures stop before the first story token. [Bonsai prefix verification](../results/forward/standard-layer/windows-stories260k-prefixes.json) · [Qwen3.5 prefix verification](../results/forward/standard-layer/windows-qwen35-4b-iq2-story-prefix.json)

The 9B checkpoint retains its native 4,096-wide hidden state with 20,480 FFN rows, of which the circuit uses 19,952. The 4B checkpoint retains its native 2,560-wide hidden state with 36,352 FFN rows. Both use packed PQ2_0 language weights and preserve the original language tensor extents byte for byte, including the original portions of the padded tensors.

## Matched story window

All captures started at ELF entry. The timed interval covered records **1,717–2,578** inclusive. The guest reserved 65,536 FP16 KV cells and held 55,454 to 55,720 live cells at the interval boundaries. The 9B run and the 27B control used the same launcher and ran in that order. The Bonsai 4B and Qwen3.5-4B captures used the same native 4B launcher in separate sessions.

| Checkpoint | Window time | Window decodes | Peak dedicated VRAM | Peak device VRAM | Peak process RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 27B control, 6.047 GB GGUF | 954.795 s | 121,982 | 6.59 GiB | 7.40 GiB | 6.64 GiB |
| 9B, 4.50 GB GGUF | 625.741 s | 102,398 | 4.94 GiB | 5.69 GiB | 4.96 GiB |
| Repaired Bonsai 4B, 4.03 GB GGUF | 474.059 s | 82,090 | 4.73 GiB | 5.79 GiB | 4.72 GiB |
| Qwen3.5-4B UD-IQ2_M, 5.12 GB GGUF | 518.781 s | 82,090 | 5.73 GiB | 7.29 GiB | 5.72 GiB |

Memory peaks are sampled across the prefix capture. System RAM is process working set. Device VRAM includes other GPU processes. The 27B and 9B comparison measures one run per checkpoint and includes different circuit layouts and action schedules. Qwen3.5-4B took 9.4% longer than the separate Bonsai 4B capture at the same decode count. These timings characterize the tested configurations and sessions.

## 4B vector exponentiation repair

In the first 4B capture, vector exponentiation wrote 0 at record 2,005 where the reference expected 127. Strengthening the adder operand signals from attention readback fixed the 12-instruction regression. Both exponentiation writes returned 127 with the existing action schedule. The repaired story prefix matched all **2,578 records and 2,760 ordered guest writes**. [Regression and trace summary](../results/forward/standard-layer/windows-stories260k-prefixes.json)

## Qwen3.5-4B UD-IQ2_M prefix validation

The **5.12 GB** Qwen3.5-4B test checkpoint combined Unsloth's separately quantized UD-IQ2_M language weights with the **18 conditioned Bonsai 4B circuit tensors**. The export retained **426 original packed tensor extents** and extended 96 feed-forward tensors with exact zero blocks. The padding-only control matched logits, KV, and recurrent state through serial histories of up to 25 tokens and batched histories of up to 64. Separate scalar and vector fixtures matched **316 trace records, 307 ordered writes, and six text handoffs** against the reference and the original Qwen control, including vector A → B → A reset. The story prefix matched **2,578 records, 2,760 ordered writes, and eight VMAT operations**. The signed FP16 weight-cache evidence matched the Bonsai 4B prefix, and the parked text state matched original Qwen3.5-4B. The story capture ended before output or handoff. [Prefix, padding, and fixture receipts](../results/forward/standard-layer/windows-qwen35-4b-iq2-story-prefix.json) · [Pinned base quantization](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523)

## Experiments at the original 4B FFN width

A standalone VMATR and VGET fixture at 2,560 lanes and 9,216 FFN rows matched 120 VMATR operations, 600 VGET results, and 2,400 ordered writes under FlashAttention on and off. It used predecoded instructions and 55,720 live F16 KV cells at synthetic occupancy. A separate partial CPU compiler census used 1,982 lanes and 4,023 rows with a synthetic microprogram ROM. The real controller, typed cache updates, and full-guest execution at 9,216 rows remain open. [Fixture and census evidence](../results/forward/standard-layer/evidence-snapshot.json)
