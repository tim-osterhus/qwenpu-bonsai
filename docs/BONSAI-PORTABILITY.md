# stories260K prefix validation across packed Bonsai checkpoints

The same stories260K guest matched **2,578 architectural trace records** on packed Bonsai **27B, 9B, and 4B** using stock Windows Prism **b12128** on an **8 GiB GTX 1080**. The three runs matched the pinned CPU reference and shared the same normalized trace and final vector state hashes. They verified eight matrix multiplies, starting at record 1,677. In each run, Bonsai's parked text position, KV cache, recurrent state, and logits matched its original packed text control. The prefix ended before story output. [Archived validation and measurements](../results/forward/standard-layer/windows-stories260k-prefixes.json)

The 9B checkpoint retains its native 4,096-wide hidden state with 20,480 FFN rows, of which the circuit uses 19,952. The 4B checkpoint retains its native 2,560-wide hidden state with 36,352 FFN rows. Both use packed PQ2_0 language weights and preserve the original language tensor extents byte for byte, including the original portions of the padded tensors.

## Matched story window

All captures started at ELF entry. The timed interval covered records **1,717–2,578** inclusive. The guest reserved 65,536 FP16 KV cells and held 55,454 to 55,720 live cells at the interval boundaries. The 9B run and the 27B control used the same launcher and ran in that order. The repaired 4B used its own launcher and ran in a separate capture.

| Checkpoint | Window time | Window decodes | Peak dedicated VRAM | Peak device VRAM | Peak process RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 27B control, 6.047 GB GGUF | 954.795 s | 121,982 | 6.59 GiB | 7.40 GiB | 6.64 GiB |
| 9B, 4.50 GB GGUF | 625.741 s | 102,398 | 4.94 GiB | 5.69 GiB | 4.96 GiB |
| Repaired 4B, 4.03 GB GGUF | 474.059 s | 82,090 | 4.73 GiB | 5.79 GiB | 4.72 GiB |

Memory peaks are sampled across the prefix capture. System RAM is process working set. Device VRAM includes other GPU processes. The 27B and 9B comparison measures one run per checkpoint and includes different circuit layouts and action schedules. The 4B capture supplies a separate observation over the same guest instructions.

## 4B vector exponentiation repair

In the first 4B capture, vector exponentiation wrote 0 at record 2,005 where the reference expected 127. Strengthening the adder operand signals from attention readback fixed the 12-instruction regression. Both exponentiation writes returned 127 with the existing action schedule. The repaired story prefix matched all **2,578 records and 2,760 ordered guest writes**. [Regression and trace summary](../results/forward/standard-layer/windows-stories260k-prefixes.json)

## Full three-token status

The **27B b12128** research checkpoint completed the three-token workload through halt, with 37,080 matching records and an exact matched-history Bonsai handoff. Full story verification for **9B** and **4B** is pending. The 9B full-run attempts stopped during CUDA startup before guest execution. The repaired 4B full-run capture was in progress when checked on November 15, 2026 (HST). [Full 27B Windows result](../results/forward/standard-layer/windows-stories260k/README.md)

The 9B and 4B checkpoints and raw captures are local research artifacts. The public summary records checkpoint identities, source report fingerprints, verifier outcomes, the parked-state comparisons, and measurements. The downloadable stock package covers a separate **27B b9596** checkpoint, with full Linux validation and Windows smoke and prefix checks. [Stock release](STOCK-PRISM-RELEASE.md) · [Download and reproduction guide](REPRODUCE-STOCK-PRISM.md)
