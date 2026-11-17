# Reconstruct the stock Prism Bonsai 2 4B checkpoint

This recipe regenerates the compiled RISC-V and vector circuit, packs it beside the original Bonsai 2 4B tensors, and applies the released vector exponentiation (VEXP) readback conditioning. It targets the [released Bonsai 2 4B GGUF](https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism/tree/c0b9de4dd5ad7057ad677fb01f7e99e4ce350e00).

A clean rebuild from a Git archive in a fresh Windows Python environment matched all **4,031,384,384 bytes** of a separately downloaded release. [Verification record](rebuild-verification.json)

The build runs on the CPU with native Windows x64 and Python 3.11. Budget 10 GiB of free scratch space on a volume that supports hard links after downloading the inputs. The optional byte comparison also requires space for the 4.03 GB released file. For executing the resulting GGUF, use the stock Windows Prism b12128 launcher and verifier in the model package.

## Rebuild

From this directory in a clean checkout, create a Python environment and download the exact original model revision. The build checks the base file size and SHA-256 before using it.

Prism ML distributes the original Bonsai 2 4B weights under Apache 2.0. [Pinned base model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/tree/50ebba61be85e5be07e8083db468ace0fc1bac43)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
curl.exe --fail --location --output Ternary-Bonsai-2-4B-PQ2_0.gguf "https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/resolve/50ebba61be85e5be07e8083db468ace0fc1bac43/Ternary-Bonsai-2-4B-PQ2_0.gguf?download=true"
.\.venv\Scripts\python.exe -B .\rebuild.py --base .\Ternary-Bonsai-2-4B-PQ2_0.gguf --output-dir .\rebuilt
```

Use a fresh output directory. The build regenerates all 18 circuit tensors from the archived compiler sources and checks the placement, coefficient stream and packed action tables. It pads the language FFNs from 9,216 to 36,352 rows while verifying all 426 original packed tensor extents. The conditioner changes 1,024 F32 coefficients in the MTP gate tensor. The final GGUF must match the pinned release size and SHA-256.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Original Bonsai 2 4B PQ2_0 | 1,146,465,120 | `d406b1e912c0d2a4c124ad10bf38a76ae662fdc4d24b176451403dc7b7f30409` |
| Combined, before conditioning | 4,031,384,384 | `31c46924342d983a60ca6654507720e29a41aa4017b607b98820b06c53bf3a29` |
| Released, after conditioning | 4,031,384,384 | `ec6d177b3af28436f8aef2c82166ccd01ee087c5aed889b13a269fa414b6c973` |

## Compare directly with the release

Download the pinned release and pass it as an extra verification input. The builder reads it only after generating the candidate and checking the result hash.

```powershell
curl.exe --fail --location --output released.gguf "https://huggingface.co/millrace-ai/qwenpu-bonsai-4b-stock-prism/resolve/c0b9de4dd5ad7057ad677fb01f7e99e4ce350e00/Bonsai-2-4B-PQ2_0-QwenPU-Stock-Prism.gguf?download=true"
.\.venv\Scripts\python.exe -B .\rebuild.py --base .\Ternary-Bonsai-2-4B-PQ2_0.gguf --output-dir .\compared --released .\released.gguf
```

The output directory contains the final GGUF and `rebuild-verification.json`. The latter records the model identities, the compiler and conditioning checks, and the number of bytes compared when using `--released`. The published record adds a `verification` section with the tested Git commit, manifest hash, and independent output hash.

## Source provenance

`frozen/` contains the compiler, exporter, conditioning code and compact input receipts with their archived relative paths. `rebuild.py` checks their digests against `source-checksums.json` and builds in its own temporary copy. The packaged signed-ROM diagnostic replaces original machine paths with an archive placeholder. The placement and exporter pins account for that redaction. The numerical inputs and target model identity remain specified by the coefficient and GGUF hashes.

`vendor/gguf/` contains the Python GGUF tooling used for packing, from the [pinned Prism llama.cpp source](https://github.com/PrismML-Eng/llama.cpp/tree/6447906bf96c1a023980d3453fb59daa6af3bb23/gguf-py). Its MIT notice is in [vendor/LICENSE](vendor/LICENSE). For guest and circuit provenance, see the [project notice](../../NOTICE) and [pinned QwenPU reference](https://github.com/divagr18/QwenPU/tree/6e2767aaa59b69ea03300c0a5199d7b36debbe50).
