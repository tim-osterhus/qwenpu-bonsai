# Bonsai with a QwenPU graph

The [forward experiment](FORWARD.md) integrates the circuit into Bonsai's CUDA decoder. This page documents the earlier shared-context experiment.

The experiment runs two QwenPU vector programs in the same native llama context as Bonsai. Each compiled QwenPU layer is a GGML graph node. The circuit uses the CPU backend, its own recurrent state, and private trace token IDs. Bonsai uses the original packed GGUF and the existing Prism CUDA DLLs. The test supplies a controlled call marker for each program.

One sparse pack supplies the graph weights and token features. Each graph step assembles an embedding from the predicted token's family and payload. A separate state snapshot supplies each program and its initial machine state. Loading that snapshot resets the circuit's recurrent memory and lag history in place. The verifier derives the expected traces independently with the reference interpreter.

After setting up the stock GGUF (`$model`) and Prism DLL directory (`$runtime`) as in [the reproduction guide](../../../docs/REPRODUCE.md), build the fixture from the Qwenputer root in PowerShell:

~~~powershell
.\experimental\bonsai-qwenpu\prepare.ps1 -InstalledRuntime $runtime
~~~

The build uses the existing Windows Visual Studio toolchain and exact Prism source revision of the installed runtime. It places the isolated checkout, DLL, probe, and pack under dist/local-bench/bonsai-in-graph. It does not start Bonsai or execute the fixture.

Run the shared-context checks with:

~~~powershell
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\validate.py --baseline $runtime --model $model
~~~

Validation runs the original Prism DLL and the modified DLL in separate processes. Each process keeps one Bonsai context across both programs. The verifier compares full logits before and after each circuit run, checks both traces against the reference, and compares saved Bonsai sequence state byte for byte across each circuit reset and run. The second program reads a RAM slot written by the first, so stale circuit memory would change its result from 6 to 9. Bonsai is then asked to report both results from the same conversation.

The runner keeps the existing Windows affinity guard and measures CPU, RAM, and VRAM. The report is saved as `dist/local-bench/bonsai-in-graph/two-program/validation.json`.

The existing one-token `stories_vx.elf` guest can also be checked in the same runtime:

~~~powershell
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\validate_story.py --model $model
~~~

The runner stops after the guest's first matrix-vector multiply and compares the free-running trace with the reference. If that passes and the measured rate projects under five minutes for the full trace, it runs the one-token guest through halt. The story report and binary traces go under `dist/local-bench/bonsai-in-graph/stories260k`.

To check autonomous dispatch:

~~~powershell
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\validate_autonomous.py --model $model
~~~

Bonsai sees two parameterless tool definitions and generates its next response greedily. The probe tokenizes user text separately from the chat template and dispatches only a complete call in the advertised format for one of the two functions. Each case starts with an empty text context. Invoked fixtures are checked against the reference, with byte-for-byte cache and text position checks before the result is passed back to Bonsai. The report goes under `dist/local-bench/bonsai-in-graph/autonomous`.

## Combined checkpoint

After the native build, run the packer with the prepared vector fixtures and one-token story snapshot available:

~~~powershell
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\pack_checkpoint.py --source $model
~~~

The packer copies Bonsai's existing tensor bytes and metadata directly into a version 3 GGUF. It appends four byte tensors for the exact QwenPU graph and the three program snapshots. It verifies the resulting file against the original checkpoint and all four input payloads before publishing it.

The modified Prism runtime recognizes this layout at model load. Creating a Bonsai context also loads the embedded graph. A circuit call selects an embedded program by ID. IDs 1 and 2 select the vector fixtures, and ID 3 selects the one-token story guest. The runtime reads each snapshot directly from the checkpoint.

~~~powershell
$checkpoint = '.\dist\local-bench\bonsai-in-graph\checkpoint\Bonsai-27B-Q1_0-QwenPU.gguf'
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\validate.py --baseline $runtime --model $model --checkpoint $checkpoint
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\validate_autonomous.py --checkpoint $checkpoint
.\.venv\Scripts\python.exe -B .\experimental\bonsai-qwenpu\validate_story.py --checkpoint $checkpoint
~~~

This layout requires the modified loader and circuit runtime. It stores Bonsai's packed language tensors and QwenPU's exact circuit weights in separate sections of the same file.

