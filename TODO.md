# Release gates

## Source and evidence

- [x] Set the original overlay code and docs under MIT. Preserve upstream component terms and attribution in `NOTICE` and `licenses/`.
- [x] Run a clean Windows forward build, the separate vector fixture, and the short story suite. Archive the checks and [runtime identity](results/forward/release-v0.1/runtime-identity.json).
- [x] Review the distribution for private paths, personal information, raw logs, and credentials. Keep reports tied to the artifact and run that produced them.

The experimental release covers three story tokens. The runtime is built from the pinned source and dependencies in the [Windows guide](docs/REPRODUCE.md).

## Short-story GGUF

- [x] Retain a GGUF containing the one- and three-token snapshots, the export manifest, and the matching patched runtime's identity.
- [x] Validate the exact GGUF through the first matrix multiply and the complete one- and three-token guests. Label the [vector mutation proof](results/forward/release-v0.1/vector-validation.json) with its own fixture hash.
- [x] Populate the model card with the final inventory, hashes, matching runtime and validation links. Include the applicable Apache, MIT, and other notices in the distribution.
- [x] Verify the downloaded GGUF hash and run the [first-VMAT smoke test](results/forward/release-v0.1/hf-download-verification.json) on the downloaded bytes from a fresh directory.

## Separate research milestones

- Extend the guest selector, probe limits, and trace comparison for a 60-token run. Record the actual token count and stop condition.
- Compare the packed CUDA lowering with upstream standard-layer operations on representative intermediate states as well as end-to-end traces.
- Check autonomous dispatch on the CUDA path and expand the comparison against stock Bonsai beyond the current logit probes.
