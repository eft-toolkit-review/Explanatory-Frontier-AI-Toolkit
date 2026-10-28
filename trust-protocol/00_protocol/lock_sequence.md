# Lock sequence (appendix A7)

1. **Before any model fitting**: evaluation share, folds, evaluation mode, search settings,
   seeds (registry rows with `fixed_before` = "Any model fitting" or "Training"). Lock with
   `python tools/lock.py create --stage training --files 00_protocol/registry.yaml`.
2. **Profile, leakage test, structure discovery, stability, fingerprints**: lock each stage's
   frozen objects (planned; the hashing mechanism is `tools/lock.py`).
3. **Single registration** after the profile is frozen and before any stimulus is shown: the
   registry, the frozen profile with SHA-256 hashes, the stimulus manifest, the primary model,
   and the exploratory plan. Archive it with a persistent identifier; write the archive hash into
   the AsPredicted entry. The lock record created here is the one `run_reopen.py` verifies.
4. **Confirmatory analysis** runs on the collected data and writes its estimates and a hash.
5. **Only then** may `07_reopen/run_reopen.py` run; CI checks that it refuses without a record.

`tools/lock.py verify` re-computes every hash and reports any file changed since locking.
Any change after a lock goes into `deviations.md`, dated, with the reason.
