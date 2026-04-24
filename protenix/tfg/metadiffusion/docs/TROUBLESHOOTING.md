# MetaDiffusion — Troubleshooting

Symptom → diagnosis → fix. Ordered by how often each comes up in
real CASP runs.

Cross-reference: commit hashes in brackets point to the fix on
`feat/metadiffusion` / `feat/metadiff-boltz-parity`.

---

## 1. "My samples don't move / fold is unchanged"

### Symptom
- Metadiff config is set, guidance is enabled, inference runs fine.
- `TFG last-step energy <TermName>: [0.0, 0.0, ...]` in the log.
- Final Rg / dist / whatever is indistinguishable from base samples.

### Common causes
1. **Warmup/cutoff window excludes the current `t`.** Protenix's
   `t` goes `1.0 → 0.0` across diffusion. Default `warmup: 0.0`,
   `cutoff: 1.0` keeps the term always on. If you set
   `warmup: 0.1, cutoff: 0.8`, the term fires only for
   `0.1 ≤ t ≤ 0.8` — the last-step energy at `t = 0` is
   correctly zero but guidance was active in the middle.
   **Confirm by setting `METADIFFUSION_DEBUG=1` and re-running:
   every `_eval` call logs its progress + activity**.
2. **CV config key mismatch.** Every CV needs its atom/region
   wired in. `distance`, `angle`, `dihedral` need
   `atom1`/`atom2`/... ; `rg`, `sasa`, `coordination` can use
   `groups` or `region1..4`. If you write the wrong field, the
   CV silently receives no mask and returns zeros.
3. **Reference-based CV with strength that's too low.** `rmsd`
   / `drmsd` / `d_tm` at strength 5 are drowned by the sampler's
   intrinsic variance (~13 Å on 210-res targets). Use strength
   ≥ 30 for rmsd/drmsd and ≥ 20 for d_tm.
4. **Zero gradient under `@torch.no_grad`** [eaed348]. If you
   pulled the code from before this fix, every CV returned
   zero grad because `requires_grad_(True)` inside the outer
   `@torch.no_grad` decorator produced graph-less leaves. Make
   sure you're on at least commit `eaed348`.

### Quick diagnostic

```bash
METADIFFUSION_DEBUG=1 python protenix/inference.py ... \
    --sample_diffusion.N_step 10   # short run for fast check
```

Grep the log:

```
grep "metadiff/<YourTerm>" inference.log | head
```

Look for: CV value changing, `|grad|` non-zero, `E_mean`
non-zero in active window.

---

## 2. "Samples explode — Rg blows up, |Cα-Cα| > 5 Å"

### Symptom
- `|Cα-Cα|` > 4.5 (stretched) or > 5 Å (broken peptide).
- Rg >> expected (2× or more).
- Sample looks like an extended coil rather than a fold.

### Causes
1. **Strength is too high for that CV's scale.** Some CVs return
   values in 10³–10⁴ (sasa, asphericity), so a nominal
   strength of 10 is actually a massive force per atom. Use
   `log_gradient: true` for those.
2. **`total_bias_clip` is too loose.** Default 5.0 is good.
   Drop to 3.0 for tight fold preservation.
3. **Using Boltz-paper strengths directly.** Our port is ~4×
   more sensitive than Boltz (different score prior); Boltz
   strength 2.0 ≈ our strength 0.5. See
   [RECIPES.md §Strength cheat-sheet](RECIPES.md#strength-cheat-sheet).

### Fix

```yaml
# bad (at our scale)
- opt: {collective_variable: pair_rmsd, target: max, strength: 2.0}

# good
- opt: {collective_variable: pair_rmsd, target: max, strength: 0.5}
```

### Special: asphericity, coordination, SASA

These have large raw gradients. Always:

```yaml
- opt: {collective_variable: asphericity, target: max,
        strength: 0.5, log_gradient: true}
```

Without `log_gradient` the gradient scales with the CV value
squared and a `strength` of 10 can put `|Cα-Cα|` above 26 Å.

---

## 3. "CUDA OOM — SASA crashes the process"

### Symptom
- `torch.OutOfMemoryError: CUDA out of memory` during SASA `_eval`.
- Typically happens when Protenix-v2 base + SASA autograd graph
  together exceed the GPU's available memory.

### Fix in order of preference

1. **Opt-in to `torch.utils.checkpoint`**:
   ```yaml
   - steer: {
       collective_variable: sasa, target: 18000,
       strength: 1.0, log_gradient: true,
       use_checkpoint: true
     }
   ```
   ~2× compute for much lower activation memory.
2. **Lower `chunk_size`**: default 32, try 16 or 8:
   ```yaml
   - steer: {collective_variable: sasa, target: 18000,
             strength: 1.0, chunk_size: 8, log_gradient: true}
   ```
3. **Reduce `n_quad`**: default 48, try 24 for rough SASA.
4. **Move to a node with more GPU memory** (pop6 has 48 GB,
   pop4 24 GB).
5. **Eliminate GPU contention**: if another process grabbed
   half the GPU (common when node is shared), SLURM might
   still assign you the same device. Pick a free GPU in your
   sbatch preamble:
   ```bash
   FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free \
       --format=csv,noheader,nounits | sort -t, -k2 -n -r \
       | head -1 | cut -d, -f1)
   export CUDA_VISIBLE_DEVICES=$FREE_GPU
   ```

---

## 4. "`atom1: A:10:CA` silently does nothing"

### Symptom
- `distance` / `angle` / `dihedral` with an `atom1..4` spec
  shows zero gradient but no error.
- Expected movement direction doesn't happen.

### Cause
- If you're on a commit before [0158c58], `atom1..4` were in the
  ignored-fields list and silently dropped. The CV received no
  indices / masks and returned zeros.

### Fix
Pull past `0158c58`. Verify:

```python
from protenix.tfg.metadiffusion.schema import build_metadiffusion_features
feats = build_metadiffusion_features(
    [{"steer": {"collective_variable": "distance",
                "atom1": "A:10:CA", "atom2": "A:100:CA",
                "target": 15, "strength": 5}}],
    my_atom_array,
)
print(feats["metadiffusion_cv_kwargs__SteeringPotential__0"])
# should contain mask1, mask2 with exactly one atom each.
```

---

## 5. "Region spec `A:5-10,20` crashes"

### Symptom
`ValueError: invalid literal for int() with base 10: '10,20'` while
loading the config.

### Cause
Before [279bc37], `_parse_region_spec` accepted only a single
residue or one `lo-hi`. Comma-separated segments crashed.

### Fix
Pull past `279bc37`. Now `A:5-10,20,30-35` is valid and
`|` 's the per-segment masks together.

---

## 6. "Hills don't persist across sbatch runs"

### Symptom
- Second run's `load_hills` logs 0 hills loaded, even though the
  first run completed.
- On-disk JSON shows more or fewer hills than expected.

### Causes + fixes

- **Missing `hills_path`**: without this, hills are in-memory only
  and die with the process. Add `hills_path: /path/to/hills.json`
  to the explore term.
- **File path not shared between processes**: make sure all your
  sbatch jobs write to the same absolute path. Don't use `$TMPDIR`.
- **Lock file not cleaned up**: the sibling `<path>.lock` file is
  created on first save and can be removed after all jobs are done
  if it somehow persisted.
- **`reset_hills()` called between runs but same instance reused**:
  on commit [279bc37] `reset_hills()` also clears the disk-path
  pointer so the next `_eval` re-reads `hills_path` from `params`.
- **`max_hills` cap exceeded on disk**: on commit [279bc37],
  `save_hills` re-applies the FIFO cap, so the JSON never grows
  past the configured `max_hills`.

---

## 7. "Diversity CV inflates `pw_RMSD` but structures are unfolded"

### Symptom
`pair_rmsd` mean increased dramatically but when you look at the
samples they're all extended coils, not different folds.

### Cause
Coil-stretched samples have inflated internal distances so
RMSD between any two of them is high. Pure RMSD measures
don't distinguish "different fold" from "different amount
of unfolding".

### Better metric
Use pairwise **TM-score** instead: bounded in [0, 1], accounts
for scale, so unfolding doesn't artificially inflate diversity.

```python
# post-hoc analysis
from protenix.tfg.metadiffusion.cv import pair_tm_cv
v, g = pair_tm_cv(coords, feats)    # shape [B]; value is 1 - mean_TM
diversity = v.mean().item()          # higher = more diverse
```

Or add `pair_tm` as the diversity CV from the start instead of
`pair_rmsd`.

### Parallel remedy
Cap `|Cα-Cα|` by dropping `total_bias_clip` to 3.0 and lowering
`strength` to 0.3 — fold is more likely to hold.

---

## 8. "Copilot review keeps failing / removed from reviewers"

### Symptom
GitHub Copilot reviewer bot is requested, starts ("copilot_work_started"),
then `review_request_removed` 12 s later. No comments posted.

### Cause
Known GitHub Copilot reviewer bot issue with some repos /
PR sizes. Not a code problem.

### Workaround
1. **Split the PR.** PR #3 (3 files + README, squash) is small
   enough that Copilot did post a `COMMENTED` review — though
   the body was "Copilot encountered an error and was unable to
   review". Try again 30-60 min later.
2. **Use Codex instead.** `codex review` covers the same ground
   more reliably. We passed 4 rounds of Codex review for the
   metadiffusion PR series.

---

## 9. "Gradient explodes / NaN in inference"

### Symptom
`Traceback: ... tensor contains NaN / inf`.

### Causes
1. **SVD on a degenerate pair.** Two identical samples in the
   batch make Kabsch SVD produce non-unique rotations. Our
   current implementations (`pair_rmsd_cv`, `pair_tm_cv`) detach
   the rotation so backward is well-defined, but if you use an
   older snapshot, pull to at least [c045965].
2. **`total_bias_clip` is None or Inf.** Set it:
   ```yaml
   - total_bias_clip: 5.0
   ```
3. **Raw autograd with checkpoint + no_grad outer scope.** If you
   see a silent crash (no traceback) plus OOM in SASA, you hit the
   checkpoint + `enable_grad` re-entry issue. Use
   `use_checkpoint: false` (the current default) or update past
   [0158c58].

---

## 10. "Config silently ignored"

### Symptom
- You set a field (e.g. `bias_tempering`, `gaussian_noise_scale`)
  and nothing happens.
- `_ADVANCED_IGNORED_FIELDS` warning in the log at startup:
  `[metadiffusion] unsupported field 'foo' in term 'bar'`.

### Cause
We intentionally warn-once on Boltz fields we haven't ported yet.
Current ignored list:
```
bias_tempering, target_from_saxs, auto_rg_scale,
gaussian_noise_scale, rmsd_groups, selection, sasa_method
```

If your field is in this set, it genuinely isn't implemented yet.
If you need it, open an issue — typically these are easy ports.

---

## Debugging checklist

When a CV isn't doing what you expect, run through:

- [ ] `METADIFFUSION_DEBUG=1` → is `_eval` fired, is CV value sane,
      is `|grad|` non-zero?
- [ ] Does `warmup/cutoff` window include the middle of diffusion?
- [ ] Is `atom1..4` / `groups` / `region1..4` resolving to atoms?
      (Add a one-off `print(feats["metadiffusion_cv_kwargs__…"])`.)
- [ ] Is strength in the right ballpark for the CV's value scale?
      (See [RECIPES.md](RECIPES.md#strength-cheat-sheet).)
- [ ] For reference-based CVs: does the reference file exist and
      match the model's atom count after `groups` filtering?
- [ ] For SASA and other memory-heavy CVs: is GPU memory available
      (nvidia-smi)?
- [ ] Are you on the latest commit (≥ `279bc37`)?

If the above doesn't fix it, the fallback is a local test:

```python
import torch
from protenix.tfg.metadiffusion.cv import my_cv
coords = torch.randn(5, N, 3)
feats = {...}  # minimum keys for the CV
value, grad = my_cv(coords, feats, **kwargs)
print(f"{value=}  |grad|_max={grad.abs().max()}")
```

If `value` is sane and `|grad|` is non-zero locally, the bug is
in the engine path; if even local is zero, the bug is in the CV or
its kwargs. From there, dig in.
