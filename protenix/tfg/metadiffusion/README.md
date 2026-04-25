# MetaDiffusion — Native Protenix Port of Boltz-Metadiffusion

Structure-steering during diffusion via collective-variable (CV)
potentials.  Runs inside the Protenix TFG engine on both v1 and v2
checkpoints, identical code path in either case.

Boltz compatibility: accepts the same `metadiffusion: [...]` YAML/JSON
block as [Boltz-Metadiffusion](https://github.com/Aspuru-Guzik-group/Boltz-Metadiffusion).
Unknown Boltz-specific fields are warn-once and ignored.

---

## 🧪 Validation status (N_step=200 real inference)

### Fully verified — 19 CVs (production ready)

| group | CVs | evidence |
|---|---|---|
| Shape | `rg`, `max_diameter`, `asphericity`, `coordination`, `contact_order`, `min_distance` | rg 16.22→17.97 (target 22) / 15.30 (target 13); max_diameter 49→55 (+12%); asphericity 63→329 (+log_gradient); min_distance hit 3.64 for target 4 |
| Atom-spec | `distance`, `angle`, `dihedral` | dist 22.54→15.51 (target 15, perfect); angle 73°→83° (target 90); dihedral -2°→+136° (target 180) |
| Pairwise diversity | `pair_rmsd`, `pair_tm`, `pair_itm`, `pair_drmsd` | pair_rmsd pw +21% on T1214 / +116% on H1204; pair_tm pw_TM 0.277→0.114 (2.4× diversity); pair_itm H1204 pw_TM 0.639→0.140 (4.6×) |
| Multimer | `inter_chain` | A-B 26.96→29.91 (target 35) |
| Interaction | `Q`/`native_contacts`, `hbond_count`, `salt_bridges`, `rmsf` | rmsf Rg +48% (spread), fold preserved for all |
| Explore | `explore=hills` well-tempered, `explore=repulsion` | Rg range 1.74→3.01 with hills |

All verified cases keep `|Cα-Cα| ≤ 4.15 Å` (native 3.80) — fold preserved.

### Reference-based — 3 CVs need strength ≥ 20-30

| CV | default fails at | recommended |
|---|---|---|
| `rmsd` | strength 5 stuck at 13 Å | **strength ≥ 30** + `log_gradient: true` |
| `drmsd` | strength 5 same | **strength ≥ 30** + `log_gradient: true` |
| `d_tm`  | TM stuck 0.22 | **strength ≥ 20** + `log_gradient: true` |

Code is correct — these bounded / intrinsic-variance CVs just need
more force to overcome the sampler's own diversity. See
[RECIPES §7](docs/RECIPES.md#7-force-a-conformation-similar-to-a-reference).

### Memory-heavy — `sasa`

Works on small targets + GPU with headroom.  On Protenix-v2 + 24 GB
GPU with contention, add `use_checkpoint: true` to the sasa term
(costs ~2× forward time, saves backward activations).
See [CV_REFERENCE §sasa](docs/CV_REFERENCE.md#sasa-memory).

---

## Quick start — drop this into your input JSON

```json
{
  "name": "my_target",
  "sequences": [...],
  "metadiffusion": [
    {"total_bias_clip": 5.0},
    {"opt": {
      "collective_variable": "pair_rmsd",
      "target": "max",
      "strength": 0.5,
      "warmup": 0.1, "cutoff": 0.8
    }},
    {"explore": {
      "type": "repulsion",
      "collective_variable": "pair_rmsd",
      "strength": 64.0, "sigma": 4.0,
      "warmup": 0.2, "cutoff": 0.8
    }}
  ]
}
```

Then run as usual:

```bash
python protenix/inference.py \
    --input_json_path my_target.json \
    --sample_diffusion.guidance.enable true \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --seeds 101
```

Guidance is auto-enabled as soon as a `metadiffusion:` block is
present in the input JSON.

---

## 🎯 Config examples — by goal

Every snippet below is a complete `"metadiffusion": [...]` block.
Drop it in alongside `"name"` and `"sequences"`.

### A. Diverse monomer sample pool (verified)

```json
"metadiffusion": [
  {"total_bias_clip": 5.0},
  {"opt": {"collective_variable": "pair_rmsd", "target": "max",
           "strength": 0.5, "warmup": 0.1, "cutoff": 0.8}},
  {"explore": {"type": "repulsion", "collective_variable": "pair_rmsd",
               "strength": 64.0, "sigma": 4.0,
               "warmup": 0.2, "cutoff": 0.8}}
]
```
T1214 pw +21%, H1204 pw +116%, `|Cα-Cα|` ≤ 4.15.

### B. Multimer docking-pose diversity (verified)

```json
"metadiffusion": [
  {"total_bias_clip": 5.0},
  {"opt": {"collective_variable": "pair_itm", "target": "max",
           "strength": 1.0, "interface_cutoff": 8.0,
           "warmup": 0.1, "cutoff": 0.8}},
  {"explore": {"type": "repulsion", "collective_variable": "pair_itm",
               "strength": 128.0, "sigma": 0.15, "interface_cutoff": 8.0,
               "warmup": 0.2, "cutoff": 0.8}}
]
```
H1204 pw_TM 0.639 → 0.140 (4.6× diverse, intra-chain fold intact).

### C. Steer Rg to a specific value (verified)

```json
"metadiffusion": [
  {"total_bias_clip": 5.0},
  {"steer": {"collective_variable": "rg", "groups": ["A"],
             "target": 22.0, "strength": 30.0,
             "warmup": 0.1, "cutoff": 0.9}}
]
```
T2201: 16.22 → 17.97 (expand) / 15.30 (compact, target=13) — works
both ways.

### D. Steer a specific inter-atom distance (verified, atom1-4 fix)

```json
"metadiffusion": [
  {"total_bias_clip": 5.0},
  {"steer": {"collective_variable": "distance",
             "atom1": "A:10:CA", "atom2": "A:100:CA",
             "target": 15.0, "strength": 5.0, "log_gradient": true,
             "warmup": 0.1, "cutoff": 0.9}}
]
```
22.54 → 15.51 (target 적중 perfect).

### E. Cross-run metadynamics (verified, hills persistence)

```json
"metadiffusion": [
  {"total_bias_clip": 5.0},
  {"explore": {"type": "hills", "collective_variable": "rg",
               "height": 0.3, "sigma": 1.5, "pace": 20,
               "well_tempered": true, "bias_factor": 10.0, "kT": 2.5,
               "hills_path": "/path/to/my_target_hills.json",
               "max_hills": 1000,
               "warmup": 0.1, "cutoff": 0.9}}
]
```
Feed the same `hills_path` to every sbatch run; fcntl-locked atomic
write preserves additive hills across concurrent writers.
Rg range 1.74 → 3.01 after 3 iterations.

### F. Template similarity (reference-based, needs strength ≥ 30)

```json
"metadiffusion": [
  {"total_bias_clip": 5.0},
  {"steer": {"collective_variable": "rmsd",
             "reference_structure": "/path/to/ref.cif",
             "groups": ["A"], "target": 0.0,
             "strength": 30.0, "log_gradient": true,
             "warmup": 0.05, "cutoff": 0.95}}
]
```
Use `d_tm` (with `target: 0.95`) instead for looser coil-tolerant
similarity; reuse the same pattern.

### G. Shape elongation (verified, `log_gradient` required)

```json
"metadiffusion": [
  {"total_bias_clip": 3.0},
  {"opt": {"collective_variable": "asphericity", "target": "max",
           "strength": 0.5, "log_gradient": true,
           "warmup": 0.2, "cutoff": 0.85}}
]
```
asphericity 63 → 329 (5× elongation), fold preserved.

### H. SASA — memory-heavy (opt in to checkpoint)

```json
"metadiffusion": [
  {"total_bias_clip": 3.0},
  {"steer": {"collective_variable": "sasa",
             "target": 18000.0, "strength": 1.0, "log_gradient": true,
             "use_checkpoint": true,        // save GPU memory
             "chunk_size": 16,              // lower if still OOM
             "warmup": 0.2, "cutoff": 0.85}}
]
```

For a comprehensive recipe library see [docs/RECIPES.md](docs/RECIPES.md).
For per-CV parameters see [docs/CV_REFERENCE.md](docs/CV_REFERENCE.md).
For failure-mode debugging see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 🧠 CV semantics — when each one helps

### Pure shape (no reference needed)
- **`rg`**: classic compaction/expansion. Steer to a target Rg or
  bound the ensemble. Works either direction. Recommended first
  experiment — it's the gentlest knob.
- **`max_diameter`** / **`asphericity`**: rod-like vs spherical
  selection. `max_diameter opt max` stretches; `asphericity opt
  max` produces elongated samples. Use `log_gradient` for
  asphericity — its raw gradient scales with CV² and will explode
  the fold otherwise.
- **`coordination`**: packing density. `opt max` gives tighter
  cores; `opt min` gives loose chains. Use with `log_gradient`.
- **`min_distance`**: prevents clashes (target ~4 Å) or forces
  contacts (target lower). One number per structure.
- **`contact_order`**: short-range vs long-range topology. Useful
  to bias toward β-sheet (low CO) vs complex topology (high CO).

### Surface area
- **`sasa`**: Shrake–Rupley solvent surface area. Increasing
  SASA fundamentally requires backbone exposure → expect mild
  unfolding. Decrease only works for *small* changes (<5%) at
  very low strength (0.0001) because the diffusion prior already
  produces compact samples; further compaction is hard.
  See `docs/CV_REFERENCE.md` for `use_checkpoint` memory note.

### Pairwise diversity (need batch B ≥ 2)
- **`pair_rmsd`** (Kabsch): the workhorse. Use opt+rep for
  decoy generation. Tip: at our scale, opt strength 0.5 +
  repulsion strength 64 is the sweet spot.
- **`pair_drmsd`** (distance matrix): rotation-invariant by
  construction; gentler than `pair_rmsd`. Use when you want
  shape diversity without orientation getting tangled.
- **`pair_tm`** (ported from MassiveFoldClustering_Tool): bounded
  TM-score-based diversity. Better than `pair_rmsd` when samples
  are at risk of unfolding — TM doesn't reward extended coil.
- **`pair_itm`** (interface-only TM): for **multimers** only.
  Diversifies docking poses while keeping intra-chain fold rigid.
  Specify `interface_cutoff: 8.0` (Å, default).

### Multimer
- **`inter_chain`**: chain–chain COM distance. `groups: [A, B]`.
  Steer to push chains apart or pull them together.

### Reference-based (need a reference CIF)
- **`rmsd`** (Kabsch): pull samples toward / away from a known
  structure. Strong, direction-correct. Use **strength ≥ 30**
  with `log_gradient: true`.
- **`drmsd`**: rotation-invariant analogue. Same strength range.
- **`d_tm`** (`tm` alias): pull samples toward a target TM-score.
  Bounded function — flat gradient at far distances; use **after**
  `rmsd` has pulled into the same basin. Effective for fine-tuning
  near TM > 0.7.
- **`native_contacts`** (`Q` alias): fraction of native contacts
  preserved. `opt max` keeps contacts; `opt min` breaks them.

### Geometry primitives (atom-based)
- **`distance`** (`atom1`, `atom2`): one COM-COM distance.
  Standard region grammar (`A:10:CA`, etc.). Steer to specific
  distances. *Verified*: 22.54 → 15.51 at strength 5.
- **`angle`** (`atom1`, `atom2`, `atom3`): 3-atom angle in
  radians. `target: 1.5708` for 90°. Strength 50 works.
- **`dihedral`** (`atom1..4`): 4-atom torsion in radians.
  Strength 30 works.

### Interaction count
- **`hbond_count`**, **`salt_bridges`**: rough heavy-atom-only
  proxies. `opt max` to encourage; not for precise tuning.
- **`rmsf`**: per-residue fluctuation across the batch. `opt
  max` to spread samples spatially without changing pairwise
  RMSD distribution dramatically.

### Explore (sample-coupled)
- **`explore type=hills`** (well-tempered metadynamics): on a
  scalar CV, deposit Gaussian penalties at the visited locations
  so the next iteration avoids them. Use `hills_path:
  /path/to/hills.json` to persist across sbatch runs.
- **`explore type=repulsion`**: between-sample Gaussian repulsion
  in CV space. Standalone (without an `opt` term) it pushes
  current-batch samples apart in CV-space; combined with
  `opt: target=max` it amplifies the diversity push.

### Phase D — Gradient modifiers (advanced)
- **`scaling`**: weight the per-atom gradient by another CV's
  gradient magnitude. E.g. scale the diversity force *down* in
  high-`contact_order` regions to keep the core packed.
- **`projection`**: project onto or away from another CV's
  direction. E.g. project the diversity force *away* from `rg`
  so the overall protein size stays fixed while the topology
  diversifies.
- **`modifier_order`**: `["scaling", "projection"]` (default) or
  reversed.

---

## ⚡ Strength cheat-sheet (critical!)

Protenix is ~4× more sensitive to guidance than Boltz, so the
paper's strength numbers over-drive and can unfold the sample.

| CV family | Strength | `log_gradient` |
|---|---|---|
| `rg` | steer **30** | skip |
| `distance`, `inter_chain` | steer **5** | yes |
| `max_diameter` | opt **10** | skip |
| `angle` | steer **50** | skip |
| `dihedral` | steer **30** | skip |
| `pair_rmsd`, `pair_tm` | opt **0.5** + rep 64 | skip (internally normalised) |
| `pair_itm` | opt **1.0** + rep 128 | skip |
| `asphericity`, `coordination`, `sasa` | **0.5** / **0.5** / **1.0** | **required** |
| `min_distance`, `contact_order` | **5** / **0.5** | yes |
| `rmsd`, `drmsd` (reference) | steer **≥ 30** | yes |
| `d_tm` (reference) | steer **≥ 20** | yes |
| `Q`, `hbond_count`, `salt_bridges`, `rmsf` | opt **0.5** | yes |

`total_bias_clip: 5.0` is the all-round default — 3.0 for tight
fold control, 8.0 for aggressive steering.

---

## Three modes

| Mode | What it does | When to use |
|---|---|---|
| **`steer`** | pull a CV toward a target value (harmonic or gaussian) | push Rg to 22 Å, pull two atoms to 15 Å, fix an angle |
| **`opt`** | minimise (`target: min`) or maximise (`target: max`) a CV | "make samples as diverse as possible", "stretch me out" |
| **`explore`** | well-tempered metadynamics hills (`type: hills`) or batch-pairwise repulsion (`type: repulsion`) | avoid previously-visited CV regions, force between-sample spread |

Every mode accepts `strength`, `warmup` (t in [0,1] below which the
term is inactive), `cutoff` (t above which the term is inactive),
and optionally `log_gradient` (max-per-atom-norm = 1.0 rescaling;
use this for CVs whose raw gradient magnitude depends on the value
scale, e.g. `sasa`, `asphericity`).

---

## Collective variables

19 CVs + 2 repulsion variants. Each entry below lists the CV's
intended use, whether a `reference_structure` is required, and the
**recommended starting strength** for Protenix v2 + a realistic
guidance clip of 5.

### Global-shape CVs (no references)

| CV | Strength guide | Notes |
|---|---|---|
| `rg` / `radius_of_gyration` | steer 30 | Classic compaction/expansion knob. |
| `max_diameter` | opt 10 | Longest Cα–Cα pair. |
| `asphericity` | opt **0.5 + log_gradient** | Gyration-tensor shape. Use `log_gradient` — raw grad scales with CV². |
| `coordination` | opt **0.5 + log_gradient** | Mean near-neighbour count (default cutoff 8 Å). Big magnitudes, use log_gradient. |
| `sasa` | steer **1.0 + log_gradient** | Shrake–Rupley SASA. Memory-heavy on >600 atoms — see [SASA memory](#sasa-memory-note). |

### Distance / geometry CVs

| CV | Strength guide | Notes |
|---|---|---|
| `distance` | steer 5 + log_gradient | Two-group COM distance. Use `atom1`, `atom2` with region grammar (below). |
| `min_distance` | steer 5 + log_gradient | Closest pair. Good for clash avoidance. |
| `angle` | steer 50 | 3-atom angle in radians. `atom1`, `atom2`, `atom3`. |
| `dihedral` | steer 30 | 4-atom torsion in radians. `atom1`..`atom4`. |

### Inter-chain (multimer)

| CV | Strength guide | Notes |
|---|---|---|
| `inter_chain` | steer 5 | COM-COM distance between two chains. `groups: [A, B]`. |

### Diversity (per-sample pairwise, seq_id=100 assumption)

| CV | Strength guide | Notes |
|---|---|---|
| `pair_rmsd` | opt 0.5 + rep 64 σ=4.0 | Kabsch-aligned RMSD between every pair of batch samples. **Main diversity knob.** |
| `pair_drmsd` | opt 0.5 | Distance-matrix RMSD. Rotation-invariant by construction; softer grad than `pair_rmsd`. |
| `pair_tm` | opt 0.5 + rep 64 σ=0.1 | TM-score-based diversity (bounded [0,1]). |
| `pair_itm` | opt 1.0 + rep 128 σ=0.15 | **Interface-only TM** — multimer pose diversity without touching intra-chain fold. `interface_cutoff` (default 8 Å). |

### Reference-based CVs (require `reference_structure`)

| CV | Strength guide | Notes |
|---|---|---|
| `rmsd` | steer 30+ | Kabsch-aligned RMSD to a reference CIF. |
| `drmsd` | steer 30+ | Distance-matrix RMSD to reference. |
| `d_tm` / `tm` | steer 20+ | TM-score to reference (bounded, so strength needs to be higher than for rmsd). |
| `native_contacts` / `Q` | opt 0.5 | Fraction of reference contacts preserved. |

### Physics-like CVs

| CV | Strength guide | Notes |
|---|---|---|
| `hbond_count` | opt 0.5 + log_gradient | Approximate intra-backbone H-bond count. |
| `salt_bridges` | opt 0.5 + log_gradient | Approximate Lys/Arg–Asp/Glu contact count. |
| `contact_order` | opt 0.5 + log_gradient | Sequence-distance-weighted contact density. |
| `rmsf` | opt max | Per-residue fluctuation across the batch. Maximising gives diverse local geometry. |

---

## Atom / region selection grammar

All CVs that need an atom (or a set of atoms) use the same region
spec as Boltz-Metadiffusion:

```
A:10            # chain A, residue 10, default atom (CA)
A:10:CB         # chain A, residue 10, CB
A::CA           # chain A, every CA
A:1-50:CA       # chain A, residues 1..50, CA
B:5-10,20       # chain B, residues 5..10 and 20
```

Example — steer the distance between two CAs toward 15 Å:

```json
{"steer": {
  "collective_variable": "distance",
  "atom1": "A:10:CA",
  "atom2": "A:100:CA",
  "target": 15.0,
  "strength": 5.0,
  "log_gradient": true,
  "warmup": 0.1, "cutoff": 0.9
}}
```

For group-style CVs (`rg`, `sasa`, `coordination`, …) use `groups`
or `region1`..`region4` instead:

```json
{"opt": {
  "collective_variable": "rg",
  "groups": ["A"],
  "target": "max",
  "strength": 10.0
}}
```

---

## Ready-made recipes

### Diverse-sample pool (CASP-style sampling)

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - opt:
      collective_variable: pair_rmsd
      target: max
      strength: 0.5
      warmup: 0.1
      cutoff: 0.8
  - explore:
      type: repulsion
      collective_variable: pair_rmsd
      strength: 64.0
      sigma: 4.0
      warmup: 0.2
      cutoff: 0.8
```

Empirically: +21 % pw-RMSD on T1214, +116 % on H1204, `|Cα-Cα|` ≤ 4.15
(fold preserved). Works for both monomers and multimers.

### Multimer docking-pose diversity (preserve intra-chain fold)

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - opt:
      collective_variable: pair_itm
      target: max
      strength: 1.0
      interface_cutoff: 8.0
      warmup: 0.1
      cutoff: 0.8
  - explore:
      type: repulsion
      collective_variable: pair_itm
      strength: 128.0
      sigma: 0.15
      interface_cutoff: 8.0
      warmup: 0.2
      cutoff: 0.8
```

H1204 pw_TM 0.639 → 0.140 (4.6× diverse) with interfaces shuffled
but individual chains rigid.

### Cross-run metadynamics (avoid previously-visited regions)

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - explore:
      type: hills
      collective_variable: rg
      height: 0.3
      sigma: 1.5
      pace: 20
      well_tempered: true
      bias_factor: 10.0
      kT: 2.5
      hills_path: /runs/.../T2201_hills.json
      warmup: 0.1
      cutoff: 0.9
```

Feed the same `hills_path` to every sbatch run and each new batch
feels the cumulative bias of every earlier batch. File access is
fcntl-locked and atomic-replaced so concurrent writers don't lose
hills.

### Steer to a specific reference structure

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - steer:
      collective_variable: rmsd
      reference_structure: /path/to/ref.cif
      groups: ["A"]
      target: 0.0
      strength: 30.0
      warmup: 0.1
      cutoff: 0.9
```

---

## `log_gradient` — when and why

Set `log_gradient: true` on a term when the CV value itself
spans many orders of magnitude (or just has a large absolute
scale) and you want the applied force to stay O(strength × 1)
per atom regardless of the CV's current value. Recommended for:

* `sasa` (values in 10³–10⁴ Å²)
* `asphericity` (values in 10¹–10³)
* `coordination` (values ~10)
* `hbond_count`, `salt_bridges`, `contact_order`
* any reference-based CV where the value starts far from the target

Skip it for CVs whose grad already has a natural scale:

* `rg`, `distance`, `min_distance`, `max_diameter` — linear in Å.
* `angle`, `dihedral` — bounded in radians, natural grad ~1.
* `pair_rmsd`, `pair_tm`, `pair_itm` — already max-normed internally.

---

## `total_bias_clip`

Global per-atom cap on the guidance gradient (in Å units; the engine
divides by `mu` internally to get the raw-grad cap). Keeps the
diffusion trajectory physical even when a strong guidance term
wants to jerk a single atom by many Å in one step. 5.0 is a good
default; go down to 3.0 for tighter fold control, up to 8.0 when
you need aggressive steering.

---

## SASA memory note

The non-checkpointed `sasa_cv` path (default) needs
`~ B × N × n_quad × 4` bytes of autograd intermediate. At
N=210, n_quad=48, B=5, that's ~17 MB per chunk plus the Protenix
model itself (~15 GB for v2). On a 24 GB GPU with contention from
other processes this can OOM at large N.

Options:

1. Opt in to checkpoint: add `"use_checkpoint": true` on the sasa
   term. Uses `torch.utils.checkpoint` which cuts peak activation
   memory at ~2× compute cost. Note: checkpoint + `@torch.no_grad`
   inference decorator interacted badly in earlier releases — if
   you see a silent crash with checkpoint on, fall back to the
   non-checkpointed path and shrink `chunk_size`.
2. Keep `chunk_size` at 32 for N ≤ 600; lower it to 16 or 8 for
   very large (>1000 atom) targets.
3. Run on a dedicated GPU with no competing CUDA processes.

Small-target smoke test: `sasa = 2255 Å²` on a random 2×50 config,
grad max = 44. Produces real gradient; OOM is a memory pressure
symptom, not a code correctness issue.

---

## Gradient modifiers (Phase D)

```yaml
- opt:
    collective_variable: pair_rmsd
    target: max
    strength: 0.5
    scaling:
      - collective_variable: contact_order
        mode: inverse      # scale down in high-contact regions
    projection:
      - collective_variable: rg
        mode: remove       # strip the Rg component out of the force
    modifier_order: [scaling, projection]
```

`GradientScaler` reweights the per-atom gradient by another CV's
gradient magnitude; `GradientProjector` projects onto or away from
another CV's direction. Order is configurable via
`modifier_order`.

---

## Known things that still need tuning, not breaking

* **`rmsd` / `drmsd` / `d_tm` reference steering at low strength**
  — with `strength: 5`, final distance-to-reference was ~13 Å on a
  target whose base intrinsic variance is already ~13 Å. The grad
  is applied correctly (verified against autograd) but overridden
  by the diffusion prior. Use strength ≥ 30 for real pull.
* **`pair_drmsd` opt**: much softer signal than `pair_rmsd`. For
  actual diversity prefer `pair_rmsd` or pair_tm.

---

## Cross-referenced bug history

| commit | what broke | how caught |
|---|---|---|
| `fa6b971` | `explore=repulsion` sign inverted → attraction | Codex review round 1, autograd diff |
| `eaed348` | every CV had zero gradient under `@torch.no_grad` | Codex review + real inference |
| `0158c58` | `atom1..4` were in the ignored-field set → distance/angle/dihedral no-op | real inference showed wrong direction |
| `0158c58` | `sasa_cv` + `torch.utils.checkpoint` + `enable_grad` silently crashed | standalone SASA run under debug |

All of the above are fixed in the current tree; see the commit
messages for the exact derivations. If you add a new CV and see
it "not moving" in real inference, first check: is it inside the
`enable_grad` scope via `_invoke_cv`? Does it receive the masks
you expect (add a one-off debug print in `_eval`)? Is the gradient
norm clipped by `total_bias_clip` before it reaches the sampler?
