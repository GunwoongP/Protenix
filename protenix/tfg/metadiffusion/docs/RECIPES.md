# MetaDiffusion — CASP Recipe Book

Copy-paste ready `metadiffusion:` blocks for common CASP and
CASP-adjacent sampling tasks. Each recipe lists the empirical
validation it was derived from — replace sequence IDs and
targets but keep the strength ratios.

Numbers in parentheses (e.g. "pw_TM 0.277→0.114") are T2201
(210-residue monomer) measurements unless noted.

See [CV_REFERENCE.md](CV_REFERENCE.md) for per-CV parameter
details and [README.md](../README.md) for the syntax overview.

---

## Recipes

1. [Diverse-sample pool for monomers](#1-diverse-sample-pool-for-monomers)
2. [Multimer docking-pose diversity (fold preserved)](#2-multimer-docking-pose-diversity-fold-preserved)
3. [Cross-run metadynamics (avoid previously-visited regions)](#3-cross-run-metadynamics-avoid-previously-visited-regions)
4. [Steer Rg to a specific value](#4-steer-rg-to-a-specific-value)
5. [Steer a specific inter-atom distance](#5-steer-a-specific-inter-atom-distance)
6. [Steer a backbone dihedral to 180°](#6-steer-a-backbone-dihedral-to-180)
7. [Force a conformation similar to a reference](#7-force-a-conformation-similar-to-a-reference)
8. [Maximise shape anisotropy (elongation)](#8-maximise-shape-anisotropy-elongation)
9. [Multi-CV composition (shape + diversity)](#9-multi-cv-composition-shape--diversity)
10. [Phase-D modifier example](#10-phase-d-modifier-example)

---

## 1. Diverse-sample pool for monomers

Use-case: a CASP target where you want 5 (or 25) samples with
maximum fold variety but all still physically reasonable.

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

**Empirical**:
- T1214 (677 res): pw_RMSD +21 %, `|Cα-Cα|` 3.94 (native-like)
- H1204 (3-chain 429 res): pw_RMSD +116 %, `|Cα-Cα|` 4.01
- Works for both monomers and multimers

**When to turn up**: if the default pool is still too similar,
raise `opt strength` to 1.0 and `explore strength` to 128 —
but watch `|Cα-Cα|` creep above 4.3.

---

## 2. Multimer docking-pose diversity (fold preserved)

Use-case: H-target with a stable interface, you want 5 samples
that dock differently without flipping the individual chains.

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

**Empirical (H1204)**: pw_TM_interface 0.639 → **0.140** (4.6×
diverse). `|Cα-Cα|` 4.31, intra-chain RMSD stays low.

**When `interface_cutoff` matters**: small interfaces (<30 Å²)
may have too few atoms within 8 Å; bump cutoff to 10–12.

---

## 3. Cross-run metadynamics (avoid previously-visited regions)

Use-case: iterative sampling — each sbatch run deposits hills
in a JSON file, the next run starts from those hills and actively
avoids them. File access is fcntl-locked and atomic so you can
run 4 sbatch arrays on the same `hills_path`.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - explore:
      type: hills
      collective_variable: rg
      height: 0.3
      sigma: 1.5
      pace: 20                    # deposit every 20 diffusion steps
      well_tempered: true
      bias_factor: 10.0
      kT: 2.5
      hills_path: /runs/…/my_target_hills.json
      max_hills: 1000             # FIFO cap
      warmup: 0.1
      cutoff: 0.9
```

**Empirical**: Rg range 1.74 → **3.01** after 3 iterations of
5 samples each on T2201.

**Note**: `max_hills: 1000` is enforced both in-memory *and*
when persisting to disk (Codex PR#3 R4), so the JSON never
grows unbounded.

---

## 4. Steer Rg to a specific value

Use-case: you know the target should have Rg ~18 Å and samples
are coming in at 16; pull toward 18.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - steer:
      collective_variable: rg
      groups: [A]
      target: 18.0
      strength: 30.0
      warmup: 0.1
      cutoff: 0.9
```

**Empirical (T2201)**:
- target 22 → Rg 16.22 → **17.97** (+1.75 Å, toward)
- target 13 → Rg 16.22 → **15.30** (-0.92 Å, toward)

**Mode = gaussian**: switch to `mode: gaussian` + `sigma: 2.0`
for a softer pull (reaches the target but doesn't punish
over-shoot).

---

## 5. Steer a specific inter-atom distance

Use-case: crystal contact, SS bond, anchor residue pair.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - steer:
      collective_variable: distance
      atom1: A:10:CA
      atom2: A:100:CA
      target: 15.0
      strength: 5.0
      log_gradient: true
      warmup: 0.1
      cutoff: 0.9
```

**Empirical**: base `dist(A:10,A:100) = 22.54 Å` → post = **15.51 Å**
(target hit exactly).

**Atom spec**: `CHAIN:RESID:ATOMNAME`, also `A:10` (CA default),
`A:5-10,20:CA` (residue ranges + singles).

**Log-gradient = true**: distance is ~10s of Å scale, so the
raw grad varies a lot; normalise.

---

## 6. Steer a backbone dihedral to 180°

Use-case: flip a loop or fix a torsion angle.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - steer:
      collective_variable: dihedral
      atom1: A:5:CA
      atom2: A:30:CA
      atom3: A:60:CA
      atom4: A:100:CA
      target: 3.1416
      strength: 30.0
      warmup: 0.1
      cutoff: 0.9
```

**Empirical**: base dihedral = -2 ° → post = **+136 °** (toward +180 °).

**Watch out**: dihedral is periodic; targets of 179 ° and -179 °
are both "almost +180 °" — pick whichever side the base sample
is already on.

---

## 7. Force a conformation similar to a reference

Use-case: you have a template (from AlphaFold, X-ray, experimental
cryo-EM) and want samples that stay close to it in RMSD or TM.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - steer:
      collective_variable: rmsd
      reference_structure: /path/to/ref.cif
      groups: [A]
      target: 0.0
      strength: 30.0                 # ≥30 needed (default 5 is too soft)
      log_gradient: true
      warmup: 0.05
      cutoff: 0.95
```

For TM-score based (tolerant of local coil differences), **use only
after rmsd has pulled samples into the same basin** — the TM curve
is flat for d ≫ d0, so the gradient is weak when samples are far
from the reference.

```yaml
  - steer:
      collective_variable: d_tm
      reference_structure: /path/to/ref.cif
      groups: [A]
      target: 0.95
      strength: 20.0
      log_gradient: true
      warmup: 0.05
      cutoff: 0.95
```

### Validated strength table (from actual measurements)

| base-to-ref distance | CV | strength | result |
|---|---|---|---|
| ~13 Å RMSD | `rmsd` + `log_gradient` | **30** | **12.94 → 3.23 Å** (strong pull) |
| ~8 Å dRMSD | `drmsd` + `log_gradient` | **30** | **7.57 → 2.51 Å** |
| TM ~0.30 → target 0.95 | `d_tm` alone | 30 | TM stayed 0.24 (too far, flat grad) |
| TM ~0.80 → target 0.95 | `d_tm` | 20 | works as fine-tuning |

**Rule of thumb**: first use `rmsd` to pull into the same basin
(RMSD < 5 Å), then switch to `d_tm` for fine-tuning if you care
about TM specifically.

---

## 8. Maximise shape anisotropy (elongation)

Use-case: target is a rod-like fragment; push samples to
elongated shapes.

```yaml
metadiffusion:
  - total_bias_clip: 3.0
  - opt:
      collective_variable: asphericity
      target: max
      strength: 0.5
      log_gradient: true           # required — raw grad ~CV²
      warmup: 0.2
      cutoff: 0.85
```

**Empirical**: asphericity 63 → **329** (5× more elongated),
`|Cα-Cα|` 3.79 (fold OK).

**Without log_gradient**: same config at `strength 20` blew Rg
up to 1515 Å. Always use `log_gradient: true` for asphericity.

---

## 9. Multi-CV composition (shape + diversity)

Use-case: you want both compaction AND diversity — e.g. force
all samples to Rg ≤ 18 while still being distinct.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - opt:
      collective_variable: pair_rmsd
      target: max
      strength: 0.5
  - steer:
      collective_variable: rg
      groups: [A]
      target: 17.0
      strength: 15.0
      warmup: 0.1
      cutoff: 0.9
```

Or add an interface cap too for multimers:

```yaml
  - steer:
      collective_variable: inter_chain
      groups: [A, B]
      target: 28.0
      strength: 3.0
```

**Note**: terms compose additively in the engine. If two terms
want opposite things (pull Rg down vs push samples apart with a
volume-sensitive CV) the effective force is their sum — watch
for cancellation.

---

## 10. Phase-D modifier example

Use-case: diversify samples but ignore the diversity force in
high-contact-density regions (core packing) — scale it down
using `contact_order` as a gate.

```yaml
metadiffusion:
  - total_bias_clip: 5.0
  - opt:
      collective_variable: pair_rmsd
      target: max
      strength: 0.5
      scaling:
        - collective_variable: contact_order
          mode: inverse              # low CO → full force; high CO → damp
          strength: 1.0
      projection:
        - collective_variable: rg
          mode: remove                # strip the Rg component so shape stays
          strength: 1.0
      modifier_order: [scaling, projection]
```

See [README §Gradient modifiers](../README.md#gradient-modifiers-phase-d)
for the scaler/projector API.

---

## Recipe selection matrix

| Your goal | Recipe |
|---|---|
| 5 diverse monomer samples | #1 |
| 5 diverse multimer poses, chains rigid | #2 |
| Iteratively push sampler to new regions | #3 |
| Fix a global-shape property | #4 (rg), #8 (asphericity) |
| Fix a local geometry | #5 (distance), #6 (dihedral) |
| Stay close to a template | #7 |
| Compose multiple constraints | #9 |
| Advanced: weight the diversity by something else | #10 |

---

## Strength cheat-sheet

| CV family | Recommended strength | Recommended `log_gradient` |
|---|---|---|
| rg / distance / max_diameter | 30 / 5 / 10 | skip |
| pair_rmsd / pair_tm | 0.5 (+ rep 64-128) | skip (internally normalised) |
| pair_itm | 1.0 (+ rep 128) | skip |
| asphericity / coordination / sasa | 0.5 / 0.5 / 1.0 | **required** |
| min_distance | 5 | recommended |
| angle | 50 | skip |
| dihedral | 30 | skip |
| rmsd / drmsd (reference) | 30+ | recommended |
| d_tm (reference) | 20+ | recommended |
| native_contacts / Q | 0.5 | recommended |
| hbond_count / salt_bridges / rmsf | 0.5 | recommended |

Remember: `total_bias_clip` caps the per-atom force in Å — 5.0 is
the all-round default. Go down (3.0) for tighter fold control, up
(8.0) for aggressive steering.
