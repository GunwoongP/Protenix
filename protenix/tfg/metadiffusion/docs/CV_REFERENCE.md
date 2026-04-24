# MetaDiffusion — Collective Variable Reference

Complete per-CV API with all parameters, expected ranges,
gradient behaviour, and recommended mode.

See [README.md](../README.md) for quickstart, recipes, and bug
history. See [RECIPES.md](RECIPES.md) for ready-made CASP configs.

---

## Table of contents

1. [Shape CVs](#shape-cvs) — rg, max_diameter, asphericity, coordination, sasa, min_distance, contact_order
2. [Pairwise diversity CVs](#pairwise-diversity-cvs) — pair_rmsd, pair_drmsd, pair_tm, pair_itm
3. [Atom / geometry CVs](#atom--geometry-cvs) — distance, angle, dihedral
4. [Multimer CVs](#multimer-cvs) — inter_chain
5. [Reference-based CVs](#reference-based-cvs) — rmsd, drmsd, d_tm / tm, native_contacts / Q
6. [Interaction CVs](#interaction-cvs) — hbond_count, salt_bridges, rmsf

---

## Shape CVs

### `rg` (a.k.a. `radius_of_gyration`)

Compaction / expansion knob. Classic Boltz usage: steer to a specific
Rg or bound the ensemble's Rg range.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` | list[str] | whole protein | chain IDs, e.g. `["A"]` |
| `region1` | str | — | alternative atom spec (`A::CA`) |

**Typical modes**: `steer` (target=Rg), `opt` (min/max).  
**Recommended strength**: `steer 30`, `opt 10`.  
**Value range**: 10–50 Å for typical proteins. Scales as `2.2·N^0.38`.  
**Log-gradient**: skip (linear in Å, natural scale).

```yaml
- steer: {collective_variable: rg, groups: [A], target: 18.0, strength: 30}
```

---

### `max_diameter`

Longest pairwise Cα distance. Complements `rg` for shape control
— Rg captures compactness, max_diameter captures elongation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` / `region1` | — | whole protein | same as rg |

**Recommended strength**: `opt 10`, `steer 5`.  
**Value range**: 40–120 Å.  
**Log-gradient**: skip.

---

### `asphericity`

Gyration-tensor eigenvalue spread. 0 = sphere, large = rod-like.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` / `region1` | — | whole protein | — |

**Recommended strength**: `opt 0.5` **+ `log_gradient: true`**.  
**Value range**: 10–500+ depending on shape. Raw gradient magnitude
scales with CV², so log_gradient is essential.  
**Log-gradient**: **required** — without it a force big enough to
move asphericity from 63 to 200 will blow up the fold.

---

### `coordination`

Mean number of near-neighbour atoms per atom (default cutoff 8 Å).
Proxy for packing density.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `contact_cutoff` | float | 8.0 | Å |
| `beta` | float | 10.0 | soft-step sharpness |

**Recommended strength**: `opt 0.5` **+ `log_gradient: true`**.  
**Value range**: 5–15 for typical proteins.

---

### `contact_order`

Sequence-separation-weighted contact density. Low CO = local
contacts (β-sheet, helix), high CO = long-range contacts
(complex topology).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `contact_cutoff` | float | 8.0 | Å |

**Recommended strength**: `opt 0.5` **+ `log_gradient: true`**.  
**Value range**: 0.03–0.25.

---

### `sasa`

Shrake–Rupley solvent-accessible surface area. Heavy CV —
see [memory note](#sasa-memory).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `probe_radius` | float | 1.4 | Å (water) |
| `atom_radius` | float | 1.9 | Å (heavy atom VDW) |
| `n_quad` | int | 48 | Fibonacci-sphere quadrature points |
| `chunk_size` | int | 32 | atoms per forward chunk |
| `use_checkpoint` | bool | `false` | opt-in to `torch.utils.checkpoint` |

**Recommended strength**: `steer 1.0` **+ `log_gradient: true`**.  
**Value range**: 5 000–25 000 Å² for 100–500 residue proteins.

#### SASA memory

The non-checkpointed path (default) builds `[B, chunk·n_quad, N]`
distance tensors. At `B=5, chunk=32, n_quad=48, N=210` peak is
~17 MB, but Protenix-v2 itself is ~15 GB — on a 24 GB GPU with
any contention you can OOM.

Options:
1. `use_checkpoint: true` — saves backward activations at ~2× forward
   cost. Known to interact with `@torch.no_grad` inference decorators
   on some PyTorch versions (silent crash); if that happens, drop
   back to no-checkpoint and shrink `chunk_size`.
2. Lower `chunk_size` to 16 or 8 for very large (>1000 atom) targets.
3. Reduce `n_quad` to 24 at the cost of angular resolution.

---

### `min_distance`

Smallest pairwise Cα–Cα distance across the structure. Useful for
explicit clash avoidance or for ensuring a minimum chain spread.

**Recommended strength**: `steer 5` **+ `log_gradient: true`**.  
**Value range**: 3.7–5.0 Å typically.

---

## Pairwise diversity CVs

Only defined for batched inputs (`B ≥ 2`) — the CV value for sample
`i` is the mean over all other samples `j` in the same batch.

### `pair_rmsd` (Kabsch)

**The main diversity knob.** Aligns each pair via Kabsch SVD then
computes RMSD on aligned positions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` / `region1` | — | whole protein | atom selection |

**Recommended**: `opt target=max strength=0.5` + `explore repulsion strength=64 sigma=4.0`.  
**Value range**: 5–30 Å.  
**Gradient normalisation**: internal max-per-atom-norm = 1.0, so
strength directly controls per-atom force. No need for log_gradient.

```yaml
- opt:     {collective_variable: pair_rmsd, target: max, strength: 0.5}
- explore: {type: repulsion, collective_variable: pair_rmsd, strength: 64, sigma: 4.0}
```

---

### `pair_drmsd` (distance-matrix)

Rotation-invariant by construction (no alignment needed). Softer
gradient than `pair_rmsd`.

**Recommended**: `opt target=max strength=2.0+`.  
**Value range**: 3–25.  
**When to use**: when you want orientation to stay free but compare
shapes; typically pair_rmsd is stronger.

---

### `pair_tm` (TM-score based)

TM-score between every pair of batch samples, using your GPU port of
the TM-align algorithm. Bounded in [0, 1] — maximising `1 - TM` is
more gentle than maximising RMSD.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d0` | float | auto from L | custom TM-score length scale |

**Recommended**: `opt target=max strength=0.5` + `explore repulsion strength=64 sigma=0.1`.  
**Value range**: 0.0–0.9 for `1 - TM`.

---

### `pair_itm` (interface-TM, multimer)

**Interface-only diversity.** Only atoms within `interface_cutoff`
of another chain contribute. Keeps intra-chain fold rigid, diversifies
docking poses.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `interface_cutoff` | float | 8.0 | Å — atom must be within this of another chain |
| `d0` | float | auto | as pair_tm |

**Recommended**: `opt target=max strength=1.0` + `explore repulsion strength=128 sigma=0.15`.  
**Multimer-only**: falls back to `pair_tm` when no chain info.

```yaml
- opt: {
    collective_variable: pair_itm, target: max, strength: 1.0,
    interface_cutoff: 10.0
  }
- explore: {
    type: repulsion, collective_variable: pair_itm,
    strength: 128, sigma: 0.15, interface_cutoff: 10.0
  }
```

---

## Atom / geometry CVs

All three take `atom1..4` region specs in the
[selection grammar](../README.md#atom--region-selection-grammar).

### `distance`

COM-COM distance between two atom selections.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atom1` | str | required | region spec, e.g. `A:10:CA` |
| `atom2` | str | required | region spec |

**Recommended**: `steer strength=5` **+ `log_gradient: true`**.  
**Value range**: 3–80 Å typically.

```yaml
- steer: {
    collective_variable: distance,
    atom1: A:10:CA, atom2: A:100:CA,
    target: 15.0, strength: 5.0, log_gradient: true
  }
```

### `angle`

3-atom angle in radians.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atom1` / `atom2` / `atom3` | str | required | `A:5:CA`, etc. |

**Recommended**: `steer strength=50`.  
**Value range**: 0–π.

```yaml
- steer: {
    collective_variable: angle,
    atom1: A:5:CA, atom2: A:50:CA, atom3: A:100:CA,
    target: 1.5708, strength: 50
  }
```

### `dihedral`

4-atom torsion in radians (right-handed).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atom1..4` | str | required | region specs |

**Recommended**: `steer strength=30`.  
**Value range**: [-π, π].

---

## Multimer CVs

### `inter_chain`

Distance between centres of mass of two chains.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` | list[str] | required | two chain IDs, `["A", "B"]` |

**Recommended**: `steer strength=5`.  
**Value range**: 15–80 Å.

```yaml
- steer: {
    collective_variable: inter_chain,
    groups: [A, B], target: 35.0, strength: 5
  }
```

---

## Reference-based CVs

All require a `reference_structure` path (CIF, mmCIF, or PDB) plus
matching chain selection via `groups` or `region1`.

### `rmsd`

Kabsch-aligned Cα RMSD to a reference structure.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference_structure` | str | required | file path |
| `groups` | list[str] | all chains | reference chain filter |

**Recommended**: `steer target=0 strength=30+` **+ `log_gradient: true`**.  
**Why strength ≥ 30**: our base samples already sit at ~13 Å RMSD
to any one sample; pulling to 0 requires overcoming the sampler's
intrinsic variance. Strength 5 is too soft.  
**Value range**: 0–30 Å.

```yaml
- steer: {
    collective_variable: rmsd,
    reference_structure: /path/to/ref.cif,
    groups: [A], target: 0.0, strength: 30, log_gradient: true
  }
```

### `drmsd`

Distance-matrix RMSD to a reference. Rotation / translation
invariant; useful for same-topology references where you only
care about internal geometry.

**Recommended**: `steer target=0 strength=30+` + `log_gradient: true`.  
**Value range**: 0–20.

### `d_tm` (alias `tm`)

TM-score to a reference (single-iteration Kabsch + logistic
weighting, ported from `MassiveFoldClustering_Tool`). Bounded
in [0, 1] so very different strengths from RMSD are needed.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference_structure` | str | required | file path |
| `d0` | float | auto from L | TM length scale |

**Recommended**: `steer target=0.95 strength=20+` + `log_gradient: true`.  
**Value range**: 0–1.

### `native_contacts` (alias `Q`)

Fraction of reference-native Cα–Cα contacts preserved.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference_structure` | str | required | — |
| `contact_cutoff` | float | 8.0 | Å |
| `beta` | float | 10.0 | soft-step sharpness |

**Recommended**: `opt target=max strength=0.5` + `log_gradient: true`.  
**Value range**: 0–1.

---

## Interaction CVs

### `hbond_count`

Approximate backbone H-bond count via heavy-atom geometry +
distance cutoff. Doesn't require explicit hydrogens.

**Recommended**: `opt target=max strength=0.5` + `log_gradient: true`.  
**Value range**: 0.1 · N – 0.5 · N for well-folded proteins.

### `salt_bridges`

Lys/Arg–Asp/Glu proximity count (heavy-atom cutoff).

**Recommended**: `opt strength=0.5` + `log_gradient: true`.

### `rmsf`

Per-residue fluctuation across the batch, summed. Maximising →
diverse per-site geometry. Doesn't require a reference.

**Recommended**: `opt target=max strength=0.5` + `log_gradient: true`.

---

## Gradient-modifier kwargs

All CVs can pass through the Phase D modifier pipeline:

```yaml
- opt:
    collective_variable: pair_rmsd
    target: max
    strength: 0.5
    scaling:
      - collective_variable: contact_order
        mode: inverse
    projection:
      - collective_variable: rg
        mode: remove
    modifier_order: [scaling, projection]
```

See [README §Gradient modifiers](../README.md#gradient-modifiers-phase-d).

---

## Cross-reference with the paper

| Boltz-Metadiffusion YAML key | Our CV name | Status |
|---|---|---|
| `rg` | ✅ `rg` | |
| `rmsd` | ✅ `rmsd` | |
| `pair_rmsd` | ✅ `pair_rmsd` | Kabsch, batched |
| `pair_rmsd_grouped` | via `region1..4` | same semantics |
| `sasa` | ✅ `sasa` | Shrake–Rupley |
| `distance` | ✅ `distance` | `atom1/atom2` |
| `inter_chain` | ✅ `inter_chain` | |
| `dihedral`, `angle` | ✅ ✅ | |
| `saxs` | ⏭ deferred | I(q) curve CV |
| `chemical_shift` | ⏭ deferred | BMRB matching |
