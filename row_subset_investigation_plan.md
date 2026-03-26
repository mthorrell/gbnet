# Row-Subset Tree Update Investigation Plan

## Goal

Make `gbnet/tests/test_row_subset_single_tree_contract.py` pass by adding a public API that lets callers choose which training rows feed a single `gb_step()` update for:

- `gbnet.xgbmodule.XGBModule`
- `gbnet.lgbmodule.LGBModule`

Target contract:

- `module.gb_step()` keeps current behavior.
- `module.gb_step(row_indices=...)` fits the next tree using only the selected rows.
- Predictions after the update are still produced for all rows in the fixed training set.

## Current State In GBNet

Observed from the current code:

- `gbnet/tests/test_row_subset_single_tree_contract.py` already defines the intended external behavior.
- `gbnet/xgbmodule.py` caches a single full training `DMatrix` in `self.dtrain` and always boosts with full-size gradients / Hessians.
- `gbnet/lgbmodule.py` caches a single full training `Dataset` in `self.train_dat` and always trains / updates against that full dataset.
- `forward()` in both modules keeps a full-size `self.FX`, so row-subsetting should affect only the tree-fit step, not the prediction tensor shape.
- Higher-level code calls `gb_step()` with no arguments today, so the new API should remain optional and backward-compatible.

## Evidence That A Public-API Approach May Work

Official backend docs suggest both libraries expose row-subsetting at the Python level:

- XGBoost Python API documents `DMatrix.slice(rindex, allow_groups=False)`.
  - Reference: https://xgboost.readthedocs.io/en/release_2.0.0/python/python_api.html
- LightGBM Python API documents `Dataset.subset(used_indices, params=None)`.
  - Reference: https://lightgbm.readthedocs.io/en/v4.4.0/pythonapi/lightgbm.Dataset.html

This means the first implementation attempt should use public Python APIs, not private internals.

## Environment Note

This workspace does not currently have `torch`, `xgboost`, or `lightgbm` installed, so backend API validation has to happen in a dependency-enabled environment before implementation details are locked down.

## Primary Questions To Answer

### Shared API Questions

- Should the public signature simply become `gb_step(row_indices=None)`?
- What input forms should be accepted initially?
  - Minimal scope: 1-D integer index array / list / tensor.
- Should indices be required to be unique?
  - Recommended initial rule: yes.
- Should indices be allowed to be unsorted?
  - Probably yes, but normalize internally.
- How should invalid input behave?
  - Out-of-range, empty, non-1-D, float dtype, duplicates.

### XGBoost Questions

- Does `self.dtrain.slice(row_indices)` work cleanly with the versions this repo supports (`xgboost>=2.0.3`)?
- Can `self.bst.boost(subset_dtrain, ..., subset_grad, subset_hess)` be called repeatedly when the booster was initialized from the full training matrix?
- Does slicing preserve feature names / types / categorical metadata if those are present?
- Do custom-objective gradients / Hessians only need to match the sliced matrix shape, or are there hidden assumptions tied to the original full matrix?

### LightGBM Questions

- Does `self.train_dat.subset(row_indices)` work cleanly with the versions this repo supports (`lightgbm>=4.3.0`)?
- Can `self.bst.update(train_set=subset_train, fobj=obj)` use a subset dataset derived from the full dataset?
- Does the subset dataset need `construct()` called explicitly?
- Does the subset preserve / inherit:
  - raw data access
  - init score
  - reference dataset linkage
  - feature metadata
- After an update on a subset dataset, does prediction on the full training data still match native LightGBM behavior?

### Feasibility / Risk Questions

- Is XGBoost straightforward while LightGBM is the real constraint?
- If LightGBM cannot accept a different per-round `train_set` through its public Python API, is there still a viable public-only fallback?
- If not, should the feature ship first for XGBoost only, or should both backends remain aligned?

## Recommended Investigation Sequence

### 1. Reproduce The Contract Test In A Real Env

Run only the standalone contract test first:

```bash
pytest gbnet/tests/test_row_subset_single_tree_contract.py -q
```

Confirm the current failure mode for both backends:

- missing `row_indices` argument
- deeper backend limitation
- both

### 2. Write Tiny Backend-Only Probes

Before changing gbnet, write minimal scripts that answer the backend questions directly.

#### XGBoost probe

Test whether this workflow succeeds and matches native subset training:

1. Build full `DMatrix(X, label=y)`.
2. Slice it with `row_indices`.
3. Train or boost one round on the slice only.
4. Predict on the full matrix.
5. Compare against training a one-round model from scratch on `X[row_indices]`.

Important checks:

- single-output first
- exact predictions match the contract fixture
- repeated alternating subsets over multiple rounds

#### LightGBM probe

Test whether this workflow succeeds and matches native subset training:

1. Build full `Dataset(X, label=y, init_score=...)`.
2. Call `subset(row_indices)`.
3. Train or update one round using only the subset.
4. Predict on the full matrix.
5. Compare against training a one-round model from scratch on `X[row_indices]`.

Important checks:

- first tree from `lgb.train(...)`
- later tree via `Booster.update(train_set=..., fobj=...)`
- subset dataset construction and raw-data retention

### 3. Lock Down Public Semantics

Assuming backend probes are positive, define the public contract as:

- `gb_step(row_indices=None)`
- `row_indices` selects training rows for the next tree only
- `forward()` and prediction shape remain unchanged
- `gb_calc()` still computes full gradients / Hessians from full `FX`
- `gb_step(row_indices=...)` slices those gradients / Hessians before handing them to the backend

Recommended initial validation rules:

- normalize to 1-D `np.int64`
- require at least one row
- require all rows in `[0, batch_size)`
- reject duplicates for v1 unless backend behavior is intentionally defined for them

### 4. Choose The Simplest Internal Design

#### XGBoost likely implementation

Candidate path:

1. Normalize `row_indices`.
2. If `row_indices is None`, keep the current path.
3. Otherwise:
   - create `subset_dtrain = self.dtrain.slice(row_indices)`
   - slice `grad` and `hess`
   - call `self.bst.boost(subset_dtrain, ...)`

Questions to resolve during implementation:

- whether a sliced `DMatrix` should be cached or rebuilt per call
- whether version-specific `boost()` signatures interact with the subset path

#### LightGBM likely implementation

Candidate path:

1. Normalize `row_indices`.
2. If `row_indices is None`, keep the current path.
3. Otherwise:
   - create `subset_train = self.train_dat.subset(row_indices)`
   - slice `grad` and `hess`
   - use `lgb.train(...)` for the first round or `self.bst.update(train_set=subset_train, fobj=obj)` for later rounds

Highest-risk point:

- whether LightGBM actually supports per-round updates on a subset `Dataset` through the public Python API while keeping the same booster history

If this fails, stop and decide whether:

- to use a more manual dataset reconstruction path from cached raw arrays
- to drop LightGBM from the first version of the feature
- to abandon the feature if backend constraints are fundamental

### 5. Add Tests In Layers

Keep the standalone contract test as the top-level acceptance test, then add smaller unit tests around edge cases.

Recommended additional tests:

- `gb_step(row_indices=full_row_indices)` matches `gb_step()`
- invalid `row_indices` are rejected clearly
- repeated calls with different subsets still produce stable behavior
- backward compatibility: existing `test_xgbmodule.py` and `test_lgbmodule.py` still pass unchanged

Optional second-wave tests:

- multi-output behavior
- tensor / list input normalization
- unsorted indices

### 6. Update User-Facing Docs

If the feature ships:

- add a short example to `README.md`
- document that this is row-subset boosting, not mini-batching
- note that the training dataset must still remain fixed across training

## Suggested Implementation Constraints

To keep scope controlled, v1 should explicitly avoid:

- changing the meaning of `forward()`
- true mini-batch training
- allowing the training dataset itself to change during training
- extending higher-level model `fit()` APIs unless there is a concrete use case

## Exit Criteria

The investigation is complete when these statements are true:

- backend-only probes show whether public APIs are sufficient for both backends
- the desired public semantics are written down and stable
- the highest-risk LightGBM question is answered with evidence
- there is a clear go / no-go decision for:
  - both backends
  - XGBoost only
  - neither
- if the answer is "go", the contract test file is enough to drive the first implementation

## Likely First Decision

The fastest path is:

1. Prove XGBoost first, because `Booster.boost()` already takes an explicit `DMatrix` each round.
2. Probe LightGBM second, because `Booster.update(train_set=...)` is the main uncertainty.
3. Only after both probes succeed, thread `row_indices` into the gbnet public API.
