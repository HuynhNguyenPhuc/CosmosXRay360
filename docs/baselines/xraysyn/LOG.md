# XRaySyn Change & Fix Log

## 2026-08-03 — Initial XRaySyn Integration & Dynamic Pose Helpers

**Phase:** 1 (Import & Environment)  
**Files:** `baselines/models/xraysyn.py`, `baselines/train/xraysyn.py`  
**Change:** Created `XRaySynWrapper` and `train_xraysyn` training script wrapping `XraySynModel` from `baselines/cloned/XraySyn/`. Implemented `_get_T_batched` and `_get_T_multi` pose matrix helpers and wrapped model initialization with a temporary `os.chdir(xraysyn_dir)` block.  
**Why:** Fixes `XraySynModel.get_T`'s hardcoded batch size of 4 (`torch.cat([T, T, T, T])`) to support arbitrary batch sizes, and allows loading `simplified_bone_absorb_2d.pt` / `simplified_tissue_absorb_2d.pt` from any working directory.  
**Verification:** Tested `XRaySynWrapper.infer_multi_views` across single and multi-batch inputs without shape or file errors.

---

## 2026-08-05 — Paper Cached & PAPER.md/CODE.md Corrected Against It

**Phase:** 0 (Resource acquisition / correctness of docs, not code)
**Files:** `docs/baselines/xraysyn/{PAPER.md,CODE.md}`, `arxiv_papers/XraySyn_2012.02407.pdf`
**Change:** The 2026-08-05 docs backfill pass initially wrote `PAPER.md` without actually
downloading/reading the paper — it cited the wrong title ("Reconstructing 3D CT Volumes from Single
2D X-ray Images using Physics-Supervised Generative Adversarial Networks", which does not exist) and
described an architecture that doesn't match the paper's real two-stage 3DPN/2DRN + CT2Xray design.
`CODE.md` separately mis-stated the repo's license as "Custom research license (Johns Hopkins
University)". Both were caught during a same-day review, the real paper (arXiv:2012.02407, "XraySyn:
Realistic View Synthesis From a Single Radiograph Through CT Priors") was downloaded into
`arxiv_papers/`, and `PAPER.md`/`CODE.md` were rewritten directly from it and cross-checked against
`baselines/cloned/XraySyn/`'s actual source (`LICENSE` is plain MIT; `DRRProjector`, `UnetGenerator`,
`NLayerDiscriminator` constructor signatures confirmed via `grep`).
**Why:** generic-pattern.md's Phase 0 rule is explicit: "never work from memory of what a method
probably does; open the PDF." This is exactly the failure mode that rule exists to prevent, and it's
worth keeping as a log entry rather than silently rewriting, since it's evidence the read-the-PDF
step needs to be checked, not assumed, when reviewing future backfills (including ones done by other
tools/agents).
**Verification:** New title/authors/venue/arXiv ID/equation numbers cross-checked against the
downloaded PDF page-by-page; license claim verified against the actual `LICENSE` file; all class/
kwarg names in the "Key Modules" mapping re-verified with `grep` against the real source files
(`drr_projector_new.py`, `unet.py`, `common.py`) rather than asserted from memory.

---

## 2026-08-05 — Widened Angular Training Range & Azimuth Alignment

**Phase:** 2 (Training Scope / Fairness & Convention Alignment)  
**Files:** `baselines/train/xraysyn.py`, `baselines/models/xraysyn.py`  
**Change:** Expanded `OTHER_POSE_THETA_Y_RANGE` in `train/xraysyn.py` from `(-0.05, 0.05)` ($\pm 9^\circ$) to `(-1.0, 1.0)` (full $360^\circ$ sweep). Switched target angle generation in `models/xraysyn.py` from `endpoint=False` to `endpoint=True`.  
**Why:** Official demo script trained `net2d` refinement on a $\pm 9^\circ$ angular neighborhood, creating a train/eval mismatch when benchmark evaluation queries $360^\circ$ novel views. Endpoint alignment matches ground-truth DiffDRR projections (`views/*.png`).  
**Verification:** Verified `net2d` trains across full angle range and `infer_multi_views` indices align with ground-truth view files.
