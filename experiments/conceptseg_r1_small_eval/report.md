# ConceptSeg-R1 small eval

## Scope

- Input package: `server_conceptseg_fine_object_runlist_v008`
- Evidence base: existing `v008` run (`90` prompt-image pairs), structured QA, object/target alignment, plus a failed live remote reconnect attempt to `scan-train`
- Sampled review set here: `12` representative items with emphasis on railing / mesh / thin-pole failure modes

## What was reused

- Local scripts: `run_server_conceptseg_r1_smoke.sh`, `run_server_conceptseg_smoke.sh`, `build_conceptseg_fine_object_runlist.py`, `validate_conceptseg_fine_object_runlist.py`
- Existing outputs: `server_conceptseg_fine_object_runlist_v008_outputs_all`, `server_conceptseg_problem40`, `server_conceptseg_fine_object_alignment_v008`
- Existing v008 package already validated locally and contains `90` items / `30` targets

## Full-run signal

- All `90/90` runs returned code `0`
- Alignment report marks `89/90` prompt-answer pairs as concept-matched, but **`30/30` targets are non-discriminative across prompts**
- `railing or thin metal guardrail` is the unstable family:
  - median red overlay ratio `0.0416`
  - p90 `0.1047`
  - max `0.4032`
  - common answers: `guardrail`, `rail`, `fence`
- `pipe or thin utility conduit` is more stable semantically, but often thickens the region instead of preserving thin geometry
- `rooftop equipment box or HVAC unit` is the cleanest family and behaves like a coarse object concept detector

## Judgment

### Can it stably segment railing / mesh / thin pole as concept regions?

Short answer: **not stably enough for fine-target mainline use**.

- On railing / mesh scenes, ConceptSeg-R1 often locks onto the **entire fence/mesh field** instead of isolating the thin structural elements.
- On very thin structures, it can also **undersegment badly** or even answer `nonexistent`.
- For pipe/pole-like scenes it is directionally useful, but the mask usually becomes a **fatter concept blob** rather than a precise thin structure.

### Relative to current SAM2+VLM

Best fit: **second-stage review / proposal signal**, not a replacement for the current mainline.

- Good at: asking "is there something fence-like / pipe-like / equipment-like here?"
- Weak at: providing the **tight, stable, topology-aware masks** needed for thin targets in production routing
- The existing alignment report's own interpretation is consistent with this: usable as constrained candidate review, not as dense semantic production output

### Most visible failure modes

1. **Over-coarse concept regions**: mesh/fence gets swallowed as one broad region.
2. **Concept drift**: `railing -> fence`, `pipe -> pole/cables`, `equipment -> duct/red box`.
3. **Background inclusion**: large chunks of rooftop/background get pulled in with the concept.
4. **Thin-structure instability**: extremely narrow poles/rails either disappear or get only a tiny sliver mask.

## Sample review verdict

| concept | pass | mixed | fail | read |
| --- | ---: | ---: | ---: | --- |
| railing / guardrail | 0 | 1 | 3 | worst family; mesh and thin rails are not stable |
| pipe / conduit | 0 | 3 | 1 | useful hint, but geometry is too thick/coarse |
| equipment / HVAC | 1 | 3 | 0 | strongest concept family, but not the target problem |

## 2026-06-16 rich-tail transfer rerun

To avoid overfitting the earlier `v008` package verdict, a fresh `12`-image
transfer set was built from the newer accepted fine-object candidates on the
`2000-2999` tail slice:

- `4` railing-rich crops from
  `railing_rich_grounded_eval_2000_2999_strict_v2`
- `4` pipe-rich crops from
  `pipe_rich_grounded_eval_2000_2999_ext17_run`
- `4` equipment-rich crops from
  `equipment_rich_grounded_eval_2000_2999_ext80_run`

Artifacts:

- runlist:
  `/root/epfs/new_route_stage1_skymask/conceptseg_rich_tail_transfer_v001/runlist.json`
- outputs:
  `/root/epfs/new_route_stage1_skymask/conceptseg_rich_tail_transfer_v001_outputs`
- QA summary:
  `/root/epfs/new_route_stage1_skymask/conceptseg_rich_tail_transfer_v001_outputs/conceptseg_rich_tail_transfer_qa.json`

Observed signal:

- `12/12` runs completed successfully
- railing prompts produced semantically consistent answers
  (`railing`, `handrail`, `rail`) with average red-overlay ratio `0.0730`
- pipe prompts stayed semantically nearby (`pipe`, `conduit`) but one sample
  drifted to `metallic`; average red-overlay ratio `0.0193`
- equipment prompts were the cleanest (`HVAC` on `4/4`) with average
  red-overlay ratio `0.0639`

Interpretation:

- This rerun confirms that ConceptSeg-R1 can act as a **concept recognizer**
  on the newer rich-tail candidates.
- It still does **not** resolve the mainline need: the output is a concept
  overlay, not a topology-stable fine mask.
- The railing family remains the most dangerous one to over-trust, because the
  semantic answer is often right while the highlighted region can still be
  broad or mesh-like.

## Remote blocker

- Tried to reconnect to `scan-train` for a live `12`-image rerun using the existing remote assets.
- SSH failed **before** remote execution with local bind/connect errors:
  - `bind 192.168.100.115: Can't assign requested address`
  - `ssh: connect to host 10.0.8.114 port 31909: failure`
- Because this failed before entering the remote shell, this is a connectivity/config blocker, not a ConceptSeg-R1 environment blocker.

Historical note:

- This connectivity issue was later resolved by fixing the local SSH bind
  configuration.
- The current verdict in this report does **not** depend on that temporary
  connection failure; it rests on the already completed `90/90` run package and
  the sampled `12`-item review set described above.

## Recommendation

- Keep ConceptSeg-R1 as a **side-track reviewer / candidate proposer** for ambiguous fine targets.
- Do **not** move it into the fine-target mainline for railing / mesh / thin pole extraction in its current form.
- If revisited, the next worthwhile experiment is not a broader rollout, but a **post-filtered second stage**:
  1. run ConceptSeg-R1 only on SAM2/VLM suspicious regions
  2. intersect with existing instance/support masks
  3. reject high-area mesh/background expansions
  4. score whether any residual thin structure signal survives
