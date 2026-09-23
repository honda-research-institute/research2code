# Halt-artifact fixtures (queue item 12, halt-reason catalog)

- `detr-3c-20260704.halt.json` — harvested VERBATIM from
  `r2c_runs/archive/detr-distill/.pipeline/stage_3c.halt` on 2026-07-10.
  This is the live 7/4 fixture the halt-reason design is built on: the
  halt reason asserts "dispatch POST timed out after 900s" while the run
  events showed the diagnostician dispatch COMPLETED, well under the
  budget, with zero writes — twice. The catalog's golden tests render the
  same halt WITH the new `halt_class`/`evidence` fields and assert the
  no-write mechanism (never the guessed timeout); the verbatim artifact
  itself anchors the old-artifact tolerance test (no `halt_class` field →
  the stock rendering, byte-for-byte the pre-catalog behavior).

- The SRL 2a companion fixture ("scaffold_package.py exit 1", the other
  7/4 fixture in halt-reason-rewrite-design.md §1) was NOT preserved on
  disk — the SRL run dir was reused and its `.pipeline/stage_1.halt` now
  records a later, different halt. The golden test reconstructs the 2a
  artifact from the design note's description (stage_2a, scaffolder exit
  1, gap-paper context); it is marked reconstructed in the test.
