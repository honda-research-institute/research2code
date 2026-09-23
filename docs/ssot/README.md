# Documentation SSOT

This directory holds maintainer-side structured facts used to refresh generated
documentation blocks. The R2C runtime does not read these files.

## Workflow

```bash
python3 scripts/refresh_ssot_docs.py validate
python3 scripts/refresh_ssot_docs.py refresh
python3 scripts/refresh_ssot_docs.py check
```

Commit SSOT YAML changes and regenerated docs together.

## Boundaries

- Runtime code remains authoritative for pipeline behavior.
- SSOT YAML carries documentation facts and mirrors selected runtime facts.
- The refresh script cross-checks mirrored facts against runtime constants.
- Marker targets overwrite only text between AUTO markers.
- File-set targets, such as `docs/generated/field-guides/`, rewrite generated
  files carrying the target's generated-file marker.

Generated blocks use this marker shape:

```markdown
<!-- BEGIN AUTO: integration-status -->
...generated content...
<!-- END AUTO: integration-status -->
```
