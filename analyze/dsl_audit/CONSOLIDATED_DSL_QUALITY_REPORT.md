# Consolidated DSL Quality Report

Generated: 2026-01-02

## Summary

This report consolidates all DSL quality issues across datasets for human review before submission.

## Issue Categories

### Existing Validation (from dsl_failures_report.json)
1. **one_hop**: DSL has only 1 qa_model() call (not multi-hop)
2. **no_compose**: Multiple calls but no dependency chain between them
3. **truncated_dsl**: DSL is incomplete/malformed
4. **yes_no_misframed**: Yes/no answer but WH-question in DSL
5. **scope_mismatch**: Specific date vs broader range mismatch

### New Audit Issues (from dsl_quality_audit.py)
6. **over_decomposition**: Simple question over-split into multi-hop DSL
7. **final_step_mismatch**: Last DSL question doesn't answer original question

---

## Dataset Statistics

| Dataset | Total | one_hop | no_compose | truncated | yes_no | scope | over_decomp | final_mismatch |
|---------|-------|---------|------------|-----------|--------|-------|-------------|----------------|
| **CRAG** | 300 | 203 (67.7%) | 24 (8.0%) | 1 | 11 | 3 | 34 (15.5%) | 33 (15.0%) |
| **HotpotQA** | 300 | 43 (14.3%) | 38 (12.7%) | 19 | 4 | 0 | 22 (7.9%) | 48 (17.1%) |
| **Frames** | 300 | 1 | 211 (70.3%) | 2 | 0 | 6 | 37 (12.3%) | 58 (19.3%) |
| **Musique** | 300 | 0 | 3 | 0 | 0 | 0 | 46 (15.3%) | 58 (19.3%) |
| **Mintaka** | 300 | 4 | 0 | 1 | 0 | 0 | 21 (7.0%) | 21 (7.0%) |
| **Bamboogle** | 125 | 0 | 1 | 0 | 0 | 0 | 5 (4.1%) | 4 (3.3%) |

---

## Critical Findings

### 1. CRAG Dataset - Severe Issues
- **67.7% one_hop**: Most "multi-hop" questions only have 1-hop DSLs
- Only 97 of 300 questions (32.3%) have valid multi-hop DSLs
- **Recommendation**: Consider excluding CRAG from multi-hop analysis or use only the ~30% with valid DSLs

### 2. Frames Dataset - No Composition
- **70.3% no_compose**: DSLs retrieve facts in parallel without dependency chains
- This breaks the assistive/incremental regime assumptions
- **Recommendation**: Review if Frames questions require compositional reasoning

### 3. Final Step Mismatch - Cross-Dataset Issue
- 15-20% of DSLs across datasets have final questions that don't clearly answer the original
- Examples: Original asks "Who X?" but final step asks "What nationality is {answer_1}?"
- **Recommendation**: Human review of flagged_for_review.csv files

---

## Files for Human Review

### Review CSVs (in analysis/dsl_audit/)
- `hotpotqa_flagged_for_review.csv` - 67 items
- `crag_flagged_for_review.csv` - 56 items
- `mintaka_flagged_for_review.csv` - 39 items
- `bamboogle_flagged_for_review.csv` - 9 items
- `musique_flagged_for_review.csv` - 94 items
- `frames_flagged_for_review.csv` - 88 items

### ID Lists (for programmatic filtering)
- `flagged_ids.json` - All flagged IDs by dataset and issue type
- `../scripts/dsl_failures_report.json` - Existing failure IDs

---

## Recommended Actions

### Immediate (for submission)
1. Filter out all `one_hop` items (they're not multi-hop)
2. Filter out all `truncated_dsl` items (malformed data)
3. Filter out `no_compose` items from Frames (70% of dataset!)

### Human Review Required
1. Review `over_decomposition` cases - some may be valid
2. Review `final_step_mismatch` cases - heuristic may have false positives
3. Decide on `yes_no_misframed` cases

### Total Unique IDs to Potentially Remove

To get the full count of unique problematic IDs, run:
```bash
python3 -c "
import json
from pathlib import Path

# Load existing failures
with open('scripts/dsl_failures_report.json') as f:
    existing = json.load(f)

# Load new audit
with open('analysis/dsl_audit/flagged_ids.json') as f:
    new_audit = json.load(f)

for dataset in ['hotpotqa', 'crag', 'mintaka', 'bamboogle', 'musique', 'frames']:
    all_ids = set()
    if dataset in existing:
        for failure_type, ids in existing[dataset].get('failures', {}).items():
            all_ids.update(ids)
    if dataset in new_audit:
        for issue_type, ids in new_audit[dataset].items():
            all_ids.update(ids)
    print(f'{dataset}: {len(all_ids)} unique problematic IDs')
"
```

---

## Next Steps

1. Human reviews CSV files and marks items as:
   - `remove` - definitely bad, filter out
   - `keep` - false positive, valid DSL
   - `uncertain` - needs discussion

2. Generate final exclusion list from human decisions

3. Re-run all downstream analysis with filtered data
