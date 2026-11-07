# MGM2 embedding preparation validation

**Overall status: PASS**

## Prepared cohort

| Item | Count |
|---|---:|
| HDF5 rows per representation | 3,896 |
| Metadata-matched rows | 3,691 |
| Excluded HDF5-only rows | 205 |
| Usable source rows | 3,689 |
| Averaged material samples | 1,226 |
| Temperature cohort | 1,191 |
| Development samples | 946 |
| Held-out test samples | 245 |
| Embedding features retained | 224 |

## Validation checks

- Passed: 91
- Warnings: 2
- Failed: 0

## Warnings requiring documentation

- **exact_duplicate_embedding_vectors_reviewed**: observed 2; Exact duplicate vectors are reported for review; they are not automatically removed because sample IDs remain distinct.
- **material_date_utc_consistent**: observed 11; Date conflicts are retained in date_utc_values and date_utc is left blank. Date is not used in the temperature model.

## Interpretation

The original and MAPseq temperature tables contain the same materials in the same frozen order, with identical targets and cruise assignments. Their only model-feature difference is the 224-dimensional MGM2 representation. This makes the later prediction comparison paired and fair.

No model was trained during this stage.
