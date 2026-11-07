# Modeling results

Each training run is written to a new subdirectory named by `RUN_ID`. Existing
nonempty run directories are never overwritten.

The run contains separate `original/` and `mapseq/` model artifacts plus a
`comparison/` folder with paired held-out metrics, figures, uncertainty
intervals, and a Markdown report. The trained CatBoost models are saved as:

```text
results/RUN_ID/original/models/catboost_temperature.cbm
results/RUN_ID/mapseq/models/catboost_temperature.cbm
```

Generated run directories are ignored by Git. This README and `.gitkeep` are
the only tracked files in this folder.
