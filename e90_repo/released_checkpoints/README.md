# `released_checkpoints/` — trained weights and bundled eval artifacts

## Trained models (policy weights)

All downloadable **PPO checkpoints** are **`*.zip` files in this directory** (not in subfolders). Examples:

- `scpg_anti_lazy_1M_best.zip`
- `mlp_clean_run005_best.zip`
- …see **[../MODELS.md](../MODELS.md)** for the full list, required `configs/*.yaml`, and links to documentation.

Load only with `evaluate.py` or `PPO.load` using the **matching** config (observation layout and policy class must align).

## Evaluation reports and videos (per model)

For **each** `.zip` above, there is a subdirectory:

**`evaluations/<same_stem_as_zip_without_suffix>/`**

For example, for `mlp_clean_run005_best.zip`:

- `evaluations/mlp_clean_run005_best/REPORT.md` — short write-up  
- `evaluations/mlp_clean_run005_best/evaluation_summary.yaml` — numeric summary  
- `evaluations/mlp_clean_run005_best/videos/episode_1.mp4`, … — supporting videos  

Regenerate all summaries and MP4s from a clean clone:

```bash
./scripts/regenerate_evaluation_media.sh
```

(run from the **repository root**)

## Training metadata

**`metadata/`** holds copies of `run_info.yaml` and `training_summary.yaml` from the original training runs (timesteps, episode counts, config paths).
