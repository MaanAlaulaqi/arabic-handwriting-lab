# Evaluate the published selection

This route runs two fixed checkpoints on the same 1,334 held-out set0 lines. It needs the public image archive, the included set0 manifests, the selected checkpoint's **`model/` directory**, and its `selection_lock.json`. The selected model is checked against the SHA-256 recorded in the lock before any test inference. On a CPU it can be very slow; a supported GPU is recommended.

## Assemble and check the selected model locally

The repository includes the five small files from the reported epoch-2 `model/` folder in `published_run/selected_model_metadata/`. Download **only** `model.safetensors` from `finetune_v2_second_epoch/checkpoint_latest/model/` in your original run. From the repository root, use its actual path here:

```bash
python -m evaluate_pretrained.prepare_selected --weights /path/to/model.safetensors
```

This standard-library-only command copies the six files to `data/selected_checkpoint/model/` and checks the complete directory fingerprint against the published selection lock. It needs no GPU, dataset, or installed Python packages. A mismatch means the model files differ; do not publish or evaluate that directory until resolved. The `data/` directory is ignored by Git. To distribute the verified model, archive the whole `model/` directory for a GitHub Release asset, then document that download in the root README. Recipients extract it so `data/selected_checkpoint/model/model.safetensors` and its five JSON files sit together.

## Run inference

From the repository root:

```bash
python -m evaluate_pretrained.run \
  --images-zip data/public_line_images.zip \
  --manifests data/set0 \
  --selected-model data/selected_checkpoint/model \
  --selection-lock published_run/final_test_v1/selection_lock.json \
  --output outputs/quick --device auto
```

Add `--base-model data/original_model` to use an offline snapshot instead of fetching the original model. Use a fresh `--output` folder when changing either model. The original set0 selection lock from the Colab run records a selected model SHA-256; this command refuses a mismatching checkpoint.

To verify **already saved** test predictions without images, model weights, or a GPU:

```bash
python -m evaluate_pretrained.verify_saved \
  --frozen published_run/final_test_v1/frozen_test_predictions.jsonl \
  --selected published_run/final_test_v1/selected_test_predictions.jsonl \
  --manifest data/set0/test.jsonl
```

The included manifest checks image names and exact reference text against the saved prediction files. This route verifies calculations from supplied predictions; it does not independently establish how those predictions were generated.

Full inference saves separate resumable prediction files and `test_comparison.json` under `--output`. Existing prediction files must have matching image names and exact references. Delete an output directory only when intentionally starting a new run; use a separate output path for a different model or split.
