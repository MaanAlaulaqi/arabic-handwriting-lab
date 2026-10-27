# Run the two-epoch training protocol

This route reproduces the **sequence**, including the frozen validation pass, first training epoch, lower-rate continuation, validation selection, and locked test comparison. It requires a CUDA GPU, enough GPU memory for this transformer, disk for the extracted images and checkpoint states, and time for 2,762 optimizer updates plus three validation and two test inference passes. Runtime depends on the GPU and storage. CPU-only full training is intentionally unsupported because the original notebook used CUDA automatic mixed precision.

From the repository root, after installing dependencies and placing the public archive and original set0 manifests:

```bash
python -m reproduce_training.run \
  --images-zip data/public_line_images.zip \
  --manifests data/set0 \
  --output outputs/full --device cuda
```

Add `--base-model data/original_model` to use a locally supplied original model snapshot. Its files are fingerprinted in the training configuration.

Epoch 1 uses seed 37 and learning rate `1e-5`; epoch 2 starts from its model weights with a **fresh optimizer and schedule**, seed 38, and learning rate `3e-6`. Both use batch size 2, eight-microbatch accumulation, AdamW, 5% warm-up, cosine decay, and a 191-token allowance derived from the training labels. Resumable checkpoints are written every 300 updates and at the end of each epoch. The selected checkpoint is chosen by the lower validation JiWER-default CER. The test manifest is read after `selection_lock.json` is written; test metrics never choose the checkpoint.

An interrupted run can resume with the same command and output directory. Do not mix a checkpoint from another configuration. GPU nondeterminism and library differences can change exact weights and scores; this workflow follows the original decisions rather than promising bitwise identical training.

The original reported run selected epoch 2 (validation CER 37.92% versus 36.55%). Test CER was 58.61% frozen versus 36.81% selected under JiWER's default edge stripping. The workflow additionally reports literal CER to expose the saved outputs' trailing-space behavior.
