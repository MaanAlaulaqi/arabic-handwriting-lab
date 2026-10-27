# Muharaf set0 ArTrOCR reproduction

This repository carries the original four Colab notebooks and two terminal workflows. Both workflows use the same `muharaf_repro/` code for split checks, 512×102 image preparation, model inference, and scoring. Neither requires Google Drive or Colab. Run the commands below from the repository root.

| Root folder | What you will find there |
| --- | --- |
| [`evaluate_pretrained/`](evaluate_pretrained/README.md) | The quicker evaluation workflow. Rerun frozen and selected test inference, or verify the saved predictions without model weights. |
| [`reproduce_training/`](reproduce_training/README.md) | The full workflow. Run the two training epochs in order, choose a checkpoint on validation, then evaluate on the locked test split. This requires substantially more compute. |
| `muharaf_repro/` | Shared Python code used by both workflows. It loads the split, prepares images, trains or runs inference, checks checkpoints, and calculates metrics. |
| `notebooks/` | The four original Colab notebooks, retained as a record of the reported experiment and its order of execution. |
| `published_run/` | Outputs from the reported run: saved predictions, selection lock, metrics, bootstrap summary, training configurations, and the small files needed to assemble the selected model. You can verify its test metrics without rerunning inference. |
| `tests/` | Small checks for image preparation, manifest consistency, split overlap, and the effect of trailing spaces on CER. |
| [`data/set0/`](data/set0/README.md) | Portable copies of the exact train, validation, and test manifests. They preserve assignments and labels while replacing paths from the original computer with `public/<filename>`. |

The commands create `outputs/` for new results. The three set0 manifests are tracked in `data/set0/`; large downloads placed elsewhere in `data/` are excluded from Git. Neither images nor weights are needed to verify the predictions already in `published_run/`.

## Inputs and installation

The published [`Muharaf-public` dataset](https://doi.org/10.5281/zenodo.11492215) by Mehreen Saeed and collaborators provides `public_line_images.zip`, containing 24,495 paired PNG/TXT line files under `public/`. Download that archive from Zenodo and place it at `data/public_line_images.zip`; the exact set0 `train.jsonl`, `validation.jsonl`, and `test.jsonl` manifests are included in `data/set0/`. The CLI checks counts (22,092 / 1,069 / 1,334), unique names, manifest-to-TXT labels, and line-disjoint splits. Generating a new random split would evaluate a different experiment.

###### _The full training workflow currently requires CUDA, so it will not run on a Mac without code adjustments._

Create a virtual environment from the repository root. Use the activation command for your terminal:

**macOS/bash**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows, PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows, Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

After activation, install PyTorch for your OS and GPU using the [official selector](https://pytorch.org/get-started/locally/). Then install the remaining packages:

```text
python -m pip install -r requirements.txt
```

The original saved training environment recorded PyTorch `2.11.0+cu130` and Transformers `5.21.0`; different hardware and kernels can yield different weights. Download access to [the original ArTrOCR checkpoint](https://huggingface.co/BushraAlmod03/ArTrOCR-HTR) is needed for inference and training. See the two workflow READMEs for commands.

The public dataset and original model have their own terms; review those before redistributing weights or images. This starter contains code, experiment notebooks, the exact portable set0 manifests, saved prediction JSONL, the selection lock, and the selected model's small configuration/tokenizer files. It does not contain the image archive, original model snapshot, or selected checkpoint weights. Its saved-prediction verification route works without either model or the images. To run inference without access to Hugging Face, supply an original model snapshot with `--base-model data/original_model`.

## Large files and repository layout

Keep the code, exact split manifests, saved predictions, and small model configuration files in Git. The selected set0 checkpoint is available as a [GitHub Release download](https://github.com/MaanAlaulaqi/arabic-handwriting-lab/releases/download/set0-selected-v1/selected_model.tar). Its archive SHA-256 is `77ddcad28a4a93ce748889ed1ca856a54c2981a86385bd1082917f83ef2869fd`.

Download `selected_model.tar` and extract it into `data/selected_checkpoint/`. The resulting files should be in `data/selected_checkpoint/model/`. The model directory's fingerprint should match `cb69c36e312d20df1600669838e96b3cd9f5bc87f300ebc956bcde2217de4aa8` in `published_run/final_test_v1/selection_lock.json`. If you already have the original `model.safetensors` instead, follow the assembly instructions in [`evaluate_pretrained/README.md`](evaluate_pretrained/README.md). Do not add weights to Git history.

The [Muharaf dataset on Zenodo](https://doi.org/10.5281/zenodo.11492215) remains the primary source for `public_line_images.zip`. Place that archive at `data/public_line_images.zip`. An unchanged copy may also be published as an optional GitHub Release mirror, with its checksum and source attribution. Do not commit the archive or its extracted PNG/TXT files to Git.
## Scoring and scope

The reported selected set0 test score uses JiWER's default CER (36.81%). JiWER strips leading/trailing spaces per line; the saved selected outputs also yielded 50.12% CER when scored literally. Both metrics are printed by the code. Word error rate, CER after removing eight basic harakat, and exact text-alignment mark counts are reported separately.

The experiment trains for two completed epochs. This is a bounded protocol, not evidence of convergence or a universal optimal epoch count. The original test set was held until validation selection. Further training should be treated as a separate study, with a new predefined protocol.
