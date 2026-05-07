# Supplementary Material for Legend

This archive contains anonymized code for the method used in the paper. The processed training data can be regenerated from public OC20 LMDB files with the included preprocessing script. It does not include written appendices, datasets, model weights, or personal information.

## Included

- `main.py`, `model.py`, `framework.py`, `config.py`, and `AdslabData.py`
- `preprocess_data.py` and `site_analysis.py` for OC20 preprocessing
- `ocpmodels/` and the EquiformerV2 adapter used by the method
- `requirements.txt`

## Not Included

- raw or processed datasets
- CatBERTa weights
- EquiformerV2 checkpoints

The omitted assets are too large to bundle inside the supplement. Place them on disk as shown below and provide their paths through the command line arguments.

## Recommended Folder Layout

```text
Supplementary_Material/
  data/
    oc20data_proc/
      train.pt
      val.pt
      val_id.pt
      val_ood_ads.pt
      val_ood_cat.pt
      val_ood_both.pt
  external/
    CatBERTa-hf/
    equiformer_v2/
      checkpoints/
  checkpoints/
  logs/
```

## External Assets

### OC20 IS2RE data

Use the official Open Catalyst Project resources for the OC20 IS2RE data:

- https://opencatalystproject.org/
- https://opencatalystproject.org/challenge.html

The processed `.pt` files may be placed directly under `data/oc20data_proc/`, or regenerated from the public OC20 LMDB data using the provided preprocessing script.

## Preprocessing

To regenerate the processed `.pt` files from OC20 LMDB data, run the preprocessing script:

```bash
python preprocess_data.py \
  --lmdb_path /path/to/oc20/lmdb \
  --mapping_path /path/to/oc20_data_mapping.pkl \
  --save_path ./data/oc20data_proc/train.pt
```

### CatBERTa text encoder

The code expects a local Hugging Face style directory containing the tokenizer and encoder weights. The public project page is:

- https://github.com/hoon-ock/CatBERTa

Obtain the released `CatBERTa-hf` directory from the project artifact or the public release, then place it at `external/CatBERTa-hf/`.

### EquiformerV2

Use the official EquiformerV2 repository and its OC20 checkpoints:

- https://github.com/atomicarchitects/equiformer_v2

Clone the repository, make the `nets/` package available, and point `--equiformer_root` to the repository root. If a public checkpoint is used, place it under `external/equiformer_v2/checkpoints/` and provide it via `--equiformer_ckpt_path`.

## Environment

Install the Python dependencies listed in `requirements.txt`.

Example:

```bash
pip install -r requirements.txt
```

## Running Training

Run commands from the `Supplementary_Material` root so the relative paths resolve correctly.
Training, validation, and explanation exports all use the pruned-motif path in this archive.
Training runs five sequential random seeds by default (`seed`, `seed+1`, ..., `seed+4`). Set `--num_runs 1` to execute a single run.

```bash
python main.py \
  --mode train \
  --num_runs 5 \
  --dataset_root ./data/oc20data_proc \
  --train_file train.pt \
  --val_file val.pt \
  --bert_path ./external/CatBERTa-hf \
  --equiformer_root ./external/equiformer_v2 \
  --equiformer_ckpt_path ./external/equiformer_v2/checkpoints/best_checkpoint.pt \
  --checkpoint ./checkpoints \
  --log_dir ./logs
```

## Running Validation

```bash
python main.py \
  --mode val \
  --dataset_root ./data/oc20data_proc \
  --val_file val.pt \
  --bert_path ./external/CatBERTa-hf \
  --equiformer_root ./external/equiformer_v2 \
  --equiformer_ckpt_path ./external/equiformer_v2/checkpoints/best_checkpoint.pt \
  --resume_checkpoint ./checkpoints/<run_name>/best_model.pth
```
