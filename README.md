# Audio Deepfake Detection with XLS-R and SLS Classifier

Reproduction and extension of **"Audio Deepfake Detection with XLS-R and SLS Classifier"** (Zhang et al., ACM Multimedia 2024).

The Selective Layer Summarization (SLS) classifier extracts attention-weighted features from all 24 transformer layers of XLS-R 300M (wav2vec 2.0), then classifies bonafide vs. spoofed speech via a lightweight fully-connected head. RawBoost data augmentation is applied during training.

## Results

| Track | Paper EER (%) | Our EER (%) | Model |
|-------|--------------|-------------|-------|
| ASVspoof 2021 DF | 1.92 | 2.14 | epoch\_2.pth |
| ASVspoof 2021 LA | 2.87 | 3.51 | epoch\_2.pth |
| In-the-Wild | 7.46 | 7.84 | epoch\_2.pth |

## Prerequisites

- NVIDIA GPU with >= 16 GB VRAM (tested on RTX 4080)
- CUDA 11.7
- Conda (Miniconda or Anaconda)

## Installation

```bash
# 1. Clone repository
git clone https://github.com/Yash-Sukhdeve/XLS-R-SLS-Deepfake-Detection.git
cd XLS-R-SLS-Deepfake-Detection

# 2. Create conda environment
conda env create -f environment.yml
conda activate SLS

# 3. Install fairseq (required for XLS-R)
# Download fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip from the
# original repository or extract from source, then:
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
cd ..

# 4. Install this package
pip install -e .
```

## Dataset Download

Download the following datasets manually:

| Dataset | Link | Purpose |
|---------|------|---------|
| ASVspoof 2019 LA | [Edinburgh DataShare](https://datashare.is.ed.ac.uk/handle/10283/3336) | Training & validation |
| ASVspoof 2021 DF | [Zenodo](https://zenodo.org/record/4835108) | Evaluation |
| ASVspoof 2021 LA | [Zenodo](https://zenodo.org/record/4837263) | Evaluation |
| ASVspoof 2021 Keys | [asvspoof.org](https://www.asvspoof.org/index2021.html) | Evaluation labels |
| In-the-Wild | [deepfake-total.com](https://deepfake-total.com/in_the_wild) | Evaluation |

## Data Directory Setup

Create symlinks to your local dataset paths:

```bash
# Training and validation data
mkdir -p data
ln -s /path/to/ASVspoof2019/LA/ASVspoof2019_LA_train data/ASVspoof2019_LA_train
ln -s /path/to/ASVspoof2019/LA/ASVspoof2019_LA_dev data/ASVspoof2019_LA_dev

# Protocol files
mkdir -p database
ln -s /path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols database/ASVspoof_LA_cm_protocols
ln -s /path/to/ASVspoof2021_DF_cm_protocols database/ASVspoof_DF_cm_protocols
```

### Evaluation Keys Directory

The `keys/` directory (included in the repo) contains In-the-Wild keys. For DF and LA evaluation, you need the ASVspoof2021 keys from [asvspoof.org](https://www.asvspoof.org/index2021.html). The evaluation script expects the following layout:

```
keys/
  in_the_wild_filelist.txt       # (included) In-the-Wild file list
  in_the_wild_key.txt            # (included) In-the-Wild labels

# For DF evaluation, pass truth_dir pointing to:
/path/to/ASVspoof2021/keys/DF/
  CM/trial_metadata.txt          # ASVspoof2021 DF trial metadata

# For LA evaluation, pass truth_dir pointing to:
/path/to/ASVspoof2021/keys/LA/
  CM/trial_metadata.txt          # ASVspoof2021 LA trial metadata
  ASV/trial_metadata.txt         # ASV trial metadata
  ASV/ASVTorch_Kaldi/score.txt   # ASV system scores
```

Example evaluation with external keys:
```bash
# DF (pass the DF keys directory as truth_dir)
python -m sls_asvspoof.evaluate --track DF scores/scores_DF.txt /path/to/ASVspoof2021/keys/DF eval

# LA (pass the LA keys directory as truth_dir)
python -m sls_asvspoof.evaluate --track LA scores/scores_LA.txt /path/to/ASVspoof2021/keys/LA eval

# In-the-Wild (uses repo's keys/ directory)
python -m sls_asvspoof.evaluate --track Wild scores/scores_Wild.txt ./keys eval
```

## Pre-trained XLS-R Model

Download the XLS-R 300M checkpoint from [fairseq/wav2vec/xlsr](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr) and place it in the repository root:

```bash
# The file should be named xlsr2_300m.pt
ls xlsr2_300m.pt
```

## Training

```bash
# Using the convenience script (recommended)
bash scripts/train.sh

# Or directly with config
python -m sls_asvspoof.train --config configs/train_df.yaml

# Override any config value via CLI
python -m sls_asvspoof.train --config configs/train_df.yaml --batch_size 10 --num_epochs 100
```

## Evaluation

Generate score files then compute EER:

```bash
# DF track
python -m sls_asvspoof.train --track=DF --is_eval --eval \
    --model_path=models/model_DF_WCE_50_5_1e-06/epoch_2.pth \
    --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt \
    --database_path=/path/to/ASVspoof2021_DF_eval/ \
    --eval_output=scores/scores_DF.txt
python -m sls_asvspoof.evaluate --track DF scores/scores_DF.txt ./keys eval

# LA track
python -m sls_asvspoof.train --track=LA --is_eval --eval \
    --model_path=models/model_DF_WCE_50_5_1e-06/epoch_2.pth \
    --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt \
    --database_path=/path/to/ASVspoof2021_LA_eval/ \
    --eval_output=scores/scores_LA.txt
python -m sls_asvspoof.evaluate --track LA scores/scores_LA.txt ./keys eval

# In-the-Wild
python -m sls_asvspoof.train --track=In-the-Wild --is_eval --eval \
    --model_path=models/model_DF_WCE_50_5_1e-06/epoch_2.pth \
    --protocols_path=keys/in_the_wild_filelist.txt \
    --database_path=/path/to/release_in_the_wild/ \
    --eval_output=scores/scores_Wild.txt
python -m sls_asvspoof.evaluate --track Wild scores/scores_Wild.txt ./keys eval
```

## Pre-trained Models

Pre-trained model checkpoints are available:
- [Google Drive](https://drive.google.com/drive/folders/13vw_AX1jHdYndRu1edlgpdNJpCX8OnrH?usp=sharing)
- [Baidu Pan](https://pan.baidu.com/s/1dj-hjvf3fFPIYdtHWqtCmg?pwd=shan)

## Configuration System

This project supports both CLI arguments and YAML config files. CLI arguments always take precedence over YAML values.

```bash
# YAML config only
python -m sls_asvspoof.train --config configs/train_df.yaml

# YAML config with CLI overrides
python -m sls_asvspoof.train --config configs/train_df.yaml --lr 0.00001

# Pure CLI (no config file)
python -m sls_asvspoof.train --track=DF --lr=0.000001 --batch_size=5 --loss=WCE --num_epochs=50
```

Available configs: `configs/train_df.yaml`, `configs/eval_df.yaml`, `configs/eval_la.yaml`, `configs/eval_wild.yaml`.

## Project Structure

```
XLS-R-SLS-Deepfake-Detection/
├── src/sls_asvspoof/          # Main package
│   ├── __init__.py
│   ├── train.py               # Training and inference entry point
│   ├── evaluate.py            # Score evaluation (EER, t-DCF)
│   ├── model.py               # SLS + XLS-R model architecture
│   ├── data_utils.py          # Dataset classes and data loading
│   ├── rawboost.py            # RawBoost data augmentation
│   └── metrics/               # Evaluation metrics
│       ├── __init__.py
│       ├── eer.py             # EER and DET curve computation
│       └── tdcf.py            # t-DCF computation
├── core_scripts/              # Utility scripts (random seed, etc.)
├── configs/                   # YAML config presets
│   ├── train_df.yaml          # Paper-matching training config
│   ├── eval_df.yaml           # DF evaluation config
│   ├── eval_la.yaml           # LA evaluation config
│   └── eval_wild.yaml         # In-the-Wild evaluation config
├── scripts/                   # Shell convenience scripts
│   ├── train.sh
│   └── eval.sh
├── keys/                      # Evaluation keys and file lists
├── experiments/               # Archived experiment results
│   ├── CHANGELOG.md
│   └── v1/                    # Baseline reproduction results
├── setup.py
├── environment.yml
├── requirements.txt
└── README.md
```

## Backward Compatibility

For users of the original flat-file layout, root-level shim files (`model.py`, `data_utils_SSL.py`, `RawBoost.py`, `eval_metrics_DF.py`, `eval_metric_LA.py`) re-export from the package with a deprecation warning.

## Citation

```bibtex
@inproceedings{zhang2024audio,
  title={Audio Deepfake Detection with XLS-R and SLS Classifier},
  author={Zhang, Qishan and Wen, Shuangbing and Hu, Tao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024},
  publisher={ACM}
}
```

## Acknowledgements

- [XLS-R](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr) (Babu et al., 2022)
- [RawBoost](https://github.com/TakHemlworksata/RawBoost) (Tak et al., Odyssey 2022)
- [ASVspoof Challenge](https://www.asvspoof.org/)
