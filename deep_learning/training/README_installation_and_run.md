# Task 3 — Arabic Context to Idiom

## Files
- `common.py`
- `train_task3_retrieval.py`
- `train_task3_mt5.py`
- `train_task3_hybrid_mt5_first.py`

## Clean environment (Windows CMD / PowerShell)

```bash
cd C:\Users\ayman\Documents\IdiomX\github_idiomX\IdiomX
py -3.11 -m venv idiomx_task3_env
idiomx_task3_env\Scripts\activate
python -m pip install --upgrade pip
```

### Install PyTorch
Choose ONE command depending on your machine.

GPU (official selector example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Install remaining libraries
```bash
pip install transformers sentence-transformers datasets sentencepiece accelerate scikit-learn pandas numpy matplotlib ipykernel
```

### Add env as Jupyter kernel
```bash
python -m ipykernel install --user --name idiomx_task3_env --display-name "Python (idiomx_task3_env)"
```

Then refresh Jupyter and switch the notebook kernel to **Python (idiomx_task3_env)**.

## Run from CMD

### Retrieval baseline
```bash
python train_task3_retrieval.py --project_root "C:\Users\ayman\Documents\IdiomX\github_idiomX\IdiomX"
```

### mT5 baseline
```bash
python train_task3_mt5.py --project_root "C:\Users\ayman\Documents\IdiomX\github_idiomX\IdiomX" --epochs 3
```

### Hybrid mT5-first
```bash
python train_task3_hybrid_mt5_first.py --project_root "C:\Users\ayman\Documents\IdiomX\github_idiomX\IdiomX"
```

## Call from notebook
```python
from pathlib import Path
from train_task3_retrieval import run_retrieval
from train_task3_mt5 import run_mt5
from train_task3_hybrid_mt5_first import run_hybrid_mt5_first

PROJECT_ROOT = Path(r"C:\Users\ayman\Documents\IdiomX\github_idiomX\IdiomX")

retrieval_result = run_retrieval(PROJECT_ROOT, Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_retrieval_baseline")
mt5_result = run_mt5(PROJECT_ROOT, Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_mt5_baseline")
hybrid_result = run_hybrid_mt5_first(PROJECT_ROOT, Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_hybrid_mt5_first")
```

## What each script saves
Each model folder saves:
- predictions CSV
- truth-table summary CSV
- metrics JSON
- text report
- demo correct / incorrect CSVs
- charts PNGs
- top confusions CSV
- saved model (for mT5)
