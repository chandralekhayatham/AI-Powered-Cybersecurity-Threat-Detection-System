# Quick Start (Windows PowerShell)

Open PowerShell in this folder and run one command at a time.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python data\generate_dataset.py
python -m src.train_model
python -m src.run_detection
python -m src.app
```

Open http://127.0.0.1:5000 in your browser. Keep the terminal open while using the dashboard.

If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

Tests: `python -m pytest -q`
