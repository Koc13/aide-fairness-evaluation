# Getting Started

This guide will help you set up the project and run your first experiment.

## Quick Setup (5 minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/Koc13/aide-fairness-evaluation.git
cd aide-fairness-evaluation
```

### 2. Create a Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Experiment exploration

### Option 1: Explore Existing Results (No API keys needed)

View pre-computed experiments and results:

```bash
# Open the income prediction notebook
jupyter notebook use-cases/income/income.ipynb
```

This notebook contains:
- Data exploration
- Baseline model results
- Fairness-aware model results
- Comparison visualizations

### Option 2: Re-run AIDE Experiments (Requires API keys)

**Warning:** This will require an API key.

```bash
# List available experiments
ls experiments/logs/

# View an experiment configuration
cat experiments/logs/income/1-income-baseline/config.yaml

# Check the best solution from a previous run
cat experiments/logs/income/1-income-baseline/best_solution.py

# View the tree visualization
open experiments/logs/income/1-income-baseline/tree_plot.html
```

To run a new AIDE experiment:

```bash
# Example: Income prediction baseline
aide data_dir="datasets/acs-income-ca" \
     goal="Predict PINCP" \
     eval="accuracy" \
     agent.steps=20
```

## Project Structure Overview

```
.
├── datasets/           # Datasets used in experiments
├── experiments/        # AIDE experiment logs and workspaces
│   ├── logs/          # Configuration, results, and visualizations
│   └── workspaces/    # Working directories with predictions
├── use-cases/          # Jupyter notebooks for each use case
│   ├── income/
│   ├── student-admission/
│   ├── hiring/
│   └── ...
├── prompts.md          # Prompts used for AIDE experiments
├── EXPERIMENTS.md      # Detailed experimental methodology
└── README.md           # Project overview
```

## Summary

1. **Explore Results**: Open notebooks in `use-cases/` to see analysis
2. **Read Methodology**: Check `EXPERIMENTS.md` for experimental design
3. **View Prompts**: See `prompts.md` for AIDE prompts used
4. **Read Papers**: Check references in individual use case overviews

