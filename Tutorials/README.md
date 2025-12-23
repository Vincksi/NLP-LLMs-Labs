# Tutorials — My Notes and How I Used These Exercises

This `Tutorials` folder contains small, focused notebooks I used to build the mathematical and architectural foundations needed for the labs.

## What I covered

- TP1 — Mathematics Optimization
  - Notebook: [TP1 - Mathematics Optimization/mathematics_optimization.ipynb](TP1%20-%20Mathematics%20Optimization/mathematics_optimization.ipynb)
  - What I practiced: gradient descent variants, loss landscapes, and practical tricks for stabilizing training.

- TP2 — Introduction to Transformers
  - Notebook: [TP2 - Introduction to Transformers/introduction_transformers.ipynb](TP2%20-%20Introduction%20to%20Transformers/introduction_transformers.ipynb)
  - What I practiced: attention mechanism, multi-head attention, positional encodings, and building block intuition.

- TP3 — Text Classification with Transformers
  - Notebook: [TP3 - Text Classification with Transformers/transformer_text_classification.ipynb](TP3%20-%20Text%20Classification%20with%20Transformers/transformer_text_classification.ipynb)
  - Training data: [TP3 - Text Classification with Transformers/train.txt](TP3%20-%20Text%20Classification%20with%20Transformers/train.txt)
  - What I practiced: preparing data, fine-tuning a transformer for classification, and evaluating results.

## How I used these tutorials

1. Read TP1 to understand optimization choices before training large models.
2. Walk through TP2 to build intuition about how transformers process tokens and attention.
3. Follow TP3 to apply the concepts and fine-tune a small transformer on the provided `train.txt` dataset.

## Quick run (Windows)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt   # if provided
jupyter notebook
```
Open the notebook you want and run cells in order. For TP3, verify `train.txt` path before starting training.

## Notes I kept
- Use small subsets when experimenting locally to reduce runtime.
- Log hyperparameters and random seeds to reproduce results.
- For transformer fine-tuning, monitor learning rate closely — small changes can destabilize training.