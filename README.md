# NLP, LLM & Text Mining — My Lab Notes

These are my hands-on lab notebooks and tutorials from the ESSEC Y3 AIDAMS course on Natural Language Processing, Large Language Models, and Text Mining. I used this repository to run experiments, record results, and learn by doing.

## How I use this repo
- I open the notebooks in Jupyter and run the cells in order to reproduce the exercises.
- I keep small datasets and output files next to each notebook so I can re-run experiments and compare results.
- I edit hyperparameters in the notebooks when I need to test alternatives or debug training runs.

## Labs (what I worked on)
- Lab 1 — Introduction to Tokenization & Embedding: see [Labs/Lab 1 - Introduction to Tokenization & Embedding/introduction_tokenization_embedding.ipynb](Labs/Lab%201%20-%20Introduction%20to%20Tokenization%20%26%20Embedding/introduction_tokenization_embedding.ipynb)
  - What I practiced: tokenization strategies, Word2Vec/GloVe/FastText embeddings, and embedding visualization.

- Lab 2 — Text Classification with Generative Models: see [Labs/Lab 2 - Text Classification with Generative Models/text_classification_generative_models.ipynb](Labs/Lab%202%20-%20Text%20Classification%20with%20Generative%20Models/text_classification_generative_models.ipynb)
  - What I practiced: zero-/few-shot classification with LLMs, prompt design, and evaluating generative classifiers.

- Lab 3 — Clustering & Modeling with LLMs: see [Labs/Lab 3 - Clustering & Modeling with LLMs/clustering_modeling_LLMs.ipynb](Labs/Lab%203%20-%20Clustering%20%26%20Modeling%20with%20LLMs/clustering_modeling_LLMs.ipynb)
  - What I practiced: creating embeddings with LLMs, semantic similarity, and document clustering.

- Lab 4 — LLM's Fine Tuning: see [Labs/Lab 4 - LLM's Fine Tuning/medical_fine_tuning.ipynb](Labs/Lab%204%20-%20LLM%27s%20Fine%20Tuning/medical_fine_tuning.ipynb) and results in [Labs/Lab 4 - LLM's Fine Tuning/evaluation_results.json](Labs/Lab%204%20-%20LLM%27s%20Fine%20Tuning/evaluation_results.json)
  - What I practiced: preparing domain-specific (medical) data, fine-tuning strategies, and model evaluation.

## Tutorials (background and small exercises)
- TP1 — Mathematics Optimization: [Tutorials/TP1 - Mathematics Optimization/mathematics_optimization.ipynb](Tutorials/TP1%20-%20Mathematics%20Optimization/mathematics_optimization.ipynb) — gradient descent, loss landscapes, and optimization tricks.
- TP2 — Introduction to Transformers: [Tutorials/TP2 - Introduction to Transformers/introduction_transformers.ipynb](Tutorials/TP2%20-%20Introduction%20to%20Transformers/introduction_transformers.ipynb) — attention, multi-head attention, and transformer basics.
- TP3 — Text Classification with Transformers: [Tutorials/TP3 - Text Classification with Transformers/transformer_text_classification.ipynb](Tutorials/TP3%20-%20Text%20Classification%20with%20Transformers/transformer_text_classification.ipynb) and training data at [Tutorials/TP3 - Text Classification with Transformers/train.txt](Tutorials/TP3%20-%20Text%20Classification%20with%20Transformers/train.txt)

## Quick start (how I run things)
1. Create and activate a virtual environment.
   - Windows:
     ```powershell
     python -m venv venv
     venv\\Scripts\\activate
     ```
2. Install dependencies (if `requirements.txt` is provided):
   ```powershell
   pip install -r requirements.txt
   ```
3. Start Jupyter and open the notebook you want:
   ```powershell
   jupyter notebook
   ```
4. Run cells in order. For heavier training, I use a GPU runtime when available.

## Tools and libraries I use
- Python 3.8+
- Jupyter Notebook
- Hugging Face Transformers
- PyTorch (or TensorFlow depending on the notebook)
- NLTK, spaCy, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## What I learned (summary)
- The practical differences between tokenization approaches and embedding methods.
- How to prompt LLMs for classification and evaluate generative outputs.
- How to build and use embeddings for clustering and semantic search.
- How to prepare data and fine-tune models for a specific domain (medical).
