# Labs — My Notes and How I Worked Through Them

This `Labs` folder contains the four hands-on lab notebooks I completed during the ESSEC Y3 AIDAMS course. I used each notebook to experiment, record results, and refine my understanding of practical NLP and LLM workflows.

## Lab overview (what I did)

- Lab 1 — Introduction to Tokenization & Embedding
  - Notebook: [introduction_tokenization_embedding.ipynb](Lab%201%20-%20Introduction%20to%20Tokenization%20%26%20Embedding/introduction_tokenization_embedding.ipynb)
  - Focus: I compared tokenization strategies, trained/inspected Word2Vec/GloVe/FastText embeddings, and visualized embedding spaces.

- Lab 2 — Text Classification with Generative Models
  - Notebook: [text_classification_generative_models.ipynb](Lab%202%20-%20Text%20Classification%20with%20Generative%20Models/text_classification_generative_models.ipynb)
  - Focus: I practiced zero-shot and few-shot classification using LLM prompts, iterated on prompts, and evaluated generated outputs.

- Lab 3 — Clustering & Modeling with LLMs
  - Notebook: [clustering_modeling_LLMs.ipynb](Lab%203%20-%20Clustering%20%26%20Modeling%20with%20LLMs/clustering_modeling_LLMs.ipynb)
  - Focus: I generated embeddings with LLMs, computed semantic similarities, and experimented with clustering algorithms for document grouping.

- Lab 4 — LLM's Fine Tuning
  - Notebook: [medical_fine_tuning.ipynb](Lab%204%20-%20LLM's%20Fine%20Tuning/medical_fine_tuning.ipynb)
  - Results: [evaluation_results.json](Lab%204%20-%20LLM's%20Fine%20Tuning/evaluation_results.json)
  - Focus: I prepared domain-specific medical data, ran fine-tuning experiments, and analyzed evaluation metrics.

## How I ran the labs

1. Activate a Python virtual environment (Windows example):
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies if a `requirements.txt` exists:
   ```powershell
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the notebook I want to run:
   ```powershell
   jupyter notebook
   ```
4. Run cells sequentially; change hyperparameters inline for experiments.

Notes: For heavier model runs I use a GPU runtime or reduce batch sizes when running locally.

## Tips I picked up

- Always inspect tokenization outputs before training models — subtle tokenization differences change results.
- Save model checkpoints and evaluation outputs (like `evaluation_results.json`) to reproduce results later.
- When prompting LLMs for classification, keep a prompt template and log examples of both good and bad outputs for debugging.
