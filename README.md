# NLP, LLM & Text Mining Labs

A comprehensive collection of hands-on labs and tutorials covering Natural Language Processing, Large Language Models, and Text Mining techniques. This course is part of the ESSEC Y3 AIDAMS program.

## Course Overview

This repository contains practical implementations and exercises exploring modern NLP techniques, from foundational concepts to advanced LLM applications. The materials progress from tokenization basics to fine-tuning transformer models.

## Labs

### Lab 1: Introduction to Tokenization & Embedding
**File:** `Labs/Lab 1 - Introduction to Tokenization & Embedding/introduction_tokenization_embedding.ipynb`

Introduction to fundamental NLP concepts:
- Text tokenization techniques
- Word embeddings and representation learning
- Word2Vec, GloVe, and FastText
- Embedding visualization and analysis

### Lab 2: Text Classification with Generative Models
**File:** `Labs/Lab 2 - Text Classification with Generative Models/text_classification_generative_models.ipynb`

Text classification using generative approaches:
- Zero-shot and few-shot classification with LLMs
- Prompt engineering for classification tasks
- Comparing different generative models
- Evaluation metrics and performance analysis

### Lab 3: Clustering & Modeling with LLMs
**File:** `Labs/Lab 3 - Clustering & Modeling with LLMs/clustering_modeling_LLMs.ipynb`

Advanced clustering and modeling techniques:
- Text clustering with LLM embeddings
- Semantic similarity analysis
- Document clustering evaluation
- Integration with LLM-generated representations

### Lab 4: LLM Fine-Tuning
**File:** `Labs/Lab 4 - LLM's Fine Tuning/medical_fine_tuning.ipynb`

Fine-tuning Large Language Models:
- Domain-specific model adaptation (medical domain)
- Training data preparation and preprocessing
- Fine-tuning strategies and hyperparameter tuning
- Model evaluation and results analysis

**Results:** `Labs/Lab 4 - LLM's Fine Tuning/evaluation_results.json`

## Tutorials

### TP1: Mathematics Optimization
**File:** `Tutorials/TP1 - Mathematics Optimization/mathematics_optimization.ipynb`

Mathematical foundations for optimization:
- Gradient descent and variants
- Loss functions and optimization landscape
- Backpropagation fundamentals
- Practical optimization techniques for ML

### TP2: Introduction to Transformers
**File:** `Tutorials/TP2 - Introduction to Transformers/introduction_transformers.ipynb`

Transformer architecture and mechanisms:
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Building blocks of modern NLP models

### TP3: Text Classification with Transformers
**File:** `Tutorials/TP3 - Text Classification with Transformers/transformer_text_classification.ipynb`

Practical text classification using transformer models:
- Fine-tuning pre-trained transformer models (BERT, DistilBERT, etc.)
- Data preparation for transformer-based classification
- Training and evaluation pipelines
- Model comparison and performance metrics

**Training Data:** `Tutorials/TP3 - Text Classification with Transformers/train.txt`

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- PyTorch or TensorFlow
- Hugging Face Transformers library
- Additional NLP libraries (NLTK, spaCy, scikit-learn, etc.)

### Installation

```bash
# Clone or download the repository
# Navigate to the project directory

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Running the Notebooks

1. Open Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

2. Navigate to the desired lab or tutorial folder

3. Open the `.ipynb` file and run cells sequentially

## Project Structure

```
NLP-LLMs-Labs/
├── Labs/
│   ├── Lab 1 - Introduction to Tokenization & Embedding/
│   ├── Lab 2 - Text Classification with Generative Models/
│   ├── Lab 3 - Clustering & Modeling with LLMs/
│   └── Lab 4 - LLM's Fine Tuning/
├── Tutorials/
│   ├── TP1 - Mathematics Optimization/
│   ├── TP2 - Introduction to Transformers/
│   └── TP3 - Text Classification with Transformers/
└── README.md
```

## 🛠️ Key Technologies

- **Deep Learning:** PyTorch / TensorFlow
- **NLP Libraries:** Hugging Face Transformers, NLTK, spaCy
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Learning Outcomes

By completing this course, you will:
- Understand fundamental NLP concepts and techniques
- Master tokenization and embedding methods
- Implement text classification pipelines
- Work with state-of-the-art transformer models
- Fine-tune LLMs for specific domains
- Evaluate and compare NLP models
- Apply optimization techniques to deep learning

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Official Documentation](https://pytorch.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)

---

**Course:** ESSEC Y3 AIDAMS - NLP, LLM & Text Mining  
**Last Updated:** December 2025
