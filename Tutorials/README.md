# NLP & LLM Tutorials

This folder contains three foundational tutorials designed to build essential skills for understanding and implementing Natural Language Processing and Transformer-based models.

## Tutorial Descriptions

### TP1: Mathematics Optimization
**File:** `mathematics_optimization.ipynb`

Master the mathematical foundations necessary for training deep learning models:
- **Gradient Descent**: Understand the fundamentals of gradient-based optimization
- **Variants**: Explore SGD, Momentum, Adam, and other optimizers
- **Loss Functions**: Learn different loss functions and their properties
- **Optimization Landscape**: Visualize and understand optimization surfaces
- **Backpropagation**: Implement and understand gradient computation
- **Learning Rates**: Discover how to choose and schedule learning rates
- **Convergence Analysis**: Study conditions for convergence and stability

**Key Concepts:**
- Derivatives and gradients
- Convexity and non-convexity
- Local vs. global optima
- Vanishing/exploding gradients
- Regularization techniques

**Why This Matters:**
Understanding optimization is crucial for:
- Training neural networks effectively
- Debugging training issues
- Improving model convergence
- Selecting appropriate learning rates and optimizers

---

### TP2: Introduction to Transformers
**File:** `introduction_transformers.ipynb`

Dive into the architecture that powers modern NLP:
- **Self-Attention Mechanism**: Learn the core innovation of transformers
- **Multi-Head Attention**: Understand parallel attention computations
- **Positional Encoding**: Discover how transformers handle position information
- **Feed-Forward Networks**: Study the non-linear transformation layers
- **Layer Normalization**: Understand normalization in deep networks
- **Encoder-Decoder Architecture**: Learn the complete transformer structure
- **Scaled Dot-Product Attention**: Implement the attention formula
- **Advantages Over RNNs**: Compare transformers to recurrent architectures

**Key Concepts:**
- Query, Key, and Value projections
- Attention weights and softmax
- Residual connections
- Model scaling and performance
- Parallelization benefits

**Why This Matters:**
Transformers are the foundation of:
- BERT, GPT, and all modern LLMs
- State-of-the-art NLP systems
- Vision transformers and multimodal models
- Understanding attention-based architectures

---

### TP3: Text Classification with Transformers
**File:** `transformer_text_classification.ipynb`  
**Training Data:** `train.txt`

Apply transformer models to a practical text classification task:
- **Model Selection**: Choose appropriate pre-trained transformer models (BERT, DistilBERT, RoBERTa)
- **Data Loading**: Process and prepare text data for transformers
- **Tokenization**: Implement subword tokenization using Hugging Face tokenizers
- **Fine-tuning**: Adapt pre-trained models to your classification task
- **Training Pipeline**: Set up training loops with proper validation
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and epochs
- **Evaluation**: Compute precision, recall, F1-score, and confusion matrices
- **Inference**: Make predictions on new text samples

**Key Concepts:**
- Transfer learning in NLP
- Pre-trained vs. fine-tuned models
- Token embeddings and special tokens
- Classification heads
- Batch processing and attention masks
- Training stability and early stopping

**Dataset:** `train.txt`
- Contains labeled text examples for classification
- Can be used for binary or multi-class classification
- Format: One sample per line with label and text

**Why This Matters:**
This tutorial demonstrates:
- Real-world application of transformers
- End-to-end machine learning pipeline
- Best practices for fine-tuning
- Practical model deployment considerations

---

## Tutorial Progression

Complete the tutorials in order for optimal learning:

```
TP1: Mathematics Optimization
    ↓ (Understand optimization fundamentals)
TP2: Introduction to Transformers
    ↓ (Learn transformer architecture)
TP3: Text Classification with Transformers
    ↓ (Apply transformers to real problems)
```

## How These Tutorials Support the Labs

| Tutorial | Supports Lab |
|----------|-------------|
| TP1 | All Labs (optimization is used everywhere) |
| TP2 | Labs 1, 2, 3 (foundation for LLM understanding) |
| TP3 | Lab 2, 3 (practical classification and embeddings) |

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Key Libraries:
  - `torch` - PyTorch deep learning framework
  - `transformers` - Hugging Face Transformers
  - `numpy`, `pandas` - Data manipulation
  - `matplotlib`, `seaborn` - Visualization
  - `scikit-learn` - Machine learning utilities
  - `tqdm` - Progress bars

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install torch transformers numpy pandas matplotlib seaborn scikit-learn tqdm
   ```

2. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Open Tutorial 1**
   - Navigate to `TP1 - Mathematics Optimization/`
   - Open `mathematics_optimization.ipynb`
   - Run cells sequentially from top to bottom

4. **Progress Through Tutorials**
   - Complete TP1, then move to TP2, then TP3
   - Take notes on key concepts
   - Experiment with parameters and hyperparameters

## Key Takeaways

1. **Mathematics Foundation**: Optimization is essential for all machine learning
2. **Architecture Knowledge**: Understanding transformers unlocks modern NLP
3. **Practical Skills**: Being able to implement end-to-end pipelines is crucial
4. **Transfer Learning**: Fine-tuning pre-trained models is the modern approach
5. **Evaluation**: Proper metrics and validation are essential for building good models


## Additional Learning Resources

### Optimization
- [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
- [Understanding Deep Learning Optimization](https://karpathy.github.io/2019/04/25/recipe/)

### Transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated BERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

### Implementation
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Model Hub: Pre-trained Models](https://huggingface.co/models)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory (OOM) | Reduce batch size, use smaller models (DistilBERT), or enable gradient accumulation |
| Slow Training | Use GPU acceleration, reduce sequence length, increase batch size |
| Poor Performance | Try different pre-trained models, adjust learning rate, increase training epochs |
| Import Errors | Run `pip install -r requirements.txt` to install all dependencies |
| Tokenizer Issues | Download models explicitly: `transformers-cli download bert-base-uncased` |

## File Structure

```
Tutorials/
├── README.md (you are here)
├── TP1 - Mathematics Optimization/
│   └── mathematics_optimization.ipynb
├── TP2 - Introduction to Transformers/
│   └── introduction_transformers.ipynb
└── TP3 - Text Classification with Transformers/
    ├── transformer_text_classification.ipynb
    └── train.txt
```
