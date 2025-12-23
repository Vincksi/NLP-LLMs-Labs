# NLP & LLM Labs

This folder contains four comprehensive laboratory exercises covering essential topics in Natural Language Processing and Large Language Models.

## Lab Descriptions

### Lab 1: Introduction to Tokenization & Embedding
**File:** `introduction_tokenization_embedding.ipynb`

Explore the foundational concepts of NLP through tokenization and embeddings:
- **Tokenization Techniques**: Learn different tokenization approaches (word-level, subword, character-level)
- **Word Embeddings**: Understand distributed representations of words
- **Embedding Models**: Study Word2Vec, GloVe, and FastText algorithms
- **Visualization**: Analyze and visualize embedding spaces using dimensionality reduction
- **Applications**: Apply embeddings to downstream NLP tasks

**Key Concepts:**
- Text preprocessing and normalization
- Vocabulary construction
- Embedding dimensions and their significance
- Semantic similarity using embeddings

---

### Lab 2: Text Classification with Generative Models
**File:** `text_classification_generative_models.ipynb`

Leverage the power of generative language models for text classification:
- **Zero-shot Classification**: Classify text without task-specific training data
- **Few-shot Learning**: Use minimal examples for rapid adaptation
- **Prompt Engineering**: Craft effective prompts for classification tasks
- **Model Comparison**: Evaluate different generative models (GPT-based, T5, etc.)
- **Performance Analysis**: Compare accuracy, efficiency, and robustness

**Key Concepts:**
- In-context learning with LLMs
- Prompt design strategies
- Cost-benefit analysis of generative vs. discriminative approaches
- Evaluation metrics for classification

---

### Lab 3: Clustering & Modeling with LLMs
**File:** `clustering_modeling_LLMs.ipynb`

Apply LLM embeddings to clustering and unsupervised learning tasks:
- **Text Clustering**: Group similar documents using LLM-generated embeddings
- **Semantic Similarity**: Measure similarity between texts
- **Clustering Algorithms**: Implement KMeans, DBSCAN, hierarchical clustering
- **Evaluation**: Use silhouette scores, Davies-Bouldin index, and other metrics
- **Visualization**: Create meaningful visualizations of clusters

**Key Concepts:**
- Dense vector representations from LLMs
- Clustering quality assessment
- Optimal cluster number determination
- Document similarity matrices

---

### Lab 4: LLM Fine-Tuning
**File:** `medical_fine_tuning.ipynb`  
**Results:** `evaluation_results.json`

Master the process of fine-tuning large language models for domain-specific tasks:
- **Medical Domain Adaptation**: Fine-tune LLMs for medical text understanding
- **Data Preparation**: Process and prepare domain-specific training data
- **Training Pipeline**: Implement efficient fine-tuning strategies
- **Hyperparameter Optimization**: Tune learning rates, batch sizes, and other parameters
- **Evaluation**: Assess model performance on domain-specific metrics
- **Results Analysis**: Review evaluation metrics saved in `evaluation_results.json`

**Key Concepts:**
- Transfer learning in NLP
- Domain-specific model adaptation
- Training stability and convergence
- Preventing catastrophic forgetting
- Domain-specific evaluation metrics

---

## Learning Progression

The labs are designed to be completed in order:

1. **Start with Lab 1** to understand how text is represented numerically
2. **Move to Lab 2** to see how modern generative models handle classification
3. **Continue with Lab 3** to apply embeddings to unsupervised learning
4. **Finish with Lab 4** to master fine-tuning for specialized applications

## Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Key Libraries:
  - `torch` / `tensorflow` - Deep learning frameworks
  - `transformers` - Hugging Face Transformers library
  - `scikit-learn` - Machine learning utilities
  - `numpy`, `pandas` - Data manipulation
  - `matplotlib`, `seaborn` - Visualization
  - `nltk`, `spacy` - NLP utilities

## 🚀 Running the Labs

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Navigate to a Lab**
   - Open the desired `.ipynb` file
   - Read through the instructions and explanations
   - Run cells sequentially (top to bottom)

4. **Experiment and Learn**
   - Modify parameters to understand their effects
   - Try different models or datasets
   - Implement variations of the exercises

## 📊 Expected Outputs

- **Lab 1**: Embedding visualizations and similarity matrices
- **Lab 2**: Classification results and model comparison tables
- **Lab 3**: Cluster visualizations and evaluation metrics
- **Lab 4**: Training curves, evaluation metrics, and saved model weights in `evaluation_results.json`

## Tips for Success

- **Start Small**: Begin with simple examples before scaling up
- **Monitor Resources**: Some fine-tuning tasks may require GPU memory
- **Read Carefully**: Each notebook contains important explanations and context
- **Experiment**: Try modifying hyperparameters and observe the effects
- **Document**: Keep notes on your findings and observations

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer Paper
- [BERT Paper](https://arxiv.org/abs/1810.04805) - Bidirectional Encoder Representations
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
