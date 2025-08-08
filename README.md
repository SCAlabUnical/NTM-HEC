# NTM-HEC: Neural Topic Modeling via Hashtag Embedding Clustering

This repository contains the codebase and experimental setup for **NTM-HEC**, a novel hashtag-centric methodology for topic modeling in social media, introduced in the paper:

> Cantini, R., Cosentino, C., Marozzo, F., Talia, D., & Trunfio, P. (2025, October). *Neural topic modeling in social media by clustering latent hashtag representations*. In 28th European Conference on Artificial Intelligence, ECAI-25. To appear.

NTM-HEC leverages corpus-specific hashtag embeddings to uncover coherent and diverse topic structures from social media content.

---

## 📁 Repository Structure

- **code/**:  
  Contains all the source files for the project:
  - **Main NTM-HEC implementation**:
    - `ntm-hec.py` – Implementation of the NTM-HEC methodology.
  - **Comparison baselines**:
    - `berTopic.py` – BERTopic model implementation.
    - `lda_.py` – Latent Dirichlet Allocation (LDA) baseline.
    - `lsa_.py` – Latent Semantic Analysis (LSA) baseline.
    - `top2Vec_.py` – Top2Vec baseline model.
  - **Supporting utilities**:
    - `main.py` – Main execution script to orchestrate experiments.
    - `metrics.py` – Metric computation and evaluation tools.
    - `preprocessing.py` – Preprocessing and data handling functions.
    - `utility.py` – Auxiliary utility functions.
    - `configuration.txt` – Configuration file to customize experimental settings.

- **requirements/**:  
  Python package requirements necessary to run the code.

- **readme/**:  
  Additional documentation and resources.

---

## 📊 Datasets

The datasets used in the experiments are publicly available:

- **Russia-Ukraine Conflict Dataset**:  
  [Kaggle - Ukraine Conflict Twitter Dataset (BwandoWando)](https://www.kaggle.com/dsv/8003074)

- **COVID-19 Anti-Vax Dataset**:  
  [Anti-Vax Twitter Dataset (Hayawi et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8648668/)

Please refer to the original article for detailed instructions on dataset preprocessing and monthly splits.

---

## 🚀 How to Run the Experiments

To replicate the experiments:

1. Install the required Python packages:

    ```bash
    pip install -r requirements/requirements.txt
    ```

2. Download and prepare the datasets according to the descriptions provided in the article.

3. Run the main experiment script:

    ```bash
    python3 code/main.py
    ```

    The experimental configuration can be adjusted by modifying the `code/configuration.txt` file.

---

## 🔬 About NTM-HEC

NTM-HEC is a three-step modular pipeline based on:

- Learning corpus-specific hashtag embeddings via Word2Vec (CBOW variant).
- Dimensionality reduction using PCA-initialized t-SNE.
- Clustering hashtag embeddings using HDBSCAN.

This approach improves topic coherence, diversity, and robustness against linguistic variability, outperforming traditional (LDA, LSA) and neural (BERTopic, Top2Vec) baselines.

---

## 📜 Citation

If you use this repository, please cite:

```bibtex
@inproceedings{Cantini2025NTMHEC,
  author    = {Riccardo Cantini and Cristian Cosentino and Fabrizio Marozzo and Domenico Talia and Paolo Trunfio},
  title     = {Neural topic modeling in social media by clustering latent hashtag representations},
  booktitle = {28th European Conference on Artificial Intelligence, ECAI-25},
  year      = {2025},
  month     = {October},
  note      = {To appear}
}
```

## 📬 Contact
For any questions or collaborations, please contact:
rcantini@dimes.unical.it
cristian.cosentino@dimes.unical.it




