# Specialty Coffee Origin Classification

## 📜 Overview
This project investigates whether **text classification models** can accurately predict the origin of specialty coffee based solely on its *Blind Assessment*.  
"Blind Assessments" are professional reviews by Q-Graders who evaluate coffee without knowing its origin.  

Using a dataset of **1,954 reviews from 31 countries** scraped from [Coffee Review](https://www.coffeereview.com), we compare traditional machine learning methods and a transformer-based model for this challenging NLP task.

---

## 📊 Goals
- **Main question**: Can coffee origin be predicted from blind tasting notes alone?
- Compare traditional models (Dummy Classifier, Multinomial Naive Bayes, NuSVC) with a transformer model (Distil-BERT).
- Understand the effects of dataset imbalance.
- Explore linguistic similarities across coffee origins.

---

## 📂 Data
- **Source**: Web-scraped from Coffee Review.
- **Size**: 1,954 single-origin reviews.
- **Labels**: Coffee origin (country).
- **Text feature**: Blind Assessment section.
- **Cleaning steps**:
  - Removed blends and undisclosed origins.
  - Standardized origin names.
  - Created two datasets:
    - **df1**: All 31 countries (imbalanced).
    - **df1_2**: Top 8 countries with ≥50 reviews (balanced subset).

---

## ⚙️ Methods

### Preprocessing
- **Traditional methods**: Lowercasing, removing stopwords & special characters, TF-IDF vectorization.
- **Deep learning**: Tokenization with Distil-BERT tokenizer.

### Models
- **Dummy Classifier** – Random baseline.
- **Multinomial Naive Bayes** – Probabilistic model.
- **NuSVC** – Support Vector Classifier with ν parameter.
- **Distil-BERT** – Transformer model fine-tuned for sequence classification.

### Training setup
- **df1**: 65% train / 35% test split, CV=2.
- **df1_2**: StratifiedShuffleSplit, CV up to 30.
- Distil-BERT fine-tuned using Hugging Face’s Trainer API.

### Evaluation
- Metric: **F1 Macro Score**.
- Confusion matrix analysis.
- TF-IDF + word cloud visualizations.

---

## Results

| Dataset | Dummy | Multinomial NB | NuSVC | Distil-BERT |
|---------|-------|----------------|-------|-------------|
| df1     | 0.03  | 0.07           | 0.08  | N/A         |
| df1_2   | 0.14  | 0.23           | **0.31** | 0.30        |

- **NuSVC** outperformed all other models on both datasets.
- Distil-BERT was close behind NuSVC on the balanced dataset.
- Overlap in flavor terms (e.g., “chocolate”, “sweet”, “acidity”) made classification harder.

---

## 🔍 Insights
- Imbalanced data reduces performance significantly.
- High overlap in descriptive terms limits model discriminability.
- The description from the blind assessment has bias, which is supported by this [study](https://onlinelibrary.wiley.com/doi/10.1111/joss.12827). This explains why all the models performed poorly.
---
