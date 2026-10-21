# ⌨️ Smart Keyboard — Next-Word Prediction with N-Gram Language Models

**Lab 02 | NLP — Natural Language Processing**

A lightweight language-modeling application that predicts the next word in a sentence using **Unigram, Bigram, and Trigram models**. The project explores probability estimation, add-k smoothing, back-off strategies, perplexity, and Top-3 accuracy through an interactive Streamlit interface.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [Features](#-features)
4. [Technologies](#-technologies)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [Running the Application](#-running-the-application)
8. [Using the Application](#-using-the-application)
9. [Dataset](#-dataset)
10. [Language Models](#-language-models)
11. [Evaluation Metrics](#-evaluation-metrics)
12. [Results and Analysis](#-results-and-analysis)
13. [Application Screenshots and Tests](#-application-screenshots-and-tests)
14. [Limitations](#-limitations)
15. [Future Improvements](#-future-improvements)
16. [Author](#-author)

---

## 🎯 Project Overview

Smart Keyboard is a small Natural Language Processing (NLP) project inspired by the word-suggestion systems found in smartphone keyboards.

The application predicts the next word based on the words already typed by the user. It uses statistical language models built from word-sequence frequencies.

For example, given the context:

`the horses eat the`

The application returns the three most probable next-word suggestions according to the trained model.

The project initially uses a small synthetic corpus generated specifically for the laboratory. It can also be extended to use an English text dataset from Hugging Face.

The application is developed in Python and provides an interactive interface through Streamlit.

## 🎓 Objectives

The main objectives of this laboratory are to:

- Understand how N-gram language models estimate the probability of the next word.
- Implement and compare Unigram, Bigram, and Trigram models.
- Understand Maximum Likelihood Estimation (MLE).
- Apply add-k (Laplace) smoothing to handle unseen word sequences.
- Explore interpolation and back-off strategies.
- Evaluate language models using perplexity and Top-3 accuracy.
- Analyze prediction errors and unseen contexts.
- Develop an interactive application with Streamlit.

## ✨ Features

### 🔮 Next-Word Prediction

- Enter an English sentence in the application.
- Generate the three most probable next-word suggestions.
- Display the estimated probability of each suggestion.
- Identify words outside the training vocabulary.
- Use back-off to fall back to a lower-order model when appropriate.

### 🧠 Language Models

- **Unigram:** predicts words without using previous words as context.
- **Bigram:** predicts the next word using the previous word.
- **Trigram:** predicts the next word using the previous two words.

### 📊 Model Evaluation

- Calculate perplexity on test sentences.
- Measure Top-3 prediction accuracy.
- Detect zero-probability tokens.
- Compare the performance of different N-gram orders.
- Examine incorrect predictions and unseen contexts.

### 📚 Dataset Management

- Use the synthetic corpus provided for the laboratory.
- Optionally load a small English text dataset from Hugging Face.
- Display examples of the training data.
- Export a preview of the dataset as a CSV file.

### 🖥️ Interactive Dashboard

The Streamlit interface includes:

- Dataset and model configuration.
- Adjustable add-k smoothing.
- Interactive next-word predictions.
- Model evaluation tables and charts.
- Prediction-error analysis.
- Dataset previews and CSV export.

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web interface |
| Pandas | Tabular data and result presentation |
| Matplotlib | Data visualization |
| Hugging Face Datasets | Optional external dataset loading |
| Collections / Counter | N-gram frequency counting |
| Git and GitHub | Version control and source-code hosting |

The core N-gram implementation can be built with the Python standard library, while the interface and optional dataset integration require additional packages.

---

## 📁 Project Structure

The following structure illustrates the expected organization. Adapt it to the files actually present in your repository.

```text
smart-keyboard/
│
├── app.py                     # Streamlit application
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration (if using uv)
├── uv.lock                     # Locked dependencies (if generated)
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusions
│
├── data/
│   ├── make_data.py            # Synthetic corpus generator
│   └── corpus.csv              # Generated corpus, if retained
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py             # N-gram models and prediction logic
│   └── evaluate.py             # Model evaluation
│
├── results/
│   ├── metrics.json            # Evaluation metrics, if generated
│   ├── failures.txt             # Prediction errors, if generated
│   └── ngram_results.png        # Evaluation plot, if generated
│
└── docs/
    └── images/
        ├── dashboard.png        # Screenshot placeholder
        ├── prediction.png       # Screenshot placeholder
        ├── evaluation.png       # Screenshot placeholder
        ├── dataset.png          # Screenshot placeholder
        └── test-results.png     # Screenshot placeholder
```

**Note:** `results/` files are generated by the evaluation pipeline described in the laboratory. The optional Hugging Face dataset may be downloaded and cached automatically rather than stored in the repository.

---

## 🚀 Installation

### Prerequisites

- Python 3.*.
- Git.
- A terminal and a code editor such as VS Code.
- An Internet connection for the optional Hugging Face dataset and deployment.



### 2. Create a virtual environment

Using `uv`:

```bash
uv venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the project uses `uv` and `pyproject.toml`:

```bash
uv sync
```

Otherwise, install the packages listed in `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

Alternatively, with a standard activated Python environment:

```bash
python -m pip install -r requirements.txt
```

### 4. Generate the synthetic dataset

If the generator is included in the repository:

```bash
python data/make_data.py
```

Using `uv`:

```bash
uv run python data/make_data.py
```

The laboratory describes a reproducible synthetic corpus of approximately 1,200 sentences distributed across 60 documents.

### 5. Evaluate the models

```bash
uv run python -m src.evaluate
```

This command runs the evaluation pipeline and, according to the laboratory specification, generates evaluation metrics, prediction-error information, and a visualization.

---

## ▶️ Running the Application

Start the Streamlit interface from the project root:

```bash
uv run streamlit run app.py
```

Or, in an activated environment:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. The default Streamlit address is usually:

`http://localhost:8501`

To stop the application, press `Ctrl + C` in the terminal.

---

## 🖥️ Using the Application

### 1. Configure the dataset

Choose between the local synthetic corpus and the optional Hugging Face dataset, if both options are implemented in your version.

For Hugging Face, begin with a small subset to keep loading and training lightweight.
<img width="1472" height="692" alt="image" src="https://github.com/user-attachments/assets/b4b9b059-6875-405b-adad-268535102a63" />


### 2. Select a model

Choose the model order:

- Unigram (1-gram)
- Bigram (2-gram)
- Trigram (3-gram)

The selected model determines how much preceding context is used to predict the next word.
<img width="294" height="493" alt="image" src="https://github.com/user-attachments/assets/fd74a01c-a171-408c-ac2c-e45a32f4eb82" />


### 3. Configure smoothing

Adjust the add-k smoothing parameter.

A smaller value gives less additional probability to unseen events, while a larger value distributes more probability mass to events that were not observed.

### 4. Enter a sentence

Example:

```text
the happy girls
```

Submit the text and inspect the Top-3 suggestions, their estimated probabilities, and the model used by the back-off strategy.

<img width="1110" height="552" alt="image" src="https://github.com/user-attachments/assets/de4e4cf1-a420-4d65-a9db-036411c3414a" />

### 5. Analyze the results

Open the evaluation section to compare perplexity and Top-3 accuracy.

Inspect the error-analysis section to identify contexts for which the expected next word was not included among the three suggestions.

<img width="1103" height="661" alt="image" src="https://github.com/user-attachments/assets/c30cc4b0-48b0-459b-9df8-5eabb4e7b42f" />

---

## 📚 Dataset

### Dataset 1 — Synthetic Laboratory Corpus

The original laboratory uses a small, synthetically generated English corpus.

According to the lab specification:

- Approximately 1,200 sentences.
- 60 documents.
- Approximately 10,000 tokens.
- Approximately 61 word types.
- A fixed random seed for reproducibility.

The corpus is generated from a fixed-seed grammar and contains deliberately structured word sequences.

This makes it suitable for understanding N-gram models, smoothing, and evaluation without requiring a large external dataset.
<img width="972" height="708" alt="image" src="https://github.com/user-attachments/assets/1695bcb0-9773-45d1-874a-3811a2ccefc4" />


### Dataset 2 — Rotten Tomatoes (Optional)

For additional experiments, the application can use the [Rotten Tomatoes dataset on Hugging Face](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).

The dataset contains English movie reviews that provide a different text distribution from the synthetic corpus.

Dataset loading example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "cornell-movie-review-data/rotten_tomatoes",
    split="train[:600]",
)

print(dataset[0])
```

The dataset can be tokenized and used to construct training and test sequences.

**Important:** Results obtained on the synthetic corpus and movie reviews should be evaluated separately. Their vocabulary, sentence structure, and language characteristics differ.

---

## 🧠 Language Models

### Unigram

A Unigram model predicts the next word using its overall frequency, without considering the preceding words.

\[
P(w)=\frac{C(w)}{N}
\]

Here, \(C(w)\) is the number of occurrences of word \(w\), and \(N\) is the total number of predicted tokens used for the estimate.

### Bigram

A Bigram model uses the immediately preceding word.

\[
P(w_i\mid w_{i-1})
=
\frac{C(w_{i-1},w_i)}{C(w_{i-1})}
\]

This allows the model to capture simple local word relationships.

### Trigram

A Trigram model uses the preceding two words.

\[
P(w_i\mid w_{i-2},w_{i-1})
=
\frac{C(w_{i-2},w_{i-1},w_i)}
{C(w_{i-2},w_{i-1})}
\]

Using more context can improve predictions, but it also increases data sparsity because some three-word sequences may never appear in training.

### Add-k Smoothing

Add-k smoothing prevents unseen events from receiving zero probability when \(k>0\).

\[
P(w\mid c)=
\frac{C(c,w)+k}{C(c)+kV}
\]

Where:

- \(C(c,w)\): number of occurrences of the context followed by the word.
- \(C(c)\): number of occurrences of the context.
- \(V\): vocabulary size.
- \(k\): smoothing parameter.

### Interpolation and Back-off

Interpolation combines probabilities from different model orders:

\[
P=\lambda_3P_3+\lambda_2P_2+\lambda_1P_1
\]

The laboratory gives the example weights \(0.6\), \(0.3\), and \(0.1\), respectively.

Back-off instead uses a lower-order model when a higher-order context is unavailable or unsuitable under the implementation's rules.

These approaches help address data sparsity and improve robustness.

---

## 📊 Evaluation Metrics

### Perplexity

Perplexity measures how well a language model predicts a sequence of words.

\[
PP=\exp\left(
-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i\mid context_i)
\right)
\]

A lower perplexity generally indicates that the model assigns higher probabilities to the observed test sequence.

Without smoothing, a single token with probability zero can make perplexity infinite.

### Top-3 Accuracy

Top-3 accuracy measures the proportion of test positions where the correct next word appears among the three suggested words.

\[
Accuracy@3=
\frac{\text{Correct predictions in Top-3}}
{\text{Total evaluated positions}}
\]

A higher Top-3 accuracy indicates that the expected word is more frequently included among the suggestions.

### Train/Test Split

The laboratory separates training and test data by document to reduce information leakage between related sentences.

For external datasets, the splitting strategy should also be checked carefully, particularly when texts may be duplicated or related.

---

## 📈 Results and Analysis

The laboratory reference results for the synthetic corpus are summarized below.

These are **reference values from the lab specification**, not guaranteed outputs from every modified application or external dataset.

| Model | Perplexity | Top-3 accuracy |
|---|---:|---:|
| Unigram, k = 1 | 24.96 | 48.0% |
| Bigram, k = 1 | 6.43 | 60.9% |
| Trigram, k = 1 | 8.30 | 64.1% |
| Trigram, k = 0.1 | 5.06 | 64.1% |
| Trigram, k = 0 | Infinite | 64.1% |
| Interpolated 3/2/1-gram, k = 0.1 | 5.15 | 64.7% |
<img width="787" height="215" alt="image" src="https://github.com/user-attachments/assets/bd573ada-41f2-4906-9c60-5b50e640b71e" />


### Interpretation

- The Unigram model performs worse because it ignores context.
- The Bigram model captures one-word context and substantially reduces perplexity.
- The Trigram model uses more context, but its performance depends on smoothing.
- Unsmoothed models can produce zero probabilities for unseen events, resulting in infinite perplexity.
- Interpolation combines different context lengths and may improve Top-3 accuracy.
- Perplexity and Top-3 accuracy measure different aspects of model quality.

### Additional Experiments

The optional smoothing experiment investigates how changing \(k\) affects perplexity.

For a rigorous experiment, use a validation set to select hyperparameters and reserve the test set for final evaluation.

---



## ⚠️ Limitations

- The original corpus is small and synthetic.
- N-gram models rely on observed frequencies and do not understand meaning in the same way as modern neural language models.
- Unseen contexts and rare sequences can reduce prediction quality.
- Add-k smoothing does not solve every data-sparsity problem.
- Results from different datasets are not directly comparable without considering their vocabulary, preprocessing, and evaluation protocols.
- The optional Rotten Tomatoes corpus consists of movie reviews, which differ from the structured synthetic sentences used in the original laboratory.

---

## 🚀 Future Improvements

Possible extensions include:

- Adding Kneser-Ney smoothing.
- Comparing interpolation and back-off strategies.
- Adding a validation set for hyperparameter selection.
- Supporting custom text files and additional datasets.
- Improving the dashboard with richer visualizations.
- Adding automated tests for probability calculations and prediction behavior.
- Evaluating the effect of vocabulary size and unknown-word handling.
- Comparing statistical N-gram models with neural language models.

---

## 👩‍💻 Author

**Author:** Imane Mekkaoui

**Project:** Lab 02 — Smart Keyboard

**Subject:** Natural Language Processing (NLP)




---

*Smart Keyboard — Exploring statistical language modeling through next-word prediction.*
