# nolimit-ds-test-mahesa

Twitter entity-level sentiment analysis using a pretrained HuggingFace transformer model.

## Dataset

- Source: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- License: CC0 Public Domain
- The dataset contains entity-level sentiment labels for tweets: Positive, Negative, Neutral, and Irrelevant.
- Irrelevant labels are remapped to Neutral per the dataset author definition.

## Models

- Classifier: cardiffnlp/twitter-roberta-base-sentiment
- Tokenizer: cardiffnlp/twitter-roberta-base-sentiment
- No fine-tuning was performed. The pretrained model is used directly for inference.

## Project Structure

```
nolimit-ds-test-mahesa/
├── notebook.ipynb
├── app.py
├── requirements.txt
├── flowchart.png
└── data/
    └── sample.csv
```

## Setup

1. Clone the repository

```
git clone https://github.com/mahesa005/nolimit-ds-test-mahesa.git
cd nolimit-ds-test-mahesa
```

2. Create a virtual environment and install dependencies

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Add the dataset files to the data/ folder

Download twitter_training.csv and twitter_validation.csv from the Kaggle link above
and place them in the data/ folder.

## Running the Notebook

Open notebook.ipynb in Jupyter or Google Colab and run all cells in order.
A sample dataset is included in data/sample.csv for quick local verification.

## Running the Streamlit App

```
streamlit run app.py
```

## Flowchart

See flowchart.png for the end-to-end pipeline diagram.
