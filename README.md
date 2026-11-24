# Stress Check

Stress Check is a small NLP project that reads a piece of text and estimates whether it looks stress-related. I started this from the `final.ipynb` experiment and kept the first usable version intentionally straightforward instead of hiding everything behind a large language model.

## What is included

- `stress.csv` - the supplied labelled dataset
- `final.ipynb` - the original experiment notebook
- `train_model.py` - cleans the rows, trains the model, and saves evaluation results
- `app.py` - a simple Streamlit page for trying the model
- `model/` - created after training; it contains the saved pipeline and metrics

## How it works

1. The text column is split into training and testing sets with the same label balance.
2. TF-IDF converts words and two-word phrases into numerical features.
3. Logistic regression learns the relationship between those features and the existing `label` column.
4. The complete pipeline is saved so the app uses the same vectorizer as training.
5. The app shows a prediction and probability, but it does not call the result a diagnosis.

In this dataset, label `1` is treated as stress-related text and label `0` as not stress-related text. That meaning comes from the data used in the original notebook, so it should be checked again if the dataset is replaced.

## Run it locally

Install Python 3.10 or newer, then open a terminal in this folder:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The browser page should open at the local Streamlit address shown in the terminal.

## Why this version is a better starting point

The original notebook trains many different models in separate cells and asks for input repeatedly. That is useful for exploration, but it makes the result hard to reproduce. Here, training and prediction share one saved pipeline, the split is stratified, and the app can be run without opening a notebook.

## Limitations

- A text classifier can be wrong. It cannot know a person's full situation.
- Social-media language, sarcasm, spelling mistakes, and very short messages can confuse it.
- The labels come from an existing dataset and do not represent a clinical assessment.
- Do not use this project for medical decisions or emergency screening.

If someone may be in immediate danger, contact local emergency services or a trusted person instead of relying on this app.

## Sensible next improvements

- Add a small manually reviewed validation set.
- Compare logistic regression with the notebook's baseline models using the same split.
- Add precision, recall, and a confusion-matrix page to the UI.
- Ask users for feedback without storing their private text.
- Add tests for empty input, missing model files, and prediction output.
