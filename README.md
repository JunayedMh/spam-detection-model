# Spam detection model

A Multinomial Naive Bayes classifier that labels SMS messages as spam or ham (not spam). The training workflow is in `model_train.ipynb`: load the data, remove duplicates, clean the text, turn it into TF-IDF features, train the model, and save it. `app.py` is a Streamlit app that loads the saved model so anyone can paste a message and check it.

Live demo: https://spam-detection-model-yjqjmpxheakldlphqbq2ab.streamlit.app/

## Results

After removing duplicate messages, the model scores 96.52% accuracy on a held-out test set of 1,034 messages (20% of the data).

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Ham   | 0.96      | 1.00   | 0.98     | 895     |
| Spam  | 1.00      | 0.74   | 0.85     | 139     |

The data is imbalanced, so accuracy alone says little: a model that always predicted ham would score about 87%. The spam numbers are the ones to watch. Every message the model called spam really was spam, so no real message was blocked in the test set. The cost is recall: the model catches 74% of the spam and misses about one in four.

## Dataset

`spam_data.csv` has 5,572 SMS messages. The notebook removes 403 duplicate messages, which leaves 5,169 unique ones:

- ham: 4,516 (about 87%)
- spam: 653 (about 13%)

The file uses two columns, `v1` (the label) and `v2` (the message text). Three more columns, `Unnamed: 2` to `Unnamed: 4`, are almost entirely empty, and the notebook drops them. The file is not valid UTF-8, so it has to be read with `encoding='latin-1'`.

## How it works

1. Load the CSV and keep only the label and message columns.
2. Rename the columns to `label` and `message`.
3. Drop duplicate messages, so repeated text does not end up in both the training and test sets.
4. Convert the labels to numbers: spam is 1, ham is 0.
5. Clean each message with `clean_text()`: lowercase it, remove digits, remove punctuation, and strip extra whitespace.
6. Turn the cleaned messages into TF-IDF vectors with `TfidfVectorizer`, using English stop word removal and a vocabulary capped at 5,000 words.
7. Split the data 80/20 into training and test sets with `random_state=43`.
8. Train a `MultinomialNB` model and evaluate it with accuracy and a classification report.
9. Check new messages with the `check_spam()` helper, which prints the prediction and the model's confidence.
10. Save the model and the vectorizer with `pickle` as `model.pkl` and `vectorizer.pkl`, which the Streamlit app loads.

### Example predictions

| Message                                             | Prediction | Probability |
| --------------------------------------------------- | ---------- | ----------- |
| Hey, are we still meeting tomorrow at 3pm?          | Not spam   | 99.28%      |
| URGENT: Your account has been compromised. Call us! | Spam       | 61.75%      |

## Streamlit app

`app.py` loads `model.pkl` and `vectorizer.pkl`, takes a pasted message, and shows whether it looks like spam along with the model's confidence. To run it locally:

```bash
streamlit run app.py
```

## Setup

You need Python 3. Install the packages listed in `requirements.txt` (streamlit, pandas, scikit-learn), plus Jupyter if you want to run the notebook:

```bash
pip install -r requirements.txt
pip install jupyter
```

Then open the notebook:

```bash
jupyter notebook model_train.ipynb
```

The data loading cell currently points to a local Windows path. Change it to the location of `spam_data.csv` on your machine, or to `spam_data.csv` if the notebook and the file sit in the same folder. The last cell of the notebook writes `model.pkl` and `vectorizer.pkl`.

## Project structure

```
spam-detection-model/
├── model_train.ipynb  # data prep, training, evaluation, saving the model
├── app.py             # Streamlit app
├── spam_data.csv      # labeled SMS messages
├── model.pkl          # saved Naive Bayes model
├── vectorizer.pkl     # saved TF-IDF vectorizer
├── requirements.txt   # packages for the deployed app
└── README.md
```

## Known limitations

- The model misses about a quarter of the spam in the test set (spam recall is 0.74).
- It was trained on one fixed dataset of English SMS messages, so it can miss spam written in newer styles. For example, a promotional message with local currency amounts and a betting site link was classified as not spam.

## Planned work

- Improve spam recall, which is the weak side of the current model.
- More work on the model and notebook in this repo.
