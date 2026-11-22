# Spam detection model

A Multinomial Naive Bayes classifier that labels SMS messages as spam or ham (not spam). The whole workflow is in one notebook, `model.ipynb`: load the data, remove duplicates, turn the text into word counts, train the model, and test it on new messages.

A Streamlit interface is planned so anyone can paste a message and check whether it is spam.

## Results

After removing duplicate messages, the model scores 98.07% accuracy on a held-out test set of 1,034 messages (20% of the data).

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Ham   | 0.99      | 0.99   | 0.99     | 895     |
| Spam  | 0.92      | 0.94   | 0.93     | 139     |

The data is imbalanced, so accuracy alone says little: a model that always predicted ham would score about 87%. The spam precision and recall are the numbers to watch. When the model says spam it is right 92% of the time, and it catches 94% of the actual spam.

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
5. Turn each message into a vector of word counts with `CountVectorizer`, using English stop word removal and a vocabulary capped at the 5,000 most frequent words.
6. Split the data 80/20 into training and test sets with `random_state=43`.
7. Train a `MultinomialNB` model and evaluate it with accuracy and a classification report.
8. Check new messages with the `check_spam()` helper, which prints the prediction and the model's confidence.

### Example predictions

| Message                                             | Prediction | Probability |
| --------------------------------------------------- | ---------- | ----------- |
| Congratulations! You won a free iPhone. Click now!  | Spam       | 99.77%      |
| Hey, are we still meeting tomorrow at 3pm?          | Not spam   | 99.97%      |
| URGENT: Your account has been compromised. Call us! | Spam       | 83.75%      |

## Setup

You need Python 3 and these packages:

```bash
pip install pandas scikit-learn jupyter
```

Then open the notebook:

```bash
jupyter notebook model.ipynb
```

The data loading cell currently points to a local Windows path. Change it to the location of `spam_data.csv` on your machine, or to `spam_data.csv` if the notebook and the file sit in the same folder.

## Project structure

```
spam-detection-model/
├── model.ipynb      # data prep, training, evaluation, predictions
├── spam_data.csv    # labeled SMS messages
└── README.md
```

## Known limitations

- The model is not saved to disk, so it has to be retrained each time the notebook runs.

## Planned work

- A Streamlit app where a user types a message and gets a spam or not spam result.
- More work on the model and notebook in this repo.
