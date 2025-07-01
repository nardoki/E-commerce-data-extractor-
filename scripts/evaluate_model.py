from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report
import pandas as pd
import torch

# Load model and tokenizer
model_path = "amharic_ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Reload validation dataset
def load_conll(path):
    sentences, tags = [], []
    with open(path, encoding="utf-8") as f:
        tokens, labels = [], []
        for line in f:
            if line.strip() == "":
                if tokens:
                    sentences.append(tokens)
                    tags.append(labels)
                    tokens, labels = [], []
            else:
                token, label = line.strip().split()
                tokens.append(token)
                labels.append(label)
        if tokens:
            sentences.append(tokens)
            tags.append(labels)
    return pd.DataFrame({"tokens": sentences, "ner_tags": tags})

valid_df = load_conll("scripts/labeled_conll.txt")[-10:]

# Rebuild label map
unique_tags = sorted({tag for doc in valid_df["ner_tags"] for tag in doc})
label2id = {label: i for i, label in enumerate(unique_tags)}
id2label = {i: label for label, i in label2id.items()}

valid_df["ner_tags"] = [[label2id[tag] for tag in seq] for seq in valid_df["ner_tags"]]
valid_dataset = Dataset.from_pandas(valid_df)

# Tokenize
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_valid = valid_dataset.map(tokenize_and_align_labels, batched=True)

trainer = Trainer(model=model, tokenizer=tokenizer)
predictions, labels, _ = trainer.predict(tokenized_valid)
preds = np.argmax(predictions, axis=2)

true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100]
              for pred, label in zip(preds, labels)]

print(classification_report(true_labels, true_preds))
