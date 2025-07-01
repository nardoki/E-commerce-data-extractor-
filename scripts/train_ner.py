from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import pandas as pd

# Load and parse CoNLL data
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

# Load training and validation data
train_df = load_conll("labeled_conll.txt")[:40]  
valid_df = train_df[-10:]  
train_df = train_df[:-10]

# Create label mappings
all_labels = sorted({label for seq in train_df["ner_tags"] for label in seq})
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}

# Convert labels to ids
train_df["ner_tags"] = [[label2id[tag] for tag in seq] for seq in train_df["ner_tags"]]
valid_df["ner_tags"] = [[label2id[tag] for tag in seq] for seq in valid_df["ner_tags"]]

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset})

# Load tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)


# Tokenize and align labels
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

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Training arguments
args = TrainingArguments(
    output_dir="amharic_ner_model",
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

trainer.train()
trainer.save_model("amharic_ner_model")
tokenizer.save_pretrained("amharic_ner_model")
