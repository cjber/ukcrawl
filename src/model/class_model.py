import evaluate
import numpy as np
import polars as pl
from datasets import Dataset
from evaluate import evaluator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

classification = pl.read_parquet(
    "./data/classification/classification.parquet",
    columns=["Primary Category", "content"],
).rename({"Primary Category": "label_name", "content": "text"})
id2label = dict(enumerate(set(classification["label_name"])))
label2id = {v: k for k, v in id2label.items()}
classification = classification.with_columns(
    pl.col("label_name").replace(label2id).cast(int).alias("label")
)


classification = Dataset.from_polars(classification)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokenized_class = classification.map(
    preprocess_function, batched=True
).train_test_split(test_size=0.2)  # type: ignore
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="weighted")


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="hfmodel",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_class["train"],
    eval_dataset=tokenized_class["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
