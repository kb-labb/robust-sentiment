from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.utils import shuffle
import evaluate
import numpy as np
from datasets import load_metric, load_dataset, Dataset
import pandas as pd
import os
import torch

torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


df1 = pd.read_csv("/home/hilhag/data_network/hillevi/sentiment_data/twitter-data.csv")
df2 = pd.read_csv("/home/hilhag/data_network/hillevi/sentiment_data/absa-imm.csv")
df3 = pd.read_csv("/home/hilhag/data_network/hillevi/sentiment_data/trustpilot.csv")
df4 = pd.read_csv("/home/hilhag/data_network/hillevi/sentiment_data/norec-chunked.csv")
df4 = pd.read_csv("/home/hilhag/data_network/hillevi/sentiment_data/norec.csv")
df5 = pd.read_csv("/home/hilhag/data_network/hillevi/sentiment_data/news.csv", on_bad_lines="skip")

df5["label"].value_counts() # sanity check

# drop id 
df1 = df1[["text", "label"]]
df2 = df2[["text", "label"]]
df3 = df3[["text", "label"]]
df4 = df4[["text", "label"]]
df5 = df5[["text", "label"]]

# experimented with different weights for the datasets based on their quality but i ended up not needing to
weight1 = 1
weight2 = 1
weight3 = 1
weight4 = 1
weight5 = 1

# combine the datasets with different weights
combined_dataset = pd.concat([df1.sample(frac=weight1), df2.sample(frac=weight2), df3.sample(frac=weight3), df4.sample(frac=weight4), df5.sample(frac=weight5)], ignore_index=True)

combined_dataset.dropna(inplace=True)

# for binary model only
# combined_dataset = combined_dataset[combined_dataset.label != 1]
# combined_dataset['label'] = combined_dataset['label'].replace([2], 1)
#print(combined_dataset["label"].value_counts())

# shuffle
combined_dataset = shuffle(combined_dataset)

binary_id2label = {0: "NEGATIVE", 1: "POSITIVE"}
binary_label2id = {"NEGATIVE": 0, "POSITIVE": 1}

multiclass_id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
multiclass_label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

dataset = Dataset.from_pandas(combined_dataset)
dataset = dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=3, id2label=multiclass_id2label, label2id=multiclass_label2id)
#model = AutoModelForSequenceClassification.from_pretrained("KBLab/megatron-bert-large-swedish-cased-165k", num_labels=2, id2label=binary_id2label, label2id=binary_label2id)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    # probabilities = tf.nn.softmax(logits)
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("KBLab/megatron-bert-large-swedish-cased-165k", model_max_length=512)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_train = dataset["train"].map(preprocess_function, batched=True)
tokenized_test = dataset["test"].map(preprocess_function, batched=True)

repo_name = "./repo-name"

training_args = TrainingArguments(
    output_dir=repo_name,
    logging_dir="./logs",
    #learning_rate=2e-2, # do default instead
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="steps",
    push_to_hub=True,
    logging_steps=10000, # reduce number of steps probably 
    save_steps=10000,
    evaluation_strategy="steps",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
print(metrics)
#trainer.save_model("sentiment_model/name")
#trainer.push_to_hub()
#metrics = trainer.evaluate()

