import datasets
from transformers import pipeline, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Load the datasets
dataset1 = datasets.load_dataset('csv', data_files='dataset1.csv')['train']
dataset2 = datasets.load_dataset('csv', data_files='dataset2.csv')['train']

# Assign weights to datasets based on their quality
weights = [0.7, 0.3]  # dataset1 has higher quality, so we assign it a higher weight

# Combine the datasets with assigned weights
dataset = dataset1.shuffle().train_test_split(test_size=0.2)['train'].select(range(500)) + dataset2.shuffle().train_test_split(test_size=0.2)['train'].select(range(200))
dataset = dataset.shuffle()

# Split the data into training and testing datasets
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2)

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define the pipeline for sentiment analysis
pipeline_sentiment = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define the training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    learning_rate=1e-5,              # learning rate
    weight_decay=0.01,               # weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end=True,
    evaluation_strategy='epoch',
    metric_for_best_model='accuracy',
    greater_is_better=True
)

# Define the function to compute the weighted loss
def compute_weighted_loss(logits, targets):
    loss = np.mean(-targets * weights[0] * np.log(logits[:, 0]) - targets * weights[1] * np.log(logits[:, 1]))
    return loss

# Define the Trainer with the weighted loss function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {'accuracy': (p['preds'].argmax(axis=1) == p['label_ids']).mean()},
    train_loss=compute_weighted_loss
)

# Train the model
trainer.train()

# Evaluate the model
result = trainer.evaluate(eval_dataset=test_dataset)
print(result)

# Use the pipeline to make predictions on new text
print(pipeline_sentiment("This is a positive review."))
print(pipeline_sentiment("This is a negative review."))
