import pandas as pd
import numpy as np

# This script prepares version 2 of the ABSA-Imm corpus from Språkbanken Text for training.
# Input is three jsonl files: train, test and dev.
# We merge these files; the split into train and test will happen later in the pipeline.
# Output is a csv file text,label

train_file = "/path/to/train/file.json"
test_file = "/path/to/test/file.json"
dev_file = "/path/to/dev/file.json"

train_df = pd.read_json(train_file, lines=True)
test_df = pd.read_json(test_file, lines=True)
dev_df = pd.read_json(dev_file, lines=True)

df = pd.concat([train_df, test_df, dev_df], ignore_index=True) # Concatenate frames

df["label"] = df["label"].apply(np.ceil).astype(str) # Round the sentiment label up and convert to string in preparation for value replacement
df["label"] = df["label"].replace(["2.0", "3.0", "4.0", "5.0"], ["1.0", "2.0", "3.0", "3.0"]) # Normalize 1-5 values on a 1-3 scale where 1 is negative, 2 is neutral and 3 is positive
df["label"] = pd.to_numeric(df["label"])
df["label"] = df["label"].astype(int)
df = df[["text", "label"]] # Throw away all data except text and label


df.to_csv("/path/to/outfile.csv") # Save as csv
