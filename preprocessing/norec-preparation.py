import pandas as pd
import numpy as np
import glob


# All files and directories ending with .txt and that don't begin with a dot:
csv_files = glob.glob("/home/adam/*.txt")




# This script processes tweets from https://github.com/niklas-palm/sentiment-classification/ in preparation for fine-tuning of a larger sentiment model.
# Indata is training set and test set. These are merged here. Split into train and test will be perfomed at a later stage in the pipeline.
# Output is csv file of text,label where 1 = negative, 2 = neutral, 3 = positive.

train_file = "../path/to/train/file.csv"
test_file = "../path/to/test/file.csv"

#train_df = pd.read_csv(train_file, delimiter = ",", quotechar = "|", encoding = "utf8")
#test_df = pd.read_csv(test_file, delimiter = ",", quotechar = "|", encoding = "utf8")

#df = pd.concat([train_df, test_df], ignore_index=True)

#df.rename(columns={'Tweet': 'text', 'Sentiment': 'label'}, inplace=True) # Rename columns

# Original data is in format: -1 = negative, 0 = neutral, 1 = positive
# Here we normalize the scale so that 1 = negative, 2 = neutral, 3 = positive

#df["label"] = df["label"].astype(str) # Convert to string in preparation for replace method
#df["label"] = df["label"].replace(["-1", "0", "1"], ["1", "2", "3"])
#df["label"] = pd.to_numeric(df["label"])
#df["label"] = df["label"].astype(int)

#df.to_csv("/path/to/out/file.csv")
