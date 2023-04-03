import pandas as pd
import numpy as np

# This script processes tweets from https://github.com/niklas-palm/sentiment-classification/ in preparation for fine-tuning of a larger sentiment model.
# Indata is training set and test set. These are merged here. Split into train and test will be perfomed at a later stage in the pipeline.
# Output is csv file of text,label where 1 = negative, 2 = neutral, 3 = positive.

#train_file = "../path/to/file.csv"
#test_file = "../path/to/test/file.csv"

train_file = "../data/raw/twitter_train_set.csv"
test_file = "../data/raw/twitter_balanced_test_set.csv"

train_df = pd.read_csv(train_file, delimiter = ",", quotechar = "|", encoding = "utf8")
test_df = pd.read_csv(test_file, delimiter = ",", quotechar = "|", encoding = "utf8")

df = pd.concat([train_df, test_df], ignore_index=True)

df["Tweet"] = df["Tweet"].str.strip()

df.rename(columns={'Tweet': 'text', 'Sentiment': 'label'}, inplace=True) # Rename columns

# Original data is in format: -1 = negative, 0 = neutral, 1 = positive
# Here we normalize the scale so that 1 = negative, 2 = neutral, 3 = positive

df["label"] = df["label"].astype(str) # Convert to string in preparation for replace method
df["label"] = df["label"].replace(["-1", "0", "1"], ["1", "2", "3"])
df["label"] = pd.to_numeric(df["label"])
df["label"] = df["label"].astype(int)


#df.to_csv("/path/to/out/file.csv")
df.to_csv("../data/processed/twitter-data.csv")


# NOTE TO SELF
# I also did this in the terminal after execution because I forgot in the actual Python code
#sed -i -r 's@^\w+,@@g' test.csv
# To remove first column of id in csv file
