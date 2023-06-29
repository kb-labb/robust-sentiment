import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This is basically an inspiration script showing some ways to perform inference on a gpu. Some modifications of this script is necessary if you're actually doing it. Inputs csv file of texts, outputs csv file with sentiment column added.

df = pd.read_csv("data.csv") # csv file of texts but if we want to actually do this on a gpu you should use a dataset

df['SENTIMENT'] = '' # initialize column SENTIMENT with empty string

use_cuda = torch.cuda.is_available()

# print some info
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"} # we don't actually nee this
label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

tokenizer = AutoTokenizer.from_pretrained('KBLab/megatron-bert-large-swedish-cased-165k')
model = AutoModelForSequenceClassification.from_pretrained('KBLab/robust-swedish-sentiment-multiclass/', num_labels=3, id2label=id2label, label2id=label2id)

def predict(i):
    """ Predict"""
    inputs = tokenizer(i, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


# getting index where text is valid
index = df.loc[df['text'].str.strip().astype(bool)].index
df.loc[index, 'SENTIMENT'] = [predict(df.loc[i]["text"]) for i in range(len(index))]

df.to_csv("sentiments.csv") # save to disk
