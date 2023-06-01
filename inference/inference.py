import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

## TODO()
## REWRITE THIS SCRIPT TO BE A MINIMAL NON-PROJECT SPECIFIC 


#df_test = pd.read_csv('test.csv').fillna('')
df = pd.read_csv("/home/hilhag/robustly-sentimental/headlines/sample.csv")
#df = df.applymap(str)

df['SENTIMENT'] = '' # initialize column SENTIMENT with empty string

#headlines = "/home/hilhag/robustly-sentimental/headlines/sample.csv"
#df = pd.read_csv(headlines, nrows=100)

use_cuda = torch.cuda.is_available()

# some info
#if use_cuda:
#    print('__CUDNN VERSION:', torch.backends.cudnn.version())
#    print('__Number CUDA Devices:', torch.cuda.device_count())
#    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
#    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# some labels
# i'm not sure if i should write the int or the string to file
# str less efficient but probably better for readability reasons
# ...you never know who will be looking at the file

id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

model_name = "/home/hilhag/robust-swedish-sentiment-multiclass/"
tokenizer = AutoTokenizer.from_pretrained('/home/hilhag/megatron-bert-large-swedish-cased-165k')
model = AutoModelForSequenceClassification.from_pretrained('/home/hilhag/robust-swedish-sentiment-multiclass/', num_labels=3, id2label=id2label, label2id=label2id)

def predict(i):
    """ Predict
    """
    inputs = tokenizer(i, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


# getting index where text is valid
index = df.loc[df['HEADLINE'].str.strip().astype(bool)].index
df.loc[index, 'SENTIMENT'] = [predict(df.loc[i]["HEADLINE"]) for i in range(len(index))]

df.to_csv('test.csv') # save to disk
# maybe do this continuously instead
