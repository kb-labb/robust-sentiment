import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#df = pd.read_csv('test.csv').fillna('')
df = pd.read_csv('test.csv').dropna()
df = df.applymap(str)

# creating column text_number initializing with ''
df['sentiment'] = ''


#headlines = "/home/hilhag/robustly-sentimental/headlines/sample.csv"
#df = pd.read_csv(headlines, nrows=100)

use_cuda = torch.cuda.is_available()

#if use_cuda:
#    print('__CUDNN VERSION:', torch.backends.cudnn.version())
#    print('__Number CUDA Devices:', torch.cuda.device_count())
#    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
#    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

model_name = "/home/hilhag/robust-swedish-sentiment-multiclass/"
tokenizer = AutoTokenizer.from_pretrained('/home/hilhag/megatron-bert-large-swedish-cased-165k')
model = AutoModelForSequenceClassification.from_pretrained('/home/hilhag/robust-swedish-sentiment-multiclass/', num_labels=3, id2label=id2label, label2id=label2id)

#for headline in headline_list:
#    inputs = tokenizer(headline, return_tensors="pt")
#    with torch.no_grad():
#>        logits = model(**inputs).logits

#    predicted_class_id = logits.argmax().item()
#    print(model.config.id2label[predicted_class_id])


def predict(i):
    inputs = tokenizer(i, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


# getting the index where text is valid
index = df.loc[df['text'].str.strip().astype(bool)].index


#for i in range(len(index)):
#    sentiment = predict(df.loc[i]["text"])
#    print(sentiment)
#    #headline = str(df.loc[i])
#    #print(headline)
# finally creating the column text_number with increment as 0, 1, 2 ...
#df.loc[index, 'text_number'] = [f'text_{i}' for i in range(len(index))]
df.loc[index, 'sentiment'] = [predict(df.loc[i]["text"]) for i in range(len(index))]

#print(type(index))
#print(index)

print(df)
# save it to disk
#df.to_csv('output2.csv')
