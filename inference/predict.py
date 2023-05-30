from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import pandas as pd

import torch
torch.cuda.is_available()

model_name = "/home/hilhag/prjs/robustly-sentimental/models/sentiment_model/sentiment-model_4/"
tokenizer = AutoTokenizer.from_pretrained("KBLab/megatron-bert-large-swedish-cased-165k")
#model = AutoModel.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

headlines = "/home/hilhag/prjs/emotional-headlines/retriever/clean/cleaned_headlines.csv_2"

#text = "Rihannas uppges gravid"
df = pd.read_csv(headlines, nrows=100)
headline_list = df["HEADLINE"].tolist()

for headline in headline_list:
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
    print(headline)
    print(classifier(headline))
    

#tokenized_text = tokenizer(text)
#print(tokenized_text)

#classifier = pipeline("sentiment-analysis", model=model_name)
#print(classifier(text))
#classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
#print(classifier(text))
#[{'label': '5 stars', 'score': 0.7273}]
