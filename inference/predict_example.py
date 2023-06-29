from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipe
line


tokenizer = AutoTokenizer.from_pretrained("KBLab/megatron-bert-large-swedish-cas
ed-165k")
model = AutoModelForSequenceClassification.from_pretrained("KBLab/robust-swedish
-sentiment-multiclass")

text = "Rihanna uppges gravid"
tokenized_text = tokenizer(text)

classifier = pipeline("sentiment-analysis", model=model)
print(classifier(text))
