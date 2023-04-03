from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "test-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

#text = "Jag misstänker att nåt är jävligt fel här. Det hela liknar nästintill en konspiration."
text = "Fantastiskt! Bra för dig, bra för världen. Otroligt fint"
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
predictions = outputs.logits.softmax(dim=-1)

predicted_class = torch.argmax(predictions, dim=-1).item()

print(predicted_class)
