from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import warnings
import json
import torch
from tqdm import tqdm
import transformers

# Compile NoReC data and outputs machine translated txt files from Norweigan to Swedish in the same format as the original corpus. 
# Hopefully we never have to run this spaghetti mess again.

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-no-sv")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-no-sv")

class CaptionDataset(Dataset):
    def __init__(self, df, tokenizer_name):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = df.loc[index, "sentence"]

        tokens = self.tokenizer(
            sentence, return_tensors="pt", padding="max_length", max_length=512
        )

        return tokens["input_ids"].squeeze_(0)


annotation_file = "/norec/data/metadata.json"
 
with open(annotation_file) as f:
    metadata = json.load(f)

sources = list(metadata.keys())
for source in sources:
    lang = metadata[source]["language"]
    if lang == "nb": # bokmål vs nynorsk
        rating = metadata[source]["rating"]
        path = metadata[source]["split"]
        file_name = "/norec/data/" + path + "/" + str(source) + ".txt"
        text = open(file_name, 'r').readlines()
        document = []
        for sentence in text:
            if sentence != "\n":
                document.append(sentence)
        df = pd.DataFrame(document, columns = ["sentence"])
        test_data = CaptionDataset(df, "Helsinki-NLP/opus-mt-no-sv")
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, prefetch_factor=2)
        model.to(device)
        a = next(iter(test_dataloader))
        with torch.no_grad():
            decoded_tokens = []
            for i, batch in enumerate(tqdm(test_dataloader)):
                output_tokens = model.generate(batch.to(device))
                decoded_tokens += tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                df["sentence_swedish"] = decoded_tokens
                swe_sents = df["sentence_swedish"].tolist()
                translated_text = " ".join(swe_sents)
                file_name_rating = source + "_" + str(rating)
                with open("./translated_texts/" + file_name_rating + ".txt", 'w') as f:
                    f.write(translated_text)
    else:
        continue
