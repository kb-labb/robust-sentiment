import json

annotation_file = "/home/hilhag/prjs/robustly-sentimental/data/raw/norec/metadata.json"

# Collect source keys
with open(annotation_file) as f:
    metadata = json.load(f)

source_keys = list(metadata.keys())

for source_key in source_keys[0:5]:
    file_name = "/home/hilhag/prjs/robustly-sentimental/data/raw/norec/translations/" + str(source_key) + ".txt"
    
