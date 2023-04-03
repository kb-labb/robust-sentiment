file = open("/home/gilleti/Documents/robustly-sentimental/data/raw/news_to_be_annotated.csv", "r")

datalist = []

for line in file:
    items = line.split("\t")
    if len(items) > 2:
        text = items[0][2:]
        text = text.strip()
        if text:
            paper = items[1]
            paper = paper.strip()
            date = items[2]
            date = date.strip()q
            thestring = text + ", " + paper + ", " + date
            datalist.append(thestring)

for string in datalist:
    print(string)