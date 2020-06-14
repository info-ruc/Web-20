
import pandas as pd

data = pd.read_csv("./spider_data.csv")
data = data[['label', 'text']]
length = data.shape[0]
print(length)
print(data.head())

train = data[:int(length*0.8)]
dev = data[int(length*0.8)+1:int(length*0.9)]
test = data[int(length * 0.9)+1:]

train.to_csv("./train.tsv", sep="\t")
dev.to_csv("./dev.tsv", sep="\t")
test.to_csv("./test.tsv",sep="\t")
