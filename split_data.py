import pandas as pd
from sklearn.model_selection import train_test_split

# dataset = pd.read_csv('./data/sentiment140.csv', delimiter=',')
# pd.DataFrame.to_csv(dataset,'overfit.tsv', sep='\t', index=False)

dataset1 = pd.read_csv('./data/all.csv', delimiter=',',encoding='latin-1')
# dataset2 = pd.read_csv('./data/sentimenttreebank.csv', delimiter=',')
# dataset3 = pd.concat([dataset1,dataset2])
print((dataset1))

remain, train_data = train_test_split(dataset1, test_size = 0.9, random_state = 1, stratify = dataset1['label'])
valid_data, test_data = train_test_split(remain, test_size = 0.5, random_state = 1,stratify = remain['label'])

a = (train_data['label'].isin(["1"]).sum())
b = (train_data['label'].isin(["0"]).sum())
c = (valid_data['label'].isin(["1"]).sum())
d = (valid_data['label'].isin(["0"]).sum())
e = (test_data['label'].isin(["1"]).sum())
f = (test_data['label'].isin(["0"]).sum())
g = (train_data['label'].isin(["0.5"]).sum())
h = (valid_data['label'].isin(["0.5"]).sum())
i = (test_data['label'].isin(["0.5"]).sum())


print("Train data | Positive:",a, "| Negative: ",b,"| Neutral: ",g)
print("Valid data | Positive:",c, " | Negative: ",d,"| Neutral: ",h)
print("Test data  | Positive:",e, "| Negative: ",f,"| Neutral: ",i)


#
# remain, train_data2 = train_test_split(dataset2, test_size = 5000, random_state = 1, stratify = dataset2['label'])
# remain, test_data2 = train_test_split(remain, test_size = 2500, random_state = 1,stratify = remain['label'])
# remain, valid_data2 = train_test_split(remain, test_size = 2500, random_state = 1, stratify = remain['label'])


pd.DataFrame.to_csv(train_data,'train.tsv', sep='\t', index=False)
pd.DataFrame.to_csv(test_data,'test.tsv', sep='\t', index=False)
pd.DataFrame.to_csv(valid_data,'validation.tsv', sep='\t', index=False)

