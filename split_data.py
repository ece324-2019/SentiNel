import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./data/sentiment140.csv', delimiter=',')
pd.DataFrame.to_csv(dataset,'overfit.tsv', sep='\t', index=False)

# train_data, remain = train_test_split(dataset, test_size = 0.36, random_state = 1, stratify = dataset['label'])
# test_data, valid_data = train_test_split(remain, test_size = (16/36), random_state = 1,stratify = remain['label'])
# remain,overfit = train_test_split(dataset, test_size = 0.005, random_state = 1, stratify = dataset['label'])

# a = (train_data['label'].isin(["1"]).sum())
# b = (train_data['label'].isin(["0"]).sum())
# c = (valid_data['label'].isin(["1"]).sum())
# d = (valid_data['label'].isin(["0"]).sum())
# e = (test_data['label'].isin(["1"]).sum())
# f = (test_data['label'].isin(["0"]).sum())
# g = (overfit['label'].isin(["1"]).sum())
# h = (overfit['label'].isin(["0"]).sum())
# print("Train data | Subjective:",a, "| Objective: ",b)
# print("Valid data | Subjective:",c, " | Objective: ",d)
# print("Test data  | Subjective:",e, "| Objective: ",f)
# print("Overfit    | Subjective:",g, "  | Objective: ",h)

# pd.DataFrame.to_csv(train_data,'train.tsv', sep='\t', index=False)
# pd.DataFrame.to_csv(test_data,'test.tsv', sep='\t', index=False)
# pd.DataFrame.to_csv(valid_data,'validation.tsv', sep='\t', index=False)
# pd.DataFrame.to_csv(overfit,'overfit.tsv', sep='\t', index=False)

