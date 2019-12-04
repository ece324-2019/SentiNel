import pandas as pd

data = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesL.txt', sep='\t')
for i in range(0, data.shape[0]):
    sentence = data['sentence']
    words = str.split(' ')
    for word in words:
        for l in list(data['result']):
            if word == l[0]:
                label += [1]
            else:
                label += [0]
    data['label'][i] = str(label)

data.to_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesFinalist.txt', sep='\t')

