import pandas as pd

data = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesFinal.txt', sep='\t')

for i in range(0, 11847):
    print(i)

    if data['result'][i] == '[]':
        print(type(data['result'][i]))
        print(data['result'][i])
        data = data.drop([i])
    else:
        for l in list(data['result']):
            if len(l[0].split(' ')) > 1:
                data = data.drop([i])
                break

data.to_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesL.txt', sep='\t', index='False')
