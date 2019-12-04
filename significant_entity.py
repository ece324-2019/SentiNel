import pandas as pd

data = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesL.txt', sep='\t')
for i in range(0, 3000):
    print(i)
    label = ''
    sal = 0.0
    thing = data['result'][i][1:len(data['result'][i])-1]
    res = thing.strip('][').split('], [')
    for l in res:
        li = l.strip('][').split(', ')
        stuff = float(li[2][1:len(li[2])])
        if stuff > sal:
            sal = stuff
            label = li[0][1:len(li[0])-1]
        data['label'][i] = label

data.to_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesFinalist.txt', sep='\t')

