from Entity_Label_Assist import sample_analyze_entity_sentiment
import pandas as pd

treebank_data = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentences.txt', delimiter='\t')
print(treebank_data.head)
print(treebank_data.shape)

for i in range(0,11853):
    sentence = treebank_data['sentence'][i]
    results = sample_analyze_entity_sentiment(sentence)
    treebank_data['result'][i] = results

treebank_data.to_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled.txt', sep='\t', index='False')
