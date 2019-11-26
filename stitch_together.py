import pandas as pd

data1 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled1.txt', delimiter='\t')
data2 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled2.txt', delimiter='\t')
data3 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled3.txt', delimiter='\t')
data4 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled4.txt', delimiter='\t')
data5 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled5.txt', delimiter='\t')
data6 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelled678.txt', delimiter='\t')
data7 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelledRIP.txt', delimiter='\t')
data8 = pd.read_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesLabelledLAST.txt', delimiter='\t')

stuff = pd.concat([data1[:1000], data2[1001:2000], data3[2001:3000], data4[3001:4000], data5[4001:5000], data6[5001:8000], data7[8001:10000], data8[10001:11855]])
stuff.to_csv('C:\\Users\\Sonali\\PycharmProjects\\ece324\\FinalProject\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentencesFinal.txt', sep='\t', index='False')