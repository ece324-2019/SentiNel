# f = open('sentiment_labels.txt','r')
# content = f.read().split()
# print(content)
# f.close()
import pandas as pd
import numpy
N=5000
with open("sentiment_labels.txt") as myfile:
    head = [next(myfile) for x in range(N)]
labels=[]
for i in range (1,len(head)):
    labels += [float(head[i].split('|')[1][:-1])]
# print(head)
# print(labels)

with open("datasetSentences.txt") as myfile:
    head = [next(myfile) for x in range(N)]
text=[]
for i in range (1,len(head)):
    text += [head[i].split('\t')[1][:-1]]
print(head)
print(text)
data = []
for i in range(N-1):
    data += [[labels[i],text[i]]]
# print(data)
df = pd.DataFrame(data)
df.to_csv('sentimenttreebank.csv', index=False)

# a = numpy.asarray([labels,text])
# numpy.savetxt("sentimenttreebank.csv", a, delimiter=",")
