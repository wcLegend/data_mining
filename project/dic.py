
import pandas as pd

topic = pd.read_csv('topic_vectors_64d.txt', header=None, sep='\t')
topic2vec = {}
print(topic.values[0])
for row in topic.values:
	#print(row[0],'====',row[1])
	topic2vec[row[0]] = list(map(float, row[1].split(' ')))  #把字符串列表转为浮点数列表
	#break
	pass
print(topic2vec['T100000'])
print(len(topic2vec))


single_word = pd.read_csv('single_word_vectors_64d.txt', header=None, sep='\t')
#for col in single_word.columns:
    #print(col, len(single_word[col].unique()))
sword2vec = {}
print(single_word.values[-1])
for row in single_word.values:
	#print(row[0],'====',row[1])
	sword2vec[row[0]] = list(map(float, row[1].split(' ')))
	#break
	pass
print(sword2vec['SW23239'])
print(len(sword2vec))


word = pd.read_csv('word_vectors_64d.txt', header=None, sep='\t')
#for col in single_word.columns:
    #print(col, len(single_word[col].unique()))
word2vec = {}
print(word.values[-1])
for row in word.values:
	#print(row[0],'====',row[1])
	word2vec[row[0]] = list(map(float, row[1].split(' ')))
	#break
	pass

print(len(word2vec))

print(word2vec['W1762829'])