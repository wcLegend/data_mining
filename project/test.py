import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sys

from imblearn.over_sampling import RandomOverSampler

train_x = list()
train_y = list()
train = pd.read_csv('invite_info_0926.txt', header=None, sep='\t')
train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
for row in train.values:
    train_x.append(list(row[0:2]))
    train_y.append(row[3])
    pass
train_x = np.array(train_x)
train_y = np.array(train_y)
print(sum(train_y==1))



sys.exit()

dataDf1=pd.DataFrame({'问题id':['Q1','Q2','Q1','Q4'], 
                     'value':[1,2,0,4],'value2':[-4,-3,-2,-1]})

dataDf1['value3'] = dataDf1['value']/dataDf1['value2']

t1 = dataDf1.groupby('问题id')['value'].agg(['describe']).reset_index()

print(t1)



a = np.array([[1,2,3,],[2,3,4]])

print(a[:,1])



df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
    
 
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])
    
 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                    index=[8, 9, 10, 11])


frames = [df1, df2, df3]
 
result = pd.concat(frames)

result['E'] = np.array(['A8', 'A9', 'A10', 'A11','A8', 'A9', 'A10', 'A11','A8', 'A9', 'A10', 'A11'])
#result['E'] = pd.DataFrame({'E': ['A8', 'A9', 'A10', 'A11','A8', 'A9', 'A10', 'A11','A8', 'A9', 'A10', 'A11']})

print(result)
sys.exit()


a = [-1.1,1.2]
a = np.array(a).astype(np.float32)
print(sum(abs(a)))

a = torch.zeros(1,64)
print(a.size())

a = {'a':1,'n':2}
print(list(a.values()))





#stake 是新建一个维度  cat是在原维度基础上拼接
a = [torch.LongTensor([[4,3,2,0]]),torch.LongTensor([[0,2,4,5]])]
print(a[0].size())
b = torch.zeros(1,4).long()

a = torch.stack([a[0],a[1],b],0)
print(a.size())
a = '-1'
f_topic_list = a.split(',') #关注的topic 
print(f_topic_list)




dataDf1=pd.DataFrame({'问题id':['Q1','Q2','Q3','Q4'], 
                     'value':[1,2,3,4],'value2':[-1,-2,-3,-4]})
dataDf2=pd.DataFrame({'问题id':['Q1','Q1','Q3','Q4','Q2','Q5'],
                     'value':[5,6,7,8,9,10]})

print(dataDf1)
print(dataDf2)

dataLfDf=dataDf1.merge(dataDf2.iloc[:,0:3], on='问题id',how='right')
print(dataLfDf)
print(dataLfDf.iloc[:,[0,2,3]])
#print(dataLfDf)
#dataLfDf.to_csv('test.txt',header=0,index =0,sep='\t')
a = pd.read_csv('test.txt', header=None, sep='\t')
a.columns =['问题id','value_x','value2','value_y']
print(a.values)
for col in a.columns:
    print(col, len(a[col].unique()))

for uid in a['问题id'].unique():
	print(uid)
	b= a.loc[a['问题id']==uid]
	b.to_csv(uid+'.txt',header=0,index =0,sep='\t')
	print(b)
	pass

q1 = pd.read_csv('Q5.txt', header=None, sep='\t')
print(q1)