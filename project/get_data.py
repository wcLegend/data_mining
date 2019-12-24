# coding: utf-8
#这部分主要是合并ans和ques，作为人物画像的补充
import pandas as pd
print('start')
"""

user_info = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
print(user_info.values[0])
user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']

user_info = user_info.iloc[:,[0,1,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]  #删去那五条无用属性
print(user_info.values[0])
for col in user_info.columns:
	print(col, len(user_info[col].unique()))





question_info = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
print(question_info.values[0])
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
for col in question_info.columns:
	print(col, len(question_info[col].unique()))





answer_info = pd.read_csv('answer_info_0926.txt', header=None, sep='\t')
print(answer_info.values[0])
answer_info.columns = ['回答id','问题id','作者id','回答创建时间','回答内容的单字编码顺序','回答内容的切词编码顺序','回答内容是否被标优','回答是否被推荐','回答是否被收入圆桌','回答是否包含图片','回答是否包含视频','回答的内容字数','回答收到的点赞数','回答收到的取赞数','回答收到的评论数','回答收藏数','回答收到的感谢数','回答收到的被举报数','回答收到的没有帮助数','回答收到的反对数']
for col in answer_info.columns:
	print(col, len(answer_info[col].unique()))
#print(answer_info[0])   #错误的语法
"""

"""
#test 测试表的链接
question_info_sample = question_info.iloc[0:20,[0,2,3,4,5,6]]
#print(question_info_sample.values)
answer_info_sample = answer_info.iloc[0:20,:]
#print(answer_info_sample.values)

answer_info_sample_plus = answer_info_sample.merge(question_info_sample, on='问题id',how='left')
for col in answer_info_sample_plus.columns:
	print(col, len(answer_info_sample_plus[col].unique()))
"""

"""
answer_info_link_plus = answer_info.merge(question_info, on='问题id',how='left')
for col in answer_info_link_plus.columns:
	print(col, len(answer_info_link_plus[col].unique()))
"""

#answer_info_link_plus.to_csv('ans_que.txt',header=0,index =0,sep='\t') #将两个数据集的级联结果合一


print("new_doc========================================================")
answer_info_link_plus = pd.read_csv('ans_que.txt', header=None, sep='\t')
answer_info_link_plus.columns =['回答id','问题id','作者id','回答创建时间','回答内容的单字编码顺序','回答内容的切词编码顺序','回答内容是否被标优','回答是否被推荐','回答是否被收入圆桌','回答是否包含图片','回答是否包含视频','回答的内容字数','回答收到的点赞数','回答收到的取赞数','回答收到的评论数','回答收藏数','回答收到的感谢数','回答收到的被举报数','回答收到的没有帮助数','回答收到的反对数','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']

for col in answer_info_link_plus.columns:
	print(col, len(answer_info_link_plus[col].unique()))


for uid in answer_info_link_plus['作者id'].unique():
	print(uid)
	b = answer_info_link_plus.loc[answer_info_link_plus['作者id']==uid]
	#b.to_csv('member_ans/'+uid+'.txt',header=0,index =0,sep='\t')
	print(b)
	break
	pass
"""
q1 = pd.read_csv('member_ans/M625498202.txt', header=None, sep='\t')
q1.columns =['回答id','问题id','作者id','回答创建时间','回答内容的单字编码顺序','回答内容的切词编码顺序','回答内容是否被标优','回答是否被推荐','回答是否被收入圆桌','回答是否包含图片','回答是否包含视频','回答的内容字数','回答收到的点赞数','回答收到的取赞数','回答收到的评论数','回答收藏数','回答收到的感谢数','回答收到的被举报数','回答收到的没有帮助数','回答收到的反对数','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
print(q1)
"""