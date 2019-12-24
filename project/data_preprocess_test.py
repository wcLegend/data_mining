import pandas as pd
from numpy import *
import numpy as np
import  torch
print('start')



class Data_Preprocess(object):
    #这里设置了长度过滤 在5-20长度间的才被留下  max_length现为35
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.topic2vec = {}
        self.word2vec = {}
        self.sword2vec = {}
        self.word2vec_emb=torch.zeros(64)
        self.sword2vec_emb=torch.zeros(64)

        self.user_dic = {}
        self.que_dic = {}

        self.train_x = list()
        self.train_y = list()
        self.val_x = list()
        self.val_y = list()

        self.train_lengths = []
        self.val_lengths = []

        self.run()


    def get_vec_dic(self):
        topic = pd.read_csv('topic_vectors_64d.txt', header=None, sep='\t')
        topic2vec = {}
        #print(topic.values[0])
        for row in topic.values:
            #print(row[0],'====',row[1])
            topic2vec[row[0]] = torch.from_numpy(array(list(map(float, row[1].split(' ')))).astype(np.float32)) #把字符串列表转为浮点数tensor
            #break
            pass
        #print(topic2vec['T100000'])
        print('topic的个数： ',len(topic2vec))       
        self.topic2vec = topic2vec

        single_word = pd.read_csv('single_word_vectors_64d.txt', header=None, sep='\t')
        #for col in single_word.columns:
            #print(col, len(single_word[col].unique()))
        sword2vec = {}
        #print(single_word.values[-1])
        for row in single_word.values:
            #print(row[0],'====',row[1])
            sword2vec[row[0]] = list(map(float, row[1].split(' ')))
            #break
            pass
        #print(sword2vec['SW23239'])
        print('单字的个数:  ',len(sword2vec))       
        
        self.sword2vec = sword2vec

        emb_0 = torch.zeros(1,64).float()
        self.sword2vec_emb = torch.from_numpy(np.array(list(sword2vec.values()))).float()#转换成numpy
        self.sword2vec_emb=torch.cat([emb_0,self.sword2vec_emb],0)
        print('单字矩阵:    ', self.sword2vec_emb.size())



        word = pd.read_csv('word_vectors_64d.txt', header=None, sep='\t')
        #for col in single_word.columns:
            #print(col, len(single_word[col].unique()))
        word2vec = {}
        #print(word.values[-1])
        for row in word.values:
            #print(row[0],'====',row[1])
            word2vec[row[0]] = list(map(float, row[1].split(' ')))
            #break
            pass        

        #print(len(word2vec))
        #print(word2vec['W1762829'])     

        self.word2vec = word2vec
        print('切词的个数:    ',len(word2vec))

        emb_0 = torch.zeros(1,64).float()
        self.word2vec_emb = torch.from_numpy(np.array(list(word2vec.values()))).float()#转换成numpy
        self.word2vec_emb=torch.cat([emb_0,self.word2vec_emb],0)
        print('切词矩阵:    ', self.word2vec_emb.size())

        pass


    def get_use_dic(self):
        #=======================构建用户画像==========================
        sex_dic = {'male':0,'female':1,'unknown':2}
        feq_dic = {'monthly':0,'daily':1,'weekly':2, 'unknown':3, 'new':4}
        user_dic = {}  #用户字典，保存所有用户的vec
        user_info = pd.read_csv('member_info_0926_64.txt', header=None, sep='\t')
        #user_info = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
        #print(user_info.values[0])
        user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
        user_info = user_info.iloc[:,[0,1,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]  #删去那五条无用属性      
        

        #user_dic是所有用户画像向量的集合，每个人是140维度，前12维是user_info的维度，后128维是关注和感兴趣两个topic的向量        
        

        #print(user_info.values[0])
        for col in user_info.columns:
            print(col, len(user_info[col].unique()))
        #print(user_info['性别'].unique())
        #print(user_info['访问评率'].unique())
        for row in user_info.values:
            #print(row[0],'====',row[1])
            #print(row[1:])
            row[1] = sex_dic[row[1]]
            row[2] = feq_dic[row[2]]
            row[8] = int(row[8][2:])
            row[9] = int(row[9][2:])
            row[10] = int(row[10][2:])
            row[11] = int(row[11][2:])
            row[12] = int(row[12][2:])
            temp = torch.from_numpy(row[1:14].astype(np.float32))  #size = 13       

            #处理用户关注话题       

            f_topic_tensor = torch.zeros(64)
            #print(row[14])
            
            f_topic_list = row[14].split(',') #关注的topic 
            for t in f_topic_list:
                if t == '-1': #不存在就直接用0向量
                    break 
                    pass
                f_topic_tensor = torch.add(f_topic_tensor,self.topic2vec[t])
                pass
            temp = torch.cat((temp,f_topic_tensor), 0)      

            #print('================================')
            #print(temp)        

            #处理用户喜欢的话题
            l_topic_list = row[15].split(',') #喜欢的topic 根据喜欢程序进行加权
            like_num_all = 0
            l_topic_tensor = torch.zeros(64)
            for t in l_topic_list:
                if t == '-1':
                    like_num_all = 1
                    break
                    pass
                t_l = t.split(':')
                if t_l[1] == 'Infinity':   #存在nan的情况
                    print('INF=============')
                    l_topic_tensor = self.topic2vec[t_l[0]]
                    like_num_all = 1
                    break
                pass
                #print(topic2vec[t_l[0]])
                #print(t_l[1])
                temp_t = self.topic2vec[t_l[0]]*float(t_l[1])   #按权重加权平均所有用户感兴趣的topic
                #print(temp_t)
                l_topic_tensor = torch.add(l_topic_tensor,temp_t)
                #print(l_topic_tensor)
                like_num_all += float(t_l[1])
                pass
            l_topic_tensor = l_topic_tensor/like_num_all
            #print('out')
            #print(l_topic_tensor)      

            #print(temp.size()) #size = 76   
            temp = torch.cat((l_topic_tensor,temp),0)
            #print(temp.size())#size = 140
            
            user_dic[row[0]] = temp   
            #break
            pass        
        

        #print(user_dic)
        self.user_dic = user_dic
        print(user_dic)
        pass

        print('user_nums:    ',len(user_dic))

    def get_que_dic(self):
        #=======================构建问题向量==========================
        que_dic = {}  #问题字典     
        max_length_sc = 30
        max_length_st = 100
        max_length_wc = 15
        max_length_wt = 50
        lengths_sw = []
        lengths_w = []      

        max_length_s = max_length_sc+max_length_st
        max_length_w = max_length_wc+max_length_wt

        question_info = pd.read_csv('question_info_0926_64.txt', header=None, sep='\t')
        #question_info = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
        #print(question_info.values[0])
        question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
        question_info = question_info.iloc[:,[0,2,3,4,5,6]]  #除了单字编码序列，后面的都可能是-1
        for col in question_info.columns:
            print(col, len(question_info[col].unique()))        

        #print(question_info.values[0])      

        for row in question_info.values:
            qid = row[0]

            row[1] = row[1].replace('-1','0').replace('SW','')
            row[1] = [int(sw) for sw in row[1].split(',')]
            row[2] = row[2].replace('-1','0').replace('W','')
            row[2] = [int(w) for w in row[2].split(',')]
            row[3] = row[3].replace('-1','0').replace('SW','')
            row[3] = [int(sw) for sw in row[3].split(',')]
            row[4] = row[4].replace('-1','0').replace('W','')
            row[4] = [int(w) for w in row[4].split(',')]
            
            information = row[1:5] #问题的描述及标题序列

            #进行单字的信息处理
            t_sw = information[0]
            #print(t_sw)
            if len(t_sw)>max_length_st:
                t_sw = t_sw[0:max_length_st]
                #print(len(t_sw))
                pass
            c_sw = information[2]
            #print(c_sw)
            if len(c_sw)>max_length_sc:
                c_sw = c_sw[0:max_length_sc]
                #print(len(c_sw))
                pass
            ts_sw = t_sw+c_sw
            #print(ts_sw)
            if len(ts_sw)<max_length_st+max_length_sc:
                #print(len(ts_sw))
                lengths_sw.append(len(ts_sw))  #激励长度
                temp = [0 for _ in range(max_length_s-len(ts_sw))]
                ts_sw = ts_sw+temp
                #ts_sw = torch.LongTensor(ts_sw)
                #print(len(ts_sw))
                pass
                    

            #进行切词的信息处理
            t_w = information[1]
            #print(t_w)
            if len(t_w)>max_length_wt:
                t_w = t_w[0:max_length_wt]
                #print(len(t_w))
                pass
            c_w = information[3]
            #print(c_w)
            if len(c_w)>max_length_wc:
                c_w = c_w[0:max_length_wc]
                #print(len(c_w))
                pass        

            ts_w = t_w+c_w
            #print(ts_w)
            if len(ts_w)<max_length_wt+max_length_wc:
                #print(len(ts_w))
                lengths_w.append(len(ts_w))  #激励长度
                temp = [0 for _ in range(max_length_w-len(ts_w))]
                ts_w = ts_w+temp
                #print(len(ts_w))
                pass
                    

            information = (ts_sw,ts_w)      


            q_topic_list = row[5].split(',')
            q_topic_tensor = torch.zeros(64)
            for t in q_topic_list:
                if t == '-1': #不存在就直接用0向量
                    break 
                    pass
                q_topic_tensor = torch.add(q_topic_tensor,self.topic2vec[t])
                pass        

            #print('================================')
            #print(q_topic_tensor)      

            que_dic[qid] =(q_topic_tensor,information) #information包括单字和切词的内容
            #break      

        #print(que_dic['Q2234111670'][1])        
        self.que_dic = que_dic

        print('question_nums:    ',len(que_dic))
        pass

    def load_train(self):
        invite = pd.read_csv('invite_info_0926_64.txt', header=None, sep='\t')
        #invite = pd.read_csv('invite_info_0926.txt', header=None, sep='\t')
        invite.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
        for row in invite.values:
            self.train_x.append(row[0:2])
            self.train_y.append(row[3])
            pass
            #break
        pass
        #print(self.train_x)
        #print(self.train_y)
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        print('trian_nums:    ',len(self.train_x))

    def run(self):
        print('Loading vocabulary.')
        self.get_vec_dic()
        self.get_que_dic()
        self.get_use_dic()
        self.load_train()


#data = Data_Preprocess('',100)
