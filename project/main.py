
import sys
import torch
from data_preprocess import Data_Preprocess
from torch import optim
import torch.nn as nn
import time
import numpy as np
from MLP import mlp
from w_encoder import wEncoder
from sw_encoder import swEncoder

def save_train_l(str,path):
    path = path
    file_l = open(path,'a',encoding='utf-8')
    file_l.write(str)
    file_l.write('\n')
    pass

if __name__ == "__main__":

	use_cuda = torch.cuda.is_available()

	data = Data_Preprocess('')
	batch_size = 64
	hidden_size = 64 
	userfea_vec_size = 13  #用户除了topic之外的特征
	word_size = len(data.word2vec)+1
	sword_size = len(data.sword2vec)+1
	weight_decay = 1e-2
	learning_rate = 0.0001
	num_iters = 20
	all_hidden_size = userfea_vec_size+hidden_size*2+hidden_size*3 #前一个是user的特征，后一个是问题特征
	fold_size = len(data.train_x)

	wEncoder = wEncoder(hidden_size,data.word2vec_emb,word_size, num_layers=1, batch_size=batch_size)
	swEncoder = swEncoder(hidden_size,data.sword2vec_emb,sword_size, num_layers=1, batch_size=batch_size)	

	mlp_a = mlp(batch_size,all_hidden_size, 2)
	mlp_criterion = nn.NLLLoss()
	mlp_optimizer = optim.Adam(mlp_a.parameters(),lr = learning_rate,weight_decay=weight_decay)	
	
	if use_cuda:
		print('cuda')
		wEncoder= wEncoder.cuda()
		swEncoder= swEncoder.cuda()
		mlp_a = mlp_a.cuda()
		mlp_criterion = mlp_criterion.cuda()
		pass	


	for epoch in range(num_iters):

		print('epoch',epoch)
		right_all = 0
		loss_all = 0
		for i in range(0, fold_size, batch_size):

			
			print(len(data.train_x))
			train_inseq = data.train_x[i : i + batch_size]
			train_out = data.train_y[i : i + batch_size]
			train_out = torch.from_numpy(np.array(train_out))#.float()
			#print(train_inseq)
			#print(train_out)		

			users = train_inseq[:,1]
			users_vec = []
			for user in users:
				users_vec.append(data.user_dic[user])
				pass
			users_vec = torch.stack([vec for vec in users_vec],0)
			print(users_vec.size())
					

			ques = train_inseq[:,0]
			ques_topic_vec = []
			ques_sw_seq = []
			ques_w_seq = []
			for que in ques:
				que_d = data.que_dic[que]
				ques_topic_vec.append(que_d[0])
				ques_sw_seq.append(que_d[1][0])
				ques_w_seq.append(que_d[1][1])
				pass	
			

			ques_topic_vec = torch.stack([vec for vec in ques_topic_vec],0)
			print('topic：==',ques_topic_vec.size())	
			

			#print(len(ques_w_seq),'=========',len(ques_w_seq[0]))
			ques_w_seq = torch.from_numpy(np.array(ques_w_seq))
			print('w',ques_w_seq.size())		

			#print(len(ques_sw_seq),'=========',len(ques_sw_seq[0]))
			ques_sw_seq = torch.from_numpy(np.array(ques_sw_seq))
			print('sw',ques_sw_seq.size())	
		
			
	
	

			if use_cuda:
				#print('cuda')
				train_out = train_out.cuda()
				users_vec = users_vec.float().cuda()
				ques_w_seq = ques_w_seq.cuda()
				ques_sw_seq = ques_sw_seq.cuda()
				ques_topic_vec = ques_topic_vec.float().cuda()
				pass		

			_,whidden = wEncoder(ques_w_seq)
			print('whidden==',whidden.size())		

			_,swhidden = swEncoder(ques_sw_seq)
			print('swhidden==',swhidden.size())		

			#print(users_vec.type())
			all_hidden = torch.cat([users_vec,ques_topic_vec,whidden.squeeze(),swhidden.squeeze()],1)
			print(all_hidden.size())	
			
			pred = mlp_a(all_hidden)
			#print(pred)		

			_,ans = pred.data.topk(1)
			print('result=====',ans.size())
			batch_size_1 = ans.size()[0]			
			right = ans.view(batch_size_1) - train_out
			right = batch_size_1-sum(abs(right.cpu().numpy())) 
			right_all += right
			print('正确的个数=====',right/batch_size_1)

			loss= mlp_criterion(pred+1e-10, train_out)
			loss_all += loss
			print('i: ',i,' ==loss: ',loss)
			str_r = 'i: '+str(i)+'===== loss: '+str(loss)
			save_train_l(str_r,'res.txt')


			mlp_optimizer.zero_grad()
			loss.backward()
			mlp_optimizer.step()
			#break
		#break
		print('==================================')
		right_rate = right_all/fold_size
		print('epoch: ',epoch,' ==loss: ',loss_all,'====right_rate: ',right_rate)

		torch.save(wEncoder.state_dict(), 'model_rel/wencoder_epoch_{}.pt'.format(epoch))
		torch.save(swEncoder.state_dict(), 'model_rel/swencoder_epoch_{}.pt'.format(epoch))
		torch.save(mlp_a.state_dict(), 'model_rel/mlp_epoch_{}.pt'.format(epoch))
