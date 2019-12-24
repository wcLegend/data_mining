import torch
import torch.nn as nn
from torch import Tensor
#from elmo import elmo_emb  #版本似乎有问题 待测试

#加入elmo_embedding 

class wEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__( self, hidden_size,word2vec,word_size, num_layers=1, batch_size=1):
        super(wEncoder, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda = False
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        #self.idx2word = idx2word  elmo有Bug所以隐去
        self.input_size = word_size
        self.embedding  = nn.Embedding(self.input_size, word2vec.size()[1])
        self.embedding.from_pretrained(word2vec)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, bidirectional=False,batch_first = True)  #


    def forward(self, input):
        """ 这里input_lengths 就是相应的长度
        input           -> (seq_len, batch, input_size)
        input_lengths   -> (Batch Size (Sorted in decreasing order of lengths))
        hidden          -> (num_layers * num_directions, batch, hidden_size)
        """


        print('w_encoder embedding input ',input.size())
        embedded = self.embedding(input) # L, B, V
        batch_size = input.size()[0]
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        hidden = self.init_hidden(batch_size)
        print('w_encoder gru hidden',hidden.size())
        outputs, hidden = self.gru(embedded, hidden)

        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        #print(outputs.size())
        # 这里也可以采用双向GRU

        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

    def init_hidden(self, batch_size=0):
        if batch_size == 0: batch_size = self.batch_size
        #result = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size) #双向
        result = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            return result.cuda()
        else:
            return result