# @Time   : 2022/4/15
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.NLBA
##########################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from recbole.model.layers import MLPLayers
from recbole.utils import InputType, FeatureSource, FeatureType
from recbole_metarec.MetaRecommender import MetaRecommender
from recbole_metarec.MetaUtils import GradCollector,EmbeddingTable

class ItemEmbedding(nn.Module):
    def __init__(self,dataset,embeddingSize,hiddenSize):
        super(ItemEmbedding, self).__init__()

        self.embeddingTable=EmbeddingTable(embeddingSize,dataset,source=[FeatureSource.ITEM])

        self.embeddingNetwork=nn.Sequential(
            nn.Linear(self.embeddingTable.getAllDim(),hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize,embeddingSize)
        )

    def forward(self,itemFeatures):
        return self.embeddingNetwork(self.embeddingTable(itemFeatures))

class NLBARec(nn.Module):
    def __init__(self,embeddingSize,hiddenDim):
        super(NLBARec, self).__init__()

        self.output_v0=nn.Parameter(torch.randn(size=(embeddingSize,1)))
        self.output_v1=nn.Parameter(torch.randn(size=(embeddingSize,1)))
        self.output_w=nn.Parameter(torch.randn(size=(hiddenDim,1)))

        self.hidden_v0=nn.Parameter(torch.randn(size=(embeddingSize,hiddenDim)))
        self.hidden_v1 = nn.Parameter(torch.randn(size=(embeddingSize, hiddenDim)))
        self.hidden_w=nn.Parameter(torch.randn(size=(embeddingSize,hiddenDim)))


    def forward(self,R0,R1,itemEmbed):
        hiddenBias=torch.matmul(R0,self.hidden_v0)+torch.matmul(R1,self.hidden_v1)
        hiddenOutput=F.relu(torch.matmul(itemEmbed,self.hidden_w)+hiddenBias)

        outputBias=torch.matmul(R0,self.output_v0)+torch.matmul(R1,self.output_v1)
        prediction=F.sigmoid(torch.matmul(hiddenOutput,self.output_w)+outputBias)

        return prediction.squeeze()

class NLBA(MetaRecommender):
    '''
    This is the recommender implement of NLBA.

    Vartak M, Thiagarajan A, Miranda C, et al. A meta-learning perspective on cold-start recommendations for items[J].
    Advances in neural information processing systems, 2017, 30.

    https://proceedings.neurips.cc/paper/2017/hash/51e6d6e679953c6311757004d8cbbba9-Abstract.html

    '''

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(NLBA, self).__init__(config,dataset)

        self.embedding_size=config['embedding']
        self.embeddingHiddenDim=config['embeddingHiddenDim']

        self.itemEmbedding=ItemEmbedding(dataset,self.embedding_size,self.embeddingHiddenDim)
        self.nlbaRec=NLBARec(self.embedding_size,config['recHiddenDim'])

    def forward(self,spt_x_item,spt_y,qrt_x_item):
        spt_x_item, qrt_x_item = self.itemEmbedding(spt_x_item), self.itemEmbedding(qrt_x_item)
        spt_y_true, spt_y_false = [], []
        for index, click in enumerate(spt_y):
            if click == 1:
                spt_y_false.append(index)
            else:
                spt_y_true.append(index)
        R0 = torch.sum(spt_x_item[spt_y_false], dim=0)
        R1 = torch.sum(spt_x_item[spt_y_true], dim=0)

        prediction_qrt_y=self.nlbaRec(R0, R1, qrt_x_item)
        return prediction_qrt_y

    def calculate_loss(self, taskBatch):
        totalLoss = torch.tensor(0.0).to(self.config.final_config_dict['device'])

        for task in taskBatch:
            spt_x_item, spt_y,qrt_x_item,qrt_y =task

            prediction_qrt_y=self.forward(spt_x_item,spt_y,qrt_x_item)
            prediction_qrt_y_neg = 1.0 - prediction_qrt_y
            prob_qrt_y = torch.stack([prediction_qrt_y_neg,prediction_qrt_y], dim=1)
            qrt_y = qrt_y - 1
            loss=F.cross_entropy(prob_qrt_y,qrt_y)

            totalLoss=totalLoss+loss

        return totalLoss,None

    def predict(self, spt_x,spt_y,qrt_x):
        prediction_qrt_y=self.forward(spt_x,spt_y,qrt_x)
        return prediction_qrt_y



