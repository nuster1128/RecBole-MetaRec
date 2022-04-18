# @Time   : 2022/4/15
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.LWA
##########################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from recbole.model.layers import MLPLayers
from recbole.utils import InputType, FeatureSource, FeatureType
from MetaRecommender import MetaRecommender
from MetaUtils import GradCollector,EmbeddingTable

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

class LWARec(nn.Module):
    def __init__(self,embeddingSize):
        super(LWARec, self).__init__()

        self.b=nn.Parameter(torch.randn(size=(1,)))
        self.w0=nn.Parameter(torch.randn(size=(embeddingSize,embeddingSize)))
        self.w1=nn.Parameter(torch.randn(size=(embeddingSize,embeddingSize)))

    def forward(self,R0,R1,itemEmbed):
        weight=torch.matmul(self.w0,R0)+torch.matmul(self.w1,R1)
        return F.sigmoid(torch.matmul(itemEmbed,weight)+self.b)

class LWA(MetaRecommender):

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(LWA, self).__init__(config,dataset)

        self.embedding_size=config['embedding']
        self.embeddingHiddenDim=config['embeddingHiddenDim']

        self.itemEmbedding=ItemEmbedding(dataset,self.embedding_size,self.embeddingHiddenDim)
        self.lwaRec=LWARec(self.embedding_size)

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

        prediction_qrt_y=self.lwaRec(R0, R1, qrt_x_item)
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



