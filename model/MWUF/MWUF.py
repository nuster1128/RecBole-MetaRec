# @Time   : 2022/4/7
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.MWUF
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

class ModelRec(nn.Module):
    def __init__(self,config,indexEmbDim,embeddingSize,dataset,hiddenDim):
        super(ModelRec, self).__init__()

        self.userEmbedding=EmbeddingTable(embeddingSize,dataset,source=[FeatureSource.USER])
        self.itemEmbedding=EmbeddingTable(embeddingSize,dataset,source=[FeatureSource.ITEM])
        self.itemIndexEmbedding=nn.Embedding(dataset.num(config['ITEM_ID_FIELD']),indexEmbDim)

        self.hiddenLayer=nn.Linear(indexEmbDim*2+embeddingSize*7,hiddenDim)
        self.outputLayer=nn.Linear(hiddenDim,2)

    def forward(self,userIndexEmb,userFeatures,itemIndex,itemFeatures):
        input_x=torch.cat([userIndexEmb,self.userEmbedding.embeddingAllFields(userFeatures),self.itemIndexEmbedding(itemIndex),self.itemEmbedding.embeddingAllFields(itemFeatures)],dim=1)

        return F.softmax(self.outputLayer(F.relu(self.hiddenLayer(input_x))))

class PreTrainModel(nn.Module):
    def __init__(self,config,dataset):
        super(PreTrainModel, self).__init__()

        self.embedding_size = config['embedding']
        self.indexEmbDim = config['indexEmbDim']

        self.userIndexEmbedding = nn.Embedding(dataset.num(config['USER_ID_FIELD']), self.indexEmbDim)
        self.f = ModelRec(config,self.indexEmbDim, self.embedding_size, dataset, config['modelRecHiddenDim'])

    def forward(self,x_userid,x_user,x_itemid, x_item):
        return self.f(self.userIndexEmbedding(x_userid),x_user,x_itemid,x_item)

class MetaNets(nn.Module):
    def __init__(self,config,embedding_size,indexEmbDim):
        super(MetaNets, self).__init__()

        self.scaleNet = nn.Sequential(
            nn.Linear(4 * embedding_size, config['scaleHiddenDim']),
            nn.ReLU(),
            nn.Linear(config['scaleHiddenDim'], indexEmbDim)
        )
        self.shiftNet = nn.Sequential(
            nn.Linear(3 * embedding_size, config['shiftHiddenDim']),
            nn.ReLU(),
            nn.Linear(config['shiftHiddenDim'], indexEmbDim)
        )

class MWUF(MetaRecommender):

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(MWUF, self).__init__(config,dataset)

        self.embedding_size = self.config['embedding']
        self.indexEmbDim = self.config['indexEmbDim']

        self.pretrainModel = PreTrainModel(config, dataset)
        self.pretrainOpt = torch.optim.SGD(self.pretrainModel.parameters(), lr=self.config['pretrainLr'])

        self.metaNets=MetaNets(config,self.embedding_size,self.indexEmbDim)

        self.userEmbeddingGradCollector=GradCollector(list(self.pretrainModel.userIndexEmbedding.state_dict().keys()))
        self.metaNetsGradCollector=GradCollector(list(self.metaNets.state_dict().keys()))

    def pretrain(self,taskBatch):
        for task in taskBatch:
            (spt_x_userid,spt_x_user,spt_x_itemid, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user,qrt_x_itemid, qrt_x_item), qrt_y =task

            predict_spt_y=self.pretrainModel(spt_x_userid,spt_x_user,spt_x_itemid, spt_x_item)
            spt_y = spt_y - 1
            spt_loss = F.cross_entropy(predict_spt_y, spt_y)

            predict_qrt_y=self.pretrainModel(qrt_x_userid,qrt_x_user,qrt_x_itemid, qrt_x_item)
            qrt_y = qrt_y - 1
            qrt_loss = F.cross_entropy(predict_qrt_y, qrt_y)

            loss = spt_loss + qrt_loss
            self.pretrainOpt.zero_grad()
            loss.backward()
            self.pretrainOpt.step()

    def calculate_loss(self, taskBatch):
        totalLoss = torch.tensor(0.0)

        for task in taskBatch:
            (spt_x_userid,spt_x_user,spt_x_itemid, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user,qrt_x_itemid, qrt_x_item), qrt_y =task
            spt_y = spt_y - 1
            qrt_y = qrt_y - 1

            spt_userIndexCold=self.pretrainModel.userIndexEmbedding(spt_x_userid)
            predict_spt_cold_y=self.pretrainModel.f(spt_userIndexCold,spt_x_user,spt_x_itemid, spt_x_item)
            spt_loss_cold=F.cross_entropy(predict_spt_cold_y,spt_y)

            spt_scale=self.metaNets.scaleNet(self.pretrainModel.f.userEmbedding.embeddingAllFields(spt_x_user))
            allItemVectors=torch.cat([self.pretrainModel.f.itemEmbedding.embeddingAllFields(spt_x_item),self.pretrainModel.f.itemEmbedding.embeddingAllFields(qrt_x_item)])
            avgItemVectors=torch.sum(allItemVectors,dim=0)/allItemVectors.shape[0]+torch.zeros(size=allItemVectors.shape)
            spt_shift=self.metaNets.shiftNet(avgItemVectors[:-10])

            spt_userIndexWarm=spt_userIndexCold*spt_scale+spt_shift
            predict_spt_warm_y=self.pretrainModel.f(spt_userIndexWarm,spt_x_user,spt_x_itemid, spt_x_item)
            spt_loss_warm = F.cross_entropy(predict_spt_warm_y, spt_y)

            qrt_userIndexCold=self.pretrainModel.userIndexEmbedding(qrt_x_userid)
            predict_qrt_cold_y=self.pretrainModel.f(qrt_userIndexCold,qrt_x_user,qrt_x_itemid, qrt_x_item)
            qrt_loss_cold=F.cross_entropy(predict_qrt_cold_y,qrt_y)

            qrt_scale=self.metaNets.scaleNet(self.pretrainModel.f.userEmbedding.embeddingAllFields(qrt_x_user))
            qrt_shift=self.metaNets.shiftNet(avgItemVectors[-10:])

            qrt_userIndexWarm=qrt_userIndexCold*qrt_scale+qrt_shift
            predict_qrt_warm_y=self.pretrainModel.f(qrt_userIndexWarm,qrt_x_user,qrt_x_itemid, qrt_x_item)
            qrt_loss_warm=F.cross_entropy(predict_qrt_warm_y,qrt_y)

            loss_cold=spt_loss_cold+qrt_loss_cold
            loss_warm=spt_loss_warm+qrt_loss_warm

            userEmbeddingGrad=torch.autograd.grad(loss_cold,self.pretrainModel.userIndexEmbedding.parameters(),create_graph=True,retain_graph=True)
            metaNetsGrad=torch.autograd.grad(loss_warm,self.metaNets.parameters())

            self.userEmbeddingGradCollector.addGrad(userEmbeddingGrad)
            self.metaNetsGradCollector.addGrad(metaNetsGrad)
            totalLoss+=loss_warm.detach()
        self.userEmbeddingGradCollector.averageGrad(self.config['train_batch_size'])
        self.metaNetsGradCollector.averageGrad(self.config['train_batch_size'])
        totalLoss /= self.config['train_batch_size']
        return totalLoss, self.userEmbeddingGradCollector.dumpGrad(),self.metaNetsGradCollector.dumpGrad()

    def predict(self, spt_x,spt_y,qrt_x):
        (spt_x_userid,spt_x_user,spt_x_itemid, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user,qrt_x_itemid, qrt_x_item) =spt_x,spt_y,qrt_x

        qrt_userIndexCold = self.pretrainModel.userIndexEmbedding(qrt_x_userid)

        qrt_scale = self.metaNets.scaleNet(self.pretrainModel.f.userEmbedding.embeddingAllFields(qrt_x_user))
        allItemVectors = torch.cat([self.pretrainModel.f.itemEmbedding.embeddingAllFields(spt_x_item),self.pretrainModel.f.itemEmbedding.embeddingAllFields(qrt_x_item)])
        avgItemVectors = torch.sum(allItemVectors, dim=0) / allItemVectors.shape[0] + torch.zeros(size=allItemVectors.shape)
        qrt_shift = self.metaNets.shiftNet(avgItemVectors[-10:])

        qrt_userIndexWarm = qrt_userIndexCold * qrt_scale + qrt_shift
        predict_qrt_warm_y = self.pretrainModel.f(qrt_userIndexWarm, qrt_x_user, qrt_x_itemid, qrt_x_item)[:,1]

        return predict_qrt_warm_y


