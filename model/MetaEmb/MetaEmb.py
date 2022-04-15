# @Time   : 2022/4/6
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.MetaEmb
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
    def __init__(self,indexEmbDim,embeddingSize,dataset,hiddenDim):
        super(ModelRec, self).__init__()

        self.userEmbedding=EmbeddingTable(embeddingSize,dataset,source=[FeatureSource.USER])
        self.itemEmbedding=EmbeddingTable(embeddingSize,dataset,source=[FeatureSource.ITEM])

        self.hiddenLayer=nn.Linear(indexEmbDim+embeddingSize*7,hiddenDim)
        self.outputLayer=nn.Linear(hiddenDim,2)

    def forward(self,indexEmb,userFeatures,itemFeatures):
        input_x=torch.cat([indexEmb,self.userEmbedding.embeddingAllFields(userFeatures),self.itemEmbedding.embeddingAllFields(itemFeatures)],dim=1)

        return F.softmax(self.outputLayer(F.relu(self.hiddenLayer(input_x))))

class PreTrainModel(nn.Module):
    def __init__(self,config,dataset):
        super(PreTrainModel, self).__init__()

        self.embedding_size = config['embedding']
        self.indexEmbDim = config['indexEmbDim']

        self.indexEmbedding = nn.Embedding(dataset.num(config['USER_ID_FIELD']), self.indexEmbDim)
        self.f = ModelRec(self.indexEmbDim, self.embedding_size, dataset, config['modelRecHiddenDim'])

class EmbeddingGenerator(nn.Module):
    def __init__(self,userEmbedding,embeddingDim,hiddenDim,indexEmbDim):
        super(EmbeddingGenerator, self).__init__()

        self.userEmbedding=deepcopy(userEmbedding)

        self.mlp=nn.Sequential(
            nn.Linear(embeddingDim * 4, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, indexEmbDim)
        )

    def forward(self,userFeatures):
        indexEmb=self.mlp(self.userEmbedding.embeddingAllFields(userFeatures))

        return indexEmb

class MetaEmb(MetaRecommender):

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(MetaEmb, self).__init__(config,dataset)

        self.embedding_size=self.config['embedding']
        self.indexEmbDim=self.config['indexEmbDim']
        self.embeddingGeneratorHiddenDim=self.config['embeddingGeneratorHiddenDim']
        self.localLr=self.config['localLr']
        self.alpha=self.config['alpha']

        self.pretrainModel=PreTrainModel(config,dataset)
        self.pretrainOpt=torch.optim.SGD(self.pretrainModel.parameters(),lr=self.config['pretrainLr'])

        self.embeddingGenerator=EmbeddingGenerator(self.pretrainModel.f.userEmbedding,self.embedding_size,self.embeddingGeneratorHiddenDim,self.indexEmbDim)
        self.metaGradCollector = GradCollector(list(self.embeddingGenerator.mlp.state_dict().keys()))

    def pretrain(self,taskBatch):
        for task in taskBatch:
            (spt_x_userid,spt_x_user, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user, qrt_x_item), qrt_y =task

            spt_x_indexEmbedding=self.pretrainModel.indexEmbedding(spt_x_userid)
            predict_spt_y=self.pretrainModel.f(spt_x_indexEmbedding,spt_x_user,spt_x_item)
            spt_y=spt_y-1
            spt_loss=F.cross_entropy(predict_spt_y,spt_y)

            qrt_x_indexEmbedding=self.pretrainModel.indexEmbedding(qrt_x_userid)
            predict_qrt_y=self.pretrainModel.f(qrt_x_indexEmbedding,qrt_x_user,qrt_x_item)
            qrt_y=qrt_y-1
            qrt_loss=F.cross_entropy(predict_qrt_y,qrt_y)

            loss=spt_loss+qrt_loss
            self.pretrainOpt.zero_grad()
            loss.backward()
            self.pretrainOpt.step()

    def forward(self,spt_x,spt_y,qrt_x):
        (spt_x_userid,spt_x_user, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user, qrt_x_item)=spt_x,spt_y,qrt_x

        phi_init = self.embeddingGenerator(spt_x_user)

        predict_spt_y = self.pretrainModel.f(phi_init, spt_x_user, spt_x_item)
        spt_y = spt_y - 1
        spt_loss = F.cross_entropy(predict_spt_y, spt_y)

        grad = torch.autograd.grad(spt_loss, phi_init)
        avgGrad = torch.sum(grad[0], dim=0) / grad[0].shape[0]
        phi_prime = (phi_init[0] - self.localLr * avgGrad) + torch.zeros(size=(qrt_x_userid.shape[0], avgGrad.shape[0]))

        predict_qrt_y = self.pretrainModel.f(phi_prime, qrt_x_user, qrt_x_item)

        return predict_qrt_y


    def calculate_loss(self, taskBatch):
        totalLoss = torch.tensor(0.0)

        for task in taskBatch:
            (spt_x_userid,spt_x_user, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user, qrt_x_item), qrt_y =task

            phi_init=self.embeddingGenerator(spt_x_user)

            predict_spt_y=self.pretrainModel.f(phi_init,spt_x_user,spt_x_item)
            spt_y = spt_y - 1
            spt_loss=F.cross_entropy(predict_spt_y,spt_y)

            grad=torch.autograd.grad(spt_loss,phi_init,create_graph=True,retain_graph=True)
            avgGrad=torch.sum(grad[0],dim=0)/grad[0].shape[0]
            phi_prime=(phi_init[0]-self.localLr*avgGrad)+torch.zeros(size=(qrt_x_userid.shape[0],avgGrad.shape[0]))

            predict_qrt_y=self.pretrainModel.f(phi_prime,qrt_x_user,qrt_x_item)
            qrt_y=qrt_y-1
            qrt_loss=F.cross_entropy(predict_qrt_y,qrt_y)

            loss=self.alpha*spt_loss+(1-self.alpha)*qrt_loss
            grad=torch.autograd.grad(loss,self.embeddingGenerator.mlp.parameters())

            self.metaGradCollector.addGrad(grad)
            totalLoss+=loss.detach()
        self.metaGradCollector.averageGrad(self.config['train_batch_size'])
        totalLoss /= self.config['train_batch_size']
        return totalLoss, self.metaGradCollector.dumpGrad()



    def predict(self, spt_x,spt_y,qrt_x):
        predict_qrt_y=self.forward(spt_x,spt_y,qrt_x)[:,1]

        return predict_qrt_y



