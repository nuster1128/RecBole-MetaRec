# @Time   : 2022/4/5
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.MAMO
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

class MAMOMemory2D():
    def __init__(self,device,k,dim,lr=None):
        self.device=device
        self.lr=lr
        self.k=k
        self.dim=dim
        self.matrix=torch.randn(size=(k,dim)).to(device)

    def attention(self,vector):
        return F.softmax(torch.matmul(self.matrix,vector))

    def indice(self,weightVector):
        return torch.matmul(weightVector,self.matrix)

    def update(self,weightVector,vector2):
        crossProduct=torch.outer(weightVector,vector2)
        self.matrix *= (1 - self.lr)
        self.matrix+= crossProduct * self.lr

class MAMOMemory3D():
    def __init__(self,device,k,dim1,dim2,lr=None):
        self.device=device
        self.lr=lr
        self.k=k
        self.dim1=dim1
        self.dim2=dim2
        self.matrix=torch.randn(size=(k,dim1,dim2)).to(device)

    def indice(self,weightVector):
        tmp=torch.reshape(self.matrix,(self.k,self.dim1*self.dim2))
        output=torch.reshape(torch.matmul(weightVector,tmp),(self.dim1,self.dim2))
        return output

    def update(self,weightVector,vector2):
        crossProduct=torch.outer(weightVector,torch.reshape(vector2,(-1,)))
        crossProduct=torch.reshape(crossProduct,self.matrix.shape)
        self.matrix *= (1 - self.lr)
        self.matrix+= crossProduct * self.lr

class MAMOEmbeddingTable(nn.Module):
    def __init__(self,embeddingSize, dataset,source,fieldNum):
        super(MAMOEmbeddingTable, self).__init__()

        self.embTable=EmbeddingTable(embeddingSize,dataset,source)
        self.network=nn.Sequential(
            nn.Linear(self.embTable.getAllDim(),int(self.embTable.getAllDim()/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.embTable.getAllDim()/2),embeddingSize)
        )

    def forward(self,interaction):
        batchX=self.embTable(interaction)
        batchX=self.network(batchX)
        return batchX

    def getProfile(self,interaction):
        return self.embTable(interaction)

class MAMORec(nn.Module):
    def __init__(self,inputDim):
        super(MAMORec, self).__init__()

        self.hiddenLayer1=nn.Linear(inputDim,int(inputDim/2),bias=False)
        self.hiddenLayer2=nn.Linear(int(inputDim/2),int(inputDim/2))
        self.outputLayer=nn.Linear(int(inputDim/2),1)

    def forward(self,x):
        x=F.leaky_relu(self.hiddenLayer1(x))
        x=F.leaky_relu(self.hiddenLayer2(x))
        return self.outputLayer(x)

def squeezeModelParams(model):
    paramList=[]
    for name,value in model.state_dict().items():
        paramList.append(torch.reshape(value,(-1,)))
    return torch.cat(paramList)

def unsqueezeModelParams(params,model):
    base=0
    state_dict=OrderedDict()
    for name,value in model.state_dict().items():
        size=value.shape
        vol=torch.prod(torch.tensor(size))
        p=params[base:base+vol]
        state_dict[name]=torch.reshape(p,size)
        base+=vol
    return state_dict

class MAMO(MetaRecommender):

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(MAMO, self).__init__(config,dataset)

        self.embedding_size=self.config['embedding']
        self.device=self.config.final_config_dict['device']

        self.taskUserEmbedding = MAMOEmbeddingTable(self.embedding_size, dataset, source=[FeatureSource.USER], fieldNum=4)
        self.taskItemEmbedding = MAMOEmbeddingTable(self.embedding_size, dataset, source=[FeatureSource.ITEM], fieldNum=3)
        self.taskMamoRec = MAMORec(self.embedding_size * 2)

        self.metaUserEmbedding = self.taskUserEmbedding.state_dict()
        self.metaItemEmbedding = self.taskItemEmbedding.state_dict()
        self.metaMamoRec = self.taskMamoRec.state_dict()

        self.userEmbeddingParamNum=squeezeModelParams(self.taskUserEmbedding).shape[0]

        self.MP = MAMOMemory2D(self.device,self.config['k'],self.taskUserEmbedding.embTable.getAllDim(),lr=self.config['alpha'])
        self.MU = MAMOMemory2D(self.device,self.config['k'],self.userEmbeddingParamNum,lr=self.config['beta'])
        self.MUI = MAMOMemory3D(self.device,self.config['k'],self.embedding_size,2*self.embedding_size,lr=self.config['gamma'])

        self.metaGradCollector = GradCollector(list(self.state_dict().keys()))

    def forward(self,spt_x_user,spt_x_item, qrt_x_user, qrt_x_item,spt_y):
        spt_y= spt_y.view(-1, 1)

        self.taskUserEmbedding.load_state_dict(self.metaUserEmbedding)
        spt_x_userProfile = self.taskUserEmbedding.getProfile(spt_x_user)[0]

        attention_u = self.MP.attention(spt_x_userProfile)
        b_u = self.MU.indice(attention_u)
        b_u = unsqueezeModelParams(b_u, self.taskUserEmbedding)

        userEmbeddingFastWeight = OrderedDict()
        for name, value in self.metaUserEmbedding.items():
            userEmbeddingFastWeight[name] = value - self.config['tau'] * b_u[name]

        self.taskUserEmbedding.load_state_dict(userEmbeddingFastWeight)
        self.taskItemEmbedding.load_state_dict(self.metaItemEmbedding)
        self.taskMamoRec.load_state_dict(self.metaMamoRec)

        MuI = self.MUI.indice(attention_u)
        mamoRecHiddenLayerStateDict = OrderedDict()
        mamoRecHiddenLayerStateDict['weight'] = MuI
        self.taskMamoRec.hiddenLayer1.load_state_dict(mamoRecHiddenLayerStateDict)

        spt_x_user, spt_x_item = self.taskUserEmbedding(spt_x_user), self.taskItemEmbedding(spt_x_item)
        spt_x = torch.cat((spt_x_user, spt_x_item), dim=1)

        predict_spt_y = self.taskMamoRec(spt_x)
        sptLoss = F.mse_loss(predict_spt_y, spt_y)

        grad = torch.autograd.grad(sptLoss, self.parameters())
        fastweight = OrderedDict()
        gradUserEmbedding = []
        paramNames = list(self.state_dict().keys())
        tmp = self.state_dict()

        for index, name in enumerate(paramNames):
            fastweight[name] = tmp[name] - self.config['rho'] * grad[index]
            if name[:len('taskUserEmbedding')] == 'taskUserEmbedding':
                gradUserEmbedding.append(torch.reshape(grad[index], (-1,)))

        self.load_state_dict(fastweight)
        gradVecForMU = torch.cat(gradUserEmbedding)

        qrt_x_user, qrt_x_item = self.taskUserEmbedding(qrt_x_user), self.taskItemEmbedding(qrt_x_item)
        qrt_x = torch.cat((qrt_x_user, qrt_x_item), dim=1)

        predict_qry_y = self.taskMamoRec(qrt_x)
        return predict_qry_y, gradVecForMU,attention_u,spt_x_userProfile,MuI

    def calculate_loss(self, taskBatch):
        
        totalLoss = torch.tensor(0.0).to(self.device)
        for task in taskBatch:
            (spt_x_user,spt_x_item),spt_y,(qrt_x_user, qrt_x_item),qrt_y = task
            
            predict_qry_y, gradVecForMU,attention_u,spt_x_userProfile,MuI=self.forward(spt_x_user,spt_x_item, qrt_x_user, qrt_x_item,spt_y)

            qrt_y=qrt_y.view(-1, 1)
            qrtLoss = F.mse_loss(predict_qry_y, qrt_y)

            self.MP.update(attention_u,spt_x_userProfile)
            self.MU.update(attention_u,gradVecForMU)
            self.MUI.update(attention_u,MuI)

            grad=torch.autograd.grad(qrtLoss,self.parameters())

            self.metaGradCollector.addGrad(grad)
            totalLoss+=qrtLoss.detach()
        
        self.metaGradCollector.averageGrad(self.config['train_batch_size'])
        totalLoss /= self.config['train_batch_size']
        return totalLoss, self.metaGradCollector.dumpGrad()

    def predict(self, spt_x,spt_y,qrt_x):
        (spt_x_user,spt_x_item),spt_y,(qrt_x_user, qrt_x_item)=spt_x,spt_y,qrt_x

        predict_qry_y,_,_,_,_=self.forward(spt_x_user,spt_x_item,qrt_x_user,qrt_x_item,spt_y)

        return predict_qry_y
