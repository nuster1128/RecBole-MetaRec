# @Time   : 2022/4/6
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.TaNP
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

class AMatrix(nn.Module):
    def __init__(self,device,rDim,k,alpha):
        super(AMatrix, self).__init__()

        self.rDim=rDim
        self.k=k
        self.alpha=alpha

        self.matrix=torch.nn.Parameter(torch.randn(size=(self.k,self.rDim))).to(device)

    def forward(self,t_i):
        vectorNorms=torch.norm(t_i-self.matrix,dim=1)
        baseNum=torch.pow(vectorNorms,2)/self.alpha+1
        fz=torch.pow(baseNum,-(self.alpha+1)/2)
        fm=torch.sum(fz)
        c_i=fz/fm
        attentionVec=torch.squeeze(torch.matmul(torch.reshape(c_i,(1,-1)),self.matrix))
        return c_i,attentionVec

class ZVectorGenerator(nn.Module):
    def __init__(self,device,rDim,zDim):
        super(ZVectorGenerator, self).__init__()

        self.device=device
        self.rDim=rDim
        self.zDim=zDim
        self.W_s=nn.Linear(rDim,rDim,bias=False)
        self.W_u=nn.Linear(rDim,zDim,bias=False)
        self.W_sigma=nn.Linear(rDim,zDim,bias=False)

    def generateEpsilon(self):
        return torch.randn(size=(self.zDim,)).to(self.device)

    def forward(self,r_i):
        new_r_i=F.relu(self.W_s(r_i))
        mu_i=self.W_u(new_r_i)
        sigma_i=torch.exp(self.W_sigma(new_r_i))
        epsilon=self.generateEpsilon()

        z_i=mu_i+epsilon*sigma_i
        return z_i,mu_i,sigma_i

class TaNPRec(nn.Module):
    def __init__(self,inputDim,oDim,hiddenDim):
        super(TaNPRec, self).__init__()

        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.oDim=oDim

        self.hiddenLinear=nn.Linear(self.inputDim,self.hiddenDim)
        self.hiddenLambda=nn.Linear(self.oDim,self.hiddenDim,bias=False)
        self.hiddenBeta=nn.Linear(self.oDim,self.hiddenDim,bias=False)

        self.outputLinear=nn.Linear(self.hiddenDim,1)
        self.outputLambda=nn.Linear(self.oDim,1,bias=False)
        self.outputBeta=nn.Linear(self.oDim,1,bias=False)

    def forward(self,input,o_i):

        hidden=F.relu(self.hiddenLambda(o_i)*self.hiddenLinear(input)+self.hiddenBeta(o_i))
        output=self.outputLambda(o_i)*self.outputLinear(hidden)+self.outputBeta(o_i)
        return output

class TaNP(MetaRecommender):
    '''
    This is the recommender implement of TaNP.

    Lin X, Wu J, Zhou C, et al. Task-adaptive neural process for user cold-start recommendation[C]
    Proceedings of the Web Conference 2021. 2021: 1306-1316.

    https://doi.org/10.1145/3442381.3449908

    '''

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(TaNP, self).__init__(config,dataset)

        self.device=self.config.final_config_dict['device']
        self.embedding_size=self.config['embedding']
        self.encoderHiddenDim=self.config['encoderHiddenDim']
        self.rDim=self.config['rDim']
        self.zDim=self.config['zDim']

        self.taskUserEmbedding=EmbeddingTable(self.embedding_size,dataset,source=[FeatureSource.USER])
        self.taskItemEmbedding=EmbeddingTable(self.embedding_size,dataset,source=[FeatureSource.ITEM])

        self.encoderMLP2=nn.Sequential(
            nn.Linear(self.taskUserEmbedding.getAllDim()+self.taskItemEmbedding.getAllDim()+1,self.encoderHiddenDim),
            nn.ReLU(),
            nn.Linear(self.encoderHiddenDim,self.rDim)
        )

        self.encoderMLP1=nn.Sequential(
            nn.Linear(self.taskUserEmbedding.getAllDim()+self.taskItemEmbedding.getAllDim()+1, self.encoderHiddenDim),
            nn.ReLU(),
            nn.Linear(self.encoderHiddenDim, self.rDim)
        )

        self.zVectorGenerator=ZVectorGenerator(self.device,self.rDim,self.zDim)
        self.taNPRec=TaNPRec(self.taskUserEmbedding.getAllDim()+self.taskItemEmbedding.getAllDim()+self.zDim,self.config['decodeHiddenDim'],self.rDim)

        self.aMatrix=AMatrix(self.device,self.rDim,self.config['k'],self.config['alpha'])

        self.metaParams=deepcopy(self.state_dict())

        self.metaGradCollector = GradCollector(list(self.state_dict().keys()))

    def kl_calculate(self,spt_mu,spt_sigma,qrt_mu,qrt_sigma):
        kl=(qrt_sigma+(qrt_mu-spt_mu)**2)/spt_sigma-1.0+torch.log(spt_sigma)-torch.log(qrt_sigma)
        return 0.5*torch.sum(kl)

    def calculate_loss(self, taskBatch):
        totalLoss = torch.tensor(0.0).to(self.device)
        C_box=[]
        self.load_state_dict(deepcopy(self.metaParams))
        for task in taskBatch:

            (spt_x_user, spt_x_item), spt_y,(qrt_x_user, qrt_x_item), qrt_y = task
            spt_y = spt_y.view(-1, 1)

            spt_x_user=self.taskUserEmbedding.embeddingAllFields(spt_x_user)
            spt_x_item=self.taskItemEmbedding.embeddingAllFields(spt_x_item)
            qrt_x_user=self.taskUserEmbedding.embeddingAllFields(qrt_x_user)
            qrt_x_item=self.taskItemEmbedding.embeddingAllFields(qrt_x_item)

            spt_input=torch.cat((spt_x_user,spt_x_item,spt_y),dim=1)
            t_ij=self.encoderMLP2(spt_input)
            t_i=torch.sum(t_ij,dim=0)/t_ij.shape[0]

            c_i,attentionVec=self.aMatrix(t_i)
            o_i=t_i+attentionVec

            # Training process
            qrt_y=qrt_y.view(-1,1)
            qrt_input=torch.cat((qrt_x_user,qrt_x_item,qrt_y),dim=1)
            r_ij=self.encoderMLP1(qrt_input)
            r_i=torch.sum(r_ij,dim=0)/r_ij.shape[0]

            qrt_z_i,qrt_mu_i,qrt_sigma_i=self.zVectorGenerator(r_i)
            o_i_batch=o_i+torch.zeros(size=(qrt_x_user.shape[0],o_i.shape[0])).to(self.device)
            qrt_z_i_batch=qrt_z_i+torch.zeros(size=(qrt_x_user.shape[0],qrt_z_i.shape[0])).to(self.device)

            qrt_input=torch.cat((qrt_x_user,qrt_x_item,qrt_z_i_batch),dim=1)

            predict_qrt_y=self.taNPRec(qrt_input,o_i_batch)

            L_ri=F.mse_loss(predict_qrt_y,qrt_y)

            spt_z_i,spt_mu_i,spt_sigma_i=self.zVectorGenerator(t_i)

            L_ci=self.kl_calculate(spt_mu_i,spt_sigma_i,qrt_mu_i,qrt_sigma_i)

            L=L_ri+L_ci

            grad=torch.autograd.grad(L,self.parameters(),create_graph=True,retain_graph=True)

            self.metaGradCollector.addGrad(grad)
            totalLoss += L.detach()

            C_box.append(c_i)
        self.metaGradCollector.averageGrad(self.config['train_batch_size'])
        totalLoss /= self.config['train_batch_size']

        C=torch.stack(C_box)
        C_sqrt=torch.pow(C,2)
        C_sum=torch.sum(C,dim=0,keepdim=True)
        temp=C_sqrt/C_sum
        temp_sum=torch.sum(temp,dim=1,keepdim=True)
        D=temp/temp_sum
        L_u=self.config['lambda']*F.kl_div(torch.log(C),D,reduction='batchmean')

        grad=torch.autograd.grad(L_u,self.parameters(),allow_unused=True)
        tempGrad=[]
        for index,value in enumerate(grad):
            if value is None:
                tempGrad.append(torch.zeros(self.metaGradCollector.gradDict[self.metaGradCollector.paramNameList[index]].shape).to(self.device))
            else:
                tempGrad.append(value)
        tempGrad=tuple(tempGrad)

        self.metaGradCollector.addGrad(tempGrad)
        totalLoss+=L_u.detach()

        return totalLoss, self.metaGradCollector.dumpGrad()

    def predict(self, spt_x,spt_y,qrt_x):
        (spt_x_user, spt_x_item), spt_y,(qrt_x_user, qrt_x_item) = spt_x, spt_y, qrt_x

        spt_y = spt_y.view(-1, 1)

        spt_x_user = self.taskUserEmbedding.embeddingAllFields(spt_x_user)
        spt_x_item = self.taskItemEmbedding.embeddingAllFields(spt_x_item)
        qrt_x_user = self.taskUserEmbedding.embeddingAllFields(qrt_x_user)
        qrt_x_item = self.taskItemEmbedding.embeddingAllFields(qrt_x_item)

        spt_input = torch.cat((spt_x_user, spt_x_item, spt_y), dim=1)
        t_ij = self.encoderMLP2(spt_input)
        t_i = torch.sum(t_ij, dim=0) / t_ij.shape[0]

        c_i, attentionVec = self.aMatrix(t_i)
        o_i = t_i + attentionVec
        o_i_batch = o_i + torch.zeros(size=(qrt_x_user.shape[0], o_i.shape[0])).to(self.device)

        spt_z_i, spt_mu_i, spt_sigma_i = self.zVectorGenerator(t_i)
        sptForQrt_z_i_batch = spt_z_i + torch.zeros(size=(qrt_x_user.shape[0], spt_z_i.shape[0])).to(self.device)

        qrt_input = torch.cat((qrt_x_user, qrt_x_item, sptForQrt_z_i_batch), dim=1)

        predict_qrt_y = self.taNPRec(qrt_input, o_i_batch)

        return predict_qrt_y



