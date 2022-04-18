# @Time   : 2022/4/7
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.MWUFTrainer
##########################
"""

from tqdm import tqdm
from copy import deepcopy
import torch
from collections import OrderedDict
from recbole.utils import FeatureSource, set_color
from recbole.data.interaction import Interaction
from recbole.utils import get_gpu_usage
from MetaTrainer import MetaTrainer

class MWUFTrainer(MetaTrainer):

    def __init__(self,config,model):
        super(MWUFTrainer, self).__init__(config,model)

        self.device=self.config.final_config_dict['device']
        self.userFields = model.dataset.fields(source=[FeatureSource.USER])
        self.itemFields = model.dataset.fields(source=[FeatureSource.ITEM])
        self.yField = model.LABEL
        self.userIdField = self.config['USER_ID_FIELD']
        self.itemIdField = self.config['ITEM_ID_FIELD']

    def taskDesolve(self, task):
        spt_x_user, spt_x_item, qrt_x_user, qrt_x_item = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for field in self.userFields:
            spt_x_user[field] = task.spt[field]
            qrt_x_user[field] = task.qrt[field]
        for field in self.itemFields:
            spt_x_item[field] = task.spt[field]
            qrt_x_item[field] = task.qrt[field]
        spt_y = task.spt[self.yField]
        qrt_y = task.qrt[self.yField]
        spt_x_userid = task.spt[self.userIdField]
        qrt_x_userid = task.qrt[self.userIdField]
        spt_x_itemid = task.spt[self.itemIdField]
        qrt_x_itemid = task.qrt[self.itemIdField]

        spt_x_user, spt_x_item, qrt_x_user, qrt_x_item = Interaction(spt_x_user), Interaction(spt_x_item), Interaction(
            qrt_x_user), Interaction(qrt_x_item)
        return (spt_x_userid,spt_x_user,spt_x_itemid, spt_x_item), spt_y,(qrt_x_userid,qrt_x_user,qrt_x_itemid, qrt_x_item), qrt_y

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        totalLoss=torch.tensor(0.0).to(self.device)
        # PreTrain
        if epoch_idx == 0:
            for ep in range(self.config['pretrainEpoch']):
                for batch_idx, taskBatch in enumerate(iter_data):
                    taskBatch = [self.taskDesolve(task) for task in taskBatch]
                    self.model.pretrain(taskBatch)
            phi_old=list(self.model.pretrainModel.userIndexEmbedding.state_dict().values())[0]
            phi_new=torch.sum(phi_old,dim=0)/phi_old.shape[0]+torch.zeros(size=phi_old.shape).to(self.device)

            newUserIndexEmbeddingParam=OrderedDict()
            for name,value in self.model.pretrainModel.userIndexEmbedding.state_dict().items():
                newUserIndexEmbeddingParam[name]=phi_new
            self.model.pretrainModel.userIndexEmbedding.load_state_dict(newUserIndexEmbeddingParam)

        # Train
        for batch_idx, taskBatch in enumerate(iter_data):
            taskBatch = [self.taskDesolve(task) for task in taskBatch]
            loss, userEmbeddingGrad,metaNetsGrad = self.model.calculate_loss(taskBatch)
            totalLoss+=loss

            newUserEmbeddingParams = OrderedDict()
            for name, params in self.model.pretrainModel.userIndexEmbedding.state_dict().items():
                newUserEmbeddingParams[name] = params - self.config['coldLossLr'] * userEmbeddingGrad[name]
            self.model.pretrainModel.userIndexEmbedding.load_state_dict(newUserEmbeddingParams)

            newMetaNetsParams=OrderedDict()
            for name,params in self.model.metaNets.state_dict().items():
                newMetaNetsParams[name]=params-self.config['warmLossLr']*metaNetsGrad[name]
            self.model.metaNets.load_state_dict(newMetaNetsParams)

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return totalLoss/(batch_idx+1)