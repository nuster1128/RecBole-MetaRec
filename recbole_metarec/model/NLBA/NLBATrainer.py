# @Time   : 2022/4/15
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.NLBA
##########################
"""

from tqdm import tqdm
from copy import deepcopy
import torch
from collections import OrderedDict
from recbole.utils import FeatureSource, set_color
from recbole.data.interaction import Interaction
from recbole.utils import get_gpu_usage
from recbole_metarec.MetaTrainer import MetaTrainer

class NLBATrainer(MetaTrainer):

    def __init__(self,config,model):
        super(NLBATrainer, self).__init__(config,model)

        self.device=self.config.final_config_dict['device']
        self.itemFields = model.dataset.fields(source=[FeatureSource.ITEM])
        self.yField = model.LABEL
        self.userIdField=self.config['USER_ID_FIELD']

        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=config['lr'])

    def taskDesolve(self, task):
        spt_x_item,qrt_x_item = OrderedDict(), OrderedDict()
        for field in self.itemFields:
            spt_x_item[field] = task.spt[field]
            qrt_x_item[field] = task.qrt[field]
        spt_y = task.spt[self.yField]
        qrt_y = task.qrt[self.yField]

        spt_x_item, qrt_x_item = Interaction(spt_x_item), Interaction(qrt_x_item)
        return spt_x_item, spt_y,qrt_x_item, qrt_y

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

        # Train
        for batch_idx, taskBatch in enumerate(iter_data):
            taskBatch = [self.taskDesolve(task) for task in taskBatch]
            loss, grad = self.model.calculate_loss(taskBatch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            totalLoss+=loss.detach()

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return totalLoss/(batch_idx+1)