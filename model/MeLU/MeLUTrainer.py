# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.MeLUTrainer
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

class MeLUTrainer(MetaTrainer):
    '''
    This is the trainer implement of MeLU.

    Lee H, Im J, Jang S, et al. Melu: Meta-learned user preference estimator for cold-start recommendation[C]
    Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019: 1073-1082.

    https://doi.org/10.1145/3292500.3330859

    Note: Temporarily, we use FOMAML instead of full MAML and will correct soon.

    '''
    def __init__(self,config,model):
        super(MeLUTrainer, self).__init__(config,model)

        self.lr = config['melu_args']['lr']
        self.xFields = model.dataset.fields(source=[FeatureSource.USER, FeatureSource.ITEM])
        self.yField = model.RATING

    def taskDesolve(self,task):
        spt_x,qrt_x=OrderedDict(),OrderedDict()
        for field in self.xFields:
            spt_x[field]=task.spt[field]
            qrt_x[field]=task.qrt[field]
        spt_y=task.spt[self.yField]
        qrt_y=task.qrt[self.yField]

        spt_x, qrt_x=Interaction(spt_x),Interaction(qrt_x)
        return spt_x, spt_y, qrt_x, qrt_y

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
        totalLoss=torch.tensor(0.0)
        for batch_idx, taskBatch in enumerate(iter_data):
            loss, grad = self.model.calculate_loss(taskBatch)
            totalLoss+=loss

            # This is SGD process.
            newParams=OrderedDict()
            for name,params in self.model.state_dict().items():
                newParams[name]=params-self.lr*grad[name]

            self.model.load_state_dict(newParams)

            self.model.keepWeightParams = deepcopy(self.model.model.state_dict())

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return totalLoss/(batch_idx+1)