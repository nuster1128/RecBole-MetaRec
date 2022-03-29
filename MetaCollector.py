import torch
from recbole.evaluator.collector import Collector
class MetaCollector(Collector):
    '''
    MetaCollector is the key component for collect data for evaluation in meta learning circumstance.

    Overall, we extend 'Collector' to 'MetaCollector'.
    The extended modification can be listed briefly as following:

    [Override] self.eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor): Collect data for evaluation.

    [Override] self.data_collect(self, train_data): Collect the evaluation resource from training data.

    '''
    def __init__(self,config):
        super(MetaCollector, self).__init__(config)

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        '''
        Collect data for evaluation.

        :param eval_pred(torch.Tensor) : Normally, it is a 1D score tensor for the query prediction in a single task.
        :param data_label(torch.Tensor) : Normally, it is a 1D score tensor for the query label in a single task.
        :return:
        '''
        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', eval_pred)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', data_label.to(self.device))

        if self.register.need('rec.topk'):
            _, eval_topk_idx = torch.topk(eval_pred, max(self.topk), dim=-1)
            _, label_topk_idx = torch.topk(data_label, max(self.topk), dim=-1)

            pos_matrix = torch.zeros_like(eval_pred, dtype=torch.int)
            pos_matrix[label_topk_idx] = 1
            pos_len_list = pos_matrix.sum(dim=0,keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=0, index=eval_topk_idx)
            result = torch.cat((pos_idx, pos_len_list),dim=0)
            result=result.unsqueeze(dim=0)
            self.data_struct.update_tensor('rec.topk', result)

    def data_collect(self, train_data):
        '''
        Collect the evaluation resource from training data.

        :param train_data: The training dataloader which contains the training data
        '''
        if self.register.need('data.num_users'):
            self.data_struct.set('data.num_users', len(train_data.getUserList()))