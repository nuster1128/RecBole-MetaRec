# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.MetaTrainer
##########################
"""

import torch
from tqdm import tqdm
from recbole.utils import set_color, get_gpu_usage,EvaluatorType
from recbole.trainer import Trainer
from recbole_metarec.MetaCollector import MetaCollector

class MetaTrainer(Trainer):
    '''
    MetaTrainer is the key component for training a meta learning method.

    Overall, we extend 'Trainer' to 'MetaTrainer'.
    If you want to implement a meta learning model, please extend this class and implement '_train_epoch()' method.
      eg. You can create MeLUTrainer(MetaTrainer) and implement its '_train_epoch()' method.

    The extended modification can be listed briefly as following:

    [Override] self.evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):

    [Abstract] self.taskDesolve(task): Desolve a task into spt_x,spt_y,qrt_x,qrt_y.

    [Abstract] self._train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False): An epoch of training.

    '''
    def __init__(self,config,model):
        super(MetaTrainer, self).__init__(config,model)

        self.eval_collector=MetaCollector(config)

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        '''
        We adapt the evaluation process with task circumstance in meta learning.
        '''
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)
        self.model.eval()

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        if self.config['eval_type'] == EvaluatorType.VALUE:
            # This is the evaluation process for value evaluation, which we intend to use spt to
            # fine-tune the meta model and predict scores of given qrts.
            for batch_idx, batched_data in enumerate(iter_data):
                for task in batched_data:
                    spt_x, spt_y, qrt_x, qrt_y=self.taskDesolve(task)
                    scores=self.model.predict(spt_x,spt_y,qrt_x).squeeze()
                    label = qrt_y.squeeze()

                    if self.gpu_available and show_progress:
                        iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

                    self.eval_collector.eval_collect(scores,label)

            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)
            self.wandblogger.log_eval_metrics(result, head='eval')

            return result
        if self.config['eval_type'] == EvaluatorType.RANKING:
            for batch_idx, batched_data in enumerate(iter_data):
                for task in batched_data:
                    spt_x, spt_y, qrt_x, qrt_y = self.taskDesolve(task)
                    scores = self.model.predict(spt_x, spt_y, qrt_x).squeeze()
                    label = qrt_y.squeeze()

                    if self.gpu_available and show_progress:
                        iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

                    self.eval_collector.eval_collect(scores, label)

            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)
            self.wandblogger.log_eval_metrics(result, head='eval')

            return result

    def taskDesolve(self,task):
        '''
        This is an abstract method which is waiting for specific model to implement.
        It desolves a task into spt_x,spt_y,qrt_x,qrt_y.

        :param task(Task): The object of class 'Task'
        :return spt_x,spt_y,qrt_x,qrt_y: The four base parts of task in meta learning.
        '''

        raise NotImplementedError()

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        '''
        This is an abstract method which is waiting for specific model to implement.
        This method indicates for an epoch of training.
        It can be called by fit().

        :param train_data:
        :param epoch_idx: The index of epoch.
        :param loss_func:
        :param show_progress:
        :return:
        '''
        raise NotImplementedError()
