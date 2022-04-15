# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.MetaRecommender
##########################
"""

from recbole.model.abstract_recommender import AbstractRecommender

class MetaRecommender(AbstractRecommender):
    '''
    MetaRecommender is the key component for implementing meta learning model.
    It is an abstract recommender for meta learning, in order to clearify 'Task' for meta learning.

    Overall, we extend 'AbstractRecommender' to 'MetaRecommender'.
    If you want to implement a meta learning model, please extend this class and implement 'calculate_loss()' method
    and 'predict()' method.
      eg. You can create MeLU(MetaRecommender) and implement its 'calculate_loss()' and 'predict()' method.

    The extended modification can be listed briefly as following:

    [Abstract] self.calculate_loss(taskBatch): Calculate the loss or the grad of the batch of tasks.

    [Abstract] self.predict(spt_x,spt_y,qrt_x): Predict the score of the query set of the task.

    '''

    def __init__(self, config, dataset):
        super(MetaRecommender, self).__init__()

        self.dataset=dataset
        self.config=config
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.RATING = config['RATING_FIELD']
        self.LABEL = config['LABEL_FIELD']

        self.embedding_size = config['embedding_size']
        self.device = config['device']

    def calculate_loss(self, taskBatch):
        '''
        Calculate the loss or the grad of the batch of tasks.
        Some meta learning model uses loss to backward.
        And some meta learning model uses grad for further calculation.

        :param tasks(List of 'Task'): The list of tasks.
        :return loss(torch.Tensor),grad(torch.Tensor): Training loss and grad of the batch of tasks. One of them can be None.
        '''
        raise NotImplementedError

    def predict(self, spt_x,spt_y,qrt_x):
        '''
        Predict the score of the query set of the task.

        :param spt_x(Interaction): Input of spt.
        :param spt_y(torch.Tensor): Rating/Label of spt.    shape: [batchsize, 1]
        :param qrt_x(Interaction): Input of qrt.
        :return scores(torch.Tensor): The predicted scores of the query set of the task.    shape: [batchsize, 1]
        '''
        raise NotImplementedError