# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.MetaDataLoader
##########################
"""

import numpy as np
import torch
from recbole.data.dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction
from MetaUtils import Task

class MetaDataLoader(AbstractDataLoader):
    '''
    MetaDataLoader is the key component for transforming dataset into task form.
    As usual, we consider each user as a task.
    Here, a batch of data refers to a batch of tasks.

    Overall, we extend 'AbstractDataLoader' to 'MetaDataLoader'.
    The extended modification can be listed briefly as following:

    [Add] self.transformToTaskFormat(): Generate 'Task' dict from dataset.

    [Add] self.generateSingleTaskForTrain(self,uid,v): Format a single task.

    [Add] self.getTaskIdList(): Generate the list of task ids in this dataset.

    [Override] self._init_batch_size_and_step(): Initialize 'train_batch_size'.

    [Override] self.pr_end(): Get the number of tasks.

    [Override] self._shuffle(): Shuffle the task.

    [Override] self._next_batch_data(): Generate a batch of tasks iteratively.

    '''
    def __init__(self,config, dataset, sampler, shuffle=False):
        super(MetaDataLoader, self).__init__(config, dataset, sampler, shuffle)
        if shuffle is False:
            self.shuffle = True
            self.logger.warning('MetaDataLoader must shuffle the data.')

        self.uid_field = dataset.uid_field
        self.user_list = self.getUserList()
        self.taskDict=self.transformToTaskFormat()

    def transformToTaskFormat(self):
        '''
        This function is used to generate 'task' dict from dataset.
        It will return 'taskDict' for this MetaDataLoader.
        During the process of this method, it will call 'self.generateSingleTaskForTrain(uid,v)' to deal with a single task(user).

        :return finalTaskDict(dict) : A dict whose keys are 'user_id' and values are corresponding 'Task' object.
        '''
        taskDict={}
        user_id_np=self.dataset.inter_feat[self.dataset.uid_field].numpy()
        item_id_np=self.dataset.inter_feat[self.dataset.iid_field].numpy()
        rating_np=self.dataset.inter_feat[self.dataset.rating_field].numpy()
        for index,uid in enumerate(user_id_np):
            iid=item_id_np[index]
            rating=rating_np[index]
            if uid not in taskDict:
                taskDict[uid]={self.dataset.iid_field:[iid],self.dataset.rating_field:[rating]}
            else:
                taskDict[uid][self.dataset.iid_field].append(iid)
                taskDict[uid][self.dataset.rating_field].append(rating)
        finalTaskDict={}
        for uid,v in taskDict.items():
            task=self.generateSingleTaskForTrain(uid,v)
            finalTaskDict[uid]=task
        return finalTaskDict

    def generateSingleTaskForTrain(self,uid,v):
        '''
        We use this function to generate a task.

        :param uid: uid from function transformToTaskFormat().
        :param v: value from function transformToTaskFormat().
        :return: An object of class Task.
        '''
        taskInfo={self.dataset.uid_field:uid}

        iids=np.array(v[self.dataset.iid_field])
        ratings=np.array(v[self.dataset.rating_field])
        spt_num,qrt_num=self.config['meta_args']['support_num'],self.config['meta_args']['query_num']
        if spt_num == 'none':
            spt_num=len(iids)-qrt_num

        index=np.random.choice(len(iids),spt_num+qrt_num,replace=False)
        sptIndex,qrtIndex=index[:spt_num],index[spt_num:]
        sptIid,sptRating=iids[sptIndex],ratings[sptIndex]
        qrtIid,qrtRating=iids[qrtIndex],ratings[qrtIndex]
        sptUid=torch.tensor([uid for _ in range(len(sptIid))])
        qrtUid=torch.tensor([uid for _ in range(len(qrtIid))])

        spt=Interaction({self.dataset.uid_field:sptUid,self.dataset.iid_field:torch.tensor(sptIid),self.dataset.rating_field:torch.tensor(sptRating)})
        spt=self.dataset.join(spt)
        qrt=Interaction({self.dataset.uid_field:qrtUid,self.dataset.iid_field:torch.tensor(qrtIid),self.dataset.rating_field:torch.tensor(qrtRating)})
        qrt=self.dataset.join(qrt)
        return Task(taskInfo,spt,qrt)

    def getTaskIdList(self):
        '''
        This function can the list of task ids in this dataset.

        :return taskIdList (1D numpy.ndarray): An 1D array of task id list.
        '''

        return np.unique(self.dataset.inter_feat[self.uid_field].numpy())

    def _init_batch_size_and_step(self):
        '''
        This function is used to initialize 'train_batch_size'.
        The 'train_batch_size' indicates the number of tasks for each training batch.
        '''
        batch_size = self.config['train_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        '''
        Get the number of tasks(users) .
        '''
        return len(self.user_list)

    def _shuffle(self):
        '''
        Shuffle the task.
        '''
        np.random.shuffle(self.user_list)

    def _next_batch_data(self):
        '''
        This function is used to generate a batch of tasks iteratively.
        :return taskBatch(list) : A list of task. The length is 'train_batch_size'.
        '''
        cur_data = self.user_list[self.pr:self.pr + self.step]
        taskBatch=[self.taskDict[uid] for uid in cur_data]
        self.pr += self.step
        return taskBatch

