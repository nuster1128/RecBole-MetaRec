# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.MetaDataset
##########################
"""

from recbole.data.dataset import Dataset
import numpy as np

class MetaDataset(Dataset):
    '''
    MetaDataset is the key component for splitting dataset by 'task.

    Overall, we extend 'Dataset' to 'MetaDataset'.
    The extended modification can be listed briefly as following:

    [Override] self.bulid(): Add 'task' keyword for 'group_by'.

    [Add] self.split_by_ratio_meta(self, ratios, group_by): Split method by 'task'.

    '''
    def __init__(self,config):
        super(MetaDataset, self).__init__(config)
        self.rating_field=config['RATING_FIELD']

    def build(self):
        '''
        In metaDataset, we add a new 'eval_args.group_by' keyword 'task' in it, which can split the dataset by user clusters.
        If we set the keywords 'task' and 'RS', it will call 'self.split_by_ratio_meta()', which we design it for the split method.
        '''
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        # ordering
        ordering_args = self.config['eval_args']['order']
        if ordering_args == 'RO':
            self.shuffle()
        elif ordering_args == 'TO':
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(f'The ordering_method [{ordering_args}] has not been implemented.')

        # splitting & grouping
        split_args = self.config['eval_args']['split']
        if split_args is None:
            raise ValueError('The split_args in eval_args should not be None.')
        if not isinstance(split_args, dict):
            raise ValueError(f'The split_args [{split_args}] should be a dict.')

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config['eval_args']['group_by']
        if split_mode == 'RS':
            if not isinstance(split_args['RS'], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == 'none':
                datasets = self.split_by_ratio(split_args['RS'], group_by=None)
            elif group_by == 'user':
                datasets = self.split_by_ratio(split_args['RS'], group_by=self.uid_field)
            elif group_by == 'task':
                datasets = self.split_by_ratio_meta(split_args['RS'], group_by=self.uid_field)
            else:
                raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
        elif split_mode == 'LS':
            datasets = self.leave_one_out(group_by=self.uid_field, leave_one_mode=split_args['LS'])
        else:
            raise NotImplementedError(f'The splitting_method [{split_mode}] has not been implemented.')

        return datasets
    def split_by_ratio_meta(self, ratios, group_by):
        '''
        This function is used to split dataset by non-intersection users clusters, which means the interactions in the output datasets
        (eg. training dataset, valid dataset and test dataset) have non-intersection user clusters.
        Split by non-intersection users is significant for user cold start task in meta learning recommendation.

        :param ratios: The split ratios of user clusters.
        :param group_by: Commonly 'task', and only by 'task' can this function be called.
        :return next_ds: [dataset_1, dataset_2 ... dataset_n]
        '''
        self.logger.debug(f'split by ratios [{ratios}], group_by=[{group_by}]')
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].numpy())
        # [[inter_id with same uid],...,[] ] This is the format of grouped_inter_feat_index.
        next_index = []
        totTask=len(grouped_inter_feat_index)
        split_ids=self._calcu_split_ids(tot=totTask, ratios=ratios)
        # print(next_index, [0] + split_ids, split_ids + [totTask])
        grouped_inter_feat_index=list(grouped_inter_feat_index)
        start,end=[0] + split_ids, split_ids + [totTask]
        for ind in range(len(ratios)):
            index=[]
            st,ed=start[ind],end[ind]
            for jnd in range(st,ed):
                for kitem in grouped_inter_feat_index[jnd]:
                    index.append(kitem)
            next_index.append(index)

        self._drop_unused_col()
        next_df = [self.inter_feat[index] for index in next_index]
        # print(len(np.unique(next_df[0]['user_id'].numpy())))     # It will output the real user number of next_df[0]

        next_ds = [self.copy(_) for _ in next_df]
        return next_ds