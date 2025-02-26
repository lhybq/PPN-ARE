#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author: yanms
# @Date  : 2021/11/1 11:38
# @Desc  :
import argparse
import os
import random
import json
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, behavior_dict, behaviors):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

        self.dislike = True

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []
        # for behavior in self.behaviors:   # 正向兴趣取样
        for behavior in self.behaviors:
            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                while np.isin(neg, self.behavior_dict['all'][str(idx + 1)]):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)
        if self.dislike:
            for behavior in ['dislike_one','dislike_sec']:    # 负兴趣采样
                items = self.behavior_dict[behavior].get(str(idx + 1), None)
                # print(items)
                if items is None or items==[]:
                    signal = [0, 0, 0]
                else:
                    pos = random.sample(items, 1)[0]
                # neg = random.randint(1, self.item_count)
                # while ~np.isin(neg, self.behavior_dict['all'][str(idx + 1)]):
                #     neg = random.randint(1, self.item_count)

                # neg = random.sample(list(self.behavior_dict['all'].get(str(idx + 1), 0)), 1)[0]

                    res=list(self.behavior_dict[self.behaviors[-1]].get(str(idx + 1), []))
                    if res == []:
                        neg =random.sample(list(self.behavior_dict['all'].get(str(idx + 1), [])), 1)[0]
                        # signal = [0, 0, 0]
                    else:
                        neg = random.choice(res)
                    signal = [idx + 1, pos, neg]
                total.append(signal)
        return np.array(total)

    def __len__(self):
        return self.user_count


class DataSet(object):

    def __init__(self, args):

        # 接受关于数据集的参数
        self.behaviors = args.behaviors # 根据数据集生成的行为名称
        self.path = args.data_path
        self.dislike = args.dislike

        # 私有方法加载数据集
        # 获取用户和物品的数量
        self.__get_count()
        # 获取用户行为对应的物品列表
        self.__get_behavior_items()
        # 获取验证集中用户交互的物品列表
        self.__get_validation_dict()
        # 加载测试集中用户交互的物品列表
        self.__get_test_dict()
        # 加载图数据
        self.__get_sparse_interact_dict()

        # 计算验证集与测试集中每个用户的交互记录数量并存储
        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict
        if self.dislike:
            # 把不喜欢的视图数据也读入
            with open(os.path.join(self.path, 'dislike_one_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict['dislike_one'] = b_dict
            with open(os.path.join(self.path, 'dislike_sec_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict['dislike_sec'] = b_dict
            # with open(os.path.join(self.path, 'dislike_thi_dict.txt'), encoding='utf-8') as f:
            #     b_dict = json.load(f)
            #     self.train_behavior_dict['dislike_thi'] = b_dict


    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """
        load graphs

        :return:
        """
        self.edge_index = {}
        self.user_behaviour_degree = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
                indices = np.vstack((row, col))
                indices = torch.LongTensor(indices)

                values = torch.ones(len(row), dtype=torch.float32)
                self.user_behaviour_degree.append(torch.sparse.FloatTensor(indices,
                                                                           values,
                                                                           [self.user_count + 1, self.item_count + 1])
                                                  .to_dense().sum(dim=1).view(-1, 1))
                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row = torch.LongTensor(row).view(-1)
                all_row.append(row)
                col = torch.LongTensor(col).view(-1)
                all_col.append(col)
                edge_index = torch.stack([row, col])
                self.edge_index[behavior] = edge_index
        # 如果不喜欢数据也有，，则生成不喜欢系数交互矩阵
        if self.dislike:
            for behavior in ['dislike_one','dislike_sec']:
                with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                    data = json.load(f)
                    row = []
                    col = []
                    for k,_ in data.items():
                        for x in data[k]:
                            row.append(int(k))
                            col.append(int(x))
                    # for line in data:
                    #     line = line.strip('\n').strip().split()
                    #     row.append(int(line[0]))
                    #     col.append(int(line[1]))
                    indices = np.vstack((row, col))
                    indices = torch.LongTensor(indices)

                    values = torch.ones(len(row), dtype=torch.float32)
                    self.user_behaviour_degree.append(torch.sparse.FloatTensor(indices,
                                                                               values,
                                                                               [self.user_count + 1,
                                                                                self.item_count + 1])
                                                      .to_dense().sum(dim=1).view(-1, 1))
                    col = [x + self.user_count + 1 for x in col]
                    row, col = [row, col], [col, row]
                    row = torch.LongTensor(row).view(-1)
                    all_row.append(row)
                    col = torch.LongTensor(col).view(-1)
                    all_col.append(col)
                    edge_index = torch.stack([row, col])
                    self.edge_index[behavior] = edge_index

        self.user_behaviour_degree = torch.cat(self.user_behaviour_degree, dim=1)
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        self.all_edge_index = torch.stack([all_row, all_col])

    # 写一个函数用于行为列表需要更新的时候，返回加入不喜欢行为的行为列表
    def behaviors_add(self):
        return self.behaviors+['dislike_one','dislike_sec']

    # 返回一个BehaviorDate类的对象，用于处理训练样本数据
    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)

    # 返回一个TestDate类的对象，用于处理验证样本数据
    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    # 返回一个BehaviorDate类的对象，用于处理训练样本数据
    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Tmall', help='')
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=1024)
    for index, item in enumerate(loader):
        print(index, '-----', item)
