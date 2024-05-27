#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_process.py
# @Author:
# @Date  : 2021/11/1 10:30
# @Desc  :
import json
import os
import random
import shutil

import numpy as np
from loguru import logger
import scipy.sparse as sp
from collections import defaultdict


def generate_dict(path, file):
    user_interaction = {}
    with open(os.path.join(path, file)) as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            user, item = int(user), int(item)

            if user not in user_interaction:
                user_interaction[user] = [item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
    return user_interaction


@logger.catch()
def generate_interact(path):
    buy_dict = generate_dict(path, 'buy.txt')
    with open(os.path.join(path, 'buy_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(buy_dict))
        total_count = sum(len(v) for v in buy_dict.values())
        print("buy_dict：", total_count)

    cart_dict = generate_dict(path, 'cart.txt')
    with open(os.path.join(path, 'cart_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(cart_dict))

    view_dict = generate_dict(path, 'view.txt')
    with open(os.path.join(path, 'view_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(view_dict))

    # 生成一个点击后的，不采取下一步操作的交互视图
    # 先将所有buy、cart集合在一起,放在new_dict中
    new_dict = cart_dict
    for dic in [buy_dict]:
        for k, v in dic.items():
            if k not in new_dict:
                new_dict[k] = v
            item = new_dict[k]
            item.extend(v)
    for k, v in new_dict.items():
        item = new_dict[k]
        item = list(set(item))
        new_dict[k] = sorted(item)
    # 先从View视图中，剔除Cart、Buy作为用户完全不喜欢的初始视图
    dislike_dic_one = {}
    for k, v in view_dict.items():
        if k not in new_dict:
            dislike_dic_one[k] = v
        else:
            dislike_dic_one[k] = [x for x in view_dict[k] if x not in new_dict[k]]
    for k, v in dislike_dic_one.items():
        item = dislike_dic_one[k]
        item = list(set(item))
        dislike_dic_one[k] = sorted(item)
    with open(os.path.join(path, 'dislike_one_dict.txt'), 'w', encoding='utf-8') as f:  # 点击后没有下一步操作的
        f.write(json.dumps(dislike_dic_one))
        # 计算总数量
        total_count = sum(len(v) for v in dislike_dic_one.values())
        print("dislike_dic_one：", total_count)

    # 不喜欢视图中再加上，收藏、加购物车后没有购买的
    dislike_dic_sec = {}
    for k, v in new_dict.items():
        if k not in buy_dict:
            dislike_dic_sec[k] = v
        else:
            dislike_dic_sec[k] = [x for x in new_dict[k] if x not in buy_dict[k]]
        if not dislike_dic_sec[k]:
            dislike_dic_sec.pop(k)
    total_count = sum(len(v) for v in dislike_dic_sec.values())
    print("dislike_dic_sec：", total_count)
    # 考虑残差的话，加上之前的dislike_one_dict，求dislike_one_dict和dislike_sec_dict的并集
    uni = defaultdict(list)
    for k, v in dislike_dic_one.items():
        uni[k].extend(v)
    for k, v in dislike_dic_sec.items():
        uni[k].extend(v)
    uni = {k: list(set(v)) for k, v in uni.items()}
    dislike_dic_sec = uni
    for k, v in dislike_dic_sec.items():
        item = dislike_dic_sec[k]
        item = list(set(item))
        dislike_dic_sec[k] = sorted(item)
    # for k, v in dislike_dic_one.items():
    #     if k not in dislike_dic_sec:
    #         dislike_dic_sec[k] = v
    #     else:
    #         item = dislike_dic_sec[k]
    #         item.extend(v)
    # # 结合后去掉重复的
    # for k, v in dislike_dic_sec.items():
    #     item = dislike_dic_sec[k]
    #     item = list(set(item))
    #     dislike_dic_sec[k] = sorted(item)
    # 如果不考虑残差的话，就不求并集
    with open(os.path.join(path, 'dislike_sec_dict.txt'), 'w', encoding='utf-8') as f:  # 加购物车后没有购买的视图
        f.write(json.dumps(dislike_dic_sec))
        total_count = sum(len(v) for v in dislike_dic_sec.values())
        print("dislike_dic_sec：", total_count)

    for k, v in new_dict.items():
        if k not in view_dict:
            view_dict[k] = v
        item = view_dict[k]
        item.extend(v)
    for k, v in view_dict.items():
        item = view_dict[k]
        item = list(set(item))
        view_dict[k] = sorted(item)
    with open(os.path.join(path, 'all_train_interact_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(view_dict))


    shutil.copyfile('buy_dict.txt', 'train_dict.txt')

    test_dict = generate_dict(path, 'test.txt')
    with open(os.path.join(path, 'test_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict))

    validation_dict = generate_dict(path, 'validation_tst.txt')
    for k, v in validation_dict.items():
        if k in test_dict:
            validation_dict[k] = [x for x in validation_dict[k] if x not in test_dict[k]]
    validation_dict = {k: v for k, v in validation_dict.items() if v !=[]}
    with open(os.path.join(path, 'validation_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(validation_dict))


def generate_all_interact(path):
    all_dict = {}
    files = ['view', 'cart', 'buy']
    for file in files:
        with open(os.path.join(path, file+'_dict.txt')) as r:
            data = json.load(r)
            for k, v in data.items():
                if all_dict.get(k, None) is None:
                    all_dict[k] = v
                else:
                    total = all_dict[k]
                    total.extend(v)
                    all_dict[k] = sorted(list(set(total)))
        with open(os.path.join(path, 'all.txt'), 'w') as w1, open(os.path.join(path, 'all_dict.txt'), 'w') as w2:
            for k, v in all_dict.items():
                for i in v:
                    w1.write('{} {}\n'.format(int(k), i))
            w2.write(json.dumps(all_dict))




if __name__ == '__main__':
    path = '.'
    generate_interact(path)
    generate_all_interact(path)

