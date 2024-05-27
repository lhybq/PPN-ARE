import scipy.sparse as sp
import numpy as np
import json
import os
import random
import shutil

import numpy as np
from loguru import logger
import scipy.sparse as sp
from collections import defaultdict
from collections import defaultdict
if __name__ == '__main__':


    # # # 读取二进制文件
    # # binary_file = 'trn_buy.csr'
    # # sparse_matrix = sp.load_npz(binary_file, allow_pickle=True)
    # #
    # # # 将稀疏矩阵转换为稠密矩阵，并将其保存为文本文件
    # # output_file = 'file.txt'
    # # dense_matrix = sparse_matrix.toarray()
    # # np.savetxt(output_file, dense_matrix, delimiter=' ', fmt='%d')
    #
    # buy_dict = {'key1': [1, 2, 3,5,65,4,564,8], 'key2': [3, 4]}
    # # cart_dict = {'key2': [5, 6], 'key3': [7, 8]}
    # # collect_dict = {'key3': [9, 10], 'key4': [11, 12]}
    # click_dict = {'key1': [1,2,3,5,13, 14], 'key5': [15,0,6,2,5,16]}
    #
    # dislike_dic_one = {}
    #
    # for k, v in click_dict.items():
    #     if k not in buy_dict:
    #         dislike_dic_one[k] = v
    #     else:
    #         dislike_dic_one[k] = [x for x in click_dict[k] if x not in buy_dict[k]]
    #
    # print(dislike_dic_one)
    #
    # result = click_dict.copy()
    # result.update(buy_dict)
    #
    # result={**buy_dict,**click_dict}
    # print(result)
    #
    # from collections import defaultdict
    # result = defaultdict(list)
    # for key, val in buy_dict.items():
    #     result[key].extend(val)
    # for key, val in click_dict.items():
    #     result[key].extend(val)
    # result = {key: list(set(val)) for key, val in result.items()}
    #
    # print(result)
    #
    #
    # A=['1','RT','TRTY','TREQQ']
    # print(A+['123'])
    # print(A)
    user_interaction = {}
    path = '.'
    file ='testdata.txt'
    with open(os.path.join(path, file)) as f:
        data = f.readlines()
        for row in data:
            listitem = row.strip().split()
            for i in range(1,listitem.__len__()):
                listitem[i]=int(listitem[i])

            if listitem[0] not in user_interaction:
                 user_interaction[listitem[0]] = listitem[1:]
                 continue
    file = 'testdata.txt'
    with open(os.path.join(path,"1"+file)) as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            item = int(item)

            if user not in user_interaction:
                user_interaction[user] = [item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
                continue


        print(listitem)
        print(listitem[1:])
        print(user_interaction)



            # if user not in user_interaction:
            #     user_interaction[user] = [item]
            # elif item not in user_interaction[user]:
            #     user_interaction[user].append(item)




































































