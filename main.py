
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from data_set import DataSet

from model_cascade import CRGCN
# from model_cascade_fuse_weight import CRGCN

from trainer import Trainer

SEED = 2023
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    # parser.add_argument('--dislike_kind',  help='', action='append')
    parser.add_argument('--dislike', type=bool, default=True) # 
    parser.add_argument('--b', type=float, default=3)  #
    parser.add_argument('--cbeta', type=float, default=0.5)  #

    # 
    parser.add_argument('--CL', type=bool, default=True)
    parser.add_argument('--CL_view', type=str, default='cart')
    parser.add_argument('--calph', type=float, default=1)

    # 
    parser.add_argument('--sample', type=bool, default=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=0.001, help='')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)

    parser.add_argument('--data_name', type=str, default='Tmall', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')

    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=3072, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=400, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')


    args = parser.parse_args()
    if args.data_name == 'Tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'cart', 'collect', 'buy']
        args.dis_behaviors = ['dislike_one', 'dislike_sec']
        # args.lr = 0.01
        # args.reg_weight = 0.001
        args.layers = [1 ,1, 1, 1, 1, 1]
        args.model_name = 'Tmall'
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['view', 'cart', 'buy']
        # args.behaviors = ['cart']
        args.dis_behaviors = ['dislike_one', 'dislike_sec']
        args.layers = [1,1,1,1,1]
        args.model_name = 'beibei'
    else:
        raise Exception('data_name cannot be None')

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = '{}_cbeta:{}_calph{}_lr:{}_epoch:{}_neg:{}_dis:{}_enb_{}_b={}_neg:buy_{}_Score+cl+neg_cl({}+SGCN),{},{}'.format(args.cbeta,args.calph,args.model_name, args.lr,args.epochs, args.reg_weight, args.dislike,args.embedding_size,args.b,args.behaviors,args.CL_view,args.layers ,TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = CRGCN(args, dataset)

    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    # trainer.evaluate(0, 12, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



