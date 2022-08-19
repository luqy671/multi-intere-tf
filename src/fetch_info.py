#coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

import faiss
import tensorflow as tf
from data_iterator import DataIterator
from model import *
from mymodel import *
from model_SINE import *
#from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book14', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--cand_num', type=int, default=400)
parser.add_argument('--model_type', type=str, default='Mine', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='default_0')

parser.add_argument('--f_mycand', action='store_false')
parser.add_argument('--f_encoder', action='store_false')
parser.add_argument('--sa_dim', type=int, default=64)
parser.add_argument('--f_trans', action='store_false')
parser.add_argument('--trans_k', type=int, default=12)
parser.add_argument('--trans_h', type=int, default=32)
parser.add_argument('--trans_p', type=float, default=1.0)
parser.add_argument('--f_auxloss', action='store_true')
parser.add_argument('--loss_k', type=float, default=1.0)

best_metric = 0

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def retrieval(sess, test_data, model, model_path, batch_size, save=True):
    
    topN = args.topN
    
    #item_embs = model.output_gru_item(sess)
    item_embs = model.output_item(sess)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    
    write_list = []
    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
   
        user_embs, intere_idx = model.output_info(sess, [hist_item, hist_mask])
        # (batch,intere_num,embed_dim) (batch,intere_num) 
        
        ni = user_embs.shape[1]
        user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]]) #(batch*intere_num,embed_dim)
        D, I = gpu_index.search(user_embs, topN) #(batch*intere_num,topN)
        
        nick_ids = [nick_id[i] for i in range(len(nick_id)) for j in range(ni)] #(batch*intere_num)
        intere_idx = np.reshape(intere_idx, [-1]) #(batch*intere_num)
        for i, u_id in enumerate(nick_ids):
            strI = [str(j) for j in I[i]]
            write_list.append(str(u_id) +','+ str(intere_idx[i]) +','+  ",".join(strI) + '\n')
        
    return write_list 


def fetch_info(
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001,
        exp_name = 'default_0'
):
    #exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = Model_Mine(item_count, args.embedding_dim, args.hidden_size, batch_size, 
                       args.num_interest, args.cand_num, maxlen, args.f_mycand, 
                       args.f_encoder, args.sa_dim, args.f_trans, args.trans_k, args.trans_h,
                       args.trans_p, args.f_auxloss, args.loss_k)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        write_list = retrieval(sess, test_data, model, best_model_path, batch_size, save=False)
    
    write_file = "analysis/"+exp_name+'_info.txt'
    with open(write_file, 'w') as f:
        for line in write_list:
            f.write(line)
                    


if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'
    
    print('DataSet: {} ({})'.format(args.dataset,args.topN))

    if args.dataset == 'elec14':
        path = './data/elec14_data/'
        item_count = 63002
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    elif args.dataset == 'book14':
        path = './data/book14_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    fetch_info(test_file=test_file, cate_file=cate_file, item_count=item_count, 
               dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate, exp_name=args.exp_name)
    
