import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.nn.rnn_cell import GRUCell
from model import *

class Model_Mine(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, cand_num, seq_len=256, f_mycand=True, f_encoder=True, sa_dim=64, f_trans=True):
        super(Model_Mine, self).__init__(n_mid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="Model_Mine")
        
        #----------------------------------my init begin---------------------------------------------
        print('*'*25 + 'This is my model' + '*'*25)
        
        print('*'*25 + ' f_mycand: {} --- f_encoder: {} ---- f_trans: {}'.format(f_mycand, f_encoder, f_trans))
        print('+'*25 + ' num_interest: {} --- cand_num: {} '.format(num_interest, cand_num))
        
        att_dim = embedding_dim
        self.W1 = tf.get_variable("W1", [embedding_dim, att_dim], trainable=True) #(embed_dim, att_dim) 
        
        if f_mycand:
            print('*'*25+'candidate genarate from eb'+'*'*25)
           
            self.W2 = tf.get_variable("W2", [num_interest, embedding_dim], trainable=True) #(intere_num, embed_dim)
            att_w = tf.einsum('ij,kj->ik',self.mid_embeddings_var, self.W2) #(item_num, intere_num)
            wmean_eb = tf.einsum('ik,ij->kj',att_w, self.mid_embeddings_var) #(intere_num, embed_dim)
            self.W3 = tf.get_variable("W3", [num_interest,cand_num], trainable=True) #(intere_num, cand_num)
            candidates = tf.einsum('ki,kj->ij',self.W3, wmean_eb) #(cand_num, embed_dim)
            self.candidates = tf.math.tanh(tf.einsum('ij,jk->ik',candidates, self.W1))#(cand_num, att_dim)
        else:
            self.candidates = tf.get_variable("candidates_pool", [cand_num, att_dim], trainable=True) #(cand_num, att_dim)
            
        if f_trans:
            print('*'*25+ 'trans_matrix: cand0[w4] X cand0 + softmax '+'*'*25)
            self.W4 = tf.get_variable("W4", [embedding_dim,embedding_dim], trainable=True)
            trans_hidden = tf.einsum('ij,jk->ik', candidates, self.W4) #(trans_h, cand_num)
            self.trans_matrix = tf.nn.softmax(tf.einsum('ij,kj->ik', trans_hidden, candidates), dim=0) #(cand_num, cand_num)
            
        #-------------------------------------my init end---------------------------------------------
             
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim]) #(batch,seq_len,embed_dim)

        self.position_embedding = \
            tf.get_variable(
                shape=[1, seq_len, embedding_dim],
                name='position_embedding') #(1,seq_len,embed_dim)
        batch_pos_eb = tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1]) #(batch,seq_len,embed_dim)
        item_list_add_pos = item_list_emb + batch_pos_eb #(batch,seq_len,embed_dim)

        #-----------------------------sa encoder begin----------------------------------------------------
        if f_encoder:
            print('*'*25+'my SA Encoder (sa dim : {})'.format(sa_dim)+'*'*25)
            SA_q = tf.nn.dropout(tf.layers.dense(item_list_add_pos, sa_dim, activation=tf.nn.tanh), keep_prob = 0.8)  # (batch,seq_len,sa_dim)
            #SA_q = tf.math.tanh(tf.einsum('ijk,kl->ijl',item_list_add_pos, self.W1))
            #SA_q = item_list_add_pos
            SA_k = SA_q
            
            SA_att_w = tf.einsum('bql,bkl->bqk', SA_q, SA_k)/tf.math.sqrt(64.0) #(batch, q, k)
            SA_atten_mask = tf.einsum('bq,bk->bqk',self.mask, self.mask) #(batch, q, k)
            SA_paddings = tf.ones_like(SA_atten_mask) * (-2 ** 32 + 1)

            SA_att_w = tf.where(tf.equal(SA_atten_mask, 0), SA_paddings, SA_att_w) #(batch, q, k)
            SA_att_w = tf.nn.softmax(SA_att_w)  #(batch, q, k)
            SA_att_w_mask = tf.where(tf.equal(SA_atten_mask, 0), tf.zeros_like(SA_atten_mask), SA_att_w) #(batch, q, k)
            
            item_eb_seq = tf.einsum('bke, bqk -> bqe', item_list_emb, SA_att_w_mask) #(batch, q, embed_dim)
            item_eb_seq_pos = item_eb_seq + batch_pos_eb #(batch, q, embed_dim)
             
        else:
            item_eb_seq = item_list_emb #(batch,seq_len,embed_dim)
            item_eb_seq_pos = item_list_add_pos #(batch,seq_len,embed_dim)
        #-----------------------------sa encoder end----------------------------------------------------------------
        
        #-----------------------------active candidates begin-----------------------------------------------
        with tf.variable_scope("activ_cand", reuse=tf.AUTO_REUSE) as scope:                        
            print('*'*25+'activ cand work'+'*'*25)
            masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1) #(batch,seq_len,embed_dim)
            item_his_eb_mean = tf.reduce_sum(item_eb_seq_pos, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9) #(batch,embed_dim)
            mean_hidden = tf.math.tanh(tf.einsum('ij,jk->ik',item_his_eb_mean,self.W1)) #(batch,att_dim)
            logits_out = tf.einsum('ij,kj->ik', mean_hidden, self.candidates) #(batch,cand_num)
            
            if f_trans:
                print('*'*25+'my trans work ' +'*'*25)
                logits_out = tf.einsum('bi,ij->bj', 
                                       logits_out, self.trans_matrix) #(batch,cand_num)   
            
            indices_K = tf.argsort(logits_out, axis=-1, direction='DESCENDING')[:,:num_interest] #(batch,intere_num)   
            self.intere_idx = indices_K
            activ_query = tf.gather(self.candidates, indices_K) #(batch,intere_num,att_dim)   
            
        #----------------------------active candidates end---------------------------------------    
            
        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.math.tanh(tf.einsum('ijk,kl->ijl',item_eb_seq_pos,self.W1)) # (batch,seq_len,att_dim)
            item_att_w =  tf.einsum('ijk,ilk->ijl', item_hidden, activ_query) # (batch,seq_len,intere_num)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1]) # (batch,intere_num,seq_len)

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1]) #(batch,intere_num,seq_len)
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1) #(batch,intere_num,seq_len)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w) #(batch,intere_num,seq_len)
            item_att_w = tf.nn.softmax(item_att_w) #(batch,intere_num,seq_len)

            interest_emb = tf.matmul(item_att_w, item_eb_seq) #(batch,intere_num,embed_dim)

        self.user_eb = interest_emb #(batch,intere_num,embed_dim)

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1])) #(batch,intere_num,1)
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1)) #(batch,intere_num)

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads) #(batch,embed_dim)  
        
        '''
        aux_loss = 0.0
        if f_auxloss:
            print('*'*25+'aux loss work (loss_k:{})'.format(loss_k) +'*'*25)
            aim_hidden = tf.math.tanh(tf.einsum('ij,jk->ik',self.item_eb, self.W1)) #(batch,att_dim)
            aim_cand_score = tf.einsum('ik,jk->ij',aim_hidden, self.candidates) #(batch,cand_num)
            aim_cand_hit = tf.math.argmax(aim_cand_score, axis=-1) #(batch)
            aux_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=aim_cand_hit, logits=logits_out))

        aux_loss *= loss_k'''
        
        self.build_sampled_softmax_loss(self.item_eb, readout)
        
    def build_sampled_softmax_loss(self, item_emb, user_emb, aux_loss=0.0):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid)) + aux_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
    def output_info(self, sess, inps):
        user_embs, intere_idx= sess.run((self.user_eb, self.intere_idx), feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs, intere_idx    # (batch,intere_num,embed_dim) (batch,intere_num) 