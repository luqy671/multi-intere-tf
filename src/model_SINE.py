import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.nn.rnn_cell import GRUCell
from model import *

class Model_SINE(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, cand_num, seq_len=256):
        super(Model_SINE, self).__init__(n_mid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="Model_SINE")
        
        #----------------------------------init begin---------------------------------------------
        
        print('*'*25 + 'This is model SINE (cand_num:{})'.format(cand_num) + '*'*25)

        self.W1 = tf.get_variable("W1", [embedding_dim, embedding_dim], trainable=True) #(embed_dim, embed_dim)
        self.W2 = tf.get_variable("W2", [embedding_dim], trainable=True) #(embed_dim)
        self.W3 = tf.get_variable("W3", [embedding_dim, embedding_dim], trainable=True) #(embed_dim, embed_dim) 
        self.W4 = tf.get_variable("W4", [embedding_dim], trainable=True) #(embed_dim)
        self.W_k1 = tf.get_variable("W_k1", [num_interest, embedding_dim, embedding_dim], trainable=True) #(intere_num, embed_dim, embed_dim)
        self.W_k2 = tf.get_variable("W_k2", [num_interest, embedding_dim], trainable=True) #(intere_num, embed_dim)
        
        self.C = tf.get_variable("conceptual_prototype_matrix", [cand_num, embedding_dim], trainable=True) #(cand_num, att_dim)
    
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.norm4 = tf.keras.layers.LayerNormalization()
        #-------------------------------------init end---------------------------------------------
        
        
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim]) #(batch,seq_len,embed_dim)

        self.position_embedding = \
            tf.get_variable(
                shape=[1, seq_len, embedding_dim],
                name='position_embedding') #(1,seq_len,embed_dim)
        batch_pos_eb = tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1]) #(batch,seq_len,embed_dim)
        item_list_add_pos = item_list_emb + batch_pos_eb #(batch,seq_len,embed_dim)
        
       
        with tf.variable_scope("Concept_activation", reuse=tf.AUTO_REUSE) as scope:     
            item_list_hidden = tf.math.tanh(tf.einsum('bte,ea->bta',item_list_add_pos,self.W1)) #(batch,seq_len,embed_dim)
            att_score = tf.einsum('bte,e->bt', item_list_hidden, self.W2) #(batch,seq_len)
            att_mask = self.mask #(batch,seq_len)
            activ_paddings = tf.ones_like(att_mask) * (-2 ** 32 + 1) #(batch,seq_len)
            att_score = tf.where(tf.equal(att_mask, 0), activ_paddings, att_score) #(batch,seq_len)
            a = tf.nn.softmax(att_score) #(batch,seq_len)
            
            z_u = tf.einsum('bte,bt->be',item_list_add_pos,a) #(batch,embed_dim)
            s_u = tf.einsum('be,ce->bc',z_u,self.C) #(batch,cand_num)
            indices_K = tf.argsort(s_u, axis=-1, direction='DESCENDING')[:,:num_interest] #(batch,intere_num)
            c_u = tf.gather(self.C, indices_K) #(batch,intere_num,embed_dim) 
            s_u_k = tf.sort(s_u, axis=-1, direction='DESCENDING')[:,:num_interest] #(batch,intere_num)
            c_u = tf.einsum('bke,bk->bke', c_u, tf.math.sigmoid(s_u_k)) #(batch,intere_num,embed_dim) 
       
        with tf.variable_scope("Interest_embedding_generation", reuse=tf.AUTO_REUSE) as scope:
            tmp_h1 = tf.einsum('bte,ea->bta',item_list_add_pos,self.W3) #(batch,seq_len,embed_dim)        
            tmp_h1 = tf.einsum('bte,bke->btk', self.norm1(tmp_h1), self.norm2(c_u)) #(batch,seq_len,intere_num)
            h1_mask = tf.tile(tf.expand_dims(self.mask, axis=2), [1, 1, num_interest]) #(batch,seq_len,intere_num)
            h1_paddings = tf.ones_like(h1_mask) * (-2 ** 32 + 1)
            tmp_h1 = tf.where(tf.equal(h1_mask, 0), h1_paddings, tmp_h1)
            P_kt = tf.nn.softmax(tmp_h1) #(batch,seq_len,intere_num)
            P_kt = tf.transpose(P_kt, [0, 2, 1]) #(batch,intere_num,seq_len)
            
            tmp_h2 = tf.math.tanh(tf.einsum('bte,kea->bkta',item_list_add_pos,self.W_k1)) #(batch,intere_num,seq_len,embed_dim)
            tmp_h2 = tf.einsum('bkte,ke->bkt',tmp_h2,self.W_k2) #(batch,intere_num,seq_len)
            h2_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_interest, 1]) #(batch,intere_num,seq_len)
            h2_paddings = tf.ones_like(h2_mask) * (-2 ** 32 + 1)
            tmp_h2 = tf.where(tf.equal(h2_mask, 0), h2_paddings, tmp_h2)
            P_tk =  tf.nn.softmax(tmp_h2)  #(batch,intere_num,seq_len)
            P = tf.math.multiply(P_kt, P_tk) #(batch,intere_num,seq_len)
            
            interest_emb = self.norm3(tf.einsum('bte,bkt->bke',item_list_add_pos, P)) #(batch,intere_num,embed_dim)
            
        with tf.variable_scope("Interest_Aggregation_Module", reuse=tf.AUTO_REUSE) as scope:
            P_u = tf.transpose(P_kt, [0, 2, 1]) #(batch,seq_len,intere_num)
            X_hat = tf.einsum('btk,bke->bte', P_u, c_u) #(batch,seq_len,embed_dim)
            
            tmp_h3 = tf.math.tanh(tf.einsum('bte,ea->bta',X_hat,self.W3)) #(batch,seq_len,embed_dim)        
            tmp_h3 = tf.einsum('bte,e->bt', tmp_h3, self.W4) #(batch,seq_len)
            h3_mask = self.mask
            h3_paddings = tf.ones_like(h3_mask) * (-2 ** 32 + 1)
            tmp_h3 = tf.where(tf.equal(h3_mask, 0), h3_paddings, tmp_h3)
            tmp_h3 = tf.nn.softmax(tmp_h3) #(batch,seq_len)
            c_apt = self.norm4(tf.einsum('bte, bt->be', X_hat, tmp_h3)) #(batch,embed_dim)
            e_u = tf.nn.softmax(tf.einsum('be,bke->bk', c_apt, interest_emb)/0.1) #(batch,intere_num)
            v_u = tf.einsum('bke, bk->be', interest_emb, e_u) #(batch, embed_dim)    
            
        self.user_eb = v_u #(batch,embed_dim)
        
        mean_C = tf.reduce_mean(self.C, axis=1, keepdims=True)
        cov_C = tf.matmul(self.C-mean_C,tf.transpose(self.C-mean_C))/tf.cast(embedding_dim, tf.float32)

        F2_C = tf.reduce_sum(tf.math.square(cov_C))
        diag_F2_C = tf.reduce_sum(tf.matrix_diag_part(tf.math.square(cov_C)))
        loss_C = 0.5*(F2_C - diag_F2_C)

        self.build_sampled_softmax_loss(self.item_eb, self.user_eb, 0.5*loss_C)
        
    def build_sampled_softmax_loss(self, item_emb, user_emb, aux_loss):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid)) + aux_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)