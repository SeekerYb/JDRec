#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "cuiyongxiong@jd.com"

import tensorflow as tf

def sample(tables, k):
    """
    Args:
      tables: tensor with shape [batch_size, candidate_item_num, sequence_length]
      k: int, number of list to sample for each table
    return:
      samples: tensor with shape [batch_size, k, sequence_length].Each element is
        the index of the item
    """
    batch_size = tf.shape(tables)[0]
    item_num = tf.shape(tables)[1]
    sequence_length = tables.shape[2]
    s=[]
    zeros = tf.zeros([k * batch_size, sequence_length])
    k_index = tf.expand_dims(tf.range(k * batch_size),1)
    tables = tf.tile(tables,[k,1,1]) # [k * batch_size, item_num, sequence_length]
    for pos in range(sequence_length):
        item_index = tf.random.categorical(tf.math.log(tables[:,:,pos]), 1) #[k * batch_size,1]
        item_index = tf.cast(item_index,tf.int32) #int64 to int32
        item_index = tf.where(tf.math.equal(item_index,item_num),item_num - pos -1, item_index)#if all elements in table is zero,item index will out of range
        s.append(item_index)
        index = tf.concat([k_index,item_index],axis=1)
        tables = tf.tensor_scatter_nd_update(tables, index, zeros)
    samples = tf.concat(s,axis=1)
    samples = tf.reshape(samples, [k,batch_size,sequence_length])
    samples = tf.transpose(samples,perm=[1,0,2]) #[batch_size, k, sequence_length]
    return samples 

def sample_greedy(tables, total_k, explore_k):
    """
    Args:
      tables: tensor with shape [batch_size, candidate_item_num, sequence_length]
      k: int, number of list to sample for each table
    return:
      samples: tensor with shape [batch_size, k, sequence_length].Each element is
        the index of the item
    """
    k = total_k
    exploit_k = total_k - explore_k
    batch_size = tf.shape(tables)[0]
    item_num = tf.shape(tables)[1]
    sequence_length = tables.shape[2]
    s=[]
    zeros = tf.zeros([k * batch_size, sequence_length])
    k_index = tf.expand_dims(tf.range(k * batch_size),1)
    explore_tables = tf.ones([explore_k * batch_size, item_num, sequence_length]) * 0.5
    tables = tf.tile(tables,[exploit_k,1,1]) # [k * batch_size, item_num, sequence_length]
    tables = tf.concat([tables,explore_tables],axis=0)
    for pos in range(sequence_length):
        item_index = tf.random.categorical(tf.math.log(tables[:,:,pos]), 1) #[k * batch_size,1]
        item_index = tf.cast(item_index,tf.int32) #int64 to int32
        #item_index = tf.clip_by_value(item_index, clip_value_min = 0, clip_value_max = item_num - pos - 1) 
        item_index = tf.where(tf.math.equal(item_index,item_num),item_num - pos -1, item_index)#if all elements in table is zero,item index will out of range
        s.append(item_index)
        index = tf.concat([k_index,item_index],axis=1)
        tables = tf.tensor_scatter_nd_update(tables, index, zeros)
    samples = tf.concat(s,axis=1)
    samples = tf.reshape(samples, [k,batch_size,sequence_length])
    samples = tf.transpose(samples,perm=[1,0,2]) #[batch_size, k, sequence_length]
    return samples 

def sample_max(tables):
    """
    Args:
      tables: tensor with shape [batch_size, candidate_item_num, sequence_length]
      k: int, number of list to sample for each table
    return:
      samples: tensor with shape [batch_size, sequence_length].Each element is
        the index of the item
    """
    batch_size = tf.shape(tables)[0]
    item_num = tables.shape[1]
    sequence_length = tables.shape[2]
    s=[]
    neg_ones = -tf.ones([batch_size, sequence_length])
    k_index = tf.expand_dims(tf.range(batch_size),1)
    for pos in range(sequence_length):
        item_index = tf.math.argmax(tables[:,:,pos],1) #[batch size]
        item_index = tf.expand_dims(item_index,1) 
        item_index = tf.cast(item_index,tf.int32) #int64 to int32
        s.append(item_index)
        index = tf.concat([k_index,item_index],axis=1)
        tables = tf.tensor_scatter_nd_update(tables, index, neg_ones)
    samples = tf.concat(s,axis=1) #[batch_size,sequence_length]
    return samples 

def sample_by_index(rerank_index,sequence_length=4):
    """
    Args:
      rerank_index: tensor with shape [batch_size, candidate_item_num]
    return:
      samples: tensor with shape [batch_size, sequence_length].Each element is
        the index of the item
    """
    return tf.math.top_k(-1 * rerank_index,sequence_length)[1]

def sample_by_pctr(pctrs,sequence_length=4):
    """
    Args:
      pctrs: tensor with shape [batch_size, candidate_item_num]
    return:
      samples: tensor with shape [batch_size, sequence_length].Each element is
        the index of the item
    """
    return tf.math.top_k(pctrs,sequence_length)[1]

##test
##t=tf.ones([10,6,4]) * 0.1
#t=tf.ones([128,40,4]) * 0.1
#out = sample(t,3)
#print(out)

