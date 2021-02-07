#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "cuiyongxiong@jd.com"

import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers,Model
from metrics import roc_auc_score
import sample_tools

tf.random.set_seed(2021)

BATCH_SIZE = 512
LENGTH = 40
TOP_LENGTH = 4

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

value_ranges = [ #field_index, value_range
        (4,47),
        (5,407),
        (6,3833),
        (7,103253),
        (8,102759),
        (9,49210),
        (10,87397),
        (15,5001),
        (16,5001),
        (17,5001),
        (18,500),
        (19,10001),
        (20,1001),
        (21,1001),
        (22,2001),
        ] + list(zip(range(23,53),[1000] * 30))


def make_parse_csv(role):
    def parse_csv(lines):
        if role == 'evaluator':
            lines = lines[:TOP_LENGTH]
        else:
            assert role == 'generator'
            lines = lines[TOP_LENGTH:]
        fields_defaults = [int()] * 11 + [float()] * 4 + [int()] * 38
        fields = tf.io.decode_csv(lines,fields_defaults)
        click = fields[0]
        rerank_index = fields[1]
        pctr = fields[11]

        day = tf.math.floordiv(tf.truncatemod(fields[3],86400 * 7),86400)
        hour = tf.math.floordiv(tf.truncatemod(fields[3],86400),3600)
        minute = tf.math.floordiv(tf.truncatemod(fields[3],3600),60)
        sec = tf.truncatemod(fields[3],60)
        
        time_features = [day, hour, minute, sec]
        
        id_features = []
        for (index, _) in value_ranges:
            if index >=23:
                id_features.append(tf.math.floormod(fields[index],1000))
            else:
                id_features.append(fields[index])
        
        float_features = [fields[11],fields[12],fields[13],fields[14]/100]
        numeric = tf.stack(float_features,axis=1)

        features = tuple(id_features + time_features + [numeric,fields[2], pctr])

        if role == 'evaluator':
            labels = tf.clip_by_value(click,clip_value_min = 0, clip_value_max = 1)
            return features,labels 
        else:
            assert role == 'generator'
            labels = rerank_index
            return features,labels
    return parse_csv

def make_dataset(filelist, role,repeat_times=1):
    dataset = tf.data.TextLineDataset(filelist)
    dataset = dataset.batch(TOP_LENGTH+LENGTH)
    dataset = dataset.map(make_parse_csv(role))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(2 * BATCH_SIZE + 1)
    dataset = dataset.repeat(repeat_times)
    return dataset

train_files = ['./data/train_0.csv','./data/train_1.csv','./data/train_2.csv','./data/train_3.csv','./data/train_4.csv','./data/train_5.csv','./data/train_6.csv','./data/train_7.csv','./data/train_8.csv']
eval_files = ['./data/test.csv']

class Generator_CTR(Model):
    
    def __init__(self,generator_class,k=32):
        super(Generator_CTR,self).__init__()
        self.generator = generator_class()
        self.evaluator = Evaluator() 
        self.evaluator.load_weights("./dumps/model/evaluator_listwise")
        self.evaluator.trainable = False
        self.k = k
    
    def call(self, inputs, training):
        policy = self.generator(inputs)[:,:,:-1]
        #sample
        sample_indexes = sample_tools.sample_greedy(policy, self.k,80) #[batch_size, k, TOP_LENGTH]
        sample_indexes = tf.expand_dims(sample_indexes,axis=-1)
        sample_indexes =tf.stop_gradient(sample_indexes)

        #gather
        batch_size = tf.shape(sample_indexes)[0]
        bs_index = tf.range(batch_size)
        bs_index = tf.broadcast_to(bs_index,[TOP_LENGTH,self.k,batch_size])
        bs_index = tf.transpose(bs_index,perm=[2,1,0])
        bs_index = tf.expand_dims(bs_index,axis=-1)
        feature_indexes = tf.concat([bs_index,sample_indexes],axis=3)
        feature_indexes = tf.reshape(feature_indexes,[batch_size * self.k * TOP_LENGTH, 2])
        evaluator_features = []
        for x in inputs:
            t = tf.gather_nd(x,feature_indexes)
            if len(t.shape) == 1:
                t = tf.reshape(t,[batch_size * self.k, TOP_LENGTH])
            else:
                assert len(t.shape) == 2
                t = tf.reshape(t,[batch_size * self.k, TOP_LENGTH,-1])
            evaluator_features.append(t)

        rank_index = tf.range(TOP_LENGTH)
        rank_index = tf.broadcast_to(rank_index,[batch_size, self.k, TOP_LENGTH])
        rank_index = tf.expand_dims(rank_index,axis=-1)
        policy_index = tf.concat([bs_index,sample_indexes,rank_index],axis=3)

        policy = tf.clip_by_value(policy,clip_value_min = 0.000001, clip_value_max = 1)
        
        sample_prop = tf.gather_nd(policy, policy_index) #shape: [batch_size, k, TOP_LENGTH]

        #inference
        pctrs = self.evaluator(evaluator_features)
        if not isinstance(self.evaluator, Evaluator_base):
            pctrs = keras.activations.sigmoid(pctrs)
        pctrs = tf.reshape(pctrs,[batch_size,self.k,TOP_LENGTH])
        pctrs = tf.stop_gradient(pctrs)

        loss = tf.reduce_sum(pctrs,axis=2) * tf.reduce_sum(tf.math.log(sample_prop),axis=2)#log(pq)=log(p)+log(q)  #[batch_size, k]
        loss = -1 / self.k * tf.reduce_sum(loss,axis = 1) # [batch_size]
        loss = tf.reduce_sum(loss)
        return loss

def Evaluator():
    inputs=[]
    concat_layers = []
    for dim in list(zip(*value_ranges))[1]:
        x = layers.Input(shape=(TOP_LENGTH,),dtype=tf.int32)
        e = layers.Embedding(input_dim=dim, output_dim=math.ceil(dim ** 0.25))(x)
        inputs.append(x)
        concat_layers.append(e)

    for dim in (7,24,60,60):
        x = layers.Input(shape=(TOP_LENGTH,),dtype=tf.int32)
        e = layers.Embedding(input_dim=dim, output_dim=math.ceil(dim ** 0.25))(x)
        inputs.append(x)
        concat_layers.append(e)

    numeric = layers.Input(shape=(TOP_LENGTH,4),dtype=tf.float32)
    inputs.append(numeric)
    concat_layers.append(numeric)
    improv = layers.Input(shape=(TOP_LENGTH,),dtype=tf.int32)
    inputs.append(improv)
    pctr = layers.Input(shape=(TOP_LENGTH,),dtype=tf.float32)
    inputs.append(pctr)

    x = tf.concat(concat_layers,2)

    for dim in [256,256,256,128]:
        x = layers.Dense(dim,tf.nn.swish)(x)
        x = layers.BatchNormalization()(x)
    x = tf.reshape(x,[-1,128 * TOP_LENGTH])
    for dim in [512,256,128,64]:
        x = layers.Dense(dim,tf.nn.swish)(x)
        x = layers.BatchNormalization()(x)
    out = layers.Dense(TOP_LENGTH)(x)

    return Model(inputs = inputs, outputs = out) 

def Generator():
    inputs=[]
    concat_layers = []
    for dim in list(zip(*value_ranges))[1]:
        x = layers.Input(shape=(TOP_LENGTH,),dtype=tf.int32)
        e = layers.Embedding(input_dim=dim, output_dim=math.ceil(dim ** 0.25))(x)
        inputs.append(x)
        concat_layers.append(e)

    for dim in (7,24,60,60):
        x = layers.Input(shape=(TOP_LENGTH,),dtype=tf.int32)
        e = layers.Embedding(input_dim=dim, output_dim=math.ceil(dim ** 0.25))(x)
        inputs.append(x)
        concat_layers.append(e)

    numeric = layers.Input(shape=(TOP_LENGTH,4),dtype=tf.float32)
    inputs.append(numeric)
    concat_layers.append(numeric)
    improv = layers.Input(shape=(TOP_LENGTH,),dtype=tf.int32)
    inputs.append(improv)
    pctr = layers.Input(shape=(TOP_LENGTH,),dtype=tf.float32)
    inputs.append(pctr)

    x = tf.concat(concat_layers,2)
    
    for dim in [64,32]:
        x = layers.Dense(dim,tf.nn.swish)(x)
        x = layers.BatchNormalization()(x)
    sum_out = tf.reduce_sum(x, axis = 1)
    for dim in [128]:
        sum_out = layers.Dense(dim,tf.nn.swish)(sum_out)
    sum_out = tf.expand_dims(sum_out, axis = 1)
    for dim in [128]:
        x = layers.Dense(dim,tf.nn.swish)(x)
    x = sum_out + x
    x = layers.BatchNormalization()(x)
    for dim in [64]:
        x = layers.Dense(dim,tf.nn.swish)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dense(TOP_LENGTH + 1)(x)
    out = tf.keras.activations.softmax(x, axis=1)

    return Model(inputs = inputs, outputs = out) 

def custom_evaluator_loss(labels,output):
    ctr_loss = tf.reduce_mean(tf.compat.v1.losses.sigmoid_cross_entropy(labels, output, label_smoothing = 0.001))
    return ctr_loss

def train_evaluator_listwise(evaluator_class=Evaluator):
    model = evaluator_class()
    model.compile(loss=custom_evaluator_loss, optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),)
    model.fit(make_dataset(train_files,'evaluator',2))
    model.save_weights('./dumps/model/evaluator_listwise')

def train_generator_ctr(generator_class=Generator,k=128):
    generator_rl = Generator_CTR(generator_class,k)
    def identify_loss(labels,output):
        return output

    generator_rl.compile(loss=identify_loss, optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),)
    generator_rl.fit(make_dataset(train_files,'generator'),)
    generator_rl.generator.save_weights('./dumps/model/generator_ctr')

def custom_generator_naive_loss(labels,output):
    labels = tf.clip_by_value(labels,clip_value_min = 0, clip_value_max = TOP_LENGTH) # total TOP_LENGTH + 1  classes
    labels = tf.cast(labels,tf.int32)
    rank_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels,output)
    return rank_loss

def train_generator_naive(generator_class=Generator):
    model = generator_class()
    model.compile(loss=custom_generator_naive_loss, optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),)
    model.fit(make_dataset(train_files,'generator'),)
    model.save_weights('./dumps/model/generator_naive')

def eval_evaluator(path,model=None):

    if not model:
        if 'pointwise' in path:
            evaluator = Evaluator_base()
        else:
            evaluator = Evaluator()
            evaluator.load_weights(path)
    else:
        evaluator = model
    click_labels=[]
    click_outs=[]
    steps = 0
    for f, l in make_dataset(eval_files,'evaluator'):
        steps = steps + 1
        click_labels.append(l.numpy())
        click_outs.append(evaluator.predict(f))
    click_label_all = np.concatenate(click_labels,axis=0)
    click_out_all = np.concatenate(click_outs,axis=0)
    if not 'pointwise' in path:
        click_out_all = sigmoid(click_out_all)
    click_auc = roc_auc_score(click_label_all,click_out_all)
    print('click auc:'+str(click_auc))
    return click_auc

class EvalGeneratorPipe(Model):

    def __init__(self, generator,evaluator):
        super(EvalGeneratorPipe,self).__init__()
        self.generator = generator
        self.evaluator = evaluator

    def call(self,inputs):
        policy = self.generator(inputs)[:,:,:-1]
        sample_indexes = sample_tools.sample_max(policy) #[batch_size, TOP_LENGTH]
        sample_indexes = tf.expand_dims(sample_indexes,axis=-1)
        batch_size = tf.shape(sample_indexes)[0]
        bs_index = tf.range(batch_size)
        bs_index = tf.broadcast_to(bs_index,[TOP_LENGTH,batch_size])
        bs_index = tf.transpose(bs_index,perm=[1,0])
        bs_index = tf.expand_dims(bs_index,axis=-1)
        feature_indexes = tf.concat([bs_index,sample_indexes],axis=2)
        feature_indexes = tf.reshape(feature_indexes,[batch_size * TOP_LENGTH, 2])
        evaluator_features = []
        for x in inputs:
            t = tf.gather_nd(x,feature_indexes)
            if len(t.shape) == 1:
                t = tf.reshape(t,[batch_size, TOP_LENGTH])
            else:
                assert len(t.shape) == 2
                t = tf.reshape(t,[batch_size, TOP_LENGTH,-1])
            evaluator_features.append(t)
        pctrs = self.evaluator(evaluator_features)
        if not isinstance(self.evaluator, Evaluator_base):
            pctrs = keras.activations.sigmoid(pctrs)
        pctrs = tf.reshape(pctrs,[batch_size,TOP_LENGTH])
        return pctrs

def eval_generator(path='./dumps/model/generator'):
    #load evaluator
    evaluator = Evaluator() 
    evaluator.load_weights("./dumps/model/evaluator_listwise")
    #load generator
    generator = Generator()
    generator.load_weights(path)
        
    pctr_values = []
    steps = 0
    pipe = EvalGeneratorPipe(generator, evaluator)
    
    for inputs,_ in make_dataset(eval_files,'generator'):
        steps = steps + 1
        pctrs = pipe.predict(inputs)
        pctr_values.append(pctrs)
    ctr_mean = np.mean(np.concatenate(pctr_values))
    print('path:'+ path)
    print("ctr_mean:"+str(ctr_mean))
    return ctr_mean

class Evaluator_base(Model):
    
    def __init(self):
        super(Evaluator_base,self).__init__()

    def call(self, inputs):
        
        pctr = inputs[-1]
        return pctr
 
         

if __name__ == "__main__":
    if len(sys.argv) < 2:
        train_evaluator_listwise()
        train_generator_naive()
        train_generator_ctr()
        eval_evaluator('./dumps/model/evaluator_pointwise')
        eval_evaluator('./dumps/model/evaluator_listwise')
        eval_generator('./dumps/model/generator_naive') 
        eval_generator('./dumps/model/generator_ctr') 
    elif sys.argv[1] == 'train':
        train_evaluator_listwise()
        train_generator_naive()
        train_generator_ctr()
    else:
        assert sys.argv[1] in ['eval','test']
        eval_evaluator('./dumps/model/evaluator_pointwise')
        eval_evaluator('./dumps/model/evaluator_listwise')
        eval_generator('./dumps/model/generator_naive') 
        eval_generator('./dumps/model/generator_ctr') 



   

