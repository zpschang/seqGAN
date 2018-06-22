# encoding=utf8

import tensorflow as tf

from model import generator_model, discriminator_model
from reader import reader
import re

gpu_rate = 0.25
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_rate)  

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

UNK_ID = 2
reader = reader('data/small/weibo_pair_train_Q.post',
    'data/small/weibo_pair_train_Q.response', 'data/words_99%.txt')

def generate_batch(post):
    post = post.decode('utf-8')
    words_post = re.split(' ', post)
    index_post = [reader.d[word] if word in reader.d else UNK_ID for word in words_post]
    return [(index_post, [])]

g_model = generator_model(vocab_size=len(reader.d),
    embedding_size=128,
    lstm_size=128,
    num_layer=4,
    max_length_encoder=40,
    max_length_decoder=40,
    max_gradient_norm=2,
    batch_size_num=20,
    learning_rate=0.001,
    beam_width=5)
d_model = discriminator_model(vocab_size=len(reader.d),
    embedding_size=128,
    lstm_size=128,
    num_layer=4,
    max_post_length=40,
    max_resp_length=40,
    max_gradient_norm=2,
    batch_size_num=20,
    learning_rate=0.001)

saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
loader = tf.train.import_meta_graph('saved/model.ckpt.meta')
loader.restore(sess, tf.train.latest_checkpoint('saved/'))
print 'load finished'

from_screen = raw_input('is input from screen: (y)/n')
from_screen = False if from_screen == 'n' else True

if not from_screen:
    file_input = open('data/small/test.post', 'r')
    bs_output = open('data/small/test_bs.response', 'w')
    sample_output = open('data/small/test_sample.response', 'w')

while True:
    if from_screen:
        post = raw_input()
    else:
        post = file_input.readline()
    batch = generate_batch(post)
    resp = g_model.generate(sess, batch, 'beam_search')
    print resp
    resp = resp[0]

    print 'beam search'
    result = ''
    for sentence in resp:
        for index in sentence:
            result += reader.symbol[index] if index >= 0 else 'unk'
            result += ' '
        result += '\n'
    result += '\n'
    if from_screen:
        print result,
    else:
        bs_output.write(result)

    resp = g_model.generate(sess, batch, 'sample')
    resp = resp[0]

    print 'sample'
    result = ''
    for word in resp:
        result += reader.symbol[word] if word >= 0 else 'unk'
        result += ''
    result += '\n'
    if from_screen:
        print result,
    else:
        sample_output.write(result)