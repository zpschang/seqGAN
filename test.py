import tensorflow as tf

from model import generator_model, discriminator_model
from reader import reader
import re
UNK_ID = 2
reader = reader('data/small/weibo_pair_train_Q.post',
    'data/small/weibo_pair_train_Q.response', 'data/words.txt')

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
sess = tf.Session()
loader = tf.train.import_meta_graph('saved/model.ckpt.meta')
loader.restore(sess, tf.train.latest_checkpoint('saved/'))
print 'load finished'

while True:
    post = raw_input()
    batch = generate_batch(post)
    resp = g_model.generate(sess, batch, 'beam_search')
    print resp
    resp = resp[0]
    for word in resp:
        for index in word:
            print reader.symbol[index] if index >= 0 else 'unk',
        print '\n',
    print '\n',