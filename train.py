import tensorflow as tf

from model import generator_model, discriminator_model
from reader import reader

reader = reader('data/small/weibo_pair_train_Q.post',
    'data/small/weibo_pair_train_Q.response', 'data/words.txt')

print len(reader.d)

g_model = generator_model(vocab_size=len(reader.d),
    embedding_size=128,
    lstm_size=200,
    num_layer=4,
    max_length_encoder=40,
    max_length_decoder=40,
    max_gradient_norm=2,
    batch_size_num=40,
    learning_rate=0.001)
#d_model = discriminator_model()

saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)

sess = tf.Session()
try:
    loader = tf.train.import_meta_graph('saved/model.ckpt.meta')
    loader.restore(sess, tf.train.latest_checkpoint('saved/'))
    print 'load finished'
except:
    sess.run(tf.global_variables_initializer())
    print 'load failed'

try:
    while reader.epoch < 10:
        g_model.pretrain(sess, reader)
    """
    d_model.update(sess, g_model, reader)

    while True:
        g_model.update(sess, d_model)
        d_model.update(sess, g_model, reader)
    """
except KeyboardInterrupt:
    saver.save(sess, 'saved/model.ckpt')
