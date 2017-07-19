import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core

GO_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

sample_times = 5

def cut(resp):
    for time in range(len(resp)):
        if resp[time] == EOS_ID:
            resp = resp[:time+1]
            break
    return resp

class generator_model():
    def __init__(self,
                vocab_size,
                embedding_size,
                lstm_size,
                num_layer,
                max_length_encoder, max_length_decoder,
                max_gradient_norm,
                batch_size_num,
                learning_rate):
        self.batch_size = batch_size_num
        self.max_length_encoder = max_length_encoder
        self.max_length_decoder = max_length_decoder
        with tf.variable_scope('g_model') as scope:
            self.encoder_input = tf.placeholder(tf.int32, [max_length_encoder, None])
            self.decoder_output = tf.placeholder(tf.int32, [max_length_decoder, None])
            self.target_weight = tf.placeholder(tf.float32, [max_length_decoder, None]) # for pretraining or updating
            self.reward = tf.placeholder(tf.float32, [max_length_decoder, None]) # for updating
            self.start_tokens = tf.placeholder(tf.int32, [None]) # for partial-sampling
            self.max_inference_length = tf.placeholder(tf.int32, []) # for inference

            self.encoder_length = tf.placeholder(tf.int32, [None])
            self.decoder_length = tf.placeholder(tf.int32, [None])
            batch_size = tf.shape(self.encoder_length)[0]

            decoder_output = self.decoder_output
            # if decoder_output have 0 dimention ???
            self.decoder_input = tf.concat([tf.ones([1, batch_size], dtype=tf.int32) * GO_ID, decoder_output[:-1]], axis=0)

            embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
            encoder_embedded = tf.nn.embedding_lookup(embedding, self.encoder_input)
            decoder_embedded = tf.nn.embedding_lookup(embedding, self.decoder_input)

            self.cell_state = tf.placeholder(tf.float32, [2*num_layer, None, lstm_size]) # for partial-sampling
            self.attention = tf.placeholder(tf.float32, [None, lstm_size])
            self.time = tf.placeholder(tf.int32)
            self.alignments = tf.placeholder(tf.float32, [None, max_length_encoder])
            
            def build_attention_state():
                cell_state = tuple([tf.contrib.rnn.LSTMStateTuple(self.cell_state[i], self.cell_state[i+1])
                    for i in range(0, 2*num_layer, 2)])
                print cell_state
                return tf.contrib.seq2seq.AttentionWrapperState(cell_state,
                    self.attention, self.time, self.alignments, tuple([]))
            partial_decoder_state = build_attention_state()

            print 'shape:', decoder_output.get_shape()
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(lstm_size)
            def multi_cell():
                return tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layer)])
            with tf.variable_scope('encoder'):
                encoder_cell = multi_cell()
                encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embedded, 
                    self.encoder_length, time_major=True, dtype=tf.float32)
            attention_state = tf.transpose(encoder_output, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(lstm_size, attention_state,
                memory_sequence_length=self.encoder_length)
            with tf.variable_scope('decoder') as decoder_scope:
                # train or evaluate
                decoder_cell = multi_cell()

                # attention wrapper
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                    attention_layer_size=lstm_size)
                decoder_init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

                helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedded, self.decoder_length, time_major=True)
                projection_layer = layers_core.Dense(vocab_size) # use_bias ?
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_init_state, output_layer=projection_layer)

                output, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True, scope=decoder_scope)
                logits = output.rnn_output
                self.result_train = tf.transpose(output.sample_id)
                self.decoder_state = decoder_state
                # inference (sample)
                helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding, 
                    start_tokens=tf.fill([batch_size], GO_ID), end_token=EOS_ID)
                decoder_sample = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper_sample, decoder_init_state,
                    output_layer=projection_layer)
                output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_sample, output_time_major=True,
                    swap_memory=True, scope=decoder_scope, maximum_iterations=self.max_inference_length)
                self.result_sample = tf.transpose(output.sample_id)
                
                # inference (partial-sample)
                helper_partial = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding,
                    start_tokens=self.start_tokens, end_token=EOS_ID)
                decoder_partial = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper_partial, partial_decoder_state,
                    output_layer=projection_layer)
                output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_partial, output_time_major=True,
                    swap_memory=True, scope=decoder_scope, maximum_iterations=self.max_inference_length)
                self.result_partial = tf.transpose(output.sample_id)

                # inference (greedy)
                helper_greedy = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, 
                    start_tokens=tf.fill([batch_size], GO_ID), end_token=EOS_ID)
                decoder_greedy = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper_greedy, decoder_init_state,
                    output_layer=projection_layer)
                output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_greedy, output_time_major=True,
                    swap_memory=True, scope=decoder_scope, maximum_iterations=self.max_inference_length)
                self.result_greedy = tf.transpose(output.sample_id)
            dim = tf.shape(logits)[0]
            decoder_output = tf.split(decoder_output, [dim, max_length_decoder-dim])[0]
            target_weight = tf.split(self.target_weight, [dim, max_length_decoder-dim])[0]
            reward = tf.split(self.reward, [dim, max_length_decoder-dim])[0]

            params = scope.trainable_variables()
            print 'shape:', logits.get_shape()
            # update for pretraining
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_output, logits=logits) # max_len * batch
            self.loss_pretrain = tf.reduce_sum(target_weight * cross_entropy) / tf.cast(batch_size, tf.float32)
            gradient_pretrain = tf.gradients(self.loss_pretrain, params)
            gradient_pretrain, _ = tf.clip_by_global_norm(gradient_pretrain, max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.opt_pretrain = optimizer.apply_gradients(zip(gradient_pretrain, params))

            # update for GAN
            one_hot = tf.one_hot(decoder_output, vocab_size)
            prob = tf.reduce_sum(one_hot * logits, axis=2)
            self.loss_generator = tf.reduce_sum(-tf.log(tf.maximum(prob, 1e-5)) * reward * target_weight) / tf.cast(batch_size, tf.float32)
            gradient_generator = tf.gradients(self.loss_generator, params)
            gradient_generator, _ = tf.clip_by_global_norm(gradient_generator, max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.opt_update = optimizer.apply_gradients(zip(gradient_generator, params))

    def all_params(self):
        with tf.variable_scope('g_model') as scope:
            total = 0
            for var in scope.trainable_variables():
                shape = var.get_shape()
                k = 1
                print shape, 
                for dim in shape:
                    k *= dim.value
                print k, var.name
                total += k
            print 'total:', total

    def pretrain(self, sess, reader):
        feed_post = [[] for _ in range(self.max_length_encoder)]
        feed_resp = [[] for _ in range(self.max_length_decoder)]
        feed_weight = [[] for _ in range(self.max_length_decoder)]
        feed_post_length = []
        feed_resp_length = []

        # read training data
        batch = reader.get_batch(self.batch_size)
        for post, resp in batch:
            feed_post_length.append(len(post))
            feed_resp_length.append(len(resp))
            for time in range(self.max_length_encoder):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)
                if time < len(resp) and resp[time] != UNK_ID:
                    feed_weight[time].append(1)
                else:
                    feed_weight[time].append(0)

        feed_dict = {}
        feed_dict[self.encoder_input] = feed_post
        feed_dict[self.decoder_output] = feed_resp
        feed_dict[self.target_weight] = feed_weight
        feed_dict[self.encoder_length] = feed_post_length
        feed_dict[self.decoder_length] = feed_resp_length
        
        result, loss, state, _ = sess.run([self.result_train, self.loss_pretrain, self.decoder_state, self.opt_pretrain], feed_dict=feed_dict)
        for sentence in result:
            for word in sentence:
                print reader.symbol[word],
                if word == EOS_ID:
                    break
            print '\n',
        print loss
        print reader.epoch, str(reader.k)+'/958640', reader.k / 958640.0
        
        
    def update(self, sess, discriminator, reader):
        # for each post, sample a response
        batch = reader.get_batch(self.batch_size)
        resp_generator = self.generate(sess, batch, 'sample')
        
        max_len = len(resp_generator[0])
        feed_reward = [] # max_len * batch_size
        feed_post = [[] for _ in range(self.max_length_encoder)]
        feed_post_length = []
        for index in range(self.batch_size):
            post = batch[index][0]
            feed_post_length.append(len(post))
            for time in range(self.max_length_encoder):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
        for t in range(max_len):
            # for each partial response, get the final hidden state
            feed_resp = [[] for _ in range(self.max_length_decoder)]
            feed_resp_length = []
            for index in range(self.batch_size):
                resp = resp_generator[index]
                resp = cut(resp)
                resp = resp[:t]
                feed_resp_length.append(len(resp))
                for time in range(self.max_length_decoder):
                    feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)
            feed_dict = {}
            feed_dict[self.encoder_input] = feed_post
            feed_dict[self.encoder_length] = feed_post_length
            feed_dict[self.decoder_output] = feed_resp
            feed_dict[self.decoder_length] = feed_resp_length

            state = sess.run(self.decoder_state, feed_dict=feed_dict)

            # from partial response, randomly sample several full responses
            start_tokens = [resp[t-1] for resp in resp_generator] if t >= 1 else [GO_ID] * self.batch_size
            mean_reward = [0 for _ in range(self.batch_size)]
            for num in range(sample_times):
                cell_state = []
                for lstm_tuple in state.cell_state:
                    cell_state = cell_state + [lstm_tuple.c, lstm_tuple.h]
                feed_dict = {}
                feed_dict[self.start_tokens] = start_tokens
                feed_dict[self.max_inference_length] = self.max_length_decoder - t
                feed_dict[self.cell_state] = cell_state
                feed_dict[self.attention] = state.attention
                feed_dict[self.time] = state.time
                feed_dict[self.alignments] = state.alignments

                output = sess.run(self.result_partial, feed_dict=feed_dict)
                # feed into disciminator and compute Q
                feed_resp = []
                for index in range(self.batch_size):
                    resp = resp_generator[index]
                    resp = cut(resp)
                    length = len(resp)
                    resp = resp[:t]
                    final_resp = resp + output[index] if length > t else resp
                    feed_resp.append(final_resp)
                feed_batch = [(batch[index][0], feed_resp[index]) for index in range(self.batch_size)]
                poss = discriminator.evaluate(sess, batch)
                for index in range(self.batch_size):
                    mean_reward[index] += poss[index] / sample_times
            feed_reward.append(mean_reward)
        feed_reward = feed_reward + [[0 for _ in self.batch_size]] * (self.max_length_decoder - max_len)

        # update generator
        feed_resp = [[] for _ in range(self.max_length_decoder)]
        feed_resp_length = []
        for index in range(self.batch_size):
            resp = resp_generator[index]
            resp = cut(resp)
            feed_resp_length.append(len(resp))
            for time in range(self.max_length_decoder):
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)
                if time < len(resp) and resp[time] != UNK_ID:
                    feed_weight[time].append(1)
                else:
                    feed_weight[time].append(0)
        feed_dict = {}
        feed_dict[self.encoder_input] = feed_post
        feed_dict[self.encoder_length] = feed_post_length
        feed_dict[self.decoder_output] = feed_resp
        feed_dict[self.decoder_length] = feed_resp_length
        feed_dict[self.reward] = feed_reward
        feed_dict[self.target_weight] = feed_weight

        loss, _ = sess.run([self.loss_generator, self.opt_update], feed_dict=feed_dict)
        print 'generator updated, loss =', loss

    def generate(self, sess, batch, mode):
        feed_post = [[] for _ in range(self.max_length_encoder)]
        feed_weight = [[] for _ in range(self.max_length_decoder)]
        feed_post_length = []
        for post, resp in batch:
            feed_post_length.append(len(post))
            for time in range(self.max_length_encoder):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
        feed_dict = {}
        feed_dict[self.encoder_input] = feed_post
        feed_dict[self.encoder_length] = feed_post_length
        feed_dict[self.max_inference_length] = self.max_length_decoder
        if mode == 'sample':
            result = sess.run(self.result_sample, feed_dict=feed_dict)
        elif mode == 'greedy':
            result = sess.run(self.result_greedy, feed_dict=feed_dict)
        return result

class discriminator_model():
    def __init__(self, vocab_size,
                embedding_size,
                lstm_size,
                num_layer,
                max_post_length, max_resp_length,
                max_gradient_norm,
                batch_size_num,
                learning_rate):
        self.batch_size = batch_size_num
        self.max_post_length = max_post_length
        self.max_resp_length = max_resp_length
        with tf.variable_scope('g_model', reuse=True):
            embedding = tf.get_variable('embedding')
        with tf.variable_scope('d_model') as scope:
            self.post_input = tf.placeholder(tf.int32, [max_post_length, None])
            self.resp_input = tf.placeholder(tf.int32, [max_resp_length, None])
            self.post_length = tf.placeholder(tf.int32, [None])
            self.resp_length = tf.placeholder(tf.int32, [None])
            self.labels = tf.placeholder(tf.int64, [None])

            batch_size = tf.shape(self.labels)[0]
            post_embedded = tf.nn.embedding_lookup(embedding, self.post_input)
            resp_embedded = tf.nn.embedding_lookup(embedding, self.resp_input)
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(lstm_size)
            def multi_cell():
                return tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layer)])
            with tf.variable_scope('encoder'):
                cell = multi_cell()
                post_output, post_state = tf.nn.dynamic_rnn(cell, post_embedded,
                    self.post_length, time_major=True, dtype=tf.float32)
            with tf.variable_scope('encoder', reuse=True):
                resp_output, resp_state = tf.nn.dynamic_rnn(cell, resp_embedded,
                    self.resp_length, time_major=True, dtype=tf.float32)
            
            def concat(lstm_tuple):
                return tf.concat([tf.concat([pair.c, pair.h], axis=1) for pair in lstm_tuple], axis=1)
            post_state_concat = concat(post_state)
            resp_state_concat = concat(resp_state)

            cell_sentence = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            init_state = cell_sentence.zero_state(batch_size, tf.float32)
            out1, state_mid = cell_sentence(post_state_concat, init_state, scope=scope)
            out2, state_final = cell_sentence(resp_state_concat, state_mid, scope=scope)

            state_final_concat = tf.concat([state_final.c, state_final.h], axis=1)
            logits = tf.layers.dense(state_final_concat, 2)
            print logits, self.labels
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            self.poss = tf.nn.softmax(logits)[:, 1]
            
            result = tf.argmax(logits, axis=1)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(result, self.labels), tf.float32))
            
            params = scope.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.opt_train = optimizer.apply_gradients(zip(gradients, params))
    def all_params(self):
        with tf.variable_scope('d_model') as scope:
            total = 0
            for var in scope.trainable_variables():
                shape = var.get_shape()
                k = 1
                print shape, 
                for dim in shape:
                    k *= dim.value
                print k, var.name
                total += k
            print 'total:', total

    def update(self, sess, generator, reader):
        batch = reader.get_batch(self.batch_size)
        resp_generator = generator.generate(sess, batch, 'greedy')
        feed_post = [[] for _ in range(self.max_post_length)]
        feed_resp = [[] for _ in range(self.max_resp_length)]
        feed_post_length = []
        feed_resp_length = []
        feed_labels = []
        for post, resp in batch:
            feed_post_length.append(len(post))
            feed_resp_length.append(len(resp))
            feed_labels.append(1)
            for time in range(self.max_post_length):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)
        
        for index in range(self.batch_size):
            post = batch[index][0]
            resp = resp_generator[index]
            resp = cut(resp)
            feed_post_length.append(len(post))
            feed_resp_length.append(len(resp))
            feed_labels.append(0)
            for time in range(self.max_post_length):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)
        
        feed_dict = {}
        feed_dict[self.post_input] = feed_post
        feed_dict[self.resp_input] = feed_resp
        feed_dict[self.post_length] = feed_post_length
        feed_dict[self.resp_length] = feed_resp_length
        feed_dict[self.labels] = feed_labels

        loss, acc, _ = sess.run([self.loss, self.acc, self.opt_train], feed_dict=feed_dict)
        print 'discriminator:', loss, acc

    def evaluate(self, sess, batch):
        feed_post = [[] for _ in range(self.max_post_length)]
        feed_resp = [[] for _ in range(self.max_resp_length)]
        feed_post_length = []
        feed_resp_length = []
        
        for post, resp in batch:
            feed_post_length.append(len(post))
            resp = cut(resp)
            feed_resp_length.append(len(resp))
            for time in range(self.max_post_length):
                feed_post[time].append(post[time] if time < len(post) else PAD_ID)
                feed_resp[time].append(resp[time] if time < len(resp) else PAD_ID)
        
        feed_dict = {}
        feed_dict[self.post_input] = feed_post
        feed_dict[self.resp_input] = feed_resp
        feed_dict[self.post_length] = feed_post_length
        feed_dict[self.resp_length] = feed_resp_length

        poss = sess.run(self.poss, feed_dict=feed_dict)
        return poss

if __name__ == '__main__':
    g = generator_model(1000, 128, 101, 4, 98, 99, 5, 20, 0.001)
    #g.all_params()
    d = discriminator_model(1000, 100, 101, 4, 40, 40, 2, 20, 0.001)
    #d.all_params()
