from __future__ import absolute_import
from __future__ import division
from data_batcher import LSTMDataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

class LSTMCharacterModel(object):
    def __init__(self,FLAGS,name='LSTMCharacterModel'):
        self.FLAGS=FLAGS
        self.name=name
        self.dataObject=LSTMDataObject(FLAGS.file_path)

        self.add_placeholders()
        self.add_embedding_layer()
        self.build_graph()
        self.add_loss()

        #opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        #self.train_step=opt.minimize(self.loss_mean)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        self.train_step = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.add_predictions()

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.keep)

    def add_placeholders(self):
        self.character_ids=tf.placeholder(tf.int32,shape=[None,None])
        self.character_label=tf.placeholder(tf.int32,shape=[None,None])
        self.keep_prob=tf.placeholder_with_default(1.0,shape=())
        self.hidden_state=tf.placeholder(tf.float32,[None,self.FLAGS.hidden_size*self.FLAGS.num_layers],name='HiddenState')

    def add_embedding_layer(self, emb_matrix=None):
        with vs.variable_scope("embeddings"):
            self.character_one_hot=tf.one_hot(self.character_ids,self.dataObject.ALPHASIZE)
            self.labels_one_hot=tf.one_hot(self.character_label,self.dataObject.ALPHASIZE)

    def build_graph(self):
        cells=[tf.contrib.rnn.GRUCell(self.FLAGS.hidden_size) for _ in range(self.FLAGS.num_layers)]
        dropcells=[tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=self.keep_prob) for cell in cells]
        multicell=tf.contrib.rnn.MultiRNNCell(dropcells,state_is_tuple=False)
        multicell=tf.contrib.rnn.DropoutWrapper(multicell,output_keep_prob=self.keep_prob)  # dropout for the softmax layer
        outputs,self.final_state=tf.nn.dynamic_rnn(cell=multicell,inputs=self.character_one_hot,dtype=tf.float32,initial_state=self.hidden_state)
        self.final_state=tf.identity(self.final_state, name='final_state')
        final_output = tf.contrib.layers.fully_connected(outputs, num_outputs=self.dataObject.ALPHASIZE,activation_fn=None)  # [batch_size,seq_len,num_chars]
        self.logits = tf.identity(final_output, name='logits')  # [batch_size,seq_len,num_chars]

    def add_loss(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_one_hot,logits=self.logits)  # batch_size,seq_len
        self.loss_mean = tf.reduce_mean(self.loss)  # batch_size

    def add_predictions(self):
        self.probabilities = tf.nn.softmax(self.logits)  # [batch_size,seq_len,num_chars]
        self.prediction = tf.argmax(self.probabilities, axis=2)  # [batch_size,seq_len]
        self.correct_pred = tf.equal(self.character_label, tf.cast(self.prediction, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

