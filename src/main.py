import tensorflow as tf
import os
import numpy as np
from src.LSTMCharacterModel import LSTMCharacterModel

tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / demo")
tf.app.flags.DEFINE_string("save_path", 'experiments/', "path and name to save the model at")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

tf.app.flags.DEFINE_string("file_path","aesop","path to txt files")
#tf.app.flags.DEFINE_string("file_path","/home/rohitapte/Desktop/tensorflow-rnn-shakespeare-master/shakespeare","path to shakespeare files")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use")
tf.app.flags.DEFINE_integer("sequence_length", 30, "Sequence length")
tf.app.flags.DEFINE_integer("num_layers",3, "Number of layers in LSTM")
tf.app.flags.DEFINE_integer("hidden_size", 512, "Size of the hidden states")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("top_probabilities",2,"Number of probabilities to sample from")
tf.app.flags.DEFINE_integer("max_gradient_norm", 5, "level to clip gradients")

tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")
tf.app.flags.DEFINE_integer("save_every", 1, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("show_every", 1, "How often to generate sample text")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

# Some GPU settings
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
model=LSTMCharacterModel(FLAGS=FLAGS,name='LSTMCharacterModel')
char2id=model.dataObject.char2id
id2char=model.dataObject.id2char
init = tf.global_variables_initializer()

def initialize_model(sess,model,expectExists=False):
    ckpt = tf.train.get_checkpoint_state(model.FLAGS.save_path)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        if expectExists:
            raise Exception("Expected model at %s but found none"%(model.FLAGS.save_path))
        else:
            print("No model found. initializing from scratch")
            sess.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

with tf.Session(config=config) as sess:
    initialize_model(sess=sess, model=model, expectExists=False)
    if model.FLAGS.mode == 'train':
        print("Training mode")
        epoch=0
        istate = np.zeros([FLAGS.batch_size, FLAGS.hidden_size * FLAGS.num_layers])  # initial zero input state
        while model.FLAGS.num_epochs == 0 or epoch <= model.FLAGS.num_epochs:
            print("Running epoch %s" % (epoch))
            loss_per_batch, batch_lengths, accuracy_per_batch = [], [], []
            for char_ids,char_labels in model.dataObject.generate_one_epoch(FLAGS.batch_size,FLAGS.sequence_length,epoch):
                input_dict={
                    model.character_ids:char_ids,
                    model.character_label:char_labels,
                    model.keep_prob:1.0-FLAGS.dropout,
                    model.hidden_state:istate,
                }
                _, loss, accuracy, prediction, istate = sess.run([model.train_step, model.loss_mean, model.accuracy, model.prediction, model.final_state],feed_dict=input_dict)
                loss_per_batch.append(loss * char_ids.shape[0])
                batch_lengths.append(char_ids.shape[0])
                accuracy_per_batch.append(accuracy * char_ids.shape[0])
            epoch+=1
            if epoch % model.FLAGS.print_every == 0:
                for i in range(1):
                    sentence = ''.join([id2char[x] for x in char_labels[-i, :].tolist()])
                    predicted = ''.join([id2char[x] for x in prediction[-i, :].tolist()])
                    print("Original: "+sentence)
                    print("\n")
                    print("Predicted: "+predicted)
                    print("--------------------------------")
            if epoch % model.FLAGS.save_every == 0:
                total_examples = float(sum(batch_lengths))
                train_loss = sum(loss_per_batch) / total_examples
                train_accuracy = sum(accuracy_per_batch) / total_examples
                print("Train Loss: %s" % (train_loss))
                print("Train Accuracy %s" % (train_accuracy))
                model.saver.save(sess, model.FLAGS.save_path + "CharacterLanguageModel")
            if epoch % model.FLAGS.show_every==0:
                print("Sample text \n")
                char_ids = np.array([[model.dataObject.char2id['L']]])
                demo_state = np.zeros([1, FLAGS.hidden_size * FLAGS.num_layers])  # initial zero input state
                count = 0
                while count<600:
                    input_dict = {
                        model.character_ids: char_ids,
                        model.keep_prob: 1.0,
                        model.hidden_state: demo_state,
                    }
                    probs, demo_state = sess.run([model.probabilities, model.final_state], feed_dict=input_dict)
                    p = np.squeeze(probs)
                    p[np.argsort(p)[:-FLAGS.top_probabilities]] = 0
                    p = p / np.sum(p)
                    char_ids = np.random.choice(model.dataObject.ALPHASIZE, 1, p=p)[0]
                    print(id2char[char_ids], end="")
                    count += 1
                    if count % 100 == 0:
                        print("")
                    char_ids = np.array([[char_ids]])

    else:
        print("Demo mode")
        sTemp=''
        char_ids=np.array([[model.dataObject.char2id['L']]])
        istate = np.zeros([1, FLAGS.hidden_size * FLAGS.num_layers])  # initial zero input state
        count=0
        while True:
            input_dict={
                model.character_ids:char_ids,
                model.keep_prob:1.0,
                model.hidden_state:istate,
            }
            probs,istate=sess.run([model.probabilities,model.final_state],feed_dict=input_dict)
            p=np.squeeze(probs)
            p[np.argsort(p)[:-FLAGS.top_probabilities]] = 0
            p = p / np.sum(p)
            char_ids=np.random.choice(model.dataObject.ALPHASIZE,1,p=p)[0]
            print(id2char[char_ids],end="")
            count+=1
            if count%100==0:
                print("")
            char_ids=np.array([[char_ids]])
