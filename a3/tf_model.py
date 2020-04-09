import tensorflow as tf
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter
import numpy as np
from tqdm import tqdm

class ParserModel():
    def __init__(self, embeddings, n_features=36,
        hidden_size=300, n_classes=3, dropout_prob=0.5):
        self.pretrained_embeddings = embeddings
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.lr = 0.0005
        self.activation = "relu"
        
    
    # def add_placeholders(self):
        self.x_placeholder = tf.placeholder(tf.int32, shape = [None,self.n_features], name = "tensor_x")
        self.y_placeholder = tf.placeholder(tf.float32, shape= [None, self.n_classes], name = "tensor_y")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
    
    def create_feed_dict(self, word_id_batch, class_batch, dropout = 0.5):
        feed_dict = {}
        feed_dict[self.x_placeholder] = word_id_batch
        feed_dict[self.y_placeholder] = class_batch
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict
    
    def add_embeddings(self):
        word_variable = tf.Variable(self.pretrained_embeddings)
        word_embeddings = tf.nn.embedding_lookup(word_variable, self.x_placeholder)
        word_embeddings = tf.reshape(word_embeddings, shape=(-1, self.n_features * self.embed_size))
        return word_embeddings

    def add_prediction(self):
        x = self.add_embeddings()
        xavier_initializer = xavier_weight_init()
        # W = tf.Variable(tf.truncated_normal([self.n_features * self.embed_size, self.hidden_size]), name = "W")
        W = tf.Variable(xavier_initializer((self.n_features * self.embed_size, self.hidden_size)),name = "W")
        b1 = tf.Variable(tf.zeros([self.hidden_size]), name = "b1")
        # U = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_classes]), name = "U")
        U = tf.Variable(xavier_initializer((self.hidden_size, self.n_classes)), name = "U")
        b2 = tf.Variable(tf.zeros([self.n_classes]), name = "b2")
        x = tf.matmul(x, W) + b1

        if self.activation == "cube":
            h = tf.pow(x, tf.constant(3, dtype = tf.float32))
        else:
            h = tf.nn.relu(x)

        h_drop = tf.nn.dropout(h, self.dropout_placeholder)
        pred = tf.matmul(h_drop, U) + b2
        return pred
    
    def add_loss(self, pred):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y_placeholder,
                logits=pred
            )
        )
        return loss

    def add_train(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op
    
    def train(self, train_data):
        with tf.Session() as session:
            pred = self.add_prediction()
            loss = self.add_loss(pred)
            train_op = self.add_train(loss)
            init = tf.global_variables_initializer()
            session.run(init)
            for epoch in tqdm(range(10)):
                total_loss = 0
                for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size = 1024)):
                    feed = self.create_feed_dict(train_x, train_y, self.dropout_prob)
                    _, l = session.run([train_op, loss], feed)
                    if i%500 == 0:
                        print("epoch {}, train loss: {}".format(epoch, l))

                

def _minibatches(data, batch_size):
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)

def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [_minibatch(d, minibatch_indices) for d in data] if list_data \
            else _minibatch(data, minibatch_indices)

def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def xavier_weight_init():
    ''' Xavier Initialization '''
    def _xavier_initializer(shape):
        '''Defines an initializer for the Xavier distribution.
        Args:
            shape: tuple or 1d array, dimension of the weight tensor
        Returns:
            tf.Tensor with specified shape sampled from the Xavier distribution 
        '''
        epsilon = np.sqrt(6 / sum(shape))
        return tf.random_uniform(shape, minval=-epsilon, maxval=epsilon)
    return _xavier_initializer

        
if __name__ == "__main__":
    debug = False
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
    p = ParserModel(embeddings)
    p.train(train_data)
