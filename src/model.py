import tensorflow as tf
import scipy.io as sc
import numpy as np
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import xgboost as xgb


# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

len_sample=1
full=7000
len_a=full/len_sample  # 6144 class1
label0=np.zeros(len_a)
label1=np.ones(len_a)
label2=np.ones(len_a)*2
label3=np.ones(len_a)*3
label4=np.ones(len_a)*4
label5=np.ones(len_a)*5
label6=np.ones(len_a)*6
label7=np.ones(len_a)*7
label=np.hstack((label0,label1,label2,label3,label4,label5,label6,label7))
label=np.transpose(label)
label.shape=(len(label),1)
print label
time1 =time.clock()
# feature = sc.loadmat("EID-M.mat")  # EID_M, with three trials, 21000 samples per sub
# all = feature['eeg_close_ubicomp_8sub']  /home/xiangzhang/matlabwork/
feature = sc.loadmat("EID-S.mat")  # EID-S, with 1 trial, 7000 samples per subject
all = feature['eeg_close_8sub_1file']

n_fea=14
all = all[0:full*8, 0:n_fea]

# EEG-S dataset is a subset of EEG_ID_label6.mat. 1 trial, 7000 samples per sub

# feature = sc.loadmat("/home/xiangzhang/matlabwork/eegmmidb/EEG_ID_label6.mat")  # 1trial, 13500 samples each subject
# all = feature['EEG_ID_label6']
# n_fea = 64
# all = all[0:21000*8, 0:n_fea]
# print all.shape
#
# a1 = all[0:7000]  # select 7000 samples from 135000
# for i in range(2,9):
#     b = all[13500*(i-1):13500*i]
#     c = b[0:7000]
#     print c.shape
#     a1 = np.vstack((a1, c))
#     print i, a1.shape
#
# all = a1
# print all.shape




time2=time.clock()
data_f1=[]
#  EEG Delta pattern decomposition
for i in range(all.shape[1]):
    x = all[:, i]
    fs = 128.0
    lowcut =0.5
    highcut = 4.0

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    data_f1.append(y)
data_f1=np.array(data_f1)
data_f1=np.transpose(data_f1)
print 'data_f1.shape',data_f1.shape
all=data_f1
time3=time.clock()
print 'PD time',time3-time2


all=np.hstack((all,label))
print all.shape

# all=combine_data
data_size=all.shape[0]

feature_all = all[:, 0:n_fea]
print all[:, n_fea:n_fea+1]


feature_all=feature_all-4200 # minus Direct Current
# z-score scaling
feature_all=preprocessing.scale(feature_all)

# min-max  unity scaling
# feature_all=preprocessing.minmax_scale(feature_all,feature_range=(0,1))
# feature_all=feature_all/sum(feature_all)
label_all = all[:, n_fea:n_fea+1]
all = np.hstack((feature_all, label_all))
print all.shape

# use the first subject as testing subject
np.random.shuffle(all)

train_data=all[0:data_size*0.875]  # 1 million samples
test_data=all[data_size*0.875:data_size]



no_fea = n_fea
n_steps = len_sample

feature_training = train_data[:, 0:no_fea]
feature_training = feature_training.reshape([train_data.shape[0], n_steps, no_fea])


feature_testing = test_data[:,0:no_fea]

feature_testing = feature_testing.reshape([test_data.shape[0], n_steps, no_fea])


label_training = train_data[:, no_fea]
label_training = one_hot(label_training)
label_testing = test_data[:, no_fea]
print label_testing
label_testing = one_hot(label_testing)


print all.shape


# batch split

a = feature_training
b = feature_testing
nodes = 30
lameda = 0.001
lr = 0.001

batch_size = int(data_size*0.125)
train_fea = []
n_group = 7
for i in range(n_group):
    f = a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)
print train_fea[0][235:237]
# exit(0)
train_label = []
for i in range(n_group):
    f = label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)


# hyperparameters
n_inputs = no_fea  # MNIST data input (img shape: 11*99)
# n_steps =  # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units=nodes
n_classes = 8  # MNIST classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs],name="x")
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights

weights = {
'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),

'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),

'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
'att': tf.Variable(tf.random_normal([n_inputs, n_hidden4_units]), trainable=True),
'att2': tf.Variable(tf.random_normal([1, batch_size]), trainable=True),

}

biases = {
'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),

'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ])),
'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),

'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
'att': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
'att2': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    print X.get_shape()
    X = tf.reshape(X, [-1, n_inputs])

    # hidden layer
    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])
    X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
    X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']

    X_in = tf.reshape(X_hidd3, [-1, n_steps, n_hidden4_units])


    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden3_units, forget_bias=1, state_is_tuple=True)
    init_state = lstm_cell_1.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell_1, X_in, initial_state=init_state, time_major=False)
    print 'outp.shape', outputs.get_shape()

    # outputs
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs

    # attention based model
    X_att2 = final_state[0]  # weights
    print X_att2.shape, X_att2.get_shape(), outputs[-1].shape
    outputs_att = tf.multiply(outputs[-1], X_att2)
    results = tf.matmul(outputs_att, weights['out']) + biases['out']

    return results,outputs[-1]

pred, Feature = RNN(x, weights, biases)


lamena = lameda
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+l2  # Softmax loss

# tf.scalar_summary('loss', cost)

lr=lr
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
pred_result =tf.argmax(pred, 1,name="pred_result")
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
rnn_s = time.clock()
with tf.Session(config=config) as sess:
    sess.run(init)
    step = 0
    while step < 2000:
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
                })
        #  early stopping
        if sess.run(accuracy, feed_dict={x: b,y: label_testing,})>0.999:
            print(
            "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
            sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }))
            break
        if step % 10 == 0:
            pp=sess.run(pred_result,feed_dict={x: b, y: label_testing})
            print "predict",pp[0:10]
            gt=np.argmax(label_testing, 1)
            print "groundtruth", gt[0:10]
            hh = sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            })
            h2=sess.run(accuracy,  feed_dict={x: train_fea[i],
                y: train_label[i]})
            print "training acc", h2
            print("The lamda is :",lamena,", Learning rate:",lr,", The step is:",step,
                  ", The accuracy is:", hh,", The train accuracy is:", h2)
            print("The cost is :",sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))
        step += 1

    endtime=time.clock()
    B=sess.run(Feature, feed_dict={
            x: train_fea[0],
            y: train_label[0],
        })
    for i in range(1,n_group):
        D=sess.run(Feature, feed_dict={
            x: train_fea[i],
            y: train_label[i],
        })
        B=np.vstack((B,D))
    B = np.array(B)
    print B.shape
    Data_train = B  # Extracted deep features
    Data_test = sess.run(Feature, feed_dict={x: b, y: label_testing})
print "RNN run time:", endtime-rnn_s

# XGBoost
xgb_s=time.clock()
xg_train = xgb.DMatrix(Data_train, label=np.argmax(label_training,1))
xg_test = xgb.DMatrix(Data_test, label=np.argmax(label_testing,1))

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob' # can I replace softmax by SVM??
# softprob produce a matrix with probability value of each class
# scale weight of positive examples
param['eta'] = 0.7

param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['subsample']=0.9
param['num_class'] =n_classes


np.set_printoptions(threshold=np.nan)
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist );
time8=time.clock()
pred = bst.predict(xg_test) ;
xgb_e=time.clock()
print 'xgb run time', xgb_e -xgb_s
print 'RNN acc', hh


