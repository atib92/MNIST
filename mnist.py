#import libs
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class data_loader():
    train_file = ""
    test_file = ""
    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file
    def load_data(self):
        train_df = pd.read_csv(self.train_file)
        test_df = pd.read_csv(self.test_file)
        train_label = train_df['label']
        train_df.drop(['label'],inplace=True,axis=1)
        train_label_vec = np.zeros([len(train_label),10],int)
        for index in range(len(train_label)):
            train_label_vec[index] = train_label[index]
        return train_df,train_label_vec,test_df
    def visualize_mnist_data(self,tr_df,ts_df,index):
        plt.imshow(tr_df.iloc[index].reshape(28,28))
        return
class network_operations():
    #def __init__(self)
    def init_weights(self,shape):
        W = tf.Variable(tf.truncated_normal(shape=shape))
        return W
    def init_bias(self,shape):
        b = tf.Variable(tf.constant(0.1,shape=shape))
        return b
    def conv(self,X,f_shape):
        F = self.init_weights(f_shape)
        b = self.init_bias([f_shape[3]])
        return tf.nn.relu(tf.nn.conv2d(X,F,strides=[1,1,1,1],padding='SAME') + b)
    def dense_layer(self,X,num_neurons):
        input_size = int(X.get_shape()[1])
        W = self.init_weights([input_size,num_neurons])
        b = self.init_bias([num_neurons])
        return tf.nn.relu(tf.matmul(X,W) + b)
    def poll(self,X):
        return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    def dropout(self,X,p):
        return tf.nn.dropout(X,keep_prob=p)
def get_next_batch(X,y,batch_size,batch_num):
    start_ix = batch_num*batch_size;
    stop_ix = min((batch_num+1)*batch_size,len(X))
    return X.iloc[start_ix:stop_ix],y[start_ix:stop_ix]
def main():
    #load_data
    d_l = data_loader("train.csv","test.csv")
    train_df,train_label,test_df = d_l.load_data()
    d_l.visualize_mnist_data(train_df,test_df,10)
    #create graph
    X = tf.placeholder(dtype=float,shape=[None,784])
    y = tf.placeholder(dtype=float,shape=[None,10])
    x_image = tf.reshape(X,[-1,28,28,1])
    net_op = network_operations()
    #Layer_1
    x_image_l1_bar = net_op.conv(x_image,[5,5,1,32])
    x_image_l1     = net_op.poll(x_image_l1_bar)
    #Layer_1
    x_image_l2_bar = net_op.conv(x_image_l1,[5,5,32,64])
    x_image_l2     = net_op.poll(x_image_l2_bar)
    #Flatten image
    flat_image = tf.reshape(x_image_l2,[-1,7*7*64])
    #Connected layer 1
    full_layer_op = net_op.dense_layer(flat_image,1024)
    #Dropout
    p = tf.placeholder(tf.float32)
    full_layer_op_drop = net_op.dropout(full_layer_op,p)
    #y_pred
    y_pred = net_op.dense_layer(full_layer_op_drop,10)
    #loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pred))
    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)
    #init global variables
    init = tf.global_variables_initializer()
    epochs = 10
    batch_size = 50
    batch_num = 0
    #start session
    X_train, X_test, y_train, y_test = train_test_split(train_df, train_label, test_size=0.33, random_state=42)    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            print(i)
            batch_x,batch_y  = get_next_batch(X_train,y_train,batch_size,batch_num)
            batch_num = batch_num + 1
            sess.run(train,feed_dict = {X:batch_x,y:batch_y,p:0.5})
            if(i%100):
                matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
                acc = tf.reduce_mean(tf.cast(matches,tf.float32))
                print(sess.run(acc,feed_dict={X:X_test,y:y_test,p:1.0}))
                print("\n");
main()
        