import tensorflow as tf
import numpy as np
import pandas as pd

class CNN(object):
    
    def __init__(self):
        learning_rate=0.001
        batch_size = 50
        n_samples=2
        
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 300,300,1])
        self.y = tf.compat.v1.placeholder(tf.float32,[None,2])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)
        
        def conv2d(x, W, b, strides=1):
            # Conv2D wrapper, with bias and relu activation
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.softmax(x)

        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
        
        self.weights = {'W_conv1':tf.Variable(tf.compat.v1.random_normal([3,3,1,128])),
               'W_conv2':tf.Variable(tf.compat.v1.random_normal([3,3,128,256])),
               'W_conv3':tf.Variable(tf.compat.v1.random_normal([3,3,256,512])),
               'W_conv4':tf.Variable(tf.compat.v1.random_normal([3,3,512,256])),
               'W_conv5':tf.Variable(tf.compat.v1.random_normal([3,3,256,128])),
               'W_fc':tf.Variable(tf.compat.v1.random_normal([10*10*128,1000])),
               'out':tf.Variable(tf.compat.v1.random_normal([1000, 2]))}
        
        self.biases = {'b_conv1':tf.Variable(tf.compat.v1.random_normal([128])),
               'b_conv2':tf.Variable(tf.compat.v1.random_normal([256])),
               'b_conv3':tf.Variable(tf.compat.v1.random_normal([512])),
               'b_conv4':tf.Variable(tf.compat.v1.random_normal([256])),
               'b_conv5':tf.Variable(tf.compat.v1.random_normal([128])),        
               'b_fc':tf.Variable(tf.compat.v1.random_normal([1000])),
               'out':tf.Variable(tf.compat.v1.random_normal([2]))}
        
        #self.x= tf.reshape(self.x, shape=[-1,300,300,1])
        self.conv1 = conv2d(self.x, self.weights['W_conv1'], self.biases['b_conv1'])
        self.conv1 = maxpool2d(self.conv1)

        self.conv2 = conv2d(self.conv1, self.weights['W_conv2'], self.biases['b_conv2'])
        self.conv2 = maxpool2d(self.conv2)

        self.conv3 = conv2d(self.conv2, self.weights['W_conv3'], self.biases['b_conv3'])
        self.conv3 = maxpool2d(self.conv3)

        self.conv4 = conv2d(self.conv3, self.weights['W_conv4'], self.biases['b_conv4'])
        self.conv4 = maxpool2d(self.conv4)
        
        self.conv5 = conv2d(self.conv4, self.weights['W_conv5'], self.biases['b_conv5'])
        self.conv5 = maxpool2d(self.conv5)

        self.fc = tf.reshape(self.conv5,[-1, 10*10*128 ])
        #fc = tf.nn.sigmoid(tf.add(tf.matmul(fc, weights['W_fc']),biases['b_fc']))
        self.fc = tf.matmul(self.fc, self.weights['W_fc'])
        self.fc=tf.add(self.fc,self.biases['b_fc'])
        self.fc= tf.nn.softmax(self.fc)
        #fc = tf.nn.dropout(fc, keep_rate)

        self.output = tf.add(tf.matmul(self.fc, self.weights['out']),self.biases['out'])
        self.y_=tf.nn.softmax(self.output)
        
        self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.y_) )
        #self.cost = tf.reduce_sum(tf.pow(self.y_-self.y,2))/(2*n_samples)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        
        def training():
            init=tf.compat.v1.initialize_all_variables()
            sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
            #sess=tf.compat.v1.Session()
            sess.run(init)
            
            #saver to save the model
            saver = tf.compat.v1.train.Saver()
            
            for i in range(1000):

                for j in range(800//batch_size):
                    sess.run(optimizer,feed_dict={x:inputX[(j*batch_size):(j*batch_size)+batch_size],y:inputY[(j*batch_size):(j*batch_size)+batch_size]})
                    #sess.run(optimizer,feed_dict={x:inputX,y:inputY})

                #logs
                cc=sess.run(cost,feed_dict={x:inputX[(j*batch_size):(j*batch_size)+batch_size],y:inputY[(j*batch_size):(j*batch_size)+batch_size]})
                print("training step:",'%04d'%(i),'cost=','{:.9f}'.format(cc))

                correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',sess.run(accuracy,feed_dict={x:inputX_test[:100], y:inputY_test[:100]}))
                
                #prediction
                pred=[]
                for k in range(1000//50):
                    pred.append(sess.run(y_,feed_dict={x:inputX_test[(k*batch_size):(k*batch_size)+batch_size]}))
                    
                pred=np.vstack(pred)
                #checkpoint the model to train again
                saver.save(sess, "./model6/model.ckpt")