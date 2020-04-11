# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:09:06 2018

@author: Monika
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 22:24:40 2018

@author: Monika
"""
import tensorflow as tf
import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path
import argparse

flags = tf.app.flags.FLAGS
num_classes=10
#batch_size=100
#learning_rate=0.001
#num_iter=200
summary_dir='graphs'
#checkpoint_file_path='checkpoints/model.ckpt-10000'
#BN=1
#init=-1



def load_data():
    train_file="train.csv"
    val_file="val.csv"
    test_file="test.csv"
#reading and normalising training,validation and test data 
    from numpy import genfromtxt
    data1 = genfromtxt(train_file, delimiter=',')
    trainData_np=data1[1:55001,1:785]
    actualLabel_train_np=data1[1:55001,785]
    trainData_np = trainData_np.astype(np.float32)
    actualLabel_train_np=actualLabel_train_np.astype(np.uint8)

    for i in range(len(trainData_np)):
        m=np.mean(trainData_np[i,:])
        c=np.std(trainData_np[i,:])
        for j in range(784):
            trainData_np[i,j]=(trainData_np[i,j]-m)/c

    actualLabel_train_np = (np.arange(10) == actualLabel_train_np[:, None]).astype(np.float32)
    
    data2 = genfromtxt(val_file, delimiter=',')
    valData_np=data2[1:5001,1:785]
    actualLabel_val_np=data2[1:5001,785]
    valData_np = valData_np.astype(np.float32)
    actualLabel_val_np=actualLabel_val_np.astype(np.uint8)

    for i in range(len(valData_np)):
        m1=np.mean(valData_np[i,:])
        c1=np.std(valData_np[i,:])
        for j in range(784):
            valData_np[i,j]=(valData_np[i,j]-m1)/c1		

    actualLabel_val_np = (np.arange(10) == actualLabel_val_np[:, None]).astype(np.float32)
		
    data3 = genfromtxt(test_file, delimiter=',')
    testData_np=data3[1:10001,1:785]
    testData_np = testData_np.astype(np.float32)

    for i in range(len(testData_np)):
        m=np.mean(testData_np[i,:])
        c=np.std(testData_np[i,:])
        for j in range(784):
            testData_np[i,j]=(testData_np[i,j]-m)/c		


    trainData_np = trainData_np.reshape(len(trainData_np), 28, 28, 1)
    valData_np = valData_np.reshape(len(valData_np), 28, 28, 1)
    testData_np = testData_np.reshape(len(testData_np), 28, 28, 1)
    return trainData_np,actualLabel_train_np,valData_np,actualLabel_val_np,testData_np




def data_aug(x_train,y_train):
    image = tf.placeholder(tf.float32, shape = (28, 28, 1))
    rot_img1 = tf.image.rot90(image,1)
    rot_img2 = tf.image.rot90(image,2)
    rot_img3 = tf.image.rot90(image,3)
    flip1 = tf.image.flip_left_right(image)
    flip2 = tf.image.flip_up_down(image)
    flip3 = tf.image.transpose_image(image)
    x_train1=x_train
    y_train1=y_train
    
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         for i in range(len(x_train)):
             rot1,rot2,rot3,fp1,fp2,fp3=sess.run([rot_img1,rot_img2,rot_img3,flip1,flip2,flip3],feed_dict={image:x_train[i,:]})
             rot1=rot1.reshape(1,28,28,1)
             rot2=rot2.reshape(1,28,28,1)
             rot3=rot3.reshape(1,28,28,1)
             fp1=fp1.reshape(1,28,28,1)
             fp2=fp2.reshape(1,28,28,1)
             fp3=fp3.reshape(1,28,28,1)
             newy=y_train[i].reshape(1,10)
             x_train1=np.append(x_train1,rot1,axis=0)
             y_train1=np.append(y_train1,newy,axis=0)
             x_train1=np.append(x_train1,rot2,axis=0)
             y_train1=np.append(y_train1,newy,axis=0)
             x_train1=np.append(x_train1,rot3,axis=0)
             y_train1=np.append(y_train1,newy,axis=0)
             x_train1=np.append(x_train1,fp1,axis=0)
             y_train1=np.append(y_train1,newy,axis=0)
             x_train1=np.append(x_train1,fp2,axis=0)
             y_train1=np.append(y_train1,newy,axis=0)
             x_train1=np.append(x_train1,fp3,axis=0)
             y_train1=np.append(y_train1,newy,axis=0)
             print(i)
    return x_train1,y_train1
    

def create_weight(shape,init):
    if init==-1:
        #W=np.random.randn(shape[0],shape[1],shape[2],shape[3])/np.sqrt(shape[2])
        #W=tf.convert_to_tensor(W, np.float32)
        #W=tf.Variable(tf.truncated_normal(shape=shape, stddev=shape[1], dtype=tf.float32))        
        W= tf.get_variable("W",shape,initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=1234,dtype=tf.float32))
    if init==2:
        #W=np.random.randn(shape[0],shape[1],shape[2],shape[3])/np.sqrt(shape[2]/2)
        #W=tf.convert_to_tensor(W, np.float32)
        W=tf.get_variable("W",shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False,seed=1234,   dtype=tf.float32) )
    if init==0:
        W=tf.get_variable("W",shape,initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in',distribution='normal',seed=1234,dtype=tf.float32) )
    return W
    
def create_weight_fc(shape,init):
    if init==-1:
        #W=np.random.randn(shape[0],shape[1])/np.sqrt(shape[0])
        #W=tf.convert_to_tensor(W, np.float32)
        #W=tf.Variable(tf.truncated_normal(shape=shape, stddev=shape[1], dtype=tf.float32))        
        W = tf.get_variable("W", shape,initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=1234,dtype=tf.float32))
    if init==2:
        #W=np.random.randn(shape[0],shape[1])/np.sqrt(shape[0]/2)
        #W=tf.convert_to_tensor(W, np.float32)
        W=tf.get_variable("W",shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False,seed=1234,  dtype=tf.float32) )
    if init==0:
        W=tf.get_variable("W",shape,initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in',distribution='normal',seed=1234,dtype=tf.float32) )
    return W   
  
    
def create_bias(shape):
    return tf.get_variable("b",shape,initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in',distribution='normal',seed=1234,dtype=tf.float32) )

def create_conv_layer(input_data, num_input_channels, num_filters, filter_shape):
    paddings = tf.constant([[0, 0,], [1, 1],[1,1],[0,0]])
    padded_x=tf.pad(input_data, paddings, "CONSTANT") 
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    #weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    W_c = create_weight(conv_filt_shape,init)
    bias = create_bias([num_filters])
    out_layer = tf.nn.conv2d(padded_x, W_c, [1, 1, 1, 1], padding='VALID')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    return out_layer

def create_pool_layer(input_data,pool_shape):
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 1, 1, 1]
    out_layer = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding='VALID')
    return out_layer

def create_fc_layer(reshape,insize,outsize):
    W_fc1 = create_weight_fc([insize, outsize],init)
    b_fc1 = create_bias([outsize])
    FC1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
    return FC1
    
def create_softmax_layer(reshape,insize,num_classes):
    W_fc2 = create_weight_fc([insize, num_classes],init)
    b_fc2 = create_bias([num_classes])
    FC2 = tf.matmul(reshape,W_fc2)+ b_fc2
    if(BN==1):
        z=tf.matmul(reshape,W_fc2)
        batch_mean2, batch_var2 = tf.nn.moments(z,[0])
        scale2 = tf.Variable(tf.ones([10]))
        beta2 = tf.Variable(tf.zeros([10]))
        epsilon = 1e-3
        FC2 = tf.nn.batch_normalization(z,batch_mean2,batch_var2,beta2,scale2,epsilon)
    return FC2    
    
    
    
def loss(FC,y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC, labels=y))
    tf.summary.scalar('cost', cost)
    return cost


def training(loss):  
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer
    
def predicted_label(y_):
    pred=tf.argmax(y_, 1)
    return pred
    
def accuracy(y,y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

    
def run():
    with tf.Graph().as_default():
        x_train, y_train,x_val,y_val,x_test = load_data()
        #x_train2,y_train2=data_aug(x_train,y_train)
        #x_train=np.load("trainary.npy")
        #y_train=np.load("trainla.npy")
        #x_val=np.load("valary.npy")
        #y_val=np.load("valla.npy")
        x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name='y')
        phase = tf.placeholder(tf.bool, name='phase')   
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        #global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.variable_scope("conv1"):
           relu1 = create_conv_layer(x,1,16,[3, 3])
        pool1 = create_pool_layer(relu1,[2, 2])
        with tf.variable_scope("conv2"):
           relu2 = create_conv_layer(pool1,16,32,[3, 3])
        pool2 = create_pool_layer(relu2,[2, 2])
        with tf.variable_scope("conv3"):
           relu3 = create_conv_layer(pool2,32,64,[3, 3])
        with tf.variable_scope("conv4"):
           relu4 = create_conv_layer(relu3,64,64,[3, 3])
        pool3 = create_pool_layer(relu4,[2, 2])
        flattened = tf.reshape(pool3, [-1, 25*25*64]) 
        with tf.variable_scope("FC1"):
           relu5=create_fc_layer(flattened,25*25*64,1024)
        #FC3=create_fc_layer(FC1,120,84)
        with tf.variable_scope("FC2"):
           soft1=create_softmax_layer(relu5,1024,num_classes)
        out = tf.nn.softmax(soft1)
        cost=loss(soft1,y)
        optimizer=training(cost)
        accuracy1=accuracy(y,out)
        #t=tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name='t')
        predtest=predicted_label(out)
        #summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        #saver = tf.train.Saver()
    
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            #writer = tf.summary.FileWriter(summary_dir, sess.graph)
            sess.run(init)
            patience=5
            min_del=0.1
            prev_loss=0
            patience_count=0
            m=len(x_train)
            bpass=55000/batch_size
            for epoch in range(num_iter):
                for i in range(bpass):
                    offset = (i * batch_size) % (len(x_train) - batch_size)
                    batch_x, batch_y = x_train[offset:(offset + batch_size), :], y_train[offset:(offset + batch_size), :]                                   
                    _,outp,cur_loss = sess.run([optimizer,out,cost],feed_dict={x: batch_x, y: batch_y,phase:0, keep_prob: 0.5})

            random_image=np.random.uniform(size=(1, 784))
            random_image=random_image.reshape([1,28,28,1])
            np.save("actualimage.npy",random_image)
            print(out.eval(feed_dict={x: random_image, keep_prob: .04}))
            print("before fooling label:")
            print(predtest.eval(feed_dict={x: random_image}))
            for digit in range(10):
                print("digit:",digit)
                process_image=random_image
                prob_digit=out[:, digit]
                gradient = tf.gradients(prob_digit, x)
                for i in range(5):
                    gradients = sess.run(gradient, {x: process_image, keep_prob: .04})
                    gradients = gradients / (np.std(gradients) + 1e-8) 
                    process_image = process_image + gradients[0]
	        print(out.eval(feed_dict={x: process_image, keep_prob: .04}))
                np.save("modimage%s.npy" % digit,process_image)
            
            val_image=x_val[0, :]
            val_image=val_image.reshape([1,28,28,1])
            np.save("actualvalimage.npy",val_image)
            print(out.eval(feed_dict={x: val_image, keep_prob: .1}))
            for digit in range(10):
                print("digit:",digit)
                process_image=val_image
                prob_digit=out[:, digit]
                gradient = tf.gradients(prob_digit, x)
                for i in range(10):
                    gradients = sess.run(gradient, {x: process_image, keep_prob: .04})
                    gradients = gradients / (np.std(gradients) + 1e-8) 
                    process_image = process_image + gradients[0]
	        print(out.eval(feed_dict={x: process_image, keep_prob: .04}))
                np.save("modifiyimage%s.npy" % digit,process_image)  
	     
		
parser=argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_itr', type=int)
parser.add_argument('--init', type=int)
parser.add_argument('--BN', type=int)

args = parser.parse_args()
learning_rate=args.lr                     #learning rate
batch_size=args.batch_size
num_iter=args.num_itr
init=args.init
BN=args.BN

if __name__ == '__main__':
	run()

    









    
