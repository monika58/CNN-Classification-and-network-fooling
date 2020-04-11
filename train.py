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


num_classes=10
num_iter=20
BN=1




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
    saver = tf.train.Saver()

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
    if init==1:      
        W= tf.get_variable("W",shape,initializer=tf.contrib.layers.xavier_initializer(uniform=False,
    seed=1234,
    dtype=tf.float32))
    if init==2:
        W=tf.get_variable("W",shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False,seed=1234,   dtype=tf.float32) )
    if init==0:
	W=tf.get_variable("W",shape,initializer=tf.variance_scaling_initializer(scale=1.0,
    mode='fan_in',
    distribution='normal',
    seed=1234,
    dtype=tf.float32) )
    return W
    
def create_weight_fc(shape,init):
    if init==1:       
        W = tf.get_variable("W", shape,initializer=tf.contrib.layers.xavier_initializer(uniform=False,
    seed=1234,
    dtype=tf.float32))
    if init==2:
        W=tf.get_variable("W",shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False,seed=1234,   dtype=tf.float32) )
    if init==0:
	W=tf.get_variable("W",shape,initializer=tf.variance_scaling_initializer(scale=1.0,
    mode='fan_in',
    distribution='normal',
    seed=1234,
    dtype=tf.float32) )
    return W    
    
    
def create_bias(shape):
    b= tf.get_variable("b",shape,initializer=tf.variance_scaling_initializer(scale=1.0,
    mode='fan_in',
    distribution='normal',
    seed=1234,
    dtype=tf.float32) )
    return b
       

def create_conv_layer(input_data, num_input_channels, num_filters, filter_shape):
    paddings = tf.constant([[0, 0,], [1, 1],[1,1],[0,0]])
    padded_x=tf.pad(input_data, paddings, "CONSTANT") 
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
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


def build_training_model(x, y,phase,keep_prob):
    with tf.variable_scope('new_gr'):
        with tf.variable_scope("conv1"):
           relu1 = create_conv_layer(x,1,64,[3, 3])
        #plot_filter("conv1/W")
        pool1 = create_pool_layer(relu1,[2, 2])
	with tf.variable_scope("conv2"):
           relu2 = create_conv_layer(pool1,64,128,[3, 3])
        pool2 = create_pool_layer(relu2,[2, 2])
        with tf.variable_scope("conv3"):
           relu3 = create_conv_layer(pool2,128,256,[3, 3])
        with tf.variable_scope("conv4"):
           relu4 = create_conv_layer(relu3,256,256,[3, 3])
        pool3 = create_pool_layer(relu4,[2, 2])
        flattened = tf.reshape(pool3, [-1, 25*25*256]) 
        with tf.variable_scope("FC1"):
           relu5=create_fc_layer(flattened,25*25*256,1024)
        with tf.variable_scope("FC2"):
           soft1=create_softmax_layer(relu5,1024,num_classes)
        out = tf.nn.softmax(soft1)
        cost=loss(soft1,y)
        optimizer=training(cost)
        accuracy1=accuracy(y,out)
        return optimizer,accuracy1,cost



def inference(x, y,phase,keep_prob):
     with tf.variable_scope('new_gr', reuse=True):
        with tf.variable_scope("conv1"):
           relu1 = create_conv_layer(x,1,64,[3, 3])
        pool1 = create_pool_layer(relu1,[2, 2])
	with tf.variable_scope("conv2"):
           relu2 = create_conv_layer(pool1,64,128,[3, 3])
        pool2 = create_pool_layer(relu2,[2, 2])
        with tf.variable_scope("conv3"):
           relu3 = create_conv_layer(pool2,128,256,[3, 3])
        with tf.variable_scope("conv4"):
           relu4 = create_conv_layer(relu3,256,256,[3, 3])
        pool3 = create_pool_layer(relu4,[2, 2])
        flattened = tf.reshape(pool3, [-1, 25*25*256]) 
        with tf.variable_scope("FC1"):
           relu5=create_fc_layer(flattened,25*25*256,1024)
        with tf.variable_scope("FC2"):
           soft1=create_softmax_layer(relu5,1024,num_classes)
        out = tf.nn.softmax(soft1)
        cost=loss(soft1,y)
        accuracy1=accuracy(y,out)
        return accuracy1,cost

def inference_test(x,phase,keep_prob):
     with tf.variable_scope('new_gr', reuse=True):
        with tf.variable_scope("conv1"):
           relu1 = create_conv_layer(x,1,64,[3, 3])
        pool1 = create_pool_layer(relu1,[2, 2])
	with tf.variable_scope("conv2"):
           relu2 = create_conv_layer(pool1,64,128,[3, 3])
        pool2 = create_pool_layer(relu2,[2, 2])
        with tf.variable_scope("conv3"):
           relu3 = create_conv_layer(pool2,128,256,[3, 3])
        with tf.variable_scope("conv4"):
           relu4 = create_conv_layer(relu3,256,256,[3, 3])
        pool3 = create_pool_layer(relu4,[2, 2])
        flattened = tf.reshape(pool3, [-1, 25*25*256]) 
        with tf.variable_scope("FC1"):
           relu5=create_fc_layer(flattened,25*25*256,1024)
        with tf.variable_scope("FC2"):
           soft1=create_softmax_layer(relu5,1024,num_classes)
        out = tf.nn.softmax(soft1)
        pred=tf.argmax(out, 1)
        return pred


		
parser=argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_itr', type=int)
parser.add_argument('--init', type=int)
parser.add_argument('--save_dir', type=str)

args = parser.parse_args()
learning_rate=args.lr                     #learning rate
batch_size=args.batch_size
init=args.init
save_dir=args.save_dir


if __name__ == '__main__':
        x_train2, y_train2,x_val,y_val,x_test = load_data()
        x_train,y_train=data_aug(x_train2,y_train2)
	print("data loaded")
        #x_train,y_train=data_aug(x_train,y_train)
	x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name='y')
        phase = tf.placeholder(tf.bool, name='phase')   
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')                                
        optimizer,accuracy_tr,loss_tr=build_training_model(x, y,0,0.4)
        accuracy_va,loss_va=inference(x, y,1,1)
        pred_t=inference_test(x,0,1)

        inita = tf.global_variables_initializer()
    
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(inita)
            patience=5
            min_del=0.1
            prev_loss=0
            patience_count=0
            curr_acc=0
            bpass=len(x_train)/batch_size
            for epoch in range(num_iter):
                train_loss=0
                for i in range(bpass):
                    offset = (i * batch_size) % (len(x_train) - batch_size)
                    batch_x, batch_y = x_train[offset:(offset + batch_size), :], y_train[offset:(offset + batch_size), :]                                   
                    _,acc_tr,cur_loss = sess.run([optimizer,accuracy_tr,loss_tr],feed_dict={x: batch_x, y: batch_y,phase:0, keep_prob: 0.04})
                    train_loss+=cur_loss
                train_loss=train_loss/bpass
                #writer.add_summary(summary,epoch)
                print(epoch, train_loss)
                validation_accuracy=0 
                val_loss=0
                bs_val=1000
                for i in range(5):
                    offset = (i * bs_val) 
                    b_x, b_y = x_val[offset:(offset + bs_val), :], y_val[offset:(offset + bs_val), :] 
                    validation_accuracy += accuracy_va.eval(feed_dict={x: b_x, y: b_y,phase:0, keep_prob: .04})
                    val_loss += loss_va.eval(feed_dict={x: b_x, y: b_y,phase:0, keep_prob: .04})
                validation_accuracy=validation_accuracy/5
                val_loss=val_loss/5
                print('Iter {} Accuracy: {}'.format(epoch, validation_accuracy))
                print('Iter {} Val loss: {}'.format(epoch, val_loss))
                if(prev_loss>0):
                    if(prev_loss-val_loss>0):
                        patience_count=0
                    else:
                        patience_count=patience_count+1
                if patience_count>patience:
                    print("early stopping")
                    break
                prev_loss=val_loss
                if(val_loss>=.93):
                    bs_test=1000
                    ytest=np.zeros((len(x_test),2))
                    for i in range(10):
                        offset = (i * bs_test) 
                        b_x = x_test[offset:(offset + bs_test), :] 
                        pred_test_label=pred_t.eval(feed_dict={x: b_x})            
                        for j in range(i*1000,(i*1000)+bs_test):
                            ytest[j,0]=j
                            ytest[j,1]=pred_test_label[j%1000]
                    #with open("Testfile.csv",'wb') as f:
                        #f.write('id,label\n')
                    with open("Testfile.csv",'w') as f:
                        np.savetxt(f,ytest,fmt='%d',delimiter=',')  
                    current=val_loss
         
            save_path = saver.save(sess, save_dir)




    
