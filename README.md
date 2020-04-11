# CNN-Classification-and-network-fooling

This model classifies MNIST fashion data with 95% accuracy 


Configuration and training details:
1. Learning Rate: 0.0001
2. Initializer: He
3. Number of epochs:20
4. Batch Size:100
5. Batch Normalization: Yes
6. Optimizer: Adam
7. Architecture :

  CONV1: Convolutional layer with 1 input channel and filter size (64 x 3 x 3
  x 3) with Padding 1 and strides 1

  POOL1 2 x 2 max-pooling layer with strides 1

  CONV2: Convolutional layer with 64 outputs, 128 outputs (filter size 128 x 64 x 3
  x 3)

  POOL2 2 x 2 max-pooling layer with strides 1

  CONV3: Convolutional layer with 128 inputs, 256 outputs with strides 1

  CONV4: Convolutional layer with 256 inputs, 256 outputs with strides 1

  POOL3: 2 x 2 max-pooling layer with strides 1

  FC1: Fully connected layer with 256 inputs, 1024 outputs

  FC2: Fully connected layer with 1024 inputs, 1024 outputs

  SOFTMAX: Softmax layer for classification: 1024 inputs, 10 outputs

8. RELU nonlinearities have been used in all convolutional layers except last layer


Network fooling is also performed on the same dataset by randomly changing some pixels in the image such that the class is
changed.
