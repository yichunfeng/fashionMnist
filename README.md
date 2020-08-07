# Adversarial Attacks and Adversarial Trainings Based on Fast Gradient Sign Method

This is the project of black-box and white-box adversarial attacks and adversarial trainings.
ref: [Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.](https://arxiv.org/abs/1412.6572)

## Requirements
* Python 3.6
* TensorFlow 2.0+
* Keras 2.0+
* Numpy
* pickle
* matplotlib

## Data Preprocess
Dataset: Fashion Mnist
Keras provides the resource of this dataset. So we could load it by:
```
fashion_mnist = keras.datasets.fashion_mnist
(train_i, train_l), (test_i, test_l) = fashion_mnist.load_data()
```

Reshape to size 28*28 and normalize the input data. Note that the first parameter in reshape function: -1 denotes that Numpy would automatically
choose the dimension based on the length of array.

```
train_images = train_i.reshape（-1,28，28,1）/ 255
test_images = test_i.reshape（-1,28，28,1）/ 255
```
Define the labels of classification and convert them to binary matrice.
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_labels = np_utils.to_categorical（train_l，num_classes = 10）
test_labels = np_utils.to_categorical（test_l，num_classes = 10）
```

## Training Model
First, we build a CNN model for the correct classification. Later, the adversarial attacks would be launched on this CNN model.

### Model Structure
1. The input layer: a 2-dimension convolution layer with 32 filters, kernal size of 5, and using the "same" padding
2. Activation layer: ReLU function
3. Pooling layer: max pooling with size 2
4. 2-dimension convolution layer: 64 filters, kernal size of 5, and using the "same" padding
5. Activation layer: ReLU function
6. Pooling layer: max pooling with size 2
7. Fully connected layers with ReLU activation function
8. Output layer: using Softmax function and mapping to 10 categories

The CNN is trained in 70 epochs with batch size 64.
The model constructing training process is illustrated in CNN_keras_fmnist.py,
or the users can simply load the trained model in fmnist_CNN.h5.
```
model = keras.models.load_model('fmnist_CNN.h5')
```

## White-box Attack

## Black-box Attack

## Adversarial Training

## Result