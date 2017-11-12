
# coding: utf-8

#                            A R T I F I C I A L   N E U R A L   N E T W O R K
#                             

# Keras is a powerful easy-to-use Python library for developing and evaluating deep learning models.
# 
# It wraps the efficient numerical computation libraries Theano and TensorFlow and allows you to define and train neural network models in a few short lines of code.
# 
# In this post, you will discover how to create your first neural network model in Python using Keras.
# 
# There is not a lot of code required, but we are going to step over it slowly so that you will know how to create your own models in the future.
# 
# The steps you are going to cover in this tutorial are as follows:
# 
# 1) Load Data.  
# 2) Define Model.  
# 3) Compile Model.  
# 4) Fit Model.  
# 5) Evaluate Model.  
# 6) Tie It All Together.  
# 
# This tutorial has a few requirements:
# 
# 1)You have Python 2 or 3 installed and configured.  
# 2)You have SciPy (including NumPy) installed and configured.  
# 3)You have Keras and a backend (Theano or TensorFlow) installed and configured.  

#                                     L E T ' S   G E T   S T A R T E D
#                                     

#  1) Load Data

# Whenever we work with machine learning algorithms that use a stochastic process (e.g. random numbers), it is a good idea to set the random number seed.
# 
# This is so that you can run the same code again and again and get the same result. This is useful if you need to demonstrate a result, compare algorithms using the same source of randomness or to debug a part of your code.
# 
# You can initialize the random number generator with any seed you like.
# 
# Dataset: http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
# 
# You can now load the file directly using the NumPy function loadtxt(). There are eight input variables and one output variable (the last column). Once loaded we can split the dataset into input variables (X) and the output class variable (Y).
# We have initialized our random number generator to ensure our results are reproducible and loaded our data. We are now ready to define our neural network model.

# In[18]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("/home/ussr/python/data/pima-indians-diabetes.csv",delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


# 2) Define Model

# Models in Keras are defined as a sequence of layers.
# 
# We create a Sequential model and add layers one at a time until we are happy with our network topology.
# 
# The first thing to get right is to ensure the input layer has the right number of inputs. This can be specified when creating the first layer with the input_dim argument and setting it to 8 for the 8 input variables.
# 
# How do we know the number of layers and their types?
# 
# This is a very hard question. There are heuristics that we can use and often the best network structure is found through a process of trial and error experimentation. Generally, you need a network large enough to capture the structure of the problem if that helps at all.
# 
# In this example, we will use a fully-connected network structure with three layers.
# 
# Fully connected layers are defined using the Dense class. We can specify the number of neurons in the layer as the first argument, the initialization method as the second argument as init and specify the activation function using the activation argument.
# 
# In this case, we initialize the network weights to a small random number generated from a uniform distribution (‘uniform‘), in this case between 0 and 0.05 because that is the default uniform weight initialization in Keras. Another traditional alternative would be ‘normal’ for small random numbers generated from a Gaussian distribution.
# 
# We will use the rectifier (‘relu‘) activation function on the first two layers and the sigmoid function in the output layer. It used to be the case that sigmoid and tanh activation functions were preferred for all layers. These days, better performance is achieved using the rectifier activation function. We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.
# 
# We can piece it all together by adding each layer. The first layer has 12 neurons and expects 8 input variables. The second hidden layer has 8 neurons and finally, the output layer has 1 neuron to predict the class (onset of diabetes or not).

# In[19]:


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#  3) Compile Model

# Now that the model is defined, we can compile it.
# 
# Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.
# 
# When compiling, we must specify some additional properties required when training the network. Remember training a network means finding the best set of weights to make predictions for this problem.
# 
# We must specify the loss function to use to evaluate a set of weights, the optimizer used to search through different weights for the network and any optional metrics we would like to collect and report during training.
# 
# In this case, we will use logarithmic loss, which for a binary classification problem is defined in Keras as “binary_crossentropy“. We will also use the efficient gradient descent algorithm “adam” for no other reason that it is an efficient default.
# 
# Finally, because it is a classification problem, we will collect and report the classification accuracy as the metric.

# In[20]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 4) Fit Model

# We have defined our model and compiled it ready for efficient computation.
# 
# Now it is time to execute the model on some data.
# 
# We can train or fit our model on our loaded data by calling the fit() function on the model.
# 
# The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the nepochs argument. We can also set the number of instances that are evaluated before a weight update in the network is performed, called the batch size and set using the batch_size argument.
# 
# For this problem, we will run for a small number of iterations (150) and use a relatively small batch size of 10. Again, these can be chosen experimentally by trial and error.

# In[21]:


# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# This is where work happens in GPU or CPU.


# 5) Evaluate Model

# We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.
# 
# This will only give us an idea of how well we have modeled the dataset (e.g. train accuracy), but no idea of how well the algorithm might perform on new data. We have done this for simplicity, but ideally, you could separate your data into train and test datasets for training and evaluation of your model.
# 
# You can evaluate your model on your training dataset using the evaluate() function on your model and pass it the same input and output used to train the model.
# 
# This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.  

# In[22]:


# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# -------------------------------------------------------------------------------------------------------------------

# 6) Make Prediction

# We can adapt the above example and use it to generate predictions on the training dataset, pretending it is a new dataset we have not seen before.
# 
# Making predictions is as easy as calling model.predict(). We are using a sigmoid activation function on the output layer, so the predictions will be in the range between 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding them.
# 

# In[23]:


# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


# Running this modified example now prints the predictions for each input pattern. We could use these predictions directly in our application if needed.

#                                         S U M M A R Y

# In this post, you discovered how to create your first neural network model using the powerful Keras Python library for deep learning.
# 
# Specifically, you learned the five key steps in using Keras to create a neural network or deep learning model, step-by-step including:
# 
# 1) How to load data.  
# 2) How to define neural network in Keras.  
# 3) How to compile a Keras model using the efficient numerical backend.  
# 4) How to train a model on data.  
# 5) How to evaluate a model on data.  
