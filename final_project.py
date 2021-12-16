# -*- coding: utf-8 -*-

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as plt

# this is based on the Jupyter notebook "Pytorch_BasicANNClass.ipynb" from Canvas
# and the "Pytorch-TrainTest.ipynb" from Canvas
class BIOF509:
    
    # here is input for the data and labels
    def __init__(self, data, labels, test_size = 0.2, n_epochs = 4):
        self.data = data
        self.labels = labels
    

    '''
    specify the size of the test dataset, and num epochs (# of times we move through the data)
    and the size of the hidden layers, and the batch sizes here
    
    The data will be split into train and test arrays using train_test_split
    then it will be split into batches with batchify
    Then it will create a neural network and train it 
    
    '''
    def train_test(self, test_size, n_epochs, hidden_dimensions, batch_size, lr):
        # split data into training/test set
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size = test_size)
        
        # create batches with batchify
        train_batches, train_label_batches = batchify(train_data, train_labels, batch_size = batch_size)
        
        # defining the neural network (Net class inside BIOF509)
        # use the length of first data point to set the length of input data
        # number of classes is equal to the number of unique values of training labels
        neural_network = BIOF509.Net(len(train_data[0]), hidden_dimensions, len(set(train_labels)))
        
        # use torch.optim to create stochastic gradient descent function
        # neural_network.parameters() reads internal info from the network
        # lr is the learning rate
        optimizer = optim.SGD(neural_network.parameters(), lr = lr)
        
        # use the nn package to create the cross entropy loss function
        loss_function = nn.CrossEntropyLoss()
        
        # tells the network that it is going to be trained 
        # and that it will calculate the information for optimization
        # this must always be called before training
        neural_network.train()
        
        accuracy_array = [] # for the purposes of plotting the accuracy of the epochs
        
        # loops once for each epoch
        for i in range(n_epochs):
            # track the num we get correct
            correct = 0
            total = 0
            
            
            # loops through each batch and feeds into the NN
            for j in range(len(train_batches)):
                # clears the previous gradients from the optimizer
                # the optimizer doesn't need to know what happened last time
                optimizer.zero_grad()
                
                batch = train_batches[j]
                labels = train_label_batches[j]
                
                # puts batch into the neural network
                # predictions: for each data point in the batch, 
                # we get something like tensor([0.3, 0.7]) where each 
                # number corresponds to the prob of a class
                predictions = neural_network(torch.tensor(batch.astype(np.float32)))
                
                labels = labels.array # must be an array, not a 'series'
                
                # put the probs into the loss function to calculate the error
                loss = loss_function(predictions, torch.LongTensor(labels))
                
                # calculates the partial derivatives needed to optimize
                loss.backward()
                
                # calculates the weight updates so the NN can update the weights
                optimizer.step()
                
               
                # extract the data from the predictions
                # then use argmax to figure out which index is the highest prob
                # if it is the 0th index, and the label is 0, add one to correct
                # if it is the 1st index, and the label is 1, add one to correct
                for n, pred in enumerate(predictions.data):
                    total += 1
                    #print(str(n))
                    #print("Prediction is: " + str(pred) + ". Label is: " + str(labels[n]))
                    #print("this is " + str(labels[n] == torch.argmax(pred)))
                    if labels[n] == torch.argmax(pred):
                        correct += 1
            
            accuracy = correct / total
            print("Accuracy for Epoch # " + str(i) + ": " + str(accuracy))
            accuracy_array.append(accuracy)
            #print("epoch: " + str(correct) + " out of " + str(total))
        print()
        
        # plot the accuracies of the epochs
        x = [0, 1, 2, 3]
        plt.pyplot.plot(x, accuracy_array, marker = "o")
        plt.pyplot.ylim()
        plt.pyplot.title('Accuracies of each epoch')
        plt.pyplot.xlabel('Epoch')
        plt.pyplot.ylabel('Accuracy')
        plt.pyplot.ylim(0,1)
        plt.pyplot.show()
                        
        
        # this tells the NN that it is going to be tested on 
        # blind test data -- it shouldn't change any internal parameters
        # this must be called before eval
        neural_network.eval()
            
        test_correct = 0
        test_total = 0
            
        # input the test data into the NN
        predictions = neural_network(torch.tensor(test_data.astype(np.float32)))
        

        # tests how many we got right
        test_labels = test_labels.array
        for n, pred in enumerate(predictions.data):
            test_total += 1
            if test_labels[n] == torch.argmax(pred):
                test_correct += 1

        print("Accuracy on test set: " + str(test_correct / test_total))
        
        print("Creating graph...")
        # create an array with the predicted labels
        predicted_labels = []
        for n, pred in enumerate(predictions.data):
            predicted_labels.append(torch.argmax(pred))
        
            
        # source: https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
        # CREATE A VISUALIZATION OF THE TEST RESULTS
        cm = confusion_matrix(test_labels, predicted_labels)
        hmap = sns.heatmap(cm, annot=True, fmt="d")
        
        # customize the graph
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
        plt.pyplot.ylabel('True label')
        plt.pyplot.xlabel('Predicted label');
        
        return neural_network
    
    
    '''
    inputs: 
        n_features -- how many features for each sample
        hidden_dimension -- number of hidden neurons
        n_classes -- number of unique labels in the data
    '''
    class Net(nn.Module):
        def __init__(self, n_features, hidden_dimension, n_classes):
            # parent class
            super(BIOF509.Net, self).__init__()
            
            # the data entry
            self.input_layer = nn.Linear(n_features,hidden_dimension)
            
            # first hidden layer
            self.layer1 = nn.Linear(hidden_dimension,hidden_dimension)
            
            # second hidden layer
            self.layer2 = nn.Linear(hidden_dimension,hidden_dimension)
            
            # output layer
            self.output_layer = nn.Linear(hidden_dimension,n_classes)
            
            # to transform to a non-linear space after each layer
            self.relu = nn.ReLU()
        
        '''
        the activation function
        inputs: a batch
        '''
        def forward(self, batch):
            # add to the input layer
            batch = self.input_layer(batch)
            
            # apply ReLU 
            batch = self.relu(batch)
            
            # first hidden layer
            batch = self.layer1(batch)
            
            # apply ReLU
            batch = self.relu(batch)
            
            # second hidden layer
            batch = self.layer2(batch)
            
            # apply ReLU
            batch = self.relu(batch)
            
            # output layer
            batch = self.output_layer(batch)
            
            # return the probability distribution via the softmax function
            return nn.functional.softmax(batch)
        
# this will turn the data into batches, with default batch size of 16
def batchify(data, labels, batch_size = 16):
    batches = []
    label_batches = []
        
    for n in range(0, len(data), batch_size):
        if n + batch_size < len(data):
            batches.append(data[n : n + batch_size])
            label_batches.append(labels[n : n + batch_size])
                
        if len(data) % batch_size > 0:
            batches.append(data[len(data)-(len(data)%batch_size):len(data)])
            label_batches.append(labels[len(data)-(len(data)%batch_size):len(data)])
        
    return batches, label_batches
    
'''
This class is for the datset and takes care of the preprocessing of the 
inputs and outs. The input required is the URL from Github of the 
CSV containing the data. 

'''
class BIOF509Dataset:
    def __init__(self, url):
        print("Initiating...")
        self.url = url
        self.data = self.preprocess_data(url)
        self.labels = self.preprocess_labels(url)
        #print("Initiated")
        
    '''
    To be able to print out the object of the dataset
    source: https://stackoverflow.com/questions/4912852/how-do-i-change-the-string-representation-of-a-python-class
    '''
    def __str__(self):
        return pd.DataFrame(self)

    '''      
    this pre-processes the data from GitHub and prepares it for the network
    Inputs: url -- the URL of the csv from Github 
    (https://raw.githubusercontent.com/yairgoldy/BNT162b2_waning_immunity/main/pos_data_days11-31_7.csv)
    '''
    def preprocess_data(self, url):
        # use pandas to read in the csv from GitHub
        print("Preprocessing...")
        data = pd.read_csv(url)
        #data = raw_data
        print("Top 5 of data:")
        print(data.head(5))
    
        # ONE HOT ENCODING
        # many variables are categorical 
        # source: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
        sector_one_hot = pd.get_dummies(data['Sector'], prefix = 'Sector')
        #print(one_hot_sector.head(5))
    
        gender_one_hot = pd.get_dummies(data['Gender'], prefix = 'Gender')
        #print(one_hot_gender.head(5))
        
        # use scikit-learn's OrdinalEncoder to convert ordinal variables
        # source: https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
    
        # create encoder object
        # categories input ensures that the months are in this order, 
        # as opposed to alphabetical order, which is the default
        #categories = np.array(['JanB', 'FebA', 'FebB', 'MarA', 'MarB', 'Apr', 'May']).reshape(-1, 1)
        vacc_encoder = OrdinalEncoder(categories = [['JanB', 'FebA', 'FebB', 'MarA', 'MarB', 'Apr', 'May']])  
        #vacc_raw = data['Vacc Period'] # read in the raw column from the csv
        # fit_transform needs a 2D array
        # source for list comprehension: https://www.w3schools.com/python/python_lists_comprehension.asp
        vacc_raw = [[x] for x in data['Vacc Period']]
        
        vacc_ordinal = vacc_encoder.fit_transform(vacc_raw) # convert to ordinal
        #print(vacc_ordinal)

        age_encoder = OrdinalEncoder(categories = [['16-39', '40-59', '60+']])
        age_raw = [[x] for x in data['Age']]
        age_ordinal = age_encoder.fit_transform(age_raw)
    
        pcr_encoder = OrdinalEncoder(categories = [['0', '1', '2+']])
        pcr_raw = [[x] for x in data['Past PCR tests']]
        pcr_ordinal = pcr_encoder.fit_transform(pcr_raw)
    
        epi_encoder = OrdinalEncoder(categories = [['07-11_07-17', '07-18_07-24', '07-25_07-31']])
        #epi_raw = data['Epi Week']
        epi_raw = [[x] for x in data['Epi Week']]
        epi_ordinal = epi_encoder.fit_transform(epi_raw)

        # create the processed data frame
        processed_input = np.concatenate([vacc_ordinal, gender_one_hot, age_ordinal, pcr_ordinal, 
                                          sector_one_hot, epi_ordinal], axis = 1)
        #print("processed input:")
        #print(processed_input[0:5,])
    
        # NORMALIZE each column based on the max in order to make sure that 
        # each feature is weighted the same 
        # must specify that axis is 0 so it normalizes each feature and not each sample
        processed_input = preprocessing.normalize(processed_input, norm = 'max', axis = 0)
        
        print("Preprocessed input")
        return processed_input
    
    '''
    This preprocesses the labels (the output layer) into more useful bins
    The raw data is normalized to positive tests per 1,000
    But for the ANN, we will bin them into 0, <=1, <=2, <=5, and >5
    input is the same URL for the raw data from Github
    '''
    def preprocess_labels(self, url):
        print("Preprocessing labels...")
        data = pd.read_csv(url)
        labels = data['Rate_Positive_1K']
        
        # use pandas 'cut' to bin the data by creating the bins and the labels
        # for the bins
        #cut_labels = ['0', '<=1', '<=2', '<=5', '>5']
        cut_labels = [0, 1, 2, 3, 4]
        cut_bins = [-np.inf, 0, 1, 2, 5, np.inf] # highest value is about 58
        # for 'cut', the default is to include the right edge: (0, 1]
        processed_labels = pd.cut(labels, bins = cut_bins, labels = cut_labels)
        print("Value counts in each bin:")
        print(processed_labels.value_counts())
 
        return processed_labels


# ***************IMPLEMENT THE NEURAL NETWORK********************
url = "https://raw.githubusercontent.com/yairgoldy/BNT162b2_waning_immunity/main/pos_data_days11-31_7.csv"

# break it down into data and labels 
dataset = BIOF509Dataset(url)
data = dataset.data
labels = dataset.labels
print("top 5 rows of data:")
print(data[0:5,])
print()
print("top 5 rows of labels:")
print(labels[0:5])

testclass = BIOF509(data, labels)
model = testclass.train_test(test_size = 0.2, n_epochs = 4, hidden_dimensions = 6, batch_size = 16, lr = 1)    
