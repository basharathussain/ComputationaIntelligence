# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:07:38 2020

@author: Bash
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:08:06 2019

@author: Bash
"""
"""
###########################
Problem #8 on google colab 
###########################
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from math import sqrt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import time 
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.optimizers import Adam, Adagrad, RMSprop 
from keras.layers.recurrent import SimpleRNN



from statsmodels.tsa.arima_model import ARIMA

from IPython.display import display # to display images
import texttable as tt

from sys import exit


from imageUtil import imgUtil
from models import SupervisedDBNRegression

### SECTION 11
RUN_ON_COLAB = 0      #0 = local, 1 = colab
RUN_ON_DATASET = 3  # 0=100-series, 1 = PeMS, 2=Shampoo, 3=Alzimers
#EPOCHS = [20, 50, 100, 200, 300, 400, 500, 1000] 
REPEATS = 1 #30     # number of times you like to repeat EPOCHs Experienet 

ALGOS =[[1001, 'LSTM'] ]

ALGOKEYS = [x[0] for x in ALGOS]
ALGOKEYS = [1001]


PMES_FILE_START_INDEX = 7
PMES_FILE_END_INDEX = 20
FLOW_LEVEL = 1    # 0 = hour level, 1 = minute level, 2 = both levels
FREEWAY_NO = '99'  #Station No 5 to filter data with
#STATION_NO = '99'  #Station No 99 to filter data with
L = 3  # window length

## Experimental Settings
NSize = 64
L = 5
EPOCHS = [1000]
# EPOCHS = [10]
BS = 128
LR = 0.01

## Experimental Settings
#NSize = 256
#L = 6
#EPOCHS = [1000]
#BS = 64
#LR = 0.2



SEED_DATE = datetime(2019, 1, 7)
dt0 = SEED_DATE 
FILE_INDICES = range(PMES_FILE_START_INDEX, PMES_FILE_END_INDEX)  # traiing from 1-27 days
Validation_dataset_size_percentage = 15 
Test_dataset_size_percentage = 15 
arr_Epochs = EPOCHS 

global_arr_Plots= []


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Split data into train, validation and test sets
def split_data(raw_data):
    data = []
    
    raw_data = np.asmatrix(raw_data) 
    data = np.array(raw_data);
    
    valid_set_size = int(np.round(Validation_dataset_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(Test_dataset_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
     #Separating the datasets
    X_train = data[:train_set_size,:-1]
    y_train = data[:train_set_size,-1]
    
    X_valid = data[train_set_size:train_set_size+valid_set_size,:-1]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1]
    
    X_test = data[train_set_size+valid_set_size:,:-1]
    y_test = data[train_set_size+valid_set_size:,-1]
    
    return [X_train, y_train, X_valid, y_valid, X_test, y_test]
    
    # Split data into train, validation and test sets
def split_data_alzimer(raw_data):
    data = []
    
    raw_data = np.asmatrix(raw_data) 
    data = np.array(raw_data);
    
    valid_set_size = int(np.round(Validation_dataset_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(Test_dataset_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
     #Separating the datasets
    X_train = data[:train_set_size,:-1]
    y_train = data[:train_set_size,-1]
    
    X_valid = data[train_set_size:train_set_size+valid_set_size,:-1]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1]
    
    X_test = data[train_set_size+valid_set_size:,:-1]
    y_test = data[train_set_size+valid_set_size:,-1]
    
    return [X_train, y_train, X_valid, y_valid, X_test, y_test]
    
    
def ReadPeMSFiles(fileIndices):
        
    datasetFolder = "H:/10. PeMS dataset/5-minStation-dist10" if RUN_ON_COLAB == 0 else "/content/drive/My Drive/Google-CoLab/Problem#8 - My Paper LSTM GRU on PeMS data/Dataset-Jan2019"
    
    file_data = []   
    for fileNo in fileIndices:
        if(fileNo < 10):
            file = open(datasetFolder + "/d10_text_station_5min_2019_01_0"+str(fileNo)+".txt","r")
        else:
            file = open(datasetFolder + "/d10_text_station_5min_2019_01_"+str(fileNo)+".txt","r")
            
        print (fileNo)
        
        previous_timestamp = -1
        #Repeat for each song in the text file
        for line in file:
          
          #Let's split the line into an array called "fields" using the ";" as a separator:
          fields = line.split(",")
              
          #and let's extract the data:
          #to_see_if_needed = fields[0]
          if(fields[3] == FREEWAY_NO):  #Station No 5 only
              
              dt = datetime.strptime(fields[0], "%m/%d/%Y %H:%M:%S")
              delta = dt - dt0
              days = delta.days
              total_seconds = days*24*60*60 + delta.seconds
              # to remove the whitespaces and one quote '
              speed = fields[9].strip("\r\n\t '")
              if speed == '':
                  speed =  '0'
              speed = float(speed)
              
              if(previous_timestamp == total_seconds): 
                  location = file_data[-1]
                  # if program manages to get
                  file_data[-1] = [location[0], fields[0], (speed + location[2] )/2.0] 
              else:              
                  # if it doesn't find 5 this
                  file_data.append([total_seconds, fields[0], speed]) 
                  previous_timestamp = total_seconds
        
        #It is good practice to close the file at the end to free up resources   
        file.close()
 
    return file_data

#print(file_data[0:10]) 
def Date_parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
    
 
#############################################
### Step: read dataset files PeMS
#############################################

if(RUN_ON_DATASET == 1):  
    df = ReadPeMSFiles(FILE_INDICES)
    ds = pd.Series([i[2] for i in df], index=[i[1] for i in df])
elif (RUN_ON_DATASET == 2):
   ds = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=Date_parser)
   df = [[x, ds[x]] for x in ds.index]
elif(RUN_ON_DATASET == 3):  
   #ds = read_csv('../../Master_Finalplus.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
   ds = read_csv('Master_Finalplus.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
   df = [[x, ds['TOTALMOD'][:][x]] for x in ds.index[0:]]
   
    
if (RUN_ON_DATASET != 0):
    #Creating a copy of the data to pass through the parser
    raw_set = df.copy()
    
    if FLOW_LEVEL == 0:
        listInHrs = [x for x in raw_set if x[0]%3600 == 0]  #every hour
        raw_set = listInHrs
    
    if FLOW_LEVEL == 0:
        listInHrs = [x for x in raw_set if x[0]%3600 == 0]  #every hour
        raw_set = listInHrs
        
     #Sep out the data
    if RUN_ON_DATASET == 3:
        X_train, y_train, X_valid, y_valid, X_test, y_test = split_data_alzimer(raw_set);
    else:
        X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(raw_set);
    
    y_train = np.array(list(map(float,y_train.tolist())))
    y_valid = np.array(list(map(float,y_valid.tolist())))
    y_test = np.array(list(map(float,y_test.tolist())))
    
    #Checking the shape of each of the data 
    print('x_train.shape = ',X_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_valid.shape = ',X_valid.shape)
    print('y_valid.shape = ', y_valid.shape)
    print('x_test.shape = ', X_test.shape)
    print('y_test.shape = ',y_test.shape)

#Plot a graph for each of train, valid and test sets
print("---------------PLOT a minute level graph first --------------------") 
if RUN_ON_DATASET == 1:
    x_axis = X_train[0:,1].tolist()
    x_axis = [datetime.strptime(x, "%m/%d/%Y %H:%M:%S") for x in x_axis]
    y_axis = y_train   
    lines=plt.plot(x_axis, y_axis, label="Per-minute real flow - train")
    plt.setp(lines, color='b', linewidth=1.2, ls='-')
       
    x_axis = X_valid[0:,1].tolist()
    x_axis = [datetime.strptime(x, "%m/%d/%Y %H:%M:%S") for x in x_axis]
    y_axis = y_valid   
    lines=plt.plot(x_axis, y_axis, label="per-minute real flow - validation")
    plt.setp(lines, color='g', linewidth=1.2, ls='-')
    
    x_axis = X_test[0:,1].tolist()
    x_axis = [datetime.strptime(x, "%m/%d/%Y %H:%M:%S") for x in x_axis]
    y_axis = y_test   
    lines=plt.plot(x_axis, y_axis, label="per-minute real flow - validation")
    plt.setp(lines, color='#ffa500', linewidth=1.2, ls='-')
     
    fig, ax = plt.subplots()   
    ax.set_xlabel('Time-interval (Date)')
    ax.set_ylabel('Traffic flow (# of Veh/5 min)')
    ax.set_title('Traffic flow rate from PeMS real data')
    plt.xticks(rotation=70)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(.02)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=.1)
    
    def date2yday(x):
        #  x is in matplotlib seconds, so they are numbers.
        y = x - mdates.date2num(datetime(2019, 1, 1))
        return y
    def yday2date(x):
        # return a matplotlib datenum (x is seconds since start of year)
        y = x + mdates.date2num(datetime(2019, 1, 1))    
        return y
    
    secaxx = ax.secondary_xaxis('top', functions=(date2yday, yday2date))
    secaxx.set_xlabel('days from year start [2019]')
     
    plt.legend(['Per-minute flow - Train dataset', 'Validation', 'Test'], loc='upper left')
    fig.set_size_inches(8, 4)
    
    plt.show()
    img = imgUtil.plt2img ( fig )
    global_arr_Plots.append([img])
    
elif RUN_ON_DATASET == 2:
    x_axis = X_train[0:,0].tolist()
    y_axis = y_train   
    
    fig, ax = plt.subplots()       
    lines=plt.plot(x_axis, y_axis, label="Per-minute real flow - train")
    plt.setp(lines, color='b', linewidth=1.2, ls='-')
       
    x_axis = X_valid[0:,0].tolist()
    y_axis = y_valid   
    lines=plt.plot(x_axis, y_axis, label="per-minute real flow - validation")
    plt.setp(lines, color='g', linewidth=1.2, ls='-')
    
    x_axis = X_test[0:,0].tolist()
    y_axis = y_test   
    lines=plt.plot(x_axis, y_axis, label="per-minute real flow - validation")
    plt.setp(lines, color='#ffa500', linewidth=1.2, ls='-')
    fig.set_size_inches(8, 4)
    
    plt.show()
    img = imgUtil.plt2img ( fig )
    global_arr_Plots.append([img])
    
elif RUN_ON_DATASET == 3:
    x_axis = X_train[0:,0].tolist()
    y_axis = y_train   
    
    fig, ax = plt.subplots()       
    lines=plt.plot(x_axis, y_axis, label="Per-minute real flow - train")
    plt.setp(lines, color='b', linewidth=1.2, ls='-')
       
    x_axis = X_valid[0:,0].tolist()
    y_axis = y_valid   
    lines=plt.plot(x_axis, y_axis, label="per-minute real flow - validation")
    plt.setp(lines, color='g', linewidth=1.2, ls='-')
    
    x_axis = X_test[0:,0].tolist()
    y_axis = y_test   
    lines=plt.plot(x_axis, y_axis, label="per-minute real flow - validation")
    plt.setp(lines, color='#ffa500', linewidth=1.2, ls='-')
    fig.set_size_inches(8, 4)
    
    plt.show()
    img = imgUtil.plt2img ( fig )
    global_arr_Plots.append([img])
    
    
#############################################
### STEP 1: Data Preprocessing for LSTM input
### Convert from time series to sliding window 
#############################################
# split a univariate sequence into samples 
# using sliding window of n_steps
def split_sequence_univariate(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def split_sequence_univariate_alzimer(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		start_ix = ((i) * n_steps)+i
		end_ix = start_ix + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[start_ix:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input/training sequence
# train_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
# valid_seq = [130, 140, 150, 160, 170, 180, 190, 200]
# test_seq = [ 180, 190, 200, 210, 220, 230, 240]

if RUN_ON_DATASET == 0:
    times = pd.date_range('2019-10-01', periods=289, freq='5min')
    times = np.array(times[0:len(train_seq)])
    ds = pd.Series([float(i) for i in train_seq], index=[i for i in times]) # used for ARIMA
else: 
    train_seq = y_train
    valid_seq = y_valid
    test_seq = y_test
    
# choose a number of time steps in sliding window
n_steps = L #3
# split into samples
if RUN_ON_DATASET == 3:
    X_train_seq, y_train_seq = split_sequence_univariate_alzimer(train_seq, n_steps)
    X_valid_seq, y_valid_seq = split_sequence_univariate_alzimer(valid_seq, n_steps)
    X_test_seq, y_test_seq = split_sequence_univariate_alzimer(test_seq, n_steps)
else:
    X_train_seq, y_train_seq = split_sequence_univariate(train_seq, n_steps)
    X_valid_seq, y_valid_seq = split_sequence_univariate(valid_seq, n_steps)
    X_test_seq, y_test_seq = split_sequence_univariate(test_seq, n_steps)

X_all = np.append(X_train_seq, X_valid_seq, axis=0) 
X_all = np.append(X_all, X_test_seq, axis=0) 
y_all = np.append(y_train_seq, y_valid_seq, axis=0)
y_all = np.append(y_all, y_test_seq, axis=0) 

# summarize the data
for i in range(min(len(X_all), 5)):
	print(X_all[i], y_all[i])


commulative_results = [];
commulative_results.append(['Method Name', 'Epochs', 'RMSE', 'MAPE%', 'MAE', "Execution Time (sec)", 'Total', 'Correct', 'Wrong'])

def draw_results():        
    tab = tt.Texttable()
    tab.header(commulative_results[0])
    for row in commulative_results[1:]:
        tab.add_row(row)
    s = tab.draw()
    print (s)



def build_vanilla_lstm_model(units, steps, features, act, opt, los):
    # define model
    model = Sequential()
    model.add(LSTM(units, 
                   activation=act,
                   return_sequences=True,
                   input_shape=(steps, features)))
    model.add(LSTM(64, return_sequences=True, activation=act))
    model.add(LSTM(32, activation=act))
    model.add(Dense(1))
    
    model.compile(optimizer=optimizer, loss=los, metrics=['accuracy'])
    model.summary()
    return model


def build_bidirectional_lstm_model(units, steps, features, act, opt, los):

# define model
    model = Sequential()
    model.add(Bidirectional(LSTM(units, activation=act),
                            return_sequences=True, 
                            input_shape=(steps, features)))
    model.add(LSTM(64, return_sequences=True, activation=act))
    model.add(LSTM(32, activation=act))
    model.add(Dense(1))
    
    model.compile(optimizer=opt, loss=los, metrics=['accuracy'])
    model.summary()
    return model

def build_gru_model(units, steps, features, act, opt, los):
    # define model
    model = Sequential()
    model.add(GRU(units, 
                   activation=act, 
                   return_sequences=True, 
                   input_shape=(steps, features)))
    model.add(GRU(64, return_sequences=True, activation=act))
    model.add(GRU(32, activation=act))
    model.add(Dense(1))
   
    #opt1 = Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss=los, metrics=['accuracy'])
    model.summary()
    return model

def buildMLP(units,  steps, out_dim, act, opt, los):
    print('Build MLP...')
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=steps))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(out_dim, activation='relu'))

    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model

def buildRNN(units,  steps, out_dim, act, opt, los):
    print('Build RNN...')
    model = Sequential()
    model.add(SimpleRNN(out_dim, input_shape=(steps, out_dim)))
   
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    model.summary()
    return model

 

for a in range(len(ALGOKEYS)):
    algo_name = [x[1] for x in ALGOS if x[0] == ALGOKEYS[a]][0]
    for j in range(len(EPOCHS)):
        start_time = time.time()   
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X_train_seq
        y = y_train_seq
        
        X = X.reshape(X.shape[0], X.shape[1], n_features)
       
        unit_size = NSize
        n_steps = L #3
        n_features = 1
        activation='relu'
        optimizer='adam'
        loss='mse'
        
        validAlgo = True 
        
        if 1001 in ALGOKEYS:
            model = build_vanilla_lstm_model(unit_size, n_steps, n_features, activation, optimizer, loss)
         #        elif 1013 in ALGOKEYS:
    #            model = buildMLP(unit_size, n_steps, n_features, activation, optimizer, loss)
        elif 1014 in ALGOKEYS:
            model = buildRNN(unit_size, n_steps, n_features, activation, optimizer, loss)        
        else:
            validAlgo = False        
        
        if(validAlgo):
    #        # fit model
            for repeat in range(REPEATS):
                #model.fit(X, y, epochs=EPOCHS[j], verbose=0)
                #model.fit(train_seq[3:], y, epochs=EPOCHS[j], verbose=0)
                model.fit(X, y, epochs=EPOCHS[j], batch_size=BS,  
                      validation_data=(X_valid_seq.reshape(X_valid_seq.shape[0], X_valid_seq.shape[1], n_features), y_valid_seq), 
                      shuffle=False)
                model.reset_states()
                
            # demonstrate prediction on train + validation + test
            arr_yhat = []
            for observation in X_all:
              #x_input = array([44, 50, 70])
              x_input = observation.reshape((1, n_steps, n_features))
              #x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], n_features))
              yhat = model.predict(x_input, verbose=2)
              arr_yhat.append(yhat[0][0])
            
            arr_y = y_all
            arr_x = [i for i in range(len(X_all))]
            
            fig, ax = plt.subplots()
            ax.set_title('Method: %s with Epochs: %d' % (algo_name, EPOCHS[j]))
            ax.set_xlabel('Observation count')
            ax.set_ylabel('Traffic flow (# of Veh/5 min)')
            plt.plot(arr_x, arr_y, 'r', label='Actual traffic flow',linewidth=1.5)
            plt.plot(arr_x, arr_yhat, '-g', label=algo_name, linewidth=1.2)
            plt.legend()            
            fig.set_size_inches(8, 4)
    
            plt.show()
            img = imgUtil.plt2img ( fig )
            global_arr_Plots.append([img])
            
            
#            global_arr_Plots.append([algo_name, EPOCHS[j], ax.canvas.draw()])
          
            print("===============Data Set=======================")
            print("Observation, Label, Prediction")
            # summarize the data
            for i in range(min(len(X_all), 15)):
            	print([a for a in X_all[i]], arr_y[i], arr_yhat[i] )
               
            
            print("===============RMSE=================")
            # report RMSE performance
            rmse = sqrt(mean_squared_error(arr_y, arr_yhat))
            mae = mean_absolute_error(arr_y, arr_yhat)
            mape = mean_absolute_percentage_error(arr_y, arr_yhat)
            print('RMSE: %.3f, MAPE: %.1f, MAE: %.3f, EPOCHS: %d, Algo: %s' % (rmse, mape, mae, EPOCHS[j], algo_name))
            
        
            # correctness
            total = 0
            correct = 0
            wrong = 0
            threshold = 5.0
            for i in range(len(arr_yhat)):
              total=total+1
              
              if((arr_y[i] - arr_yhat[i]) < threshold):
                correct=correct+1
              else:
                wrong=wrong+1
            
            end_time = time.time()
            
            
            commulative_results.append([algo_name, EPOCHS[j], round(rmse, 2),round(mape, 2), round(mae, 2), round(end_time - start_time, 2), total, correct, wrong])
            
        draw_results()  
        print("===============End of "+algo_name+"=======================")

if 1011 in ALGOKEYS:   #DBN
    algo_name = [x[1] for x in ALGOS if x[0] == 1011][0]
    for j in range(len(EPOCHS)):
        start_time = time.time()   
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X_train_seq
        y = y_train_seq
        
        # Data scaling
        min_max_scaler = MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        
        # Training
        regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                            learning_rate_rbm=0.01,
                                            learning_rate=0.01,
                                            n_epochs_rbm=EPOCHS[j], #20,
                                            n_iter_backprop=200,
                                            batch_size=16,
                                            activation_function='relu')
        regressor.fit(X, y)
        
        # Test
        X_test = min_max_scaler.transform(X_all)
        arr_yhat = regressor.predict(X_test)
     
        arr_y = y_all
        arr_x = [i for i in range(len(X_all))]
        arr_yhat = [round(i[0], 3) for i in arr_yhat]
        
        fig, ax = plt.subplots()
        ax.set_title('Method: %s with Epochs: %d' % (algo_name, EPOCHS[j]))
        ax.set_xlabel('Observation count')
        ax.set_ylabel('Traffic flow (# of Veh/5 min)')
        plt.plot(arr_x, arr_y, 'r', label='Actual traffic flow',linewidth=1.5)
        plt.plot(arr_x, arr_yhat, '-g', label=algo_name, linewidth=1.2)
    #    plt.rcParams["figure.figsize"] = (16,6)
        plt.legend()
        fig.set_size_inches(8, 4)
    
        plt.show()
        img = imgUtil.plt2img ( fig )
        global_arr_Plots.append([img])
                
        print("===============Data Set=======================")
        print("Observation, Label, Prediction")
        # summarize the data
        for i in range(min(len(X_all), 15)):
        	print([a for a in X_all[i]], arr_y[i], arr_yhat[i])
           
        
        print("===============RMSE=================")
        # report RMSE performance
        rmse = sqrt(mean_squared_error(arr_y, arr_yhat))
        mae = mean_absolute_error(arr_y, arr_yhat)
        mape = mean_absolute_percentage_error(arr_y, arr_yhat)
        print('RMSE: %.3f, MAPE: %.1f, MAE: %.3f, EPOCHS: %d, Algo: %s' % (rmse, mape, mae, EPOCHS[j], algo_name))
        
    
        # correctness
        total = 0
        correct = 0
        wrong = 0
        threshold = 5.0
        for i in range(len(arr_yhat)):
          total=total+1
          
          if((arr_y[i] - arr_yhat[i]) < threshold):
            correct=correct+1
          else:
            wrong=wrong+1
        
        end_time = time.time()
        
        
        commulative_results.append([algo_name, EPOCHS[j], round(rmse, 2), round(mape, 2), round(mae, 2), round(end_time - start_time, 2), total, correct, wrong])
    
    draw_results()  
    print("===============End of "+algo_name+"=======================")

 
if 1013 in ALGOKEYS:   # MLP
    algo_name = [x[1] for x in ALGOS if x[0] == 1013][0]
    for j in range(len(EPOCHS)):
        start_time = time.time()   
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X_train_seq
        y = y_train_seq
        
        X = X.reshape(X.shape[0], X.shape[1], n_features)
       
        unit_size = NSize
        n_steps = L
        n_features = 1
        activation='relu'
        optimizer='adam'
        loss='mse'
    
        model = buildMLP(unit_size, X.shape[0], n_features, activation, optimizer, loss)
    
        model.fit(X, y, epochs=EPOCHS[j], verbose=0)
    
    #        # fit model
    #        for repeat in range(REPEATS):
    #            model.fit(X, y, epochs=EPOCHS[j], verbose=0)
    #            model.fit(X, y, epochs=EPOCHS[j], batch_size=64, 
    #                  validation_data=(X_valid_seq.reshape(X_valid_seq.shape[0], X_valid_seq.shape[1], n_features), y_valid_seq), 
    #                  shuffle=False)
    #            model.reset_states()
    #           
        #model.fit(X, y, epochs=EPOCHS[j], verbose=0)
    #        model.fit(X, y, epochs=EPOCHS[j], batch_size=64, 
    #                  validation_data=(X_valid_seq.reshape(X_valid_seq.shape[0], X_valid_seq.shape[1], n_features), y_valid_seq), 
    #                  shuffle=False)
             
            
        # demonstrate prediction on train + validation + test
        arr_yhat = []
        for observation in X_all:
          #x_input = array([44, 50, 70])
          x_input = observation.reshape((1, n_steps, n_features))
          #x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], n_features))
          yhat = model.predict(x_input, verbose=0)
          arr_yhat.append(yhat[0][0])
        
        arr_y = y_all
        arr_x = [i for i in range(len(X_all))]
        
        fig, ax = plt.subplots()
        ax.set_title('Method: %s with Epochs: %d' % (algo_name, EPOCHS[j]))
        ax.set_xlabel('Observation count')
        ax.set_ylabel('Traffic flow (# of Veh/5 min)')
        plt.plot(arr_x, arr_y, 'r', label='Actual traffic flow',linewidth=1.5)
        plt.plot(arr_x, arr_yhat, '-g', label=algo_name, linewidth=1.2)
    #    plt.rcParams["figure.figsize"] = (16,6)
        plt.legend()
        fig.set_size_inches(8, 4)
    
        plt.show()
        img = imgUtil.plt2img ( fig )
        global_arr_Plots.append([img])
        
        print("===============Data Set=======================")
        print("Observation, Label, Prediction")
        # summarize the data
        for i in range(min(len(X_all), 15)):
        	print([a for a in X_all[i]], arr_y[i], arr_yhat[i] )
           
        
        print("===============RMSE=================")
        # report RMSE performance
        rmse = sqrt(mean_squared_error(arr_y, arr_yhat))
        mae = mean_absolute_error(arr_y, arr_yhat)
        mape = mean_absolute_percentage_error(arr_y, arr_yhat)
        print('RMSE: %.3f, MAPE: %.1f, MAE: %.3f, EPOCHS: %d, Algo: %s' % (rmse, mape, mae, EPOCHS[j], algo_name))
        
    
        # correctness
        total = 0
        correct = 0
        wrong = 0
        threshold = 5.0
        for i in range(len(arr_yhat)):
          total=total+1
          
          if((arr_y[i] - arr_yhat[i]) < threshold):
            correct=correct+1
          else:
            wrong=wrong+1
        
        end_time = time.time()
        
        
        commulative_results.append([algo_name, EPOCHS[j], round(rmse, 2),round(mape, 2), round(mae, 2), round(end_time - start_time, 2), total, correct, wrong])
        
    draw_results()  
    print("===============End of "+algo_name+"=======================")


if 1007 in ALGOKEYS:
    algo_name = [x[1] for x in ALGOS if x[0] == 1007][0]
    
    start_time = time.time()   
    
    # fit model
    model = ARIMA(ds, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())
    
    ##################ROLLING FORECAST ARIMA#########
    X = ds.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
    	model = ARIMA(history, order=(5,1,0))
    	model_fit = model.fit(disp=0)
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    	print('predicted=%.3f, expected=%.3f' % (yhat, obs))
        
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
   ###################
    arr_y = test
    arr_yhat = predictions
    rmse = sqrt(mean_squared_error(arr_y, arr_yhat))
    mae = mean_absolute_error(arr_y, arr_yhat)
    mape = mean_absolute_percentage_error(arr_y, arr_yhat)
    print('RMSE: %.3f, MAPE: %.1f, MAE: %.3f, EPOCHS: %d, Algo: %s' % (rmse, mape, mae, EPOCHS[j], algo_name))
    ##################
    # plot
    fig, ax = plt.subplots()
    ax.set_title('Method: %s with Epochs: %d' % (algo_name, EPOCHS[j]))
    ax.set_xlabel('Observation count')
    ax.set_ylabel('Traffic flow (# of Veh/5 min)')
    plt.plot(test)
    plt.plot(predictions, color='red')
    fig.set_size_inches(8, 4)

    plt.show()
    img = imgUtil.plt2img ( fig )
    global_arr_Plots.append([img])
     
    ##################ROLLING FORECAST ARIMA#########
    # https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7
    rolling_mean = ds.rolling(window = 6).mean()
    rolling_std = ds.rolling(window = 6).std()
    plt.plot(ds, color = 'blue', label = 'Original')
    plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.show()
    
    end_time = time.time()
    commulative_results.append([algo_name, 'N/A', round(rmse, 2),round(mape, 2), round(mae, 2), round(end_time - start_time, 2), -1, -1, -1])
    draw_results()  
    print("===============End of "+algo_name+"=======================")
 

   

def build_cnn_lstm_model():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def build_convLstm_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()    
    return model

# univariate lstm example
if 2001 in ALGOKEYS or 2002 in ALGOKEYS:
    for a in range(len(ALGOKEYS)):
        algo_name = [x[1] for x in ALGOS if x[0] == ALGOKEYS[a]][0]

    #############################################
    ## Data pre-processing startes here
    # choose a number of time steps in sliding window
    n_steps = 4
    # split into samples
    
    X_train_seq, y_train_seq = split_sequence_univariate(train_seq, n_steps)
    X_valid_seq, y_valid_seq = split_sequence_univariate(valid_seq, n_steps)
    X_test_seq, y_test_seq = split_sequence_univariate(test_seq, n_steps)
    
    X_all = np.append(X_train_seq, X_valid_seq, axis=0) 
    X_all = np.append(X_all, X_test_seq, axis=0) 
    y_all = np.append(y_train_seq, y_valid_seq, axis=0)
    y_all = np.append(y_all, y_test_seq, axis=0) 
    
    # summarize the data
    for i in range(min(len(X_all), 5)):
    	print(X_all[i], y_all[i])

    X = X_train_seq
    y = y_train_seq
    
    n_features = 1
    n_seq = 2
    n_steps = 2
    X = X.reshape(X.shape[0], n_seq, n_steps, n_features)
     
    # summarize the data
    for i in range(min(len(X), 5)):
    	print(X[i], y[i])
    # ---------------------------------|
    #   sample|  label   |  prediction |
    # ---------------------------------|
    #   X     |  y       |  yhat       | << given data
    #   Xunk  |  yunk    |  yunkhat    | << unknown samples 
    # ---------------------------------|
    #   Xall  |  yall    |  yallhat    | << unknown samples 
    # ---------------------------------|
    
    
    #############################################
    ## Data pre-processing ends here
    #############################################
    
    for j in range(len(arr_Epochs)):
        start_time = time.time() 
        
       
        # define model
        if 2001 in ALGOKEYS:
            model = build_cnn_lstm_model()
        elif 2002 in ALGOKEYS:
            model = build_convLstm_model()

        # fit model
        model.fit(X, y, epochs=arr_Epochs[j], verbose=0)
        # demonstrate prediction
        arr_yhat = []
        for observation in X_all:
          #x_input = array([44, 50, 70])
          #x_input = observation.reshape((1, n_steps, n_features))0
          x_input = observation.reshape((1, n_seq, n_steps, n_features))
          #x_input = x_input.reshape((x_input.shape[0], x_input.shape[1], n_features))
          yhat = model.predict(x_input, verbose=0)
          arr_yhat.append(yhat[0][0])
        
        arr_y = y_all
        arr_x = [i for i in range(len(X_all))]
        
        fig, ax = plt.subplots()
        ax.set_title('Method: %s with Epochs: %d' % (algo_name, EPOCHS[j]))
        ax.set_xlabel('Observation count')
        ax.set_ylabel('Traffic flow (# of Veh/5 min)')
        plt.plot(arr_x, arr_y, 'r', label='Actual traffic flow',linewidth=2.0)
        plt.plot(arr_x, arr_yhat, '-g', label=algo_name, linewidth=2.0)
    #    plt.rcParams["figure.figsize"] = (16,6)
        plt.legend()
        fig.set_size_inches(8, 4)
    
        plt.show()
        img = imgUtil.plt2img ( fig )
        global_arr_Plots.append([img])

        
        
        print("===============Data Set=======================")
        print("Observation, Label, Prediction")
        # summarize the data
        for i in range(min(len(X_all), 15)):
        	print([a for a in X_all[i]], arr_y[i], arr_yhat[i] )
           
        
        print("===============RMSE=================")
        # report RMSE performance
        rmse = sqrt(mean_squared_error(arr_y, arr_yhat))
        mae = mean_absolute_error(arr_y, arr_yhat)
        mape = mean_absolute_percentage_error(arr_y, arr_yhat)
        print('RMSE: %.3f, MAPE: %.1f, MAE: %.3f, EPOCHS: %d, Algo: %s' % (rmse, mape, mae, EPOCHS[j], algo_name))
       
    
        # correctness
        total = 0
        correct = 0
        wrong = 0
        threshold = 2.0
        for i in range(len(arr_yhat)):
          total=total+1
          
          if((arr_y[i] - arr_yhat[i]) < threshold):
            correct=correct+1
          else:
            wrong=wrong+1
        
        end_time = time.time()
        
        
        commulative_results.append([algo_name, EPOCHS[j], round(rmse, 2),round(mape, 2), round(mae, 2), round(end_time - start_time, 2), total, correct, wrong])
    
    
    print("===============overall results=================")
   
    draw_results()  
    print("===============End of "+algo_name+"=======================")
 
    
##POST PROCESSING
algo_list = [i[0] for i in  commulative_results[1:]]
# get distinct values
algo_set = set(algo_list)
# from set to list
algo_set = list(algo_set)

box_data_desc = []
box_data_dict = {}
for i in range(len(algo_set)):
    algo_result = [x for x  in commulative_results if x[0] == algo_set[i]]
    print("===============# " + str(i+1) + ": boxplot analysis of "+algo_set[i]+"=================")
    error_scores_rmse = [i[2] for i in  algo_result]
    
    box_data_dict[algo_set[i]] = error_scores_rmse
    
    results = DataFrame()
    results['rmse'] = error_scores_rmse
    print(results.describe())
    box_data_desc.append(results.describe())


fig, ax = plt.subplots()
bp = ax.boxplot(box_data_dict.values(), patch_artist=True)
ax.set_xticklabels(box_data_dict.keys())
ax.set_xlabel('NN Algorithm')
ax.set_ylabel('RMSE values')
ax.set_title('BoxPlot')
plt.xticks(rotation=70)

box_colors = ['#5975A4', '#5F9E6E', '#B55D60', '#857AAA', '#ED008C']
k = 0
## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = box_colors[k] )
    k = k + 1

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
              
#     # summarize results
#     results = DataFrame()
#     results['rmse'] = error_scores
#     print(results.describe())
#     results.boxplot()
#     #plt.gcf().set_size_inches(18, 6)
    
plt.gcf().set_size_inches(18, 6)   
plt.show()

print("===============overall results=================")
for i in range(len(commulative_results)):
	print(commulative_results[i]) 
    
 
for i, d in enumerate(commulative_results):
    line = '|'.join(str(x).ljust(12) for x in d)
    print(line)
    if i == 0:
        print('-' * len(line))



for k in range(len(global_arr_Plots)):
    img = global_arr_Plots[k][0] 
    display(img)
