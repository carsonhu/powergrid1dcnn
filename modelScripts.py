# multivariate multi-step 1d cnn example
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
import numpy as np

def adjustSplit(Xtrains, Ytrains, split):
  # base split is 200-600
  # split is < 200
  YtrainsNew = np.concatenate([Xtrains[:,:,0][:,split:200], Ytrains], axis=1)
  XtrainsNew = Xtrains[:,0:split,:]
  return XtrainsNew, YtrainsNew

def trainModel(trainData, testData, io_shape, features):
  n_steps_in, n_steps_out = io_shape
  trainData, testData = adjustSplit(trainData, testData, io_shape[0])
  model = Sequential()
  model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, features)))
  #model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, features),
  #                 kernel_regularizer=tensorflow.keras.regularizers.l2(0.003)))
  # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='causal', strides=2, input_shape=(n_steps_in, features)))
  # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='causal', strides=4, input_shape=(n_steps_in, features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(50, activation='relu'))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(trainData, testData, epochs=200)
  return model
def evalModel(Xtest, Ytest, model, io_shape, features):
  Xtest, Ytest = adjustSplit(trainData, testData, io_shape[0])
  Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], features))
  Ytest = Ytest.reshape((Ytest.shape[0], Ytest.shape[1]))
  from sklearn.metrics import mean_squared_error

  x_input = array(Xtest)
  #x_input = x_input.reshape((1, n_steps_in, n_features))
  yhat = model.predict(x_input, verbose=0)
  #x_input = x_input.reshape((x_input.shape[0], x_input.shape[1]))
  from sklearn.metrics import mean_squared_error, mean_absolute_error
  mse = mean_squared_error(Ytest,yhat)
  return mse
  #rms = mean_squared_error(Ytest, yhat)
  #rms
def evalModels(Xtest,Ytest, models, features, singleModel=False):
  total = 0
  vals = []
  toPrint = []
  
  for i in Xtest.keys():    
    if singleModel:
      val = evalModel(Xtest[i],Ytest[i], models, features)
    else:
      val = evalModel(Xtest[i],Ytest[i], models[i], features)
    toPrint.append((i,val))
    total += val
  toPrint = sorted(toPrint, key = lambda x: x[0])
  for j in toPrint:
    print(j)
  print("Total: ", total / len(list(Xtest.keys())))  
  return vals
def plotLines(model, Xtest, Ytest, bus, title):
  yhat = model.predict(array(Xtest), verbose=0)
  import matplotlib.pyplot as plt
  fig, axs = plt.subplots(3, 3)
  axs = axs.ravel()
  fig.suptitle('9 events for bus ' + str(bus) +', Model: ' + str(title))

  ids = [9, 19, 29, 39, 49, 59, 69, 79, 89]
  for i in range(len(ids)):
    axs[i].plot(yhat[ids[i],:],label="Predicted")
    axs[i].plot(Ytest[ids[i],:], label="Actual")
  axs[0].legend()

def plotLinesWholeModel(model, Xtest, Ytest, title):
  yhat = model.predict(array(Xtest), verbose=0)
  import matplotlib.pyplot as plt
  fig, axs = plt.subplots(3, 3)
  axs = axs.ravel()
  fig.suptitle('9 events for ' + str(title))

  ids = [1, 19, 27, 39, 48, 58, 67, 79, 89]
  for i in range(len(ids)):
    axs[i].plot(yhat[ids[i],:],label="Predicted")
    axs[i].plot(Ytest[ids[i],:], label="Actual")
  axs[0].legend()
  figure = plt.gcf()  # get current figure
  figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
  plt.savefig('filename.png', bbox_inches='tight') # bbox_inches removes extra white spaces  
  