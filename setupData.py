import random
import scipy.io
import glob, os
import numpy as np
from numpy import hstack
from numpy import array
from sklearn.model_selection import train_test_split

def setupData(directory):
  files = glob.glob(directory + "*.mat")
  indexLength = len(files)
  dat = list(range(1,indexLength))
  print(len(dat))
  random.seed(2)
  random.shuffle(dat)

  XtrainsFull, Xtrains, YtrainsFull, Ytrains, XtestsFull, Xtests, YtestsFull, Ytests = ([list() for i in range(6)] for j in range(8))
  start_index = 110
  end_trainIndex=start_index+200
  end_testIndex=end_trainIndex+600
  #print(Xtrains)
  for neighborIndex, neighborArr in enumerate(Lneighbors):
    for loopIndex, index in enumerate(dat):             
        if loopIndex % 10 == 0:
          print(loopIndex)

        file = files[index]
        mat = scipy.io.loadmat(file)
          # key is index, values 
          # we add array that consists of each one
        
        for ind, row in enumerate(neighborArr):
          firstrow = []
          rowBus = row[0]
          theRow = abs(mat['bus_v'][rowBus,:])
          theRow = theRow.reshape((len(theRow), 1))
          firstrow.append(theRow)
          for item in row[1:]:          
            theRow = abs(mat['cur'][int(item),:])
            theRow = theRow.reshape((len(theRow), 1))
            firstrow.append(theRow)
          
          firstrow = hstack( firstrow )
          #firstrow = array(firstrow)
          if loopIndex < 0.8 * len(dat):
              XtrainsFull[neighborIndex].append( [rowBus, firstrow[start_index:end_trainIndex,:]])
              YtrainsFull[neighborIndex].append( [rowBus, firstrow[end_trainIndex:end_testIndex,0]])    
              # First value will always be same
          else:
              XtestsFull[neighborIndex].append( [rowBus, firstrow[start_index:end_trainIndex,:]])
              YtestsFull[neighborIndex].append( [rowBus, firstrow[end_trainIndex:end_testIndex,0]])
    Xtrains[neighborIndex] = array([row[1] for row in XtrainsFull[neighborIndex]])
    Ytrains[neighborIndex] = array([row[1] for row in YtrainsFull[neighborIndex]])
    Xtests[neighborIndex] = array([row[1] for row in XtestsFull[neighborIndex]])
    Ytests[neighborIndex] = array([row[1] for row in YtestsFull[neighborIndex]])

    return XtrainsFull, YtrainsFull, XtestsFull, YtestsFull