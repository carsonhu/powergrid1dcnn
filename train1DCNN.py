from numpy import array
from sklearn import preprocessing
import numpy as np
import copy
import modelScripts

def standardizeData(XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent):
    # for getting all the 1 neighbor currents
    Xtrains, Ytrains, Xtests,  Ytests = ({} for j in range(4))

    for i in range(6):
        Xtrains[i] = array([row[1] for row in XtrainsCurrent[i]])
        Ytrains[i] = array([row[1] for row in YtrainsCurrent[i]])
        Xtests[i] = array([row[1] for row in XtestsCurrent[i]])
        Ytests[i] = array([row[1] for row in YtestsCurrent[i]])
    
    # standardize current
    Xtrains1 = copy.deepcopy(Xtrains)
    Xtests1 = copy.deepcopy(Xtests)
    for i in range(5):
        scaler1 = preprocessing.StandardScaler()
        scaler2 = preprocessing.StandardScaler()
        XtrainC = Xtrains[i][:,:,1:]
        XtestC = Xtests[i][:,:,1:]
        Xtrains1[i][:,:,1:] = scaler1.fit_transform(XtrainC.reshape(XtrainC.shape[0] * XtrainC.shape[2], XtrainC.shape[1] )).reshape(XtrainC.shape)
        Xtests1[i][:,:,1:] = scaler2.fit_transform(XtestC.reshape(XtestC.shape[0] * XtestC.shape[2], XtestC.shape[1] )).reshape(XtestC.shape)
    return Xtrains1, Ytrains, Xtests1, Ytests



def get1NeighborCurrents(XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent):
    # for getting all the 1 neighbor currents

    Xtrains, Ytrains, Xtests,  Ytests, XtrainsNoC, XtestsNoC = ({} for j in range(6))

    for j in range(6):
        for id in set([row[0] for row in XtrainsCurrent[j]]):
            Xtrains[id] = array([row[1] for row in XtrainsCurrent[j] if row[0] == id])
            Ytrains[id] = array([row[1] for row in YtrainsCurrent[j] if row[0] == id])
            Xtests[id] = array([row[1] for row in XtestsCurrent[j] if row[0] == id])
            Ytests[id] = array([row[1] for row in YtestsCurrent[j] if row[0] == id])
            XtrainsNoC[id] = array([row[1][:,0] for row in XtrainsCurrent[j] if row[0] == id])
            XtestsNoC[id] = array([row[1][:,0] for row in XtestsCurrent[j] if row[0] == id])

    # modifying Xtrains
    for id in Xtrains:
        Xtrains1 = Xtrains[id]
        Xtests1 = Xtests[id]
        scaler1 = preprocessing.StandardScaler()
        scaler2 = preprocessing.StandardScaler()
        XtrainC = Xtrains1[:,:,1:]
        XtestC = Xtests1[:,:,1:]
        Xtrains1[:,:,1:] = scaler1.fit_transform(XtrainC.reshape(XtrainC.shape[0] * XtrainC.shape[2], XtrainC.shape[1] )).reshape(XtrainC.shape)
        Xtests1[:,:,1:] = scaler2.fit_transform(XtestC.reshape(XtestC.shape[0] * XtestC.shape[2], XtestC.shape[1] )).reshape(XtestC.shape)

    return Xtrains, Ytrains, Xtests, Ytests

def loadCurrentData():
    XtrainsCurrent=np.load('data/XtrainFullMultiNeighbor2.npy', allow_pickle=True)
    YtrainsCurrent=np.load('data/YtrainFullMultiNeighbor2.npy', allow_pickle=True)
    XtestsCurrent=np.load('data/XtestFullMultiNeighbor2.npy', allow_pickle=True)
    YtestsCurrent=np.load('data/YtestFullMultiNeighbor2.npy', allow_pickle=True)
    return XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent

if __name__ == "__main__":
    XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent = loadCurrentData()
    XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent = get1NeighborCurrents(XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent)

    # Xtrains, Ytrains, Xtests, Ytests = standardizeData(XtrainsCurrent, YtrainsCurrent, XtestsCurrent, YtestsCurrent)
