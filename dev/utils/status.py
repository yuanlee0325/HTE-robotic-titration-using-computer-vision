# check OT2's status
import os
import pandas as pd


def get_status(path,fname='pi2msi_copy.txt'):
    fname=os.path.join(path,fname)
    isFile = os.path.isfile(fname)
    if isFile:
        df= pd.read_csv(fname,header=None)
        isTransferSuccessful=True if (df.iloc[0,0]=='finished!') else False
    else:
        df=None
        isTransferSuccessful=False
    print('file exists? : {}'.format(isFile))
    print('liq transfer successful? {}'.format(isTransferSuccessful))
    return (isFile,isTransferSuccessful)


def get_changes(changes):
    
    for (_,y) in changes:
            fname=y.split('\\')[-1]
            print(fname)
    isChanged = True if fname == 'msi2pi.csv' else False
    isFinished = False
    if isChanged : 
        df = pd.read_csv(y)
        isFinished = (df.status[0] == 'finished')   
    return (isChanged and isFinished)