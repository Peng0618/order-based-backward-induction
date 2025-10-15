# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:04:13 2019

@author: xupeng
"""

import matlab
import matlab.engine
import numpy as np

engine = matlab.engine.start_matlab()
#  engine = matlab.engine.start_matlab("-desktop")  ！！

#engine.sqrt(2.)



#State = engine.initialBNT(matlab.double([1]))
#print(State)



#engine.cd(r'C:\Users\xupen\Google Drive\PythonScripts\BayesNet\MatlabCode', nargout=0)
#engine.loadBNT(nargout = 0)
#engine.cd(r'C:\Users\xupen\Google Drive\PythonScripts\BayesNet\MatlabCode', nargout=0)


engine.cd(r'D:\GoogleDrive\PythonScripts\BayesNet\MatlabCode', nargout=0)
engine.loadBNT(nargout = 0)
engine.cd(r'D:\GoogleDrive\PythonScripts\BayesNet\MatlabCode', nargout=0)



#N = 12

Evidence = [0]*12
#Evidence[2] = 1
InitPriorVector = engine.inferenceBN(Evidence)
print(InitPriorVector)

PyPriorVector = np.asarray(InitPriorVector).flatten() 

engine.exit()