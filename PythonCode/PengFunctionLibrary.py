# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:47:15 2020

@author: xp
"""
import numpy as np

import pickle

#from BNet_model import BNet



from spyder_kernels.utils.nsview import (is_supported,get_supported_types)
import shelve



class PengFunction:
    def __init__(self):
        self.a = 0
        
        
    def generateDecisionSet(self, Time, LevelNum):
        
        DecisionSet = np.zeros((LevelNum**Time, Time))                  
        
        TotalCount = 0
        for i in range(2**Time):     
            TempCount = TotalCount
            for j in reversed(range(Time)):
                Remainder = TempCount%LevelNum
                DecisionSet[i,j] = Remainder
                
                TempCount = int(TempCount/LevelNum)
                
            TotalCount = TotalCount + 1
                
        return DecisionSet
    
    
    def saveGlobalVariables(self, filename):

        VariableList = []
        
        my_shelf = shelve.open(filename,'n') # 'n' for new
        # with shelve.open(filename,'n') as my_shelf:
        for key in dir():
            try:
                my_shelf[key] = dir()[key]
                
                VariableList = VariableList + key

            except:
                print('Save Error ERROR:'.format(key))
                pass;
        
        my_shelf.close()
        
        return VariableList

    
    def loadGlobalVariables(self, filename):
        
        my_shelf = shelve.open(filename)
        
        for key in my_shelf:
            try:
                globals()[key]=my_shelf[key]
            except:
                #print('Save Error ERROR:'.format(key))
                pass;
        
        my_shelf.close()


if __name__ == "__main__":
    # maze game
    P=PengFunction()
    DSet = P.generateDecisionSet(3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                