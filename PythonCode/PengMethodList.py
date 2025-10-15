# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:47:15 2020

@author: xp
"""
import numpy as np


import shelve


import pickle


        
def generateDecisionSet(Time, LevelNum):
        
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
    


# not usable
def saveGlobalVariables(AllVariables,filename):
    
    
    VariableList = []
        
    my_shelf = shelve.open(filename,'n') # 'n' for new
    
    # with shelve.open(filename,'n') as my_shelf:
    
    #AllVariables = globals()
    for key in AllVariables:
        
        try:
            
            my_shelf[key] = AllVariables[key]                       
            
            print(str(key))
            
            VariableList = VariableList + [key]

        except:
            print('Save Error ERROR:'.format(key))
            #pass;
        
    my_shelf.close()
        
    return VariableList



# not usable
def loadGlobalVariables(filename):
        
    my_shelf = shelve.open(filename)
        
    for key in my_shelf:
        try:
            globals()[key]=my_shelf[key]
        except:
            #print('Save Error ERROR:'.format(key))
            pass;
        
    my_shelf.close()


    


def saveDictVariables(Dict,filename):

    with open(filename,"wb") as f:
        pickle.dump(Dict,f)
    
    
    
def loadDictVariables(filename):

    with open(filename,'rb') as f:
        Dict2  = pickle.loads(f.read())
        
    return Dict2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                