import os 
import sys
import dill

import pandas as pd
import numpy as np

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try: 
        report = {}
        
        for model_name,model in models.items():
            para = params[model_name]
            
            grid = GridSearchCV(estimator=model,param_grid=para,cv=3,verbose=1)
            grid.fit(X_train,y_train)
            
            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)    
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)  
            
            report[model_name] = test_model_score
            
        return report        
    except Exception as e:
        raise CustomException(e,sys)
    