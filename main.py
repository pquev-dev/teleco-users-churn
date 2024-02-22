import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import joblib

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,jaccard_score,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier,plot_importance

from imblearn.over_sampling import SMOTE


#TODO: Aplicar un reajuste de valores 0 y 1 

class GenerateModel():
    
    def __init__(self,x,y):
        self.models = {
            'lgr' : LogisticRegression,
            'tree' : DecisionTreeClassifier,
            'rf' : RandomForestClassifier,
            'xgboost' : XGBClassifier,
            'gbc' : GradientBoostingClassifier,
            'knn' : KNeighborsClassifier
            
        }
        self.params = {
            'lgr' : {
                'max_iter' : 1000,
                'C' : 0.1,
                'class_weight' : {0 :1 , 1: 2}
       
            },
            'tree' : {
                'max_depth' : 5
            },
            'rf' : {
                'n_estimators' : 500,
                'max_depth' : 5,
                'random_state' : 42
            },
            'xgboost' : {
                #TODO: implementar hiperparametros
            },
            'gbc' : {},
            'knn' : {'n_neighbors' : 5}
        }
        self.x = x 
        self.y = y
    
    def plot_confussion_matrix(self,y_real,y_predict,name_model):
        name_path = f"./results_metrics/confussion_matrix-{name_model}.jpg"
        sns.heatmap(confusion_matrix(y_real,y_predict),annot=True,fmt='d')
        plt.title(f"confussion matrix - {name_model}")
        plt.savefig(name_path)
        plt.close()
    
    def training_model(self,):
        
        X_train,X_test,y_train,y_test = train_test_split(self.x,self.y,random_state=42,test_size=0.2)
        
        best_score = 0 
        best_model = None 
        model_name = ""
        
        resultados = []
        
        for n,m in self.models.items():
            print(f"entrenando modelo : {n}")
            
            model = m(**self.params[n])
            scores = cross_val_score(model,X_train,y_train,cv=3)
            print(f"precision media : {scores.mean()}")
            
            model = model.fit(X_train,y_train)
            
            predict = model.predict(X_test)
            
            f1_s = f1_score(y_test,predict)
            jaccard_s = jaccard_score(y_test,predict)
            accuracy_s = accuracy_score(y_test,predict)
            promedio_diff = (f1_s + jaccard_s + accuracy_s)/3

            resultados.append({
                'model_name' : n,
                'accuracy_score' : accuracy_s,
                'f1_score' : f1_s,
                'jaccard_score' : jaccard_s,
                'diff' :  promedio_diff,
                'scores_cross_val' : scores.mean()
            })
            
            ##plotear matriz de confusion
            self.plot_confussion_matrix(y_test,predict,n)

            if n == 'xgboost':
                plot_importance(model)
                plt.savefig('./results_metrics/plot_importance_xgboost.jpg')
            
            ##obtener promedio de ambas metricas 
            
            if best_score < promedio_diff:
                best_score = promedio_diff
                best_model = model
                model_name = n 
                
        ##transformar arreglo a dataframe 
        df_resultados = pd.DataFrame(resultados)        
        df_resultados.to_csv("./results_metrics/metricas_modelos.csv")
        
        ##exportar modelo 
        joblib.dump(best_model,f"./out/churn_clients_model-{model_name}.pkl")
        
    
class Main():
    
    def __init__(self,data):
        self.data = data
        
    def run(self,features,target):
                
        print(self.data[target].unique)
        
        self.data[target] = self.data[target].map({
            'Yes' : 1,
            'No' : 0
        })
        
        X = self.data[features]
        y = self.data[target]
        
        X['TotalCharges'] = X['TotalCharges'].astype('float')
            
        print(X.isnull().sum())
        
        ##trasformar valores categoricos a numericos 
        X_t = pd.get_dummies(X)
        
        ##balanceo de clases
        smt = SMOTE(random_state=123)
        X,y = smt.fit_resample(X_t,y)
       
        X = np.c_[X]
        ##generar modelo
        model = GenerateModel(X,y)
        model.training_model()
        
if __name__ == "__main__":
    
    #config 
    df = pd.read_csv('../../datasets/Telco-Customer-Churn.csv')
    
    #Eliminar caracteres
    eliminar_caracteres = [' ', '']
    df = df.loc[~df["TotalCharges"].isin(eliminar_caracteres),]
    
    features = list(df.drop(['Churn','customerID'],axis=1).columns)
    target = "Churn"
    
    main = Main(df)
    main.run(features,target)
    
        
        
        