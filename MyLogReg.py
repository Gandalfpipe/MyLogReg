import pandas as pd
import numpy as np
import random                   
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg():
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate:float = 0.1,
                 metric: str = None,
                 reg: str = None, 
                 l1_coef: float = 0, 
                 l2_coef: float = 0,
                 sgd_sample = None,
                 random_state = 42) -> None:
        self.n_iter  = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.eps = 1e-15
        self.metric = metric
        self.mv = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        pass
    
    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-z))
    
    def log_loss(self,y_vector: np.ndarray, y_predict: np.ndarray,weighted_vector) -> float:
        if self.reg == None:
            return (-1 * np.mean(y_vector * np.log(self.sigmoid(y_predict) + self.eps) + (1 - y_vector) * np.log(1 + self.eps - self.sigmoid(y_predict))))
        elif self.reg == 'l1':
            return -1 * np.mean(y_vector * np.log(self.sigmoid(y_predict) + self.eps) + (1 - y_vector) * np.log(1 + self.eps - self.sigmoid(y_predict))) + self.l1_coef * np.sum(np.absolute(weighted_vector))
        elif self.reg == 'l2':
            return (-1 * np.mean(y_vector * np.log(self.sigmoid(y_predict) + self.eps) + (1 - y_vector) * np.log(1 + self.eps - self.sigmoid(y_predict)))) + self.l2_coef * np.sum(np.square(weighted_vector))
        elif self.reg == 'elasticnet':
            return (-1 * np.mean(y_vector * np.log(self.sigmoid(y_predict) + self.eps) + (1 - y_vector) * np.log(1 + self.eps - self.sigmoid(y_predict)))) + self.l2_coef * np.sum(np.square(weighted_vector)) + self.l1_coef * np.sum(np.absolute(weighted_vector))
        
        
    def grad(self,num_rows: int, y_vector: np.ndarray, y_predict: np.ndarray, X_matrix: np.ndarray, weighted_vector) -> np.ndarray:
        gradient = 0
        if self.reg == None:
            gradient = ((1/(num_rows)) * np.dot(self.sigmoid(y_predict) - y_vector, X_matrix))
        
        if self.reg == 'l1':
            gradient =  ((1/(num_rows)) * np.dot(self.sigmoid(y_predict) - y_vector, X_matrix) + self.l1_coef * np.sign(weighted_vector))
        
        if self.reg == 'l2':
            gradient = ((1/(num_rows)) * np.dot(self.sigmoid(y_predict) - y_vector, X_matrix) + self.l2_coef * 2 * weighted_vector)
        
        if self.reg == 'elasticnet':
            gradient = ((1/(num_rows)) * np.dot(self.sigmoid(y_predict) - y_vector, X_matrix) + self.l2_coef * 2 * weighted_vector + self.l1_coef * np.sign(weighted_vector))
        
        return gradient
        
    def get_coef(self) -> np.ndarray:
        return self.weights[1:]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_matrix = X.to_numpy()
        unit_column = np.ones(X.shape[0])
        X_matrix = np.insert(X_matrix,0,unit_column,axis=1)
        
        return self.sigmoid(np.dot(X_matrix,self.weights))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_matrix = X.to_numpy()
        unit_column = np.ones(X.shape[0])
        X_matrix = np.insert(X_matrix,0,unit_column,axis=1)
        y = self.sigmoid(np.dot(X_matrix, self.weights))
        y = np.where(y > 0.5, 1 , 0)
        
        return y
    
    def errors(self,y_vector: np.ndarray,y_predict: np.ndarray) -> tuple:
        y_predict = self.sigmoid(y_predict)
        y_predict = np.where(y_predict > 0.5,1,0)
        return np.count_nonzero(np.logical_and(y_predict == y_vector, y_predict == 1) == True), np.count_nonzero(np.logical_and(y_predict != y_vector, y_predict < y_vector) == True), np.count_nonzero(np.logical_and(y_predict != y_vector, y_predict > y_vector) == True), np.count_nonzero(np.logical_and(y_predict == y_vector, y_predict == 0) == True)
    
    def accuracy(self,y_vector: np.ndarray, y_predict: np.ndarray) -> float:
        return (self.errors(y_vector,y_predict)[0] + self.errors(y_vector,y_predict)[3])/(self.errors(y_vector,y_predict)[0] + self.errors(y_vector,y_predict)[1] + self.errors(y_vector,y_predict)[2] + self.errors(y_vector,y_predict)[3])

    def precision(self,y_vector: np.ndarray,y_predict: np.ndarray) -> float:
        return self.errors(y_vector,y_predict)[0]/(self.errors(y_vector,y_predict)[0] + self.errors(y_vector,y_predict)[2])
    
    def recall(self,y_vector: np.ndarray, y_predict: np.ndarray) -> float:    
        return self.errors(y_vector,y_predict)[0]/(self.errors(y_vector,y_predict)[0] + self.errors(y_vector,y_predict)[1])
        
    def f_measure_metric(self,y_vector: np.ndarray, y_predict: np.ndarray) -> float:
        return (2 * self.precision(y_vector,y_predict) * self.recall(y_vector,y_predict))/(self.precision(y_vector,y_predict) + self.recall(y_vector,y_predict))
    
    def roc_auc(self,y_vector: np.ndarray, y_predict: np.ndarray) -> float:
        y_predict = self.sigmoid(y_predict)
        y_predict = np.round(y_predict, decimals=10)
        table = np.column_stack((y_predict,y_vector))
        table = table[table[:, 0].argsort()[::-1]]
        N = 0
        P = 0
        s = 0
        subsum = 0
        i = 0
        while i < table.shape[0]:
            j=0
            subsum = 0
            if table[i,1] == 1:
                P += 1
            elif table[i,1] == 0:
                N += 1 
                while j < table.shape[0]:
                    if (table[j,0] == table[i,0]) and (table[j,1] == 1):
                        subsum += 0.5
                    elif (table[j,1] == 1) and (j < i):
                        subsum += 1

                    j += 1
    
            i += 1
            s += subsum
        
        return s/(N * P)
    
    def count_metric(self,y_vector: np.ndarray, y_predict: np.ndarray) -> float:
        if self.metric == None:
            pass
        elif self.metric == 'accuracy':
            return self.accuracy(y_vector,y_predict)
        elif self.metric == 'precision':
            return self.precision(y_vector,y_predict)
        elif self.metric == 'recall':
            return self.recall(y_vector,y_predict)
        elif self.metric == 'f1':
            return self.f_measure_metric(y_vector,y_predict)
        elif self.metric == 'roc_auc':
            return self.roc_auc(y_vector,y_predict)
    
    def get_best_score(self) -> float:
        return self.mv
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False) -> None:
        num_rows = X.shape[0]
        num_columns = X.shape[1]
        unit_column = np.ones(num_rows)
        X_matrix = X.to_numpy()
        y_vector = y.to_numpy()
        X_matrix = np.insert(X_matrix,0,unit_column,axis=1)
        weighted_vector = np.ones(num_columns + 1)
        random.seed(self.random_state)
        i = 1
        
        if self.sgd_sample == None:
            while i <= self.n_iter:
                y_predict = np.dot(X_matrix, weighted_vector)
                loss = self.log_loss(y_vector,y_predict,weighted_vector)
                grad = self.grad(num_rows, y_vector, y_predict, X_matrix,weighted_vector)
                if isinstance(self.learning_rate, float) == True:
                    weighted_vector = np.add(weighted_vector,(-1 * self.learning_rate) * grad)
                else:
                    rate = self.learning_rate(i)
                    weighted_vector = np.add(weighted_vector,(-1 * rate) * grad)
                if verbose and i % verbose == 0:
                    metric = self.count_metric(y_vector,y_predict)
                    print(f'{i} | loss: {loss}| {self.metric} : {metric}')
            
                i += 1
       
        if isinstance(self.sgd_sample, int) == True:
            while i <= self.n_iter:
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                X_mini_matrix = np.take(X_matrix,sample_rows_idx,axis=0)    
                y_mini_vector = np.take(y_vector,sample_rows_idx)
                y_predict = np.dot(X_matrix, weighted_vector)
                y_mini_predict = np.take(y_predict,sample_rows_idx)
                loss = self.log_loss(y_vector,y_predict,weighted_vector)
                grad = self.grad(num_rows=self.sgd_sample,y_vector=y_mini_vector,y_predict=y_mini_predict, X_matrix=X_mini_matrix, weighted_vector=weighted_vector)                        
                if isinstance(self.learning_rate, float) == True:
                    weighted_vector = np.add(weighted_vector,(-1 * self.learning_rate) * grad)
                else:
                    rate = self.learning_rate(i)
                    weighted_vector = np.add(weighted_vector,(-1 * rate) * grad)
                if verbose and i % verbose == 0:
                    metric = self.count_metric(y_vector,y_predict)
                    print(f'{i} | loss: {loss}| {self.metric} : {metric}')
            
                i += 1
                
        if isinstance(self.sgd_sample, float) == True:
            while i <= self.n_iter:
                part = round(self.sgd_sample * num_rows)
                sample_rows_idx = random.sample(range(X.shape[0]), part)
                X_mini_matrix = np.take(X_matrix,sample_rows_idx,axis=0)    
                y_mini_vector = np.take(y_vector,sample_rows_idx)
                y_predict = np.dot(X_matrix, weighted_vector)
                y_mini_predict = np.take(y_predict,sample_rows_idx)
                loss = self.log_loss(y_vector,y_predict,weighted_vector)
                grad = self.grad(num_rows=part,y_vector=y_mini_vector,y_predict=y_mini_predict, X_matrix=X_mini_matrix, weighted_vector=weighted_vector)                        
                if isinstance(self.learning_rate, float) == True:
                    weighted_vector = np.add(weighted_vector,(-1 * self.learning_rate) * grad)
                else:
                    rate = self.learning_rate(i)
                    weighted_vector = np.add(weighted_vector,(-1 * rate) * grad)
                if verbose and i % verbose == 0:
                    metric = self.count_metric(y_vector,y_predict)
                    print(f'{i} | loss: {loss}| {self.metric} : {metric}')
            
                i += 1
                
        
        self.weights = weighted_vector
        y_predict = np.dot(X_matrix, weighted_vector)
        self.mv = self.count_metric(y_vector,y_predict)
        pass
