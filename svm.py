

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn import svm
import pandas as pd
import numpy as np
import util

class MovieSVM():

    def __init__(self, threshold, delta):
        self.threshold = threshold
        self.e = delta
        self.accuracy = []

# se ajusta a la SVM de acuerdo con el algoritmo descrito

    def fit(self, A, V):
 

        #N = self._buildNegative(A)
        #A = self._applyThreshold(A, self.threshold)

        T = np.copy(A)
        A = np.copy(A)
        for i in range(len(V)):
            for j in range(len(V[i])):
                if V[i, j] >= self.threshold:
                    V[i, j] = 1
                elif V[i, j] != 0:
                    V[i, j] = 0
                else:
                    V[i, j] = -1 
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i,j] >= self.threshold:
                    A[i,j] = 1
                    T[i,j] = 1
                elif A[i,j] != 0:
                    A[i,j] = 0
                    T[i,j] = 0
                else:
                    A[i,j] = np.random.randint(0, 2, size=1)[0]
                    T[i,j] = -1
        
        print(A.shape, np.count_nonzero(A==0), np.count_nonzero(A==1)) 
        print(T.shape, np.count_nonzero(T==0), np.count_nonzero(T==1)) 
        print(V.shape, np.count_nonzero(V==0), np.count_nonzero(V==1)) 
        #return []
        
        totalValidation = np.count_nonzero(V!=-1)
        iteration = 0
            
        svms = [ svm.SVC() for i in range(len(A[0]))  ]
        #print(len(svms))
        self.accuracy = []
        acc_prev, acc_k = 0, 2*self.e
        #print(np.delete(A, 1, axis=1).shape)    
        #print()
        self.train_accuracy = []
        while acc_k - acc_prev > self.e:
            train_correct = 0
            total_train = 0
            start_time = time.time()
            iteration += 1
            for i in range(len(svms)):
                print("train:", str(i) + "/" + str(len(svms)), end='\r')
                X = np.delete(A, i, axis=1)
                Y = A[:,i]
        
                try:
                    svms[i].fit(X, Y)
                except:
                    dummy = 0
        
                A[:, i] = svms[i].predict(X)
                for j in range(len(A[:,i])):
                    if T[j,i] != -1 and A[j,i] != T[j,i]:
                        #A[j,i] = T[j,i]     
                        total_train += 1
                    elif T[j,i] != -1:
                        train_correct += 1   
                        total_train += 1
            self.train_accuracy.append((train_correct*1.0)/total_train)
            # calcular la precisión de iteración

            countMatched = 0
            
            # ir a través de cada columna, predecir esa columna, verificar la coincidencia en V
            for i in range(len(svms)):
                print("Validar:", str(i) + "/" + str(len(svms)),self.train_accuracy, end='\r')
                X = np.delete(V, i, axis=1)
                #print(X.shape)
                Y = V[:,i]
                Yhat = svms[i].predict(X)
                #print(Yhat)####
                countMatched += np.sum(Yhat==Y)
            #print(X,Y,Yhat)
            acc_prev = acc_k
            acc_k = (countMatched*1.0)/totalValidation
            
            self.accuracy.append(acc_k)
            print("\n - Iteracion:", iteration, "\n - precision:", acc_k*100, "\n error:", acc_k - acc_prev, "\n Tiempo (Seg):", time.time() - start_time)
        return self.accuracy, self.train_accuracy

                
        # mientras acc_k - acc_k-1> e:
             # para cada columna, yo
                 # calcula svm - X = [A [:, i-1], A [i + 1 ,:]], Y = A [:, i]
                 # guardar svm en svms [i]
                 # predecir valores
                 # reemplazar A [:, i] = predicciones
            
             # predecir valores para cada punto de prueba
             # acc_k = exactitud
            
    def countCorrect(self,T, A):
        count = np.sum(T==A)
        return count


    #def countTrue(A, N)

    # crea una matriz de la misma forma que A
     # Todas las posiciones de valores en A se reemplazan por -1
     # Todas las 0 posiciones en A se reemplazan con 0, 1
    def _buildNegative(self, A, thresh):
        #N = np.copy(A)
        N = np.random.randint(2, size=A.shape)
        N = N - 5*A
        N[N < 0] = -1
        return N

    def _applyTreshold(A, thresh):
        A[A >= thresh] = 2*thresh
        A[A < thresh] = 0
        return A

if __name__== "__main__":
    NUM_MOVIES = 500
    Data = util.load_data_matrix()
    A = Data[:200, :NUM_MOVIES]
    movieSVM = MovieSVM(3.5, .01)
    V = Data[201:, :NUM_MOVIES]
    v_non_zero = np.count_nonzero(V)
    for i in range(len(A[0]) - 1, 0, -1):
        if np.count_nonzero(A[:,i]) == 0:
            A = np.delete(A, i, axis=1)
            V = np.delete(V, i, axis=1)
    
    accuracy, train_accuracy = movieSVM.fit(A, V)
    #print("\n\n sparsity:", 1 - (v_non_zero*1.0)/(V.shape[0] * V.shape[1] * 1.0))
    print("\n\nValores de precisión terminados:", accuracy)
    print("\n\nprecision de Training ", train_accuracy)

