import numpy as np
import pandas as pd

STUDENT_NAME = 'UTTEJ REDDY PAKANATI, SAI ANURAG NEELISETTY, KRISHNA KANTH MUTTA' 

STUDENT_ID = '20875894, 20911061, 20919166'


def Fun_sigmoid(z): #sigmoid functon
    return 1 / (1 + np.exp(-z))

def Forward_propagation(x_data, weights, layers):
    activations, layer_input = [x_data], x_data
    for j in range(layers):
        activation = Fun_sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Adding the bias
    
    return activations

def Max_function(out):
    x, inx = out[0], 0
    for i in range(1, len(out)):
        if(out[i] > x):
            x, inx = out[i], i
    return inx

def Prediction(x_data, weights):
    layers_whole = len(weights)
    x_data = np.append(1, x_data) # Adding bias
    activation_op = Forward_propagation(x_data, weights, layers_whole)    
    out_final = activation_op[-1].A1
    inx = Max_function(out_final)
    # Initialize prediction vector to zeros
    y_pred = [0 for i in range(len(out_final))]
    y_pred[inx] = 1 
    return y_pred

def Accuracy(X, weights):
    result=[]
    for i in range(len(X)):
        x = X[i]
        guess = Prediction(x, weights)
        result.append(guess)
    result1=np.array(result)
    return result1


def test_mlp(data_file):
    import pickle
    with open('A1_G55.pkl','rb') as pickle_file:
        weights= pickle.load(pickle_file)
    df_test = pd.read_csv(data_file,header=None)
    arr_test = df_test.to_numpy()
    y_pred=Accuracy(arr_test, weights)
    return y_pred
