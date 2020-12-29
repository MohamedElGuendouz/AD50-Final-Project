#%% LINEAR REGRESSION
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Data/data_merged.csv')
x=df.iloc[:,1:26]
y=df.loc[:,['nb_ope']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

mean_s_lin = mean_squared_error(y_test, predictions)
print("Mean squared error Linear Regression = ",mean_s_lin)
mean_a_lin = mean_absolute_error(y_test, predictions)
print("Mean absolute error Linear Regression = ",mean_a_lin)

# %% RECCURENT NEURAL NETWORK FUNCTIONS AND IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

n_obs= 100
rnn_structure=(50,50,50,50)

def build_series(inputs,n_obs):
    A=[]
    for v in range(inputs.shape[1]):
        X = []
        for i in range(n_obs, len(inputs)):
            X.append(inputs[i-n_obs:i, v])
        X=np.array(X)
        A.append(X)
    A=np.swapaxes(np.swapaxes(np.array(A),0,1),1,2)
    return A

def build_regressor(layers,input_shape,rnn_type='LSTM',dropout=0.2,optimizer='adam',loss_function='mean_squared_error'):
    regressor = Sequential()
    n_layers=len(layers)

    if rnn_type=='LSTM':
        regressor.add(LSTM(units = layers[0], return_sequences = True, input_shape = input_shape))
    else:
        regressor.add(GRU(units = layers[0], return_sequences = True, input_shape = input_shape))
    regressor.add(Dropout(dropout))

    if n_layers>1:
        for i in range(1,n_layers-1):
            if rnn_type=='LSTM':
                regressor.add(LSTM(units = layers[i], return_sequences = True))
            else:
                regressor.add(GRU(units = layers[i], return_sequences = True))
            regressor.add(Dropout(dropout))

    if rnn_type=='LSTM':
        regressor.add(LSTM(units = layers[-1]))
    else:
        regressor.add(GRU(units = layers[-1]))

    regressor.add(Dropout(dropout))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = optimizer, loss = loss_function)
    return regressor

def plot_predictions(real,predicted):
    plt.plot(real[:,25], color = 'red', label = 'Real Interventions')
    plt.plot(predicted, color = 'blue', label = 'Predicted Interventions')
    plt.title('Interventions Prediction')
    plt.xlabel('Semaines')
    plt.ylabel('Interventions')
    plt.legend()
    plt.show()

print("Done")
# %% RNN GRU

df = pd.read_csv('Data/data_merged.csv')

sc = StandardScaler()
training_set=df.iloc[:,1:]
training_set_scaled=sc.fit_transform(training_set)

test_set=df.iloc[:,1:]
test_set_scaled=sc.fit_transform(test_set)

x_train= build_series(training_set_scaled,n_obs)
y_train=np.array(training_set_scaled[n_obs:,25])

regressor=build_regressor(rnn_structure,(x_train.shape[1], x_train.shape[2]),rnn_type='GRU',dropout=0.2)
regressor.fit(x_train, y_train, epochs = 50, batch_size = 40)

dataset = np.concatenate((training_set_scaled,test_set_scaled),axis=0)
inputs = dataset[len(training_set_scaled)-n_obs:]
x_test=build_series(inputs,n_obs)

predictions = regressor.predict(x_test)

plot_predictions(test_set_scaled, predictions)

mean_s_gru = mean_squared_error(test_set_scaled[:,25], predictions)
print("Mean squared error GRU = ", mean_s_gru)
mean_a_gru = mean_absolute_error(test_set_scaled[:,25], predictions)
print("Mean absolute error GRU = ", mean_a_gru)

# %% RNN LSTM

df = pd.read_csv('Data/data_merged.csv')

sc = StandardScaler()
training_set=df.iloc[:,1:]
training_set_scaled=sc.fit_transform(training_set)

test_set=df.iloc[:,1:]
test_set_scaled=sc.fit_transform(test_set)

x_train= build_series(training_set_scaled,n_obs)
y_train=np.array(training_set_scaled[n_obs:,25])

regressor=build_regressor(rnn_structure,(x_train.shape[1], x_train.shape[2]),rnn_type='LSTM',dropout=0.2)
regressor.fit(x_train, y_train, epochs = 50, batch_size = 40)

dataset = np.concatenate((training_set_scaled,test_set_scaled),axis=0)
inputs = dataset[len(training_set_scaled)-n_obs:]
x_test=build_series(inputs,n_obs)

predictions = regressor.predict(x_test)

plot_predictions(test_set_scaled, predictions)

mean_s_lstm = mean_squared_error(test_set_scaled[:,25], predictions)
print("Mean squared error LSTM = ",mean_s_lstm)
mean_a_lstm = mean_absolute_error(test_set_scaled[:,25], predictions)
print("Mean absolute error LSTM = ",mean_a_lstm)
# %%
