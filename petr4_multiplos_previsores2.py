from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')
base=base.dropna()
base_treinamento=base.iloc[:,1:7].values
normalizador=MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada=normalizador.fit_transform(base_treinamento)

#até aqui é igual para todas os tipos de IAs

qtd_dias=90
qtd_total_registros=1242
qtd_colunas=6
previsores=[]
preco_real=[]
 
for i in range(qtd_dias,qtd_total_registros):
    previsores.append(base_treinamento_normalizada[i-qtd_dias:i,0:qtd_colunas])
    preco_real.append(base_treinamento_normalizada[i,0])

previsores, preco_real=np.array(previsores),np.array(preco_real);
previsores = np.reshape(previsores, (previsores.shape[0],previsores.shape[1],1))

regressor=Sequential()
#criação da rede neural LSTM com 4 camadas com 100 células de memória ou 50 células
#input shape enviado só para a primeira camada
regressor.add(LSTM(units = 100, return_sequences=True, input_shape=(previsores.shape[1],1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='sigmoid'))

regressor.compile(optimizer='adam',loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
