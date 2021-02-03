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

regressor=Sequential()
regressor.add(LSTM(units = 100, return_sequences=True, input_shape=(previsores.shape[1],6)))
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

#para treinamento dps que o treinamento para de melhorar os resultados
es= EarlyStopping(monitor='loss',min_delta=1e-10,patience=10, verbose=1)
#reduz a tx de aprendizagem quando 
rlr=ReduceLROnPlateau(monitor='loss',factor=0.2,patience=5, verbose=1)
mcp=ModelCheckpoint(filepath='pesos.h5',monitor='loss',save_best_only=True,verbose=1)
regressor.fit(previsores,preco_real,epochs=100,batch_size=32,callbacks=[es,rlr,mcp])
base_teste=pd.read_csv('petr4_teste.csv')
preco_real_teste=base_teste.iloc[:,1:2].values 
frames=[base,base_teste]


# base_completa=pd.concat((base['Open'],base_teste['Open']),axis=0)