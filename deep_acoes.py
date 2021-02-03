 # -*- coding: utf-8 -*-
"""
Para a previsão eu preciso de no mínimo 4 dados

Aprendizagem supervisionada
"""
#pip install keras
#pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#carregamento e entendimento da base de dados

#ler valores do arquivo petr4_treinamento
base = pd.read_csv('petr4_treinamento.csv')
#apagar valores nulos
base=base.dropna()
#selecionar valores que queremos fazer as previsões
base_treinamento=base.iloc[:,1:2].values
#normalizar os valores para colocar em uma escala de 0 a 1 e conseguir usar a lógica das tabelas verdades
normalizador=MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada=normalizador.fit_transform(base_treinamento)

#Estrutura da base para previsão temporal: 

#1) definir o intervalo de tempo
qtd_dias=90
qtd_total_registros=1242
#valores anteriores 
previsores=[]
 #valores atuais
preco_real=[]
 
for i in range(qtd_dias,qtd_total_registros):
    #adicionar valores(0->qtd_total_registros-90) do banco de dados no vetor de previsores
    previsores.append(base_treinamento_normalizada[i-qtd_dias:i,0])
    #adicionar valores(90->qtd_total_registros) do banco de dados no vetor de preço real
    preco_real.append(base_treinamento_normalizada[i,0])


#transformar os dados em numpy
previsores, preco_real=np.array(previsores),np.array(preco_real);
#tabela de previsores: cada linha tem 90 colunas- ultimos 90 dias
#como estamos usando 1242 valores cada linha é uma atualizacao dos dados(FIFO- first in first out)

#previsores t(0)->t(89) preco_real:t(90)

#colocando o formato do array para o que o pro o keras deseja
#ultimo atributo(1) é a quantidade de atributos que gostariamos de trabalhar
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

regressor.add(Dense(units=1, activation='linear'))

regressor.compile(optimizer='rmsprop',loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs=100, batch_size=32)


#efetivamente fazer as previsões
base_teste=pd.read_csv('petr4_teste.csv')
#pega na matriz só a coluna que realmente queremos
preco_real_teste=base_teste.iloc[:,1:2].values 
#unir base de treino com a base de teste
base_completa=pd.concat((base['Open'],base_teste['Open']),axis=0)
entradas =base_completa[ -len(base_teste)-90:].values
entradas=entradas.reshape(-1,1)
entradas=normalizador.transform(entradas)

X_teste=[]
for i in range(90,112):
    X_teste.append(entradas[i-90:i,0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste,(X_teste.shape[0],X_teste.shape[1],1))
previsoes=regressor.predict(X_teste)
previsoes=normalizador.inverse_transform(previsoes)
previsoes.mean()
preco_real_teste.mean()