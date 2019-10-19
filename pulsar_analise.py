import pandas as pd
base = pd.read_csv('pulsar_stars.csv')


base.loc[pd.isnull(base['Meanoftheintegratedprofile'])]
base.loc[pd.isnull(base['Standarddeviationoftheintegratedprofile'])]
base.loc[pd.isnull(base['Excess kurtosis of the integrated profile'])]
base.loc[pd.isnull(base['Skewnessoftheintegratedprofile'])]
base.loc[pd.isnull(base['MeanoftheDM-SNRcurve'])]
base.loc[pd.isnull(base['Standard deviation of the DM-SNR curve'])]
base.loc[pd.isnull(base['Excess kurtosis of the DM-SNR curve'])]
base.loc[pd.isnull(base['Skewness of the DM-SNR curve'])]

previsores = base.iloc[:, 0:8]
classe = base.iloc[:, 8]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.20, random_state=6)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


#Arvore de Decisão
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)



#Random Forest
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


#KNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Regressão Logistica
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(random_state = 1)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#SVC
from sklearn.svm import SVC
classificador = SVC(kernel = 'linear', random_state = 1)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Rede Neural
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.0000010,
                              solver = 'adam',
                              hidden_layer_sizes=(100),
                              activation='tgnh')
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Resultado
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste, previsoes)







