import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

baseDados = pd.read_excel(r'dry_bean_dataset.xlsx')
valores_baseDados = baseDados.values

X = valores_baseDados[:, 0:16]
Y = valores_baseDados[:, 16]

X_treino, X_validacao, Y_treino, Y_validacao = train_test_split(
    X, Y, test_size=0.2, random_state=1)

# Modelos que avaliaremos para escolher entre os dois melhores
modelo_LDA = LinearDiscriminantAnalysis()
modelo_CART = DecisionTreeClassifier()

modelo_LDA.fit(X_treino, Y_treino)
modelo_CART.fit(X_treino, Y_treino)

predicoes_LDA = modelo_LDA.predict(X_validacao)
predicoes_CART = modelo_CART.predict(X_validacao)

print("Acurácia LDA: " + str(accuracy_score(Y_validacao, predicoes_LDA)))
print("Acurácia CART: " + str(accuracy_score(Y_validacao, predicoes_CART)))

print("\nMatriz de confusão LDA:")
print(confusion_matrix(Y_validacao, predicoes_LDA))

print("\nMatriz de confusão CART:")
print(confusion_matrix(Y_validacao, predicoes_CART))

print("\nClassificação LDA:")
print(classification_report(Y_validacao, predicoes_LDA))

print("\nClassificação CART:")
print(classification_report(Y_validacao, predicoes_CART))

# Escolher dataset
# Treinar dois modelos de AM
# Avaliar em relação a quality
# Escolher uma tecnica de hyper-parametrizao -> grid search ou random search
# Análise de desempenho em relação a outros trrabalhos com o msm dataset

# Treinou o modelo? Beleza!! Fazer 3 cenários ou mais
# - na resolução de tarefas como classificação, regressoa,. clusterização, predição etc

# ------------------------- SLIDES -----------------------------------------
# Criar slides para apresentação do trabalho. Os slides devem conter no mínimo: Slide
# inicial (identificação da UEM, do curso, disciplina, professor, título do trabalho, nomes e
# RAs dos alunos), alguma fundamentação teórica, contextualização do dataset utilizado,
# hyper-parametrização e avaliação do modelo, 3 cenários de uso do modelo, referências
# bibliográficas.
