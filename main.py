import numpy as np
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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def apresentaResultados(name, Y_validacao, predicoes):
    print("Acurácia " + name + ": " + str(accuracy_score(Y_validacao, predicoes)))

    print("\nMatriz de confusão " + name + ": ")
    print(confusion_matrix(Y_validacao, predicoes))

    print("\nClassificação " + name + ": ")
    print(classification_report(Y_validacao, predicoes))


def getPredicoes(modelo, params, X_validacao, X_treino, Y_treino):
    search_LDA = HalvingGridSearchCV(
        modelo, params, random_state=0).fit(X_treino, Y_treino)

    return search_LDA.predict(X_validacao)


def main():
    baseDados = pd.read_excel(r'dry_bean_dataset.xlsx')
    valores_baseDados = baseDados.values

    indexInicial = 0
    indexMeio = int(len(valores_baseDados)/2)
    indexFinal = len(valores_baseDados) - 1

    cenariosExtras = np.array([valores_baseDados[indexInicial],
                               valores_baseDados[indexMeio], valores_baseDados[indexFinal]])

    valores_baseDados = np.delete(valores_baseDados, indexFinal, axis=0)
    valores_baseDados = np.delete(valores_baseDados, indexMeio, axis=0)
    valores_baseDados = np.delete(valores_baseDados, indexInicial, axis=0)

    X = valores_baseDados[:, 0:16]
    Y = valores_baseDados[:, 16]

    X0 = cenariosExtras[:, 0:16]
    Y0 = cenariosExtras[:, 16]

    X_treino, X_validacao, Y_treino, Y_validacao = train_test_split(
        X, Y, test_size=0.2, random_state=1)

    # Modelos que avaliaremos para escolher entre os dois melhores
    modelo_LDA = LinearDiscriminantAnalysis()
    modelo_CART = DecisionTreeClassifier()
    modelo_SVC = SVC()

    param_grid_LDA = {"solver": ['svd', 'lsqr', 'eigen']}

    predicoes_LDA = getPredicoes(
        modelo_LDA, param_grid_LDA, X_validacao, X_treino, Y_treino)

    predicoes_LDA0 = getPredicoes(
        modelo_LDA, param_grid_LDA, X_validacao, X_treino, Y0)

    apresentaResultados('LDA 0 ', X_validacao, predicoes_LDA0)

    param_grid_CART = {"criterion": [
        "gini", "entropy"], "splitter": ["best", "random"]}

    predicoes_CART = getPredicoes(
        modelo_CART, param_grid_CART, X_validacao, X_treino, Y_treino)

    # param_grid_SVC = {"kernel": [
    #   "linear", "poly", "sigmoid", "precomputed"], "C": [pow(10, 5), pow(10, 6), pow(10, 7)]}

    # predicoes_SVC = getPredicoes(
    #   modelo_SVC, param_grid_SVC, X_validacao, X_treino, Y_treino)

    #apresentaResultados('LDA', Y_validacao, predicoes_LDA)
    #apresentaResultados('CART', Y_validacao, predicoes_CART)
    #apresentaResultados('SVC', Y_validacao, predicoes_SVC)


main()
