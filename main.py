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


def getPredicoes(modelo, params, X_treino, Y_treino, X_validacao):
    search_LDA = HalvingGridSearchCV(
        modelo, params, random_state=0).fit(X_treino, Y_treino)

    return search_LDA.predict(X_validacao)


def main():
    base_dados = pd.read_excel(r'dry_bean_dataset.xlsx')
    valores_base_dados = base_dados.values

    index_inicial = 0
    index_meio_1 = int(len(valores_base_dados)/3)
    index_meio_2 = int((len(valores_base_dados)/3) * 2)
    index_final = len(valores_base_dados) - 1

    cenarios_extras = np.array([valores_base_dados[index_inicial],
                               valores_base_dados[index_meio_1],
                               valores_base_dados[index_meio_2],
                               valores_base_dados[index_final]])

    valores_base_dados = np.delete(valores_base_dados, index_final, axis=0)
    valores_base_dados = np.delete(valores_base_dados, index_meio_2, axis=0)
    valores_base_dados = np.delete(valores_base_dados, index_meio_1, axis=0)
    valores_base_dados = np.delete(valores_base_dados, index_inicial, axis=0)

    X = valores_base_dados[:, 0:16]
    Y = valores_base_dados[:, 16]

    X0 = np.array(cenarios_extras[:, 0:16])
    Y0 = np.array(cenarios_extras[:, 16])

    X_treino, X_validacao, Y_treino, Y_validacao = train_test_split(
        X, Y, test_size=0.2, random_state=1)

    # Modelos que avaliaremos para escolher entre os dois melhores
    modelo_LDA = LinearDiscriminantAnalysis()
    modelo_CART = DecisionTreeClassifier()
    modelo_SVC = SVC()

    param_grid_LDA = {"solver": ['svd', 'lsqr', 'eigen']}

    predicoes_LDA = getPredicoes(
        modelo_LDA, param_grid_LDA, X_treino, Y_treino, X_validacao)

    predicoes_LDA_extra = getPredicoes(
        modelo_LDA, param_grid_LDA, X_treino, Y_treino, X0)

    apresentaResultados('LDA 0', Y0, predicoes_LDA_extra)

    # param_grid_CART = {"criterion": [
    #     "gini", "entropy"], "splitter": ["best", "random"]}
    #
    # predicoes_CART = getPredicoes(
    #     modelo_CART, param_grid_CART, X_validacao, X_treino, Y_treino)

    # param_grid_SVC = {"kernel": [
    #   "linear", "poly", "sigmoid", "precomputed"], "C": [pow(10, 5), pow(10, 6), pow(10, 7)]}

    # predicoes_SVC = getPredicoes(
    #   modelo_SVC, param_grid_SVC, X_validacao, X_treino, Y_treino)

    #apresentaResultados('LDA', Y_validacao, predicoes_LDA)
    #apresentaResultados('CART', Y_validacao, predicoes_CART)
    #apresentaResultados('SVC', Y_validacao, predicoes_SVC)


main()
