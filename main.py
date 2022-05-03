import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def apresentaResultados(name, Y_validacao, predicoes):
    print('\n--------------------------- ' +
          name + ' ---------------------')

    # Apresentação da acurácia obtiad dado o modelo.
    print("Acurácia: " + str(accuracy_score(Y_validacao, predicoes)))

    # Apresentação da matriz de confusão
    print("\nMatriz de confusão: ")
    print(confusion_matrix(Y_validacao, predicoes))

    # Apresentação da precisão, recall, f1-score e support
    print("\nClassificação: ")
    print(classification_report(Y_validacao, predicoes, zero_division=1))

    print('--------------------------------------------------------\n')


def calcularClassificacao(modelo, params, X_treino, Y_treino, X_validacao):
    # Aplicação da hiperparametrização a partir do:
    # - Modelo
    # - Parâmetros definidos
    # - X_validação -> Para o caso do dataset completo, será 20% da base de teste
    #               -> Para o caso dos cenários extras, será os 4 cenários de uso de teste

    hyper_param_model = HalvingGridSearchCV(
        modelo, params, random_state=0)

    fitted_model = hyper_param_model.fit(X_treino, Y_treino)

    return fitted_model.predict(X_validacao)


def main():
    # Leitura da base de dados dos 7 tipos diferentes de feijão seco.
    base_dados = pd.read_excel(r'dry_bean_dataset.xlsx')
    valores_base_dados = base_dados.values

    # Separação dos cenários de uso extras, utilizados separadamente
    # para testes.
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

    # Divisão do dataset sem os 4 cenários de uso
    X0 = np.array(cenarios_extras[:, 0:16])
    Y0 = np.array(cenarios_extras[:, 16])

    # Divisão do dataset com 80% para treino e 20% para teste
    X_treino, X_validacao, Y_treino, Y_validacao = train_test_split(
        X, Y, test_size=0.2, random_state=1)

    # Modelos utilizados para treinanamento.
    modelo_LDA = LinearDiscriminantAnalysis()
    modelo_CART = DecisionTreeClassifier()

    # Parâmetrização do LDA para aplicação do GridSearch.
    param_grid_LDA = {"solver": ['svd', 'lsqr', 'eigen']}

    # Calculando classificação do modelo utilizando LDA e dataset de treino com 80%
    # e dataset de teste com 20%
    predicoes_LDA = calcularClassificacao(
        modelo_LDA, param_grid_LDA,
        X_treino, Y_treino, X_validacao
    )

    # Calculando classificação do modelo utilizando LDA e dataset
    # sem os 4 cenários de uso definidos
    predicoes_LDA_extra = calcularClassificacao(
        modelo_LDA, param_grid_LDA,
        X_treino, Y_treino, X0
    )

    # Apresentando os resultados obtidos para os dois casos acima
    # utilizando o algoritmo LDA.
    apresentaResultados('LDA - Dataset completo', Y_validacao, predicoes_LDA)
    apresentaResultados('LDA - 4 cenários de uso', Y0, predicoes_LDA_extra)

    # Parâmetrização do CART para aplicação do GridSearch.
    param_grid_CART = {"criterion": [
        "gini", "entropy"], "splitter": ["best", "random"]}

    # Calculando classificação do modelo utilizando CART e dataset de treino com 80%
    # e dataset de teste com 20%
    predicoes_CART = calcularClassificacao(
        modelo_CART, param_grid_CART,
        X_treino, Y_treino, X_validacao
    )

    # Calculando classificação do modelo utilizando CART e dataset
    # sem os 4 cenários de uso definidos
    predicoes_CART_extra = calcularClassificacao(
        modelo_CART, param_grid_CART,
        X_treino, Y_treino, X0
    )

    # Apresentando os resultados obtidos para os dois casos acima
    # utilizando o algoritmo CART.
    apresentaResultados('CART - Dataset completo',
                        Y_validacao, predicoes_CART
                        )

    apresentaResultados('CART - 4 cenários de uso',
                        Y0, predicoes_CART_extra
                        )


# Chamanda principal do programa.
if __name__ == '__main__':
    main()
