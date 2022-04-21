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


dataset = pd.read_excel(r'dry_bean_dataset.xlsx')
dataset_values = dataset.values

X = dataset_values[:, 0:16]
Y = dataset_values[:, 16]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.10, random_state=1)

# Modelos que avaliaremos para escolher entre os dois melhores
modelos = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
           ('LDA', LinearDiscriminantAnalysis()),
           ('KNN', KNeighborsClassifier()),
           ('CART', DecisionTreeClassifier()),
           ('NB', GaussianNB()),
           ('SVM', SVC(gamma='auto'))]


resultados = []
nomes = []

for nome, modelo in modelos:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    results = cross_val_score(modelo, X_train, Y_train,
                              cv=kfold, scoring='accuracy')
    resultados.append(results)
    nomes.append(nome)
    print('%s: %f (%f)' % (nome, results.mean(), results.std()))

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
