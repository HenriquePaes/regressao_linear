import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importando os dados
pd.set_option('display.max_columns', 21) # garante q o pandas mostre todas as colunas
file = pd.read_csv('./kc_house_data.csv')

# mostrando os 10 primeiros dados
# print(file.head(10))

# excluindo features irrelevantes
file.drop('id', axis=1, inplace=True)
file.drop('date', axis=1, inplace=True)
file.drop('zipcode', axis=1, inplace=True)
file.drop('lat', axis=1, inplace=True)
file.drop('long', axis=1, inplace=True)

file.head()

# definindo variaveis preditoras e variavel target
y = file['price']
x = file.drop('price', axis=1)

# separando os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=10)

# criando o modelo
modelo = LinearRegression()
modelo.fit(x_treino, y_treino)

# calculando o coeficiente R2
resultado = modelo.score(x_teste, y_teste)
print('Resultado: {:.2f}'.format(resultado * 100))
