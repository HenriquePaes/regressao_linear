from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

# gerando a massa de dados
x, y = make_regression(n_samples=200, n_features=1, noise=30)

# plotando os dados
#plt.scatter(x, y)
#plt.show()

# Criação do modelo
modelo = LinearRegression()

modelo.fit(x, y)
print(modelo.intercept_) # coeficiente linear
print(modelo.coef_) # coeficiente angular

# mostrando o resultado
plt.scatter(x, y)
xreg = np.arange(-3, 4, 1)
plt.plot(xreg, (modelo.coef_ * xreg) + (modelo.intercept_), color='red')
plt.title('Dados completo')
plt.show()

# Realizando a divisão dos nossos dados em dados de treino e dados de teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30)

# Criando o modelo
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste) # coeficiente de determinação R2
print('Resultado: {:.2f}%'.format(resultado * 100))

# mostrando o grafico dos dados de treino
plt.scatter(x_treino, y_treino)
xreg = np.arange(-3, 4, 1)
plt.plot(xreg, (modelo.coef_ * xreg) + (modelo.intercept_), color='red')
plt.title('Dados de Treino')
plt.show()
