from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np

# gerando uma massa de dados:
x, y = make_regression(n_samples=200, n_features=1, noise=30)

# mostrando no gráfico
#plt.scatter(x, y)
#plt.show()

# criação do modelo
modelo = LinearRegression()

modelo.fit(x,y)
print(modelo.intercept_) # referece ao coeficiente linear
print(modelo.coef_) # coeficiente angular

# mostrando o resultado
plt.scatter(x, y)
xreg = np.arange(-3, 4, 1)
plt.plot(xreg, (modelo.coef_ * xreg) + (modelo.intercept_), color='red') # gráfico regressão
plt.show()
