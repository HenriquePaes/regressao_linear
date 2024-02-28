import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

# Lendo a base de dados
base = pd.read_csv('../recursos/mt_cars.csv')

# Imprimindo as informações da base e alguns registros
print(base.shape)
# print(base.head())

# removendo a coluna Unnamed: 0
base = base.drop(['Unnamed: 0'], axis=1)
# print(base.head())

# Visualizando a matriz de correlação
corr = base.corr()
# sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f') # parametro annot se True, mostra os valores
# plt.show()

# Gerando o gráfico de algumas colunas que são bem correlacionadas
# column_pairs = [('mpg','cyl'), ('mpg','disp'), ('mpg','hp'), ('mpg','wt'), ('mpg','drat'), ('mpg','vs')]
# n_plots = len(column_pairs)
# fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6,4 * n_plots))
# for i, pair in enumerate(column_pairs):
#     x_col, y_col = pair
#     sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
#     axes[i].set_title(f'{x_col} vs {y_col}')
#
# plt.tight_layout()
# plt.show()

# aic 156.6 | bic 162.5
# modelo = sm.ols(formula='mpg ~ wt + disp + hp', data=base)

# aic 165.1 | bic 169.5
# modelo = sm.ols(formula='mpg ~ disp + cyl', data=base)

# aic 179.1 | bic 183.5
modelo = sm.ols(formula='mpg ~ drat + vs', data=base)
modelo = modelo.fit()
print(modelo.summary())

# Analisando os residuais do modelo
residuos = modelo.resid
plt.hist(residuos, bins=20) # bins é o numero de eixos
plt.xlabel('Residuos')
plt.ylabel('Frequencia')
plt.title('Histograma de Resíduos')
plt.show()

# Gerando gráfico de normalidade
stats.probplot(residuos, dist='norm', plot=plt)
plt.title('Q-Q Plot de Residuos')
plt.show()

# Executar o teste de shapiro-wilk, teste de hipotese
stat, pval = stats.shapiro(residuos)
print(f'Shapiro-Wilk statística: {stat:.3f}, p-value: {pval:.3f}')
# hipotese nula: Os dados estão normalmente distribuídos
# p <= 0.05 rejeito a hipótese nula, (não estão normalmente distribuídos)
# p > 0.05 não é possível rejeitar a Hipótese Nula


