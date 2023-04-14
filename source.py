import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn
import yfinance as yf
import statsmodels.api as sm
import os

#---------------------------         Correlation Matrix           -------------------------------------------------------
#
def show_corr_map(dataframe):
	corr = dataframe.corr(method='pearson', numeric_only=True)
	print("----------------------------------------------------------------\n")
	print(corr)
	print("\n----------------------------------------------------------------")
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True
	seaborn.heatmap(corr, cmap=seaborn.color_palette("Spectral", as_cmap=True), vmax=1.0, vmin=-1.0, mask=mask, linewidth=2.5)
	plt.yticks(rotation=0)
	plt.xticks(rotation=90)
	plt.show()

#--------------------------        Ordinary Last-Squares model         -----------------------------------
#
def ols(dataframe, y, drop_vars):
	drop_vars.append(y)
	model = sm.OLS(dataframe[y], sm.add_constant(dataframe.drop(drop_vars, axis=1)))
	results = model.fit()
	print(results.summary())
	res = pd.DataFrame(data=[results.params, results.pvalues], index = ['Coef','p-val']).T
	summary = f"\n<Dep Variable: {y}>\nR-Squared: {results.rsquared}\nR-Squared Adj: {results.rsquared_adj}\n-----------------------------\n{res}\n"
	print(summary)

#--------------------------            Historical Charts                 --------------------------------------------
''' 
Shows historical charts, with either cumulative or normal percentage changes, 
as well as with different timeframes  
'''
def show_historical_chart(dataframe, symbols, cum=False, resample=None):
	interval = 'Daily'

	if cum:
		dataframe = ((dataframe+1).cumprod()-1)
	if resample != None:
		dataframe = dataframe.resample(resample).last()
		if resample == 'W':
			interval = 'Weekly'
		elif resample == 'M':
			interval = 'Monthly'
		else:
			interval = 'Yearly'

	interval = interval + " Cumulative" if cum==True else interval

	fig, ax = plt.subplots(figsize=(12,6))
	for symbol in symbols:
		ax.plot(dataframe[symbol].index, dataframe[symbol], label=symbol)
	ax.set_xlabel('Date')
	ax.set_ylabel(f'{interval} Returns')
	ax.legend()
	ax.set_title(f'{interval} Returns')
	ax.xaxis.set_major_locator(mdates.YearLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
	plt.show()

#--------------------------------------------------------------------------------------------------


# Clear console on every run
os.system('CLS')

# Import Data
data = pd.read_csv("pct_changes_d.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index("Date")

# Show charts of gold prices in relation to SP500 and DJI over different timeframes, cumulative / non-cumulative
show_historical_chart(data, ['^XAU', '^GSPC', '^DJI'])
show_historical_chart(data, ['^XAU', '^GSPC', '^DJI'], resample='M')
show_historical_chart(data, ['^XAU', '^GSPC', '^DJI'], resample='W', cum=True)
show_historical_chart(data, ['^XAU', '^GSPC', '^DJI'], resample='Y', cum=True)

# Show charts of gold prices in relation to gold mining companies over different timeframes, cumulative / non-cumulative
show_historical_chart(data, ['^XAU', 'GOLD', 'NEM', 'KGC'], resample='M')
show_historical_chart(data, ['^XAU', 'GOLD', 'NEM', 'KGC'], resample='Y', cum=True)

# Correlation matrix
show_corr_map(data)

# Ordinary Least-Squares model with gold prices as the dependent variables
input("press any keys to compute next model...")
ols(data, '^XAU', ['NEM','GOLD','KGC'])
input("press any keys to compute next model...")
ols(data, 'GOLD', ['^GSPC', '^DJI', 'KGC', 'NEM'])
input("press any keys to compute next model...")
ols(data, 'NEM', ['^GSPC', '^DJI', 'KGC', 'GOLD'])
input("press any keys to compute next model...")
ols(data, 'KGC', ['^GSPC', '^DJI', 'GOLD', 'NEM'])
input("press any keys to compute next model...")
ols(data, 'GOLD', ['NEM','KGC'])
input("press any keys to compute next model...")
ols(data, 'NEM', ['GOLD','KGC'])
input("press any keys to compute next model...")
ols(data, 'KGC', ['NEM', 'GOLD'])