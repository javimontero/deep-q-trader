# Author: Javier Montero 
# License: MIT


import pandas as pd
import numpy as np

class Data:
	def __init__ (self):
		self.rawdata = pd.read_csv ("dataset/OHLCV-201807.csv")
		self.rawdata.timestamp = pd.to_datetime (self.rawdata.timestamp)
		self.rawdata['price'] = self.rawdata.close.values # copy clean price data
		
		self.cols = list (self.rawdata.columns)
		self.cols.remove ('timestamp')
		self.cols.remove ('price')

	def log_returns (self, cols):
		'''
		log property 
		log (a/b) = log(a) - log(b)
		'''
		c = 0.000001 #constant to prevent log(0)
		self.rawdata[cols]=np.log(self.rawdata[cols]+c).diff() #calculate log returns

	def zscore(self, x, window):
		'''
		Si hay una serie larga (>window) sin cambios, esta funcion devuelve NaN, np.inf y np.NINF
			'''
		r = x.rolling(window=window)
		m = r.mean().shift(1)
		s = r.std(ddof=0).shift(1)
		z = (x-m)/s
		return z

	def clip_outliers (self, c, threshold):
		self.rawdata.loc[self.rawdata[c] > threshold, c] = threshold
		self.rawdata.loc[self.rawdata[c] < -1*threshold, c] = -1*threshold

	def encode_timestamp (self):
		# Encode time (hour and minute) sine cosine
		self.rawdata['ts_m_s'] = np.sin(2 * np.pi * self.rawdata.timestamp.dt.minute/59) # minute sine
		self.rawdata['ts_m_c'] = np.cos(2 * np.pi * self.rawdata.timestamp.dt.minute/59) # minute cosine

		self.rawdata['ts_h_s'] = np.sin(2 * np.pi * self.rawdata.timestamp.dt.hour/23) # hour sine
		self.rawdata['ts_h_c'] = np.cos(2 * np.pi * self.rawdata.timestamp.dt.hour/23) # hour cosine

		# Encode time (dow - day of week) one hot encoding
		dow = pd.get_dummies (self.rawdata.timestamp.dt.dayofweek, prefix = 'dow')
		self.rawdata = self.rawdata.join (dow)

	def preprocess (self):
		"""Preprocess: log returns + z-score(60) + clip outliers to 3"""
		# self.rawdata['price'] = self.rawdata.close.values # copy clean price data
		
		# cols = list (self.rawdata.columns)
		# cols.remove ('timestamp')
		# cols.remove ('price')
		
		self.log_returns (self.cols) # log returns

		for c in self.cols:
			self.rawdata[c] = self.zscore (self.rawdata[c], 60) # Z-Score normalization
			#self.clip_outliers (c,  3) # clip outliers

		# Clean nan, inf and -inf
		self.rawdata.replace([np.inf, -np.inf], np.nan)
		self.rawdata.fillna(0,inplace=True)

		self.encode_timestamp()

