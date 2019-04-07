# Author: Javier Montero 
# License: MIT


import pandas as pd
import numpy as np
import pickle

class Data:
	''' Order book dataset'''
	def __init__ (self,load_preprocessed=False):
		self.rawdata = pickle.load (open('dataset/ob_201807.pk','rb'))
		self.rawdata['price_returns'] = self.rawdata.price.values 
		self.rawdata = self.rawdata.replace ([np.inf, -np.inf], np.nan)
		self.rawdata = self.rawdata.dropna()
		self.rawdata.timestamp = pd.to_datetime (self.rawdata.timestamp)
		self.rawdata.index = pd.RangeIndex (start=0, stop = len(self.rawdata.index), step=1)

	def log_returns (self, cols):
		'''
		log property 
		log (a/b) = log(a) - log(b)
		'''
		c = 0.000001 #constant to prevent log(0)
		self.rawdata[cols]=np.log(self.rawdata[cols]+c).diff() #calculate log returns

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

	def shifted_returns(self,d):
		'''shif values to be all positives, then calculate return'''
		min = abs(d.min())
		d = d + min
		return d.diff()

	def shifted_log_returns(self,d):
		'''shif values to be all positives, then calculate log return'''
		min = abs(d.min())
		d = d + min # shift
		return np.log (d+0.00001).diff()
		
	def preprocess (self):
		
		self.rawdata.delta_depth = self.shifted_log_returns (self.rawdata.delta_depth)
		self.rawdata.delta_vol   = self.shifted_log_returns (self.rawdata.delta_vol)
		self.rawdata.wspread     = self.shifted_log_returns (self.rawdata.wspread)
		self.rawdata.slope       = self.shifted_log_returns (self.rawdata.slope)
		self.rawdata.price_returns = np.log(self.rawdata.price_returns+0.00001).diff() #calculate log returns


		self.rawdata = self.rawdata.replace ([np.inf, -np.inf], np.nan)
		self.rawdata = self.rawdata.dropna()

		self.encode_timestamp()