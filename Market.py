# Author: Javier Montero 
# License: MIT

import pandas as pd
import numpy as np
#from Data_ohlcv import Data

class Position():
    def __init__ (self):
        self.side = 1
        self.price = 0
        self.qty = 0
        self.upnl = 0 # Unrealized PnL

class Market(): ## Adaptado a DQN
    def __init__(self, data, startDateTime, endDateTime, data_lenght, funds, max_episode_draw, rwd_function, flip_position=False, fee=0.00075):
        self.INITIAL_FUNDS = funds
        self.balance = self.INITIAL_FUNDS
        self.flip_position = flip_position

        self.MAXDRAWDOWN = max_episode_draw # Maximum drawdown to finish the episode
        
        self.reward = getattr (self,rwd_function)

        #action space logic
        self._sell = 0
        self._hold = 1
        self._buy  = 2
        
        self._short  = 0
        self._none = 1
        self._long  = 2

        self.BACK_DATA = data_lenght

        self.position = Position()
        
        self.tpnl = 0 # transaction PnL

        # order dataset
        columns = ['timestamp', 'side', 'qty', 'price','fee']
        #self.orders = pd.DataFrame(columns=columns)
        #self.orders = self.orders.fillna(0) 
        
        # transaction dataset
        columns = ['timestamp','amount', 'balance']
        #self.transactions = pd.DataFrame (columns=columns)

        self.FEE = fee 

        # timestamps de inicio y final de la simulación
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        
        self.data = data
        
        # Steps
        columns = ['timestamp','side', 'price', 'upnl', 'tpnl', 'reward','balance']
        self.steps = pd.DataFrame (columns=columns)
        self.steps = self.steps.fillna(0)
        
        # BITMEX API timestamp format to python datetime
        #self.data['timestamp'] = pd.to_datetime (self.data.timestamp)

        # index of startDate

        self.startIndex = self.data.index [self.data['timestamp'] >= startDateTime][0]
        self.endIndex = self.data.index [self.data['timestamp'] >= endDateTime][0]
        self.runIndex = self.startIndex # timestamp del momento en el que se encuentra el simulador

        
    def save_transaction (self, timestamp, amount):
        # Add to transaction dataset
        # self.transactions = self.transactions.append (pd.Series([timestamp, amount], 
        #                                                 index = self.transactions.columns),
        #                                                  ignore_index = True)
        self.transactions = self.transactions.append({'timestamp': timestamp, 
                                                         'amount': amount,
                                                        'balance': 0},
                                                         ignore_index=True)
        # Update BALANCE OR NOT....
    
    # def calc_profit_loss (self, entry_side, entry_price, exit_side, exit_price):
    #     '''
    #     Traditional PnL calculation = (entryPrice * exitSide) + (exitPrice * entrySide)
    #     ASSUME: -1 : SELL(SHORT) | 0  : NONE | 1  : BUY(LONG)
    #     '''
    #     entry_fee = entry_price*self.FEE
    #     exit_fee  = exit_price *self.FEE
    #     # maps 0=SELL|1=HOLD|2=BUY to -1=SELL|0=HOLD|1=BUY
    #     entry_side -=1 
    #     exit_side  -=1
    #     return (entry_price * exit_side) + (exit_price * entry_side) - entry_fee - exit_fee

    def calc_unrealized_pnl (self, entry_side, entry_price, actual_price):
        '''Unrealized PnL includes the exit fee'''
        # if entry_side == 2: side = 0 
        # if entry_side == 0: side = 2
        # return self.calc_profit_loss (entry_side = entry_side, 
        #                              entry_price = entry_price, 
        #                              exit_side   = side, 
        #                              exit_price  = actual_price)
        
        if entry_side == 2: exit_side = 0 
        if entry_side == 0: exit_side = 2
        # Invert position
        entry_side -=1 
        exit_side  -=1
        return (entry_price * exit_side) + (actual_price * entry_side) - (actual_price*self.FEE)

    def execute_order (self, side, qty):
    	# Number of qty =1 always.
        self.tpnl = 0
        if self.position.side == self._none and side == self._hold: # do nothing
            return

        actual_price = self.data.price[self.runIndex] # close price
        timestamp = self.data.timestamp[self.runIndex] # current runIndex timestamp

        # Position open - Action hold. Calculate Unrealized PnL
        if self.position.side != self._none and side == self._hold:
            # Unrealized PnL calculation
            self.position.upnl = self.calc_unrealized_pnl (entry_side   = self.position.side,
                                                           entry_price  = self.position.price,
                                                           actual_price = actual_price)
            return                                     
            
        # Save the order
        
        # self.orders = self.orders.append({
        #     'timestamp':timestamp, 
        #     'side': side, 
        #     'qty': qty, 
        #     'price': actual_price,
        #     'fee': self.FEE*actual_price},
        #                                 ignore_index=True)
        #self.orders.loc[self.id_trade] = [timestamp, side, contracts, actual_price]
        #self.id_trade +=1
     
        # Position open - Action close position or Flip position
        # Flip position not allowed. Close open position, update Realized PnL and save transaction
        if (self.position.side == self._long and side ==self._sell) or (self.position.side==self._short and side==self._buy) and (self.flip_position==False):   
            # self.position.upnl = self.calc_unrealized_pnl (entry_side   = self.position.side,
            #                                                entry_price  = self.position.price,
            #                                                actual_price = actual_price)
            self.tpnl = self.calc_unrealized_pnl (entry_side   = self.position.side,
                                                  entry_price  = self.position.price,
                                                  actual_price = actual_price)

            #self.save_transaction (self.data.timestamp[self.runIndex], self.position.upnl)
            #self.tpnl = self.position.upnl
            
            # update balance
            self.balance += self.tpnl
            
            # close position
            self.position.upnl = 0 # Zero Unrealized PnL
            self.position.qty  = 0
            self.position.side = self._none # Close position 
            self.price         = 0
            return
            
        # Position None - Action buy or sell. Open position with new Buy or Sell order
        if self.position.side == self._none and side != self._hold:
            # Open position
            self.position.side = side
            self.qty = qty
            self.position.price = actual_price

            self.tpnl     = - actual_price * self.FEE
            self.balance += self.tpnl
            return
        
    def reward_01 (self):
        '''Reward_01: Log returns of portfolio value (balance + unrealized pnl)'''
        actual = self.balance + self.position.upnl
        if self.steps.shape[0] > 2:
            previous = self.steps.balance[-1:].values[0] + self.steps.upnl[-1:].values[0]
            reward = np.log (actual/previous)
        else:
            reward =0
        return reward
    
    def reward_02 (self):
        '''Reward_02: PnL at position close, otherwise 0'''
        reward = self.tpnl
        return reward
    
    def reward_03 (self):
        '''Reward_03: Actual balance - Previous balance'''
        if self.steps.shape[0] > 2:
            reward = self.balance - self.steps[-1:].balance.values[0]
        else:
            reward = 0
        return reward
    def reward_04(self):
        '''Reward_04: sigmoid (balance - balance1) or sigmoid (reward_03)'''
        if self.steps.shape[0] > 2:
            reward = self.balance - self.steps[-1:].balance.values[0]
            reward = 1/(1+np.exp(-reward))
        else:
            reward = 0
        return reward


    def step(self, action):
        done = False # end of dataset        
        
        self.execute_order (action , qty = 1)  # Execute order

        reward = self.reward()
         
        # End Episode if maximum drawdown reached
        if self.balance < self.INITIAL_FUNDS -(self.INITIAL_FUNDS * self.MAXDRAWDOWN):
             done = True
             reward = -1
        
        # Save steps
        self.steps = self.steps.append ({
            'timestamp': self.data.timestamp[self.runIndex],
            'side': self.say_action(action),
            'qty': 1,
            'price': self.data.price[self.runIndex],
            'upnl': self.position.upnl ,
            'tpnl': self.tpnl,
            'reward': reward,
            'balance': self.balance}, ignore_index = True)
 
        self.runIndex += 1 # Next sample
        if self.runIndex == self.endIndex: # Dataset end?
            done = True
    
        return self.get_state(), reward, done, None
       
    def getRuntime (self):
        return self.data.timestamp[self.runIndex]
    
    def say_action (self,action):
        if action==0: return "sell"
        if action==1: return ""
        if action==2: return "buy"
    
    # Return number of days of the simulation period
    def getNumdays (self):
        num = self.data.timestamp[self.endIndex] - self.data.timestamp[self.startIndex] 
        return num.days
        
    # def get_historic_states (self, _index, backsamples):
    #     d = self.data[['open', 'high', 'low','close','volume']][_index-backsamples+1:_index+1]
    #     return np.array (d.values.flatten(order='F'))

    def position_one_hot (self, position):
        ''' Dirty one hot encoding of position'''
        if position == 0: return [1,0,0]
        if position == 1: return [0,1,0]
        if position == 2: return [0,0,1]
    
    def get_state_size (self):
        ''' Return the size of the state'''
        self.runIndex = self.startIndex
        print (self.get_state())
        return len(self.get_state())

    def get_state(self): 
        """ Generate features forming the state
        
        STATE = Position + timestamp + price data
        =====
            position (3)              : one hot encondig
            timestamp (11) 
                hour (2) & minute (2) : cyclical sine-cosine encoding
                day of week (7)       : one hot encoding
            OHLCV data (5) x historic : data
        """
        
        position = self.position_one_hot(self.position.side)
        cols = ['ts_m_s','ts_m_c','ts_h_s','ts_h_c','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6']
        tstamp   = self.data[cols][self.runIndex:self.runIndex+1] # timestamp
        cols.append ('price')
        cols.append ('timestamp')
        # All other columns (drop encoded timestamps columns, original timestamp and price)
        data = self.data [self.data.columns.difference (cols)] [self.runIndex-self.BACK_DATA+1:self.runIndex+1]
        #tstamp   = self.data[['ts_m_s','ts_m_c','ts_h_s','ts_h_c','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6']][self.runIndex:self.runIndex+1]
        #ohlcv    = self.data[['open', 'high', 'low','close','volume']][self.runIndex-self.BACK_DATA+1:self.runIndex+1]
        #return np.concatenate ((position, tstamp.values.flatten(), ohlcv.values.flatten()))
        return np.concatenate ((position, tstamp.values.flatten(), data.values.flatten()))
        
    
    def getNumSamples (self):
        return self.endIndex-self.startIndex
    
    # def _set_rpnl(self):
    # 	return self.INITIAL_FUNDS

    def reset(self):
        # go to dataset start
        self.runIndex = self.startIndex
        # empty datasets
        #self.transactions = self.transactions.iloc[0:0] 
        #self.orders = self.orders.iloc[0:0]
        self.steps = self.steps.iloc[0:0]

        self.balance = self.INITIAL_FUNDS
        self.tpnl = 0

        return self.get_state()
    