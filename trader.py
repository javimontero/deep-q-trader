#!/usr/bin/env python

'''
Author: Javier Montero 
License: MIT
'''

import numpy as np
from datetime import datetime, timedelta
import timeit
import pandas as pd
import pickle
import os
import sys
import yaml
import argparse

from Market import Market
from agents.DQNAgent import Agent

from utils import is_legal_action
from utils import describe

def train_agent (env, agent, debug_steps = None, batch_size=10,filename='results', T=10, episodes=2):
    done = False
    if debug_steps==True: os.system ("mkdir log")
    results = pd.DataFrame (columns = ['episode', 'reward', 'balance', 'steps','total_steps'])
    

    total_steps = 0 
    for e in range(episodes):
        start_episode = timeit.default_timer()
        state = env.reset()
        state = np.reshape(state, [1, len(state)])
        steps = 0
        while True:
            action,_ = agent.act_legal(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, len(next_state)])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model ()
                break
       
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if steps % T == 0: # Update target value once in T steps
                agent.update_target_model ()

            #print step avances
            if steps % 100 == 0:
                sys.stdout.write("steps: {}/{} balance: {:.2f}\r".format(steps, 
                    env.getNumSamples(), 
                    env.balance) )
                sys.stdout.flush()

            steps += 1
            total_steps += 1

        if debug_steps == True:
            filename_steps = "log/steps-{:03d}.pk".format(e)
            pickle.dump (env.steps, open (filename_steps, 'wb')) # dump results
        
        results = results.append ({'episode': e, 
                                        'reward'     : env.steps.reward.sum(), 
                                        'balance'    : env.balance,
                                        'steps'      : env.steps.shape[0],
                                        'total_steps': total_steps}, 
                                        ignore_index=True)
        pickle.dump (results, open (filename, 'wb')) # dump results    
        stop_episode=timeit.default_timer()
        print ("epi: {}, tot_steps: {}, balance: {:.2f}, cum_rewards: {:.2f}, epi_steps: {}, steps/sec: {:.2f} trx: {}".format (
            e, 
            total_steps,
            env.balance, 
            env.steps.reward.sum(),
            env.steps.shape[0],
            env.steps.shape[0]/timedelta(seconds=stop_episode-start_episode).total_seconds(),
            env.steps[env.steps.tpnl != 0].timestamp.count()))
        
        pickle.dump (results, open (filename, 'wb')) # dump results
        agent.save ("model.hdf5") # dump model

# LAUNCH TRAINING
def main (args):
    # Reading config file
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    _results_filename = cfg['config']['run_filename']
    _startDate = cfg['config']['startDate']
    _endDate   = cfg['config']['endDate']

    #data_lenght = args.lag # Number of historic samples
    #state_size = 5 * data_lenght + 3 + 11 # (OHLC * past samples) + position_side_one_hot_encoded (3) + timestamp (11 dimension)
   

    if args.dataset == 'ohlcv': from Data_ohlcv import Data
    if args.dataset == 'order_book': from Data_ob import Data

    price_data = Data (load_preprocessed = args.load_preprocessed)
    num_features = len (price_data.rawdata.columns) - 2 # Features except timestamp and price
    state_size = num_features * args.lag + 3 + 11 # (OHLC * past samples) + position_side_one_hot_encoded (3) + timestamp (11 dimension)
    price_data.preprocess()

    env = Market(price_data.rawdata, _startDate, _endDate, 
                data_lenght      =args.lag, 
                funds            =args.funds, 
                max_episode_draw =args.drawdown, 
                rwd_function     =args.reward,
                flip_position    =args.flip,
                fee              =args.fee)
    action_size = 3
    agent = Agent(74, action_size, args.memory_size)

    #describe(env, agent, args)

    start = timeit.default_timer()
    train_agent(env, agent, debug_steps = True, batch_size=args.batch_size,filename=_results_filename, T= args.T, episodes=args.episodes)
    stop = timeit.default_timer()  
    print ("Runtime: ", str(timedelta(seconds=stop-start)))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='order_book',type=str, help='Dataset: ohlcv or order_book', dest='dataset')
    parser.add_argument('-T', default=10, type=int, help='Update weights every', dest='T')
    parser.add_argument('-l', default=10, type=int, help='Number of historic records to use', dest='lag') 
    parser.add_argument('--funds', default=500, type=int, help='Initial funds', dest='funds') 
    parser.add_argument('--memory', default=7200, type=int,help='Memory size', dest='memory_size') 
    parser.add_argument('--drawdown', default=0.2, type=float, help='Maximum drawdown per episode', dest='drawdown') 
    parser.add_argument('--batch', default=5, type=int,help='Batch size', dest='batch_size') 
    parser.add_argument('--episodes', default=2, type=int,help='Number of episodes', dest='episodes') 
    parser.add_argument('--reward', default='reward_01', type=str,help='Reward function',dest='reward')
    parser.add_argument('--flip', default=False, type=bool, help='Allow flip position',dest='flip')
    parser.add_argument('--fee', default=False, type=float, help='Market fees (BitMEX default 0.00075)',dest='fee')
    parser.add_argument('--load_preprocessed', default=True, type=bool, help='Load preprocessed data or not',dest='load_preprocessed')

    # parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    # parser.add_argument('--rom_path', default='./atari_roms', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    # parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    # parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    # parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    # parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float, help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    # parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    # parser.add_argument('--entropy', default=0.02, type=float, help="Strength of the entropy regularization term (needed for actor-critic)", dest="entropy_regularisation_strength")
    # parser.add_argument('--clip_norm', default=3.0, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    # parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    # parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    # parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    # parser.add_argument('--max_local_steps', default=5, type=int, help="Number of steps to gain experience from before every update.", dest="max_local_steps")
    # parser.add_argument('--arch', default='NIPS', help="Which network architecture to use: from the NIPS or NATURE paper", dest="arch")
    # parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    # parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    # parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    # parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    # parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)