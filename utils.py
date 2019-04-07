'''
Author: Javier Montero 
License: MIT
'''

import http.client, urllib
import os

def send_pushover_qtrader (msg):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
    urllib.parse.urlencode({
        "token": "",
        "user": "",
        "message": msg,
      }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()

def current_folder_name():
	fullpath = os.getcwd()
	return fullpath.split(os.sep)[-1]

def is_legal_action (action, position):
    if action == 0 and position == 0:
        return False
    if action == 2 and position == 2:
        return False
    return True

def describe(env, agent, args):
    print (" ")
    print ("##################### Training description #####################")
    print ("Samples: ", env.getNumSamples())
    print ("From {} to {}".format (env.startDateTime, env.endDateTime))
    print ("Backsamples: {}  State size: {} Funds: {} Drawdown: {}".format (
                                            args.lag, 
                                            env.state_size, 
                                            env.INITIAL_FUNDS, 
                                            env.MAXDRAWDOWN))
    print ("Gamma: {}  Alpha: {} Memory: {} Batch size: {} Episodes: {}".format (
                                            agent.gamma, 
                                            agent.learning_rate, 
                                            args.memory_size, 
                                            args.batch_size, 
                                            args.episodes))
    print ('Reward: {}'.format (env.reward.__doc__))
    print (agent.model.summary())
    print ("################################################################")
    