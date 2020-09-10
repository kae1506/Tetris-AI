"""
Main File
"""
import math
import numpy as np

import DeulingDQN as dddqn
import DQN as dqn
import DoubleDQN as ddqn
import env as env
import argparse
import matplotlib.pyplot as plt

def plotLearning(x, scores, filename, lines=None):
    plt.plot(x, scores)
    plt.xlabel("Game", color="C0")
    plt.ylabel("Avg_scores", color="C0")
    plt.title("Tetris AI")
    plt.savefig(filename)

LR = 0.0005
INPUT_SHAPE = (20,30,2)
N_ACTIONS = 4

parser = argparse.ArgumentParser(description='TETRIS AI WOHO BABY LESSSGOOOOO')
parser.add_argument('algo', metavar='N', type=str, nargs='+',
                    help='which algorithm to use')
parser.add_argument('filename', metavar='N', type=str, nargs='+',
                    help='Filename for plot')
parser.add_argument('n_games', metavar='N', type=int, nargs='+',
                    help='number of episodes to play')

agent = None
args = parser.parse_args()
if args.algo[0] == "dqn":
    agent = dqn.Agent(LR, INPUT_SHAPE, N_ACTIONS)

elif args.algo[0] == "ddqn":
    agent = ddqn.Agent(LR, INPUT_SHAPE, N_ACTIONS)

elif args.algo[0] == "dddqn":
    agent = dddqn.Agent(LR, INPUT_SHAPE, N_ACTIONS)

filename = args.filename[0]

if __name__ == '__main__':
    env = env.TetrisEnv()
    scores, avg_scores = [],[]
    high_score = -math.inf
    n_games = args.n_games[0]
    # for i in range(n_games):
    #     done = False
    #     obs = env.reset()
    #     score = 0
    #     while not done:
    #         action = agent.choose_action(obs)
    #         obs_, reward, done, info = env.step(action)
    #         score += reward
    #         agent.memory.add(obs=obs, act=action, rew=reward, next_obs=obs_, done=done)
    #         agent.learn()
    #
    #         obs = obs_
    #
    #     scores.append(score)
    #     avg_score = np.mean(scores[:-100])
    #     avg_scores.append(avg_score)
    #     high_score = max(high_score, score)
    #
    #     print(f"Episode: {i}, Score: {score}, Avg Score: {avg_score}, High Score: {high_score}")
#
# a = [i for i in range(n_games)]
# plotLearning(a, avg_scores, filename)
# plt.show()