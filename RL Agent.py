# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:41:07 2022

@author: Jasper Havenhand
"""
import gym
env = gym.make('Alien-v0', render_mode='human')
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while True:
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        t += 1
env.close()