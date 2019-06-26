import gym
from gym import spaces
from env import CustomEnv

from baselines.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: CustomEnv()])





