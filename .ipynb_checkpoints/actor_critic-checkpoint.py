from collections import defaultdict
import random

import numpy as np
import torch
from torch import FloatTensor as FT, tensor as T

class A2C:
    def __init__(
        self,
        env,
        actor,
        critic,
        n_actns, 
        actor_optmz, 
        critic_optmz,
        mdl_pth='../models/a2c',
        log_freq=100,
        hyprprms={},
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.n_actns = n_actns
        self.actor_optmz = actor_optmz
        self.critic_optmz = critic_optmz
        self.log_freq = log_freq
        self.mdl_pth = mdl_pth
        self.hyprprms = hyprprms
        self.gamma = self.hyprprms.get('gamma', 0.95),
        self.step_sz = self.hyprprms.get('step_sz', 0.001)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'cum_reward': 0,
            },
        )
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'cum_reward': 0,
            },
        )
        
    @staticmethod
    def _normalise(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        arr -= mean
        arr /= (std + 1e-5)
        return arr
        
        
    def _get_returns(self, trmnl_state_val, rewards, gamma=1, normalise=True):
        R = trmnl_state_val
        returns = []
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R 
            returns.append(R)
    
        returns = returns[::-1]
        if normalise:
            returns = self._normalise(returns)
            
        return FT(returns)
    
    def _get_action(self, policy):
        actn = T(policy.sample().item())
        actn_log_prob = policy.log_prob(actn).unsqueeze(0)
        return actn, actn_log_prob
        
    def train(self):
        exp = []
        state = self.env.reset()
        ep_ended = False
        ep_reward = 0
        state = FT(state)
        
        while not ep_ended:
            policy = self.actor(state)
            actn, actn_log_prob = self._get_action(policy)
            state_val = self.critic(state)
                
            _, reward, done, nxt_state, ep_ended = self.env.step(action=actn.item())
            nxt_state = FT(nxt_state)
            exp.append((nxt_state, state_val, T([reward]), actn_log_prob))
            ep_reward += reward
            
            state = nxt_state
            
        states, state_vals, rewards, actn_log_probs = zip(*exp)
        actn_log_probs = torch.cat(actn_log_probs)
        state_vals = torch.cat(state_vals)
        trmnl_state_val = self.critic(state).item()
        returns = self._get_returns(trmnl_state_val, rewards).detach()
        
        
        adv = returns - state_vals
        actn_log_probs = actn_log_probs
        actor_loss = (-1.0 * actn_log_probs * adv.detach()).mean()
        critic_loss = adv.pow(2).mean()
        net_loss = (actor_loss + critic_loss).mean()
        
        self.actor_optmz.zero_grad()
        self.critic_optmz.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optmz.step()
        self.critic_optmz.step()
        
        return net_loss, ep_reward
    
    def run(self, ep=1000):
        for ep_no in range(ep):
            ep_loss, ep_reward = self.train()
            
            self.logs[ep_no]['reward'] = ep_reward
            if ep_no > 0:
                self.logs[ep_no]['cum_reward'] += \
                self.logs[ep_no-1]['cum_reward']
            
            if ep_no % self.log_freq == 0:
                print(f'Episode: {ep_no}, Loss: {ep_loss}, Avg. Reward: {ep_reward}')
            