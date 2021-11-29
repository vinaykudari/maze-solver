from collections import defaultdict
import random

import numpy as np
import torch
from torch import FloatTensor as FT, tensor as T

class A2C_CNN:
    def __init__(
        self,
        env,
        actor,
        critic,
        n_actns, 
        actor_optmz, 
        critic_optmz,
        mdl_pth='../models/a2c_cnn',
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
                'avg_reward': 0,
            },
        )
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'avg_reward': 0,
            },
        )
        
    @staticmethod
    def _normalise(arr):
        mean = arr.mean()
        std = arr.std()
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
            return self._normalise(torch.cat(returns))
        
        return torch.cat(returns)
    
    def _get_action(self, policy):
        actn = T(policy.sample().item())
        actn_log_prob = policy.log_prob(actn).unsqueeze(0)
        return actn, actn_log_prob
        
    def train(self):
        exp = []
        state = self.env.reset()
        ep_ended = False
        ep_reward = 0
        state = state[1]
        z = state.shape[0]
        x = state.shape[1]
        y = state.shape[2]
        state = torch.from_numpy(state)
        state = torch.reshape(state,(1,z,x,y))
        state = state.float()
        
        while not ep_ended:
            policy = self.actor(state)
            actn, actn_log_prob = self._get_action(policy)
            state_val = self.critic(state)
                
            _, reward, done, nxt_state, ep_ended = self.env.step(actn.item())
            nxt_state = nxt_state[1]
            z = nxt_state.shape[0]
            x = nxt_state.shape[1]
            y = nxt_state.shape[2]
            nxt_state = torch.from_numpy(nxt_state)
            nxt_state = torch.reshape(nxt_state,(1,z,x,y))
            nxt_state = nxt_state.float()

            exp.append((nxt_state, state_val, T([reward]), actn_log_prob))
            ep_reward += reward
            
            state = nxt_state
            
        states, state_vals, rewards, actn_log_probs = zip(*exp)
        actn_log_probs = torch.cat(actn_log_probs)
        state_vals = torch.cat(state_vals)
        trmnl_state_val = self.critic(state).item()
        returns = self._get_returns(trmnl_state_val, rewards, False).detach()
        
        
        adv = returns - state_vals
        actn_log_probs = actn_log_probs
        actor_loss = (-1.0 * actn_log_probs * self._normalise(adv.detach())).mean()
        critic_loss = adv.pow(2).mean()
        net_loss = (actor_loss + critic_loss).mean()
        
        self.actor_optmz.zero_grad()
        self.critic_optmz.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optmz.step()
        self.critic_optmz.step()
        
        return net_loss, ep_reward
    
    def evaluate(self, ep=None):
        if not ep:
            ep = self.eval_ep

        for ep_no in range(ep):
            state = self.env.reset()
            state = FT(state)
            ep_ended = False
            ep_reward = 0
            ts = 0

            while not ep_ended and ts < 200:
                policy = self.actor(state)
                actn, actn_log_prob = self._get_action(policy)
                _, reward, done, nxt_state, ep_ended = self.env.step(actn.item())
                ep_reward += reward
                state = FT(nxt_state)

            self.eval_logs[ep_no]['reward'] = ep_reward
    
    def run(self, ep=1000):
        rewards = []
        for ep_no in range(ep):
            ep_loss, ep_reward = self.train()

            rewards.append(ep_reward)
            avg_reward = np.mean(rewards[-50:])
            self.logs[ep_no]['reward'] = ep_reward
            self.logs[ep_no]['avg_reward'] = avg_reward
            
            if ep_no % self.log_freq == 0:
                print(f'Episode: {ep_no}, Loss: {ep_loss}, Avg. Reward: {ep_reward}')
            