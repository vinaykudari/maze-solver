from collections import defaultdict, deque
import random

import numpy as np
import torch
from torch import FloatTensor as FT, tensor as T

class ReplayMemory:
    def __init__(self, size):
        self.current_size = 0
        self.queue = deque(maxlen=size)
        
    def _get_current_size(self):
        return self.current_size
    
    def can_sample(self, size):
        return self.current_size >= size
    
    def store(self, transition):
        self.current_size += 1
        self.queue.append(transition)
        
    def sample(self, size):
        if not self.can_sample(size):
            raise Exception('Cannot sample, not enough experience')
        
        return random.sample(self.queue, size)

class DQN:
    def __init__(
        self,
        env,
        target_net,
        policy_net,
        optimizer, 
        loss_func,
        log_freq=10,
        tau = 1e-3,
        train_freq=5,
        w_sync_freq=5,
        batch_size=10,
        memory_size=5000,
        gamma=0.95,
        step_size=0.001,
        episodes=1000,
        eval_episodes=50,
        epsilon_start=0.3,
        epsilon_decay=0.9996,
        epsilon_min=0.01,
        negative_rewards=[-0.75, -0.85, -15.0],
    ):
        self.env = env
        self.gamma = np.float64(gamma)
        self.n_states = self.env.observation_space.n
        self.states = self.env.states
        self.n_actions = self.env.action_space.n
        self.actions = self.env.actions
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(size=memory_size)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.step_size = step_size
        self.tau = tau
        self.episodes = episodes
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.negative_rewards = negative_rewards
        self.eval_episodes = eval_episodes
        self.w_sync_freq = w_sync_freq
        self.train_freq = train_freq
        self.log_freq = log_freq
        self.batch_no = 0
        
        # initialize action-value function
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions),
        )
        
        # initialize random policy
        self.policy = {
            state:np.random.choice(self.actions) for state in self.states
        }
        
        # initialize traning logs
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'cumulative_reward': 0,
                'epsilon': None
            },
        )
        
        #initialize evaluation logs
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'cumulative_reward': 0,
                'epsilon': None
            },
        )
       
    def _clip_reward(self, reward):
        return (2 * (
            reward - self.env.min_reward
        ) / (self.env.max_reward - self.env.min_reward)) - 1
    
    def _get_action_probs(self, state, epsilon):
        # initialize episilon probability to all the actions
        probs = np.ones(self.n_actions) * (epsilon / self.n_actions)
        action_values = self.policy_net.forward(state.unsqueeze(0))
        best_action = torch.argmax(action_values)
        # initialize 1-epsilon probability to the greedy action
        probs[best_action] = 1 - epsilon + (epsilon / self.n_actions)
        return probs
        
    def _get_action(self, state, epsilon):
        action = np.random.choice(
            self.actions, 
            p=self._get_action_probs(
                state,
                epsilon,
            ),
        ) 
        
        return action, self.actions.index(action)
    
    def _store_transition(self, transition):
        self.replay_memory.store(transition)
        
    def _update_policy(self, state, Q_values):
        # update policy
        self.policy[tuple(state)] = self.actions[
            np.argmax(Q_values)
        ]

    def _train_one_batch(self, transitions, epsilon):
        states, actions, rewards, next_states, goal_achieved = zip(*transitions)
        states = FT(states)
        next_states = FT(next_states)
        actions = T([actions]).view(-1, 1)
        rewards = T([rewards]).view(-1, 1)
        goal_achieved = T([goal_achieved]).view(-1, 1).float()
        
        Q_values = self.policy_net(states)
        
        predictions = Q_values.gather(1, actions)
        labels_next = torch.max(self.target_net(next_states), dim=1).values.view(-1, 1).detach()            
        labels = rewards + (self.gamma * labels_next * (1 - goal_achieved))
        
        loss = self.loss_func(predictions, labels)
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        rand = random.randint(0, self.batch_size-1)
        self._update_policy(states[rand].detach().numpy(), Q_values[rand].detach().numpy())
        
        return loss
        
    def _sync_weights(self, soft=False):
        if not soft:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # @TODO: Implement soft updates
            pass
        
    def run(self):
        epsilon = self.epsilon_start
        for episode_no in range(self.episodes):
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            episode_ended = False
            self.logs[episode_no]['epsilon'] = epsilon
            episode_reward = 0
            episode_loss = 0
            timestep = 0
            state = self.env.reset()
            
            while not episode_ended:
                action, action_idx = self._get_action(FT(state), epsilon)
                _, reward, done, next_state, episode_ended = self.env.step(action=action)
                
                episode_reward += reward
                
                self._store_transition(
                    [state, action_idx, reward, next_state, done]
                )
                
                if self.replay_memory.current_size > self.memory_size:
                    transitions = self.replay_memory.sample(size=self.batch_size)
                    episode_loss += self._train_one_batch(transitions, epsilon)
                    
                    if self.batch_no % self.w_sync_freq == 0:
                        self._sync_weights()
                    self.batch_no += 1
                    
                timestep += 1
                state = next_state
                
            if episode_no % self.log_freq == 0 and self.replay_memory.current_size > self.memory_size:
                print(f'Episode: {episode_no}, Reward: {episode_reward}, Loss: {episode_loss}')
                
            # save logs for analysis
            self.logs[episode_no]['reward'] = episode_reward
            if episode_no > 0:
                self.logs[episode_no]['cumulative_reward'] += \
                self.logs[episode_no-1]['cumulative_reward']
                
    def evaluate_one_episode(self, e_num=None, policy=None):
        action_seq = []
        timestep = 0
        done = False
        state = self.env.reset()
        if not policy:
            policy = self.policy
            
        while not done:
            action, reward, goal, state, done = self.env.step(
                action=self.policy[state],
            )
            timestep += 1
            
            if e_num is not None:
                self.eval_logs[e_num]['reward'] += reward
                self.eval_logs[e_num]['cumulative_reward'] = self.eval_logs[e_num]['reward']
                self.eval_logs[e_num]['goal_achieved'] = goal
            
            action_seq.append(action)
            
        return timestep, action_seq
    
    def evaluate(self, policy=None):
        if not policy:
            policy = self.policy
        
        for n in range(self.eval_episodes):
            timesteps, _ = self.evaluate_one_episode(n, policy)
            self.eval_logs[n]['timesteps'] = timesteps
            
            if n > 0:
                self.eval_logs[n]['cumulative_reward'] += \
                self.eval_logs[n-1]['cumulative_reward']
        