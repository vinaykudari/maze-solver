from collections import defaultdict
import numpy as np

class MonteCarlo:
    def __init__(
        self, 
        env,
        gamma=0.01,
        episodes=1000,
        eval_episodes=50,
        epsilon_start=0.8,
        epsilon_decay=0.9999,
        epsilon_min=0.01,
        negative_rewards=[-0.75, -0.85, -5.0],
        max_eval_timesteps=20,
    ):
        self.env = env
        self.gamma = gamma
        self.n_states = self.env.observation_space.n
        self.states = self.env.states
        self.n_actions = self.env.action_space.n
        self.actions = self.env.actions
        self.episodes = episodes
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eval_episodes = eval_episodes
        
        # initialize traning logs
        self.logs = defaultdict(
            lambda: {
                'bad_state_count': 0,
                'timesteps': 0,
                'goal_achieved': False,
                'reward': 0,
                'cumulative_reward': 0,
                'epsilon': None,
            },
        )
        
        #initialize evaluation logs
        self.eval_logs = defaultdict(
            lambda: {
                'bad_state_count': 0,
                'timesteps': 0,
                'goal_achieved': False,
                'reward': 0,
                'cumulative_reward': 0,
            },
        )
        self.negative_rewards = negative_rewards
        self.max_eval_timesteps = max_eval_timesteps
        
        
        # initialize random policy
        self.policy = {
            state:np.random.choice(self.actions) for state in self.states
        }
    
        # initialize action-value function
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions),
        )
        
        # initialize state-action counts
        self.sa_count = defaultdict(
            lambda: np.zeros(self.n_actions),
        )
        
    def _get_action_probs(self, Q_s, epsilon):
        # initialize episilon probability to all the actions
        probs = np.ones(self.n_actions) * (epsilon / self.n_actions)
        best_action = np.argmax(Q_s)
        # initialize 1-epsilon probability to the greedy action
        probs[best_action] = 1 - epsilon + (epsilon / self.n_actions)
        return probs
        
    def _get_action(self, state, epsilon):
        # select action based in epsilon greedy fashion 
        action = np.random.choice(
            self.actions, 
            p=self._get_action_probs(
                self.Q[state],
                epsilon,
            ),
        ) 
        
        return action, self.actions.index(action)
    
    def _simulate_one_episode(self, n, epsilon):
        state = self.env.reset()
        episode = []
        timesteps = 0
        episode_ended = False
        
        while not episode_ended:
            action, a_idx = self._get_action(state, epsilon)
            _, reward, goal, current_state, episode_ended = self.env.step(action=action)
            
            if reward in self.negative_rewards:
                self.logs[n]['bad_state_count'] += 1
            
            # save logs for analysis
            self.logs[n]['reward'] += reward
            self.logs[n]['cumulative_reward'] = self.logs[n]['reward']
            self.logs[n]['goal_achieved'] = goal
                
            # save timestep information
            episode.append((state, a_idx, reward))
            state = current_state
            timesteps += 1
            
        self.logs[n]['timesteps'] = timesteps
        
        return episode
    
    def _update_Q(self, episode):
        states, actions, rewards = zip(*episode)
        states_actions = list(zip(states, actions))
        G = 0
        
        for timestep in range(len(episode)-1, -1, -1):
            state, action, reward = episode[timestep]
            G = reward + self.gamma*G
            # one step monte carlo
            if (state, action) not in states_actions[:timestep]:
                old_Q = self.Q[state][action]
                self.sa_count[state][action] += 1
                alpha = 1 / self.sa_count[state][action]
                self.Q[state][action] = old_Q + alpha * (G - old_Q)
                
                # update policy using Q value function
                self.policy[state] = self.actions[
                    np.argmax(self.Q[state])
                ]
            
                
    def run(self):
        epsilon = self.epsilon_start
        for episode_no in range(self.episodes):
            # set lower limit for epsilon
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            self.logs[episode_no]['epsilon'] = epsilon
            episode = self._simulate_one_episode(episode_no, epsilon)
            self._update_Q(episode)
            
            if episode_no > 0:
                self.logs[episode_no]['cumulative_reward'] += \
                self.logs[episode_no-1]['cumulative_reward']
            
        return self.policy, self.Q
    
    def evaluate(self, policy=None):
        if not policy:
            policy = self.policy
        
        for n in range(self.eval_episodes):
            done = False
            state = self.env.reset()
            timesteps = 0
            
            while not done:
                _, reward, goal, state, done = self.env.step(
                    action=self.policy[state],
                )
                timesteps += 1
                
                if reward in self.negative_rewards:
                    self.eval_logs[n]['bad_state_count'] += 1
                    
                self.eval_logs[n]['reward'] += reward
                self.eval_logs[n]['cumulative_reward'] = self.eval_logs[n]['reward']
                self.eval_logs[n]['goal_achieved'] = goal
                
            self.eval_logs[n]['timesteps'] = timesteps
            
            if n > 0:
                self.eval_logs[n]['cumulative_reward'] += \
                self.eval_logs[n-1]['cumulative_reward']