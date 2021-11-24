import copy
from collections import defaultdict
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import FloatTensor as FT, tensor as T
from torch.optim import Adam

from policy_net import PolicyNetwork
from q_net import QNetwork
from replay_buffer import ReplayBuffer


DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


class SAC:
    def __init__(
        self,
        env,
        name,
        input_dim,
        networks={},
        optmzrs={},
        log_freq=10,
        hyprprms={},
        save_mdls=True,
        load_mdls=False,
    ):
        self.env = env
        self.env_name = name
        self.input_dim = input_dim
        self.action_space = env.action_space
        self.hyprprms = hyprprms
        self.eps = self.hyprprms.get('eps', 1e-6)
        self.lr = self.hyprprms.get('lr', 5e-4)
        self.gamma = self.hyprprms.get('gamma', 0.99)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
        self.mem_sz = self.hyprprms.get('mem_sz', 5000)
        self.critic_sync_f = self.hyprprms.get('critic_sync_f', 5)
        self.tau = self.hyprprms.get('tau', 1e-2)
        self.save_mdls = save_mdls
        self.load_mdls = load_mdls
        self.memory = ReplayBuffer(self.mem_sz)
        self.tgt_entropy = -self.action_space.n

        # alpha network
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Adam(params=[self.log_alpha], lr=self.lr)

        # policy network
        self.policy = networks.get(
            'policy_net',
            PolicyNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
                eps=self.eps,
            ),
        )
        self.policy_optmz = optmzrs.get(
            'policy_optmz',
            Adam(self.policy.parameters(), lr=self.lr),
        )

        # critic network
        self.critic_a = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_a_optmz = optmzrs.get(
            'critic_a_optmz',
            Adam(self.critic_a.parameters(), lr=self.lr),
        )

        self.critic_b = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_b_optmz = optmzrs.get(
            'critic_b_optmz',
            Adam(self.critic_b.parameters(), lr=self.lr),
        )
        
        self._load_models()

        # target critic network
        self.critic_a_tgt = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_a_tgt.load_state_dict(self.critic_a.state_dict())
        self.critic_a_tgt_optmz = optmzrs.get(
            'critic_a_optmz',
            Adam(self.critic_a_tgt.parameters(), lr=self.lr),
        )

        self.critic_b_tgt = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_b_tgt.load_state_dict(self.critic_b.state_dict())
        self.critic_b_tgt_optmz = optmzrs.get(
            'critic_b_optmz',
            Adam(self.critic_b_tgt.parameters(), lr=self.lr),
        )

        self.log_freq = log_freq
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

    def _get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = T(np.array([state])).to(DEVICE)
        else:
            state = torch.cat([state]).to(DEVICE)
        actions, _, _ = self.policy.sample(state)
        return actions.cpu().detach().numpy()

    def _sync_weights(self, src, tgt, tau=None):
        if tau is None:
            tau = self.tau

        tgt_weights = dict(tgt.named_parameters())
        src_weights = dict(src.named_parameters())

        for name in src_weights:
            src_weights[name] = tau * src_weights[name].clone() + \
                    (1-tau) * tgt_weights[name].clone()

        tgt.load_state_dict(src_weights)

    def _save_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
        alpha_path=None,
    ):
        path = Path(f'../models/{self.env_name}/')
        if not path.exists():
            path.mkdir()

        if policy_path is None:
            policy_path = path/'actor'

        if critic_a_path is None:
            critic_a_path = path/'critic_a'

        if critic_b_path is None:
            critic_b_path = path/'critic_b'

        self.policy.save(policy_path)
        self.critic_a.save(critic_a_path)
        self.critic_b.save(critic_b_path)

    def _load_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
        tgt_value_path=None,
        value_path=None,
    ):
        print('loading models....')
        path = Path(f'../models/{self.env_name}/')

        if policy_path is not None:
            policy_path = path/'actor'
            self.policy.load(policy_path)

        if critic_a_path is not None:
            critic_a_path = path/'critic_a'
            self.critic_a.load(critic_a_path)

        if critic_b_path is not None:
            critic_b_path = path/'critic_b'
            self.critic_b.load(critic_b_path)

    def _train_alpha_net(self, probs, log_probs):
        temp = torch.sum(log_probs * probs, dim=1)
        alpha_loss = - (self.log_alpha.exp() * (temp.detach() + self.tgt_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        return alpha_loss.cpu().detach().numpy()

    def _train_critic_net(self, states, actions, rewards, nxt_states, dones):
        with torch.no_grad():
            actions, probs, log_probs = self.policy.sample(nxt_states)
            q1_target = self.critic_a_tgt(nxt_states)
            q2_target = self.critic_b_tgt(nxt_states)
            etp_min_q_vals = torch.min(q1_target, q2_target) - self.alpha * log_probs
            nxt_v_target = (probs * (etp_min_q_vals)).sum(1).unsqueeze(-1)
            q_backup = rewards.unsqueeze(-1) + (self.gamma * (~dones.unsqueeze(-1)) * nxt_v_target)

        pred_q1_values = self.critic_a(states).gather(1, actions.unsqueeze(-1))
        pred_q2_values = self.critic_b(states).gather(1, actions.unsqueeze(-1))

        critic_a_loss = 0.5 * F.mse_loss(pred_q1_values, q_backup)
        critic_b_loss = 0.5 * F.mse_loss(pred_q2_values, q_backup)

        self.critic_a_optmz.zero_grad()
        self.critic_b_optmz.zero_grad()

        critic_a_loss.backward(retain_graph=True)
        critic_b_loss.backward()

        self.critic_a_optmz.step()
        self.critic_b_optmz.step()

        return ((critic_a_loss + critic_b_loss)/2).cpu().detach().numpy()

    def _train_policy_net(self, states, alpha):
        actions, probs, log_probs = self.policy.sample(states)
        with torch.no_grad():
            pred_q1_values = self.critic_a(states)
            pred_q2_values = self.critic_b(states)

        min_q_values = torch.min(pred_q1_values, pred_q2_values)
        policy_loss = (probs * (self.alpha * log_probs - min_q_values)).sum(1).mean()

        self.policy_optmz.zero_grad()
        policy_loss.backward()
        self.policy_optmz.step()

        return policy_loss.cpu().detach().numpy(), probs, log_probs

    def train(self, ep_no):
        states, actions, rewards, nxt_states, dones = \
            self.memory.sample(self.mem_sz)
   
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones)
        actions = torch.from_numpy(actions).float()
        states = torch.from_numpy(states).float()
        nxt_states = torch.from_numpy(nxt_states).float()

        curr_alpha = self.alpha
        critic_loss = self._train_critic_net(
            states,
            actions,
            rewards,
            nxt_states,
            dones,
        )
        policy_loss, probs, log_probs = self._train_policy_net(
            states,
            curr_alpha,
        )
        alpha_loss = self._train_alpha_net(probs, log_probs)

        if ep_no % self.critic_sync_f:
            self._sync_weights(self.critic_a, self.critic_a_tgt)
            self._sync_weights(self.critic_b, self.critic_b_tgt)

        return alpha_loss, critic_loss, policy_loss
    
    def evaluate(self, ep=None):
        if not ep:
            ep = self.eval_ep

        for ep_no in range(ep):
            state = self.env.reset()
            ep_ended = False
            ep_reward = 0
            ts = 0

            while not ep_ended and ts < 200:
                action = self._get_action(state)
                nxt_state, reward, ep_ended, _ = self.env.step(action)
                ep_reward += reward
                state = nxt_state

            self.eval_logs[ep_no]['reward'] = ep_reward

    def run(self, ep=1000):
        print('collecting experience...')
        rewards = []

        if self.load_mdls:
            self._load_models(
                policy_path=f'models/{env_name}/policy',
                critic_a_path=f'models/{env_name}/critic_a',
                critic_b_path=f'models/{env_name}/critic_b',
                value_path=f'models/{env_name}/value',
                tgt_value_path=f'models/{env_name}/tgt_value',
            )

        for ep_no in range(ep):
            state = self.env.reset()
            ep_ended = False
            ep_reward = 0
            a_loss, c_loss, p_loss = 0, 0, 0
            ts = 0

            while not ep_ended and ts < 200:
                # changes for maze env
                action_idx = self._get_action(state)
                if isinstance(state, np.ndarray):
                    action_idx = action_idx[0]
                action = self.env.actions[action_idx]
                _, reward, _, nxt_state, ep_ended = self.env.step(action)
                ep_reward += reward
                self.memory.add((state, action_idx, reward, nxt_state, ep_ended))
                state = nxt_state

                if self.memory.curr_size > self.mem_sz:
                    a_loss, c_loss, p_loss = self.train(ep_no)
                    if ep_no % 100:
                        self._save_models()
                ts += 1

            rewards.append(ep_reward)
            avg_reward = np.mean(rewards[-50:])
            self.logs[ep_no]['reward'] = ep_reward
            self.logs[ep_no]['avg_reward'] = avg_reward

            if ep_no % self.log_freq == 0:
                if self.memory.curr_size > self.mem_sz:
                    print(f'Episode: {ep_no}, Reward: {ep_reward}, Avg. Reward: {avg_reward}, Policy Loss={round(float(p_loss), 2)}')
                else:
                    print(ep_no, end='..')





