from gym import Env as OpenAIEnv
from gym import spaces
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np

class Rewards:
    GOAL = 20.0
    WALL = -5.0
    OUT_OF_BOUND = -0.75
    VISITED = -0.85
    LEGAL = -0.05

class GridEnv(OpenAIEnv):
    def __init__(
        self,
        maze,
        is_stochastic,
        action_transitions,
        max_timesteps=300,
    ):
        self.maze = maze
        self.w, self.h = np.shape(maze)
        self.is_stochastic = is_stochastic
        self.n_actions = len(action_transitions)
        self.action_transitions = action_transitions
        self.current_action = None
        self.observation_space = spaces.Discrete(self.w*self.h)
        self.action_space = spaces.Discrete(self.n_actions)
        self.states = [(i, j) for j in range(self.w) for i in range(self.h)]
        self.actions = list(self.action_transitions)
        self.max_timesteps = max_timesteps
        self.timestep = 0
        self.state = maze
        self.visited = {(i, j):False for j in range(self.w) for i in range(self.h)}
        self.agent_pos = np.array([0, 0])
        self.current_state = (self.agent_pos[0], self.agent_pos[1])
        self.max_timesteps = max_timesteps
        
        # set random goal position
        self.goal_pos = np.array([self.w-1, self.h-1])
        
        # set colors for visualization
        self.state[self.agent_pos[0]][self.agent_pos[1]] = 0.2
        self.state[self.goal_pos[0]][self.goal_pos[1]] = 0.4
        
        fig, ax, mesh = self._set_figure(
            self.state,
            self.w,
            self.h,
            self.observation_space.n,
            self.n_actions,
            show_fig=True,
        )
        
    @staticmethod
    def _set_figure(
        grid,
        w,
        h,
        states,
        actions,
        n=1,
        m=1,
        figsize=(5, 5),
        show_fig=False
    ):
        fig, ax = plt.subplots(n, m, figsize=figsize)
        mesh = []
        
        # for single plot
        if n == 1 and m == 1:
            # Set grid size
            ax.set_xticks(np.arange(0, w, 1))
            ax.set_yticks(np.arange(0, h, 1))
            mesh.append(ax.pcolormesh(grid, cmap ='tab20c'))
            plt.grid()
        else:
            for i in range(n):
                for j in range(m):
                    ax[i][j].set_xticks(np.arange(0, w, 1))
                    ax[i][j].set_yticks(np.arange(0, h, 1))
                    mesh.append(ax[i][j].pcolormesh(grid, cmap ='tab20c'))
                    ax[i][j].grid()
                    
        plt.title(
            f'#states: {states}, #actions: {actions}',
            y=-0.01,
            color='gray',
        )
        
        if not show_fig:
            plt.close(fig)
        
        return fig, ax, mesh
    
    def _move_agent(self, x, y):
        goal_achieved = False
        # reward when the agent doesnt ends up in goal state
        reward = Rewards.LEGAL
        
        if self.state[self.agent_pos[0]][self.agent_pos[1]] == 0.65:
            self.state[self.agent_pos[0]][self.agent_pos[1]] = 0.0
        
        # check if the current agent position is a wall or goal state
        if self.state[self.agent_pos[0]][self.agent_pos[1]] != 0.0 and \
         self.state[self.agent_pos[0]][self.agent_pos[1]] != 0.4:
            # set different color to visited position
            self.state[self.agent_pos[0]][self.agent_pos[1]] = 0.35
            
        
        # clip out-of-bound positions to the edges
        if x >= self.h:
            # reward when the agent goes out of bound
            reward = Rewards.OUT_OF_BOUND
            x = self.h-1
        else:
            x = max(0, x)
            
        if y >= self.w:
            # reward when the agent goes out of bound
            reward = Rewards.OUT_OF_BOUND
            y = self.w-1
        else:
            y = max(0, y)
        
        # set new position
        self.agent_pos[0], self.agent_pos[1] = x, y
        
        # check if the agent's new position is a wall
        if self.state[x][y] == 0.0:
            self.state[x][y] = 0.65
            # reward when the agent hits the wall
            reward = Rewards.WALL
        else:   
            # set agent color in the visualisation
            self.state[x][y] = 0.2
            
        
        if self.visited[(x, y)]:
            # reward when the agent is traverses the visited states
            reward = min(Rewards.VISITED, reward)
        
        
        if (self.agent_pos == self.goal_pos).all():
            goal_achieved = True
            # reward when the agent reaches goal state
            reward = Rewards.GOAL
            
        return reward, goal_achieved
        

    
    def _render_plots(self, n, m, figsize):
        fig, ax, mesh = self._set_figure(
            self.state,
            self.w,
            self.h,
            self.observation_space.n,
            self.n_actions,
            n,
            m,
            figsize, 
            show_fig=True,
        )
        count = 0
        for i in range(n):
            for j in range(m):
                self._update_fig(_, mesh[count], ax[i][j])
                count += 1
        
        plt.show()
        
    
    def _update_fig(self, i, mesh, ax):
        action, reward, _, _, _ = self.step()
        self.state[self.goal_pos[0]][self.goal_pos[1]] = 0.4
        mesh.set_array(self.state.flatten())
        ax.set_title(
            f'timestep: {self.timestep} | reward: {reward} | action: {action}')
        return mesh
    
    def _perform_action(self, action):
        self.current_action = action
        can_agent_move = True
        goal_achieved = False
        
        if self.is_stochastic:
            # make a move according to the transition probability                 
            can_agent_move = np.random.choice(
                [True, False], p=[
                    self.action_transitions[action],
                    1-self.action_transitions[action],
                ],
            )
            if can_agent_move == False:
                reward = Rewards.LEGAL
                
        if can_agent_move:
            if action.lower() == 'w':
                reward, goal_achieved = self._move_agent(
                    self.agent_pos[0]+1, self.agent_pos[1]
                )
            elif action.lower() == 's':
                reward, goal_achieved = self._move_agent(
                    self.agent_pos[0]-1, self.agent_pos[1]
                )
            elif action.lower() == 'a':
                reward, goal_achieved = self._move_agent(
                    self.agent_pos[0], self.agent_pos[1]-1
                )
            elif action.lower() == 'd':
                reward, goal_achieved = self._move_agent(
                    self.agent_pos[0], self.agent_pos[1]+1
                )

        return reward, goal_achieved
        
    def reset(self):
        self.timestep = 0
        self.goal_pos = np.array([self.w-1, self.h-1])
        self.agent_pos = np.array([0, 0])
        
        # set colors for visualization
        self.state[self.agent_pos[0]][self.agent_pos[1]] = 0.2
        self.state[self.goal_pos[0]][self.goal_pos[1]] = 0.4
        
        self.current_state = (self.agent_pos[0], self.agent_pos[1])
        self.visited = {(i, j):False for j in range(self.w) for i in range(self.h)}
        
        return self.current_state
    
    def step(self, action=None):
        done = False
        if not action:
            # select random action if no action is provided
            action_idx = np.random.choice(self.action_space.n)
            action = self.actions[action_idx]
        elif action not in self.actions:
            raise Exception('Action not defined for current environment')
        
        reward, goal_achieved = self._perform_action(action)
                                                                   
        self.timestep += 1
        self.current_state = (self.agent_pos[0], self.agent_pos[1])
        self.visited[(self.agent_pos[0],self.agent_pos[1])] = True
        
        if self.max_timesteps:
            if self.timestep >= self.max_timesteps:
                done = True
                
        if goal_achieved:
            done = True
                                                                   
        return action, reward, goal_achieved, self.current_state, done
    
    def render(self, show=False):
        fig, ax, mesh = self._set_figure(
            self.state,
            self.w,
            self.h,
            self.observation_space.n,
            self.n_actions,
            show_fig=True,
        )
#         plt.arrow(0.25, 0.5, dx=0.4, dy=0, head_width=0.1, color='white')
        if show:
            plt.show()
        
    def animate(self, filename='animation.mp4'):
        fig, ax, mesh = self._set_figure(
            self.state,
            self.w,
            self.h,
            self.observation_space.n,
            self.n_actions,
        )
        ani = FuncAnimation(
            fig,
            self._update_fig,
            frames=self.max_timesteps,
            interval=160,
            fargs=(mesh[0], ax, ), 
        )
        ani.save(filename)
        return ani