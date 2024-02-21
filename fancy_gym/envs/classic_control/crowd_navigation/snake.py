from typing import SupportsFloat, Union, Tuple, Optional, Any, Dict
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


class SnakeEnv(gym.Env):
    """
    Snake env for testing purposes.

    Args:
        width (int): width of the environment (in meters)
        hieght (int): hieght of the environment (in meters)
    """
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        **kwargs
    ):
        super(SnakeEnv, self).__init__()
        self.MAX_EPISODE_STEPS = 400
        self.WIDTH = width
        self.HEIGHT = height
        self.num_cells_x = kwargs.get('num_cells_x', 10)
        self.num_cells_y = kwargs.get('num_cells_y', 10) 
        self._start_env_vars()

        self.action_space = spaces.Discrete(4)

        # State should contain all grid cells with the indication of what's inside
        self.observation_space = spaces.Box(
            low=0, high=3,  #  : empty, 1 : snake head, 2 : snake body, 3 : goal
            shape=(self.num_cells_y, self.num_cells_x),
            dtype=int
        )   

        # containers for plotting
        self.metadata = {'render.modes': ["human"]}
        self.fig = None
        self._steps = 0
        self._score = 0
        self._is_collided = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super(SnakeEnv, self).reset(seed=seed, options=options)
        self._start_env_vars()
        self._steps = 0
        self._score = 0
        self._is_collided = False
        return self._get_obs().copy(), {}

    def _start_env_vars(self):
        self._snake_body = [np.array([self.num_cells_x // 2, self.num_cells_y // 2])]
        self._add_goal()
        

    def _add_goal(self):
        fruit_placed = False
        while not fruit_placed:
            potential_pos = np.random.randint(0, self.num_cells_x), np.random.randint(0, self.num_cells_y)
            if not any(np.array_equal(potential_pos, part) for part in self._snake_body):
                self._goal_pos = potential_pos
                fruit_placed = True
    
    def _get_reward(self):
        if self._is_collided:
            return -20, {"reason": "collision"}
        elif np.array_equal(self._snake_body[0], self._goal_pos):
            return 20, {"reason": "fruit"}
        else:
            return -1, {"reason": "moved"}


    def _get_obs(self):
        grid = np.zeros((self.num_cells_y, self.num_cells_x), dtype=int)
        
        grid[self._snake_body[0][1], self._snake_body[0][0]] = 1
        
        for part in self._snake_body[1:]:
            grid[part[1], part[0]] = 2
            
        grid[self._goal_pos[1], self._goal_pos[0]] = 3
        return grid

    
    def _check_collisions(self, new_head_pos):
        # Walls
        if (new_head_pos[0] < 0 or new_head_pos[0] >= self.num_cells_x or
                new_head_pos[1] < 0 or new_head_pos[1] >= self.num_cells_y):
            return True 

        # Own body
        if any(np.array_equal(new_head_pos, part) for part in self._snake_body[1:]):
            return True

        return False

    
    def _update_agent_pos(self, action: int):
        direction_map = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0])    # Right
        }

        if isinstance(action, np.ndarray):
            action = action.item() 

        direction = direction_map[action]

        new_head_pos = self._snake_body[0] + direction

        # Checks the collisions before updating the position
        if self._check_collisions(new_head_pos):
            self._is_collided = True
            return self._get_reward()

        self._snake_body.insert(0, new_head_pos)

        reward, info  = self._get_reward()
        # Eats the fruit
        if np.array_equal(new_head_pos, self._goal_pos):
            self._score += 1
            self._add_goal() 
        else:
            # Remove the last segment of the body to simulate the movement
            self._snake_body.pop()
        
        return reward, info
    
    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_xlim(-0.5, self.num_cells_x - 0.5)
            self.ax.set_ylim(-0.5, self.num_cells_y - 0.5)
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            
            self.grid_display = np.zeros((self.num_cells_y, self.num_cells_x))
            self.img = self.ax.imshow(self.grid_display, cmap=ListedColormap(['#808080', 'green', 'blue', 'red']), vmin=0, vmax=3)
            
        self.grid_display = np.zeros((self.num_cells_y, self.num_cells_x))
        self.grid_display[self._snake_body[0][1], self._snake_body[0][0]] = 1  # Tête du serpent
        for part in self._snake_body[1:]:
            self.grid_display[part[1], part[0]] = 2  # Corps du serpent
        self.grid_display[self._goal_pos[1], self._goal_pos[0]] = 3  # Fruit
        
        self.img.set_data(self.grid_display)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    
    def step(self, action: int):
        """
        Args:
            action: action to be taken by the agent

        Returns:
            Tuple of (obs, reward, done, info) with type np.ndarray
        """
        
        reward, info = self._update_agent_pos(action)
        self._steps += 1

        
        done = self._terminate()

        obs = self._get_obs()


        return obs, reward, done, False, info


    def _terminate(self):
        return self._is_collided

    def close(self):
        super(SnakeEnv, self).close()
        del self.fig
