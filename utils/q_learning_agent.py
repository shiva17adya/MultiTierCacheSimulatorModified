import random
import time
from typing import List, Optional

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        seed: Optional[int] = None,
    ):
        """Q-Learning Agent for ARC parameter 'p' adaptation.

        Args:
            num_states: Maximum number of distinct state values (representing p from 0 to num_states - 1).
            num_actions: Range parameter defining the discrete action adjustment interval [-num_actions, num_actions - 1].
            learning_rate: Learning rate alpha (default 0.1).
            discount_factor: Discount factor gamma (default 0.99).
            epsilon: Exploration probability epsilon (default 0.1).
            seed: Optional random seed for reproducible exploration.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_states = max(1, int(num_states))
        self.num_actions = max(1, int(num_actions))

        self.action_space: List[int] = list(range(-self.num_actions, self.num_actions))
        self.q_table = np.zeros([self.num_states, len(self.action_space)], dtype=float)

        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_decay_value = 0.001
        self.rewards: List[int] = []

    def get_next_state(
        self,
        current_p: int,
        current_t1_len: int,
        current_b1_len: int,
        current_b2_len: int,
        hit: str,
    ) -> int:
        state_idx = max(0, min(int(current_p), self.num_states - 1))

        if random.random() < self.epsilon:
            action_idx = random.randrange(len(self.action_space))
        else:
            action_idx = int(np.argmax(self.q_table[state_idx]))

        action_value = self.action_space[action_idx]
        new_state = max(0, min(state_idx + action_value, self.num_states - 1))

        if hit == "t1" or hit == "t2":
            reward = 100
        elif hit == "b1":
            reward = -10 if current_b1_len < current_b2_len else -1
        elif hit == "b2":
            reward = -10 if current_b1_len > current_b2_len else -1
        elif hit == "Miss":
            reward = -100
        else:
            reward = 0

        self.rewards.append(reward)

        best_future_q = float(np.max(self.q_table[new_state]))
        td_target = reward + self.discount_factor * best_future_q
        td_error = td_target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.learning_rate * td_error

        return new_state
