import random
import time
from typing import List, Optional, Dict

import numpy as np


class QLearningAgent:
    CACHE_EVENTS = ("t1", "t2", "b1", "b2", "Miss")
    EVENT_MAP: Dict[str, int] = {
        "t1": 0,
        "t2": 1,
        "b1": 2,
        "b2": 3,
        "Miss": 4,
        "miss": 4,
    }

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        epsilon_decay_value: float = 0.0,
        min_epsilon: float = 0.01,
        include_cache_state: bool = False,
    ):
        """Q-Learning Agent for ARC parameter 'p' adaptation.

        Args:
            num_states: Maximum number of distinct p state values (representing p from 0 to num_states - 1).
            num_actions: Range parameter defining the discrete action adjustment interval [-num_actions, num_actions - 1].
            learning_rate: Learning rate alpha (paper default 0.1).
            discount_factor: Discount factor gamma (paper default 0.99).
            epsilon: Exploration probability epsilon (default 0.1).
            seed: Optional random seed for reproducible exploration.
            epsilon_decay_value: Amount to decay epsilon after each decision step.
            min_epsilon: Lower bound for decayed epsilon (default 0.01).
            include_cache_state: Whether to incorporate cache events (T1, T2, B1, B2, Miss)
                                 into the state representation per paper Section 3.4.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_p_states = max(1, int(num_states))
        self.num_actions = max(1, int(num_actions))
        self.include_cache_state = bool(include_cache_state)

        # action_space contains discrete signed delta_p adjustment values:
        # e.g., for num_actions = 10, action_space spans [-10, -9, ..., 0, ..., 9]
        self.action_space: List[int] = list(range(-self.num_actions, self.num_actions))

        # Total rows in Q-table:
        # If include_cache_state is True: num_p_states * 5 event states
        # If False: num_p_states (backward compatible with 1D parameter space)
        if self.include_cache_state:
            self.total_states = self.num_p_states * len(self.CACHE_EVENTS)
        else:
            self.total_states = self.num_p_states

        self.num_states = self.total_states  # Maintain backward compatibility attribute

        # Q-table dimensions: [total_states, len(action_space)]
        self.q_table = np.zeros([self.total_states, len(self.action_space)], dtype=float)

        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.epsilon_decay_value = float(epsilon_decay_value)
        self.min_epsilon = float(min_epsilon)
        self.rewards: List[int] = []

    def get_state_index(self, p: int, hit: str) -> int:
        """Map parameter p and cache event to a valid Q-table row index."""
        clamped_p = max(0, min(int(p), self.num_p_states - 1))
        if self.include_cache_state:
            event_idx = self.EVENT_MAP.get(hit, 4)
            return clamped_p * len(self.CACHE_EVENTS) + event_idx
        return clamped_p

    def get_next_state(
        self,
        current_p: int,
        current_t1_len: int,
        current_b1_len: int,
        current_b2_len: int,
        hit: str,
    ) -> int:
        """Select an action, update the Q-table, decay epsilon, and return the new p.

        Args:
            current_p: Current ARC target size parameter p.
            current_t1_len: Current length of active list T1.
            current_b1_len: Current length of ghost list B1.
            current_b2_len: Current length of ghost list B2.
            hit: Cache event identifier ('t1', 't2', 'b1', 'b2', 'Miss').

        Returns:
            new_p: Updated value of p, guaranteed to be within [0, num_p_states - 1].
        """
        clamped_p = max(0, min(int(current_p), self.num_p_states - 1))
        state_idx = self.get_state_index(clamped_p, hit)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_idx = random.randrange(len(self.action_space))
        else:
            action_idx = int(np.argmax(self.q_table[state_idx]))

        action_value = self.action_space[action_idx]

        # Calculate new ARC target parameter p
        new_p = max(0, min(clamped_p + action_value, self.num_p_states - 1))

        # Reward calculation strictly following paper Section 3.4
        hit_lower = str(hit).lower()
        if hit_lower in ("t1", "t2"):
            reward = 100
        elif hit_lower == "b1":
            reward = -10 if current_b1_len < current_b2_len else -1
        elif hit_lower == "b2":
            reward = -10 if current_b1_len > current_b2_len else -1
        elif hit_lower in ("miss", "none"):
            reward = -100
        else:
            reward = 0

        self.rewards.append(reward)

        # Estimate maximum future Q value from new_p
        if self.include_cache_state:
            base_row = new_p * len(self.CACHE_EVENTS)
            future_slice = self.q_table[base_row : base_row + len(self.CACHE_EVENTS), :]
            best_future_q = float(np.max(future_slice))
        else:
            best_future_q = float(np.max(self.q_table[new_p]))

        # Standard Bellman temporal difference update
        td_target = reward + self.discount_factor * best_future_q
        td_error = td_target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.learning_rate * td_error

        # Progression: Apply epsilon decay towards exploitation
        if self.epsilon_decay_value > 0.0 and self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_value)

        return new_p
