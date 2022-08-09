import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

# fmt: off
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_EVERY = 2        # number of steps between updates
NB_LEARN = 3            # number of steps to learn with current trajectory
# fmt: on

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, index, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            index (int): index of agent
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.index = index = torch.tensor([index]).to(device)
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            num_agents * state_size, num_agents * action_size, random_seed
        ).to(device)
        self.critic_target = Critic(
            num_agents * state_size, num_agents * action_size, random_seed
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, actions_target, actions_pred):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            actions_target:
            actions_pred:
        """
        states, actions, rewards, next_states, dones = experiences
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        # ---------------------------- update critic ----------------------------
        actions_target = torch.cat(actions_target, dim=1).to(device)

        Q_targets_next = self.critic_target(
            next_states.reshape(next_states.shape[0], -1),
            actions_target.reshape(next_states.shape[0], -1),
        )

        # Compute Q targets for current states (y_i)
        Q_targets = rewards.index_select(1, self.index).squeeze(1) + (
            gamma * Q_targets_next * (1 - dones.index_select(1, self.index).squeeze(1))
        )

        # Compute critic loss
        Q_expected = self.critic_local(
            states.reshape(states.shape[0], -1), actions.reshape(actions.shape[0], -1)
        )

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = torch.cat(actions_pred, dim=1).to(device)

        actor_loss = -self.critic_local(
            states.reshape(states.shape[0], -1),
            actions_pred.reshape(actions_pred.shape[0], -1),
        ).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class MultiAgents:
    """Create multiple agents that interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Initialize agents
        self.agents = [
            Agent(state_size, action_size, i, num_agents, random_seed)
            for i in range(num_agents)
        ]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.time_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)

        self.time_step = (self.time_step + 1) % UPDATE_EVERY

        if len(self.memory) > BATCH_SIZE and self.time_step == 0:
            for i in range(NB_LEARN):
                for agent in self.agents:
                    # Learn from random sample
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)

                    # update target networks
                    agent.soft_update(agent.critic_local, agent.critic_target, TAU)
                    agent.soft_update(agent.actor_local, agent.actor_target, TAU)

    def learn(self, experiences, agent, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions_target = [
            agent_i.actor_target(
                states.index_select(1, torch.tensor([i]).to(device)).squeeze(1)
            )
            for i, agent_i in enumerate(self.agents)
        ]

        agent_action_pred = agent.actor_local(
            states.index_select(1, agent.index).squeeze(1)
        )
        actions_pred = [
            agent_action_pred
            if i == agent.index.numpy()[0]
            else actions.index_select(1, torch.tensor([i]).to(device)).squeeze(1)
            for i, agent_i in enumerate(self.agents)
        ]

        agent.learn(experiences, gamma, actions_target, actions_pred)

    def act(self, states, add_noise=True):
        actions = [
            np.squeeze(agent.act(np.expand_dims(state, axis=0), add_noise), axis=0)
            for agent, state in zip(self.agents, states)
        ]

        return np.stack(actions)

    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def save_weights(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'agent{i}_checkpoint__actor.pth')
            torch.save(agent.critic_local.state_dict(), f'agent{i}_checkpoint__critic.pth')    


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(
            self.size
        )
        self.state = x + dx

        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.stack([e.action for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.stack([e.reward for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.stack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
