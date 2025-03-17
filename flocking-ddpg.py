import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import socket
import struct
import matplotlib.pyplot as plt

# Hyperparameters
MAX_MEMORY = 30000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0001
CRITIC_LR = 0.0002
NUM_DRONES = 3
STATE_DIM = 18
ACTION_DIM = 2
NUM_EPISODES = 1000
MAX_STEPS = 500

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300 + action_dim, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=MAX_MEMORY):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DDPG Agent
class FlockingDDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.replay_buffer = ReplayBuffer()
        self.soft_update(1.0)
        self.ou_noise = OUNoise(action_dim)  # Add OU noise
        self.value_losses = []  # To store value losses
        self.policy_losses = []  # To store policy losses

    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        if add_noise:
            noise = self.ou_noise.sample()
            action += noise
        action = np.clip(action, -1, 1)  # Clip action to valid range
        return action

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        # Sample batch
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + GAMMA * self.target_critic(next_states, next_actions)

        # Compute critic loss
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.value_losses.append(critic_loss.item())  # Log value loss

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.policy_losses.append(actor_loss.item())  # Log policy loss

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(TAU)

    def soft_update(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# Unity Socket Communication using our custom protocol
class UnitySocket:
    def __init__(self, host='127.0.0.1', port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print('Waiting for Unity connection...')
        self.conn, _ = self.sock.accept()
        print('Connected to Unity')


    def receive_state(self):
        data = b''
        # receive the states of all drones at once, reducing socket calls
        expected = NUM_DRONES * STATE_DIM * 4
        while len(data) < expected:
            packet = self.conn.recv(expected - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data += packet
        state = struct.unpack('f' * NUM_DRONES * STATE_DIM, data)
        print(f"Received States: {state}")
        return state

    def send_action(self, action):
        # send computed actions for all the drones together to Unity
        data = struct.pack('f' * NUM_DRONES * ACTION_DIM, *action)
        self.conn.sendall(data)

    def close(self):
        self.conn.close()
        self.sock.close()


# Reward Calculation
def calculate_reward(drone_state, drone_next_state):
    d1_t, d1_t1 = drone_state[3], drone_next_state[3]
    d2_t1, d3_t1 = drone_next_state[5], drone_next_state[7]
    min_obstacle_dist = min(drone_next_state[8:])
    transition_reward = np.tanh(0.2 * (10 - drone_next_state[1])) * (d1_t - d1_t1)
    mutual_reward = (3 * np.exp(0.05 * (d2_t1 - 20)) + 3 * np.exp(
        0.05 * (d3_t1 - 20))) if 10 <= d2_t1 <= 50 and 10 <= d3_t1 <= 50 else -5
    obstacle_penalty = -5 if min_obstacle_dist < 10 else 0
    step_penalty = -3
    reward = transition_reward + mutual_reward + obstacle_penalty + step_penalty
    return reward


# Function to plot losses
def plot_losses(value_losses, policy_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(value_losses, label="Value Loss (Critic)")
    plt.plot(policy_losses, label="Policy Loss (Actor)")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Value Loss and Policy Loss Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    agent = FlockingDDPG(STATE_DIM, ACTION_DIM)
    unity_socket = UnitySocket()

    for episode in range(NUM_EPISODES):
        print("Starting episode", episode)
        total_reward = 0
        for t in range(MAX_STEPS):
            print(f"Step: {t}")
            states = unity_socket.receive_state()
            # STATE_DIM+1 to account for episode info as well (whether done or not)
            # 1 means episode done, 0 means not done.

            states = [states[STATE_DIM*i:STATE_DIM*(i+1)] for i in range(NUM_DRONES)]

            actions_tmp = [agent.select_action(state, add_noise=True) for state in states]  # Add noise during training
            actions = []
            for row in actions_tmp:
                for action in row:
                    actions.append(action)
            unity_socket.send_action(actions)
            next_states = unity_socket.receive_state()
            next_states = [next_states[STATE_DIM*i:STATE_DIM*(i+1)] for i in range(NUM_DRONES)]

            over = False
            status = "[reached target]"
            # -1 => collision
            #  0 => going on
            #  1 => reached target
            for i in range(NUM_DRONES):
                if next_states[i][-1] != 0:
                    over = True
                    if next_states[i][-1] == -1:
                        status = "[collision]"
                reward = calculate_reward(states[i], next_states[i])
                total_reward += reward
                agent.replay_buffer.add((states[i], actions_tmp[i], reward, next_states[i]))
            agent.train()
            if over:
                print(f"Episode ended: {status}")
                break
        print(f"Episode {episode}, Total Reward: {total_reward}")
        print("========")

    unity_socket.close()

    # Plot losses after training
    plot_losses(agent.value_losses, agent.policy_losses)


if __name__ == "__main__":
    main()