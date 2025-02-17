import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import socket
import struct

# Hyperparameters
MAX_MEMORY = 30000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0001
CRITIC_LR = 0.0002
NUM_DRONES = 3
STATE_DIM = 17
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

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        return action
    
    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert lists of numpy arrays to single numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + GAMMA * self.target_critic(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(TAU)
    
    def soft_update(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Unity Socket Communication
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
        while len(data) < STATE_DIM * 4:
            packet = self.conn.recv(STATE_DIM * 4 - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data += packet
        return struct.unpack('f' * STATE_DIM, data)
        
    def send_action(self, action):
        data = struct.pack('f' * ACTION_DIM, *action)
        self.conn.sendall(data)
    
    def check_termination_signal(self):
        if self.conn.recv(1, socket.MSG_PEEK) == b'\xFF':
            self.conn.recv(1)
            return True
        return False
    
    def close(self):
        self.conn.close()
        self.sock.close()


# Reward Calculation
def calculate_reward(drone_state, drone_next_state):
    reward = 0
    # Example reward components for each drone
    d1_t, d1_t1 = drone_state[3], drone_next_state[3]  # Distance to target
    d2_t1, d3_t1 = drone_next_state[5], drone_next_state[7]  # Distances to neighbors
    min_obstacle_dist = min(drone_next_state[8:])  # Closest obstacle distance
    
    transition_reward = np.tanh(0.2 * (10 - drone_next_state[1])) * (d1_t - d1_t1)
    mutual_reward = (3 * np.exp(0.05 * (d2_t1 - 20)) + 3 * np.exp(0.05 * (d3_t1 - 20))) if 10 <= d2_t1 <= 50 and 10 <= d3_t1 <= 50 else -5
    obstacle_penalty = -5 if min_obstacle_dist < 10 else 0
    step_penalty = -3
    
    reward = transition_reward + mutual_reward + obstacle_penalty + step_penalty
    
    return reward # Average reward across all drones

def main():
    agent = FlockingDDPG(STATE_DIM, ACTION_DIM)
    unity_socket = UnitySocket()
    
    for episode in range(NUM_EPISODES):
        print("Starting episode", episode)
        total_reward = 0
        for t in range(MAX_STEPS):
            states, actions, next_states, rewards = [], [], [], []
            for drone_id in range(NUM_DRONES):
                state = unity_socket.receive_state()
                action = agent.select_action(state)
                print(f"({action})")
                unity_socket.send_action(action)
                next_state = unity_socket.receive_state()
                reward = calculate_reward(state, next_state)
                total_reward += reward
                # print("moving drone", drone_id)
                agent.replay_buffer.add((state, action, reward, next_state))
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                
            agent.train()

            # Check for termination signal from Unity
            if unity_socket.check_termination_signal():
                print("Episode terminated early")
                break
        
        print(f"Episode {episode}, Total Reward: {total_reward}")
    
    unity_socket.close()

if __name__ == "__main__":
    main()
