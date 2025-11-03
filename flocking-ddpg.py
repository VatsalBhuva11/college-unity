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
STATE_DIM = 17  # Per-drone state dimension (without termination flag)
ACTION_DIM = 2
NUM_EPISODES = 1000
MAX_STEPS = 500
ATTENTION_DIM = 64  # Dimension for attention mechanism

# Actor Network (per drone - decentralized execution)
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


# Attention-based Critic Network (per drone - centralized training)
class AttentionCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, attention_dim=ATTENTION_DIM):
        super(AttentionCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.attention_dim = attention_dim
        
        # State encoder for self
        self.self_state_encoder = nn.Linear(state_dim, attention_dim)
        
        # State encoder for other agents
        self.other_state_encoder = nn.Linear(state_dim, attention_dim)
        
        # Attention mechanism
        self.attention_query = nn.Linear(attention_dim, attention_dim)
        self.attention_key = nn.Linear(attention_dim, attention_dim)
        self.attention_value = nn.Linear(attention_dim, attention_dim)
        
        # Action encoder
        self.action_encoder = nn.Linear(action_dim, attention_dim)
        
        # Final layers combining all information
        self.fc1 = nn.Linear(attention_dim * 2, 300)  # self + attended others
        self.fc2 = nn.Linear(300 + attention_dim, 400)  # + action
        self.fc3 = nn.Linear(400, 1)

    def forward(self, self_state, self_action, other_states):
        """
        Args:
            self_state: (batch_size, state_dim) - current drone's state
            self_action: (batch_size, action_dim) - current drone's action
            other_states: (batch_size, num_other_agents, state_dim) - other drones' states
        Returns:
            Q-value: (batch_size, 1)
        """
        batch_size = self_state.shape[0]
        
        # Encode self state
        self_encoded = torch.relu(self.self_state_encoder(self_state))  # (batch, attention_dim)
        
        # Encode other states
        num_others = other_states.shape[1]
        other_encoded = torch.relu(self.other_state_encoder(
            other_states.view(-1, self.state_dim)
        ))  # (batch * num_others, attention_dim)
        other_encoded = other_encoded.view(batch_size, num_others, self.attention_dim)
        
        # Compute attention
        query = self.attention_query(self_encoded).unsqueeze(1)  # (batch, 1, attention_dim)
        key = self.attention_key(other_encoded)  # (batch, num_others, attention_dim)
        value = self.attention_value(other_encoded)  # (batch, num_others, attention_dim)
        
        # Attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.attention_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, 1, num_others)
        
        # Apply attention
        attended_others = torch.bmm(attention_weights, value).squeeze(1)  # (batch, attention_dim)
        
        # Encode action
        action_encoded = torch.relu(self.action_encoder(self_action))  # (batch, attention_dim)
        
        # Combine self state and attended others
        combined_state = torch.cat([self_encoded, attended_others], dim=1)  # (batch, attention_dim * 2)
        
        # Process through network
        x = torch.relu(self.fc1(combined_state))
        x = torch.cat([x, action_encoded], dim=1)  # (batch, 300 + attention_dim)
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


# ATT-DDPG Agent (Attention-based DDPG with per-drone agents)
class FlockingATTDDPG:
    def __init__(self, state_dim, action_dim, num_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Create per-drone actor and critic networks
        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [AttentionCritic(state_dim, action_dim, num_agents - 1) for _ in range(num_agents)]
        self.target_critics = [AttentionCritic(state_dim, action_dim, num_agents - 1) for _ in range(num_agents)]
        
        # Create optimizers for each agent
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=ACTOR_LR) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=CRITIC_LR) for critic in self.critics]
        
        # Per-agent replay buffers
        self.replay_buffers = [ReplayBuffer() for _ in range(num_agents)]
        
        # Per-agent noise generators
        self.ou_noises = [OUNoise(action_dim) for _ in range(num_agents)]
        
        # Initialize target networks
        self.soft_update_all(1.0)
        
        # Loss tracking
        self.value_losses = [[] for _ in range(num_agents)]
        self.policy_losses = [[] for _ in range(num_agents)]

    def select_action(self, agent_id, state, add_noise=True):
        """Select action for a specific agent (decentralized execution)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actors[agent_id](state_tensor).detach().numpy()[0]
        if add_noise:
            noise = self.ou_noises[agent_id].sample()
            action += noise
        action = np.clip(action, -1, 1)  # Clip action to valid range
        return action

    def train(self, agent_id):
        """Train a specific agent (centralized training with attention)"""
        if self.replay_buffers[agent_id].size() < BATCH_SIZE:
            return
        
        # Sample batch for this agent
        batch = self.replay_buffers[agent_id].sample(BATCH_SIZE)
        # Unpack: (self_state, self_action, other_states, reward, next_self_state, next_other_states, done)
        (self_states, self_actions, other_states_list, rewards, 
         next_self_states, next_other_states_list, dones) = zip(*batch)
        
        # Convert to tensors
        self_states = torch.FloatTensor(np.array(self_states))
        self_actions = torch.FloatTensor(np.array(self_actions))
        other_states = torch.FloatTensor(np.array(other_states_list))  # (batch, num_others, state_dim)
        rewards = torch.FloatTensor(np.array(rewards)).reshape(-1, 1)
        next_self_states = torch.FloatTensor(np.array(next_self_states))
        next_other_states = torch.FloatTensor(np.array(next_other_states_list))  # (batch, num_others, state_dim)
        dones = torch.FloatTensor(np.array(dones)).reshape(-1, 1)
        
        # Compute target Q-value using target critic with attention
        with torch.no_grad():
            next_self_action = self.target_actors[agent_id](next_self_states)
            target_q = rewards + GAMMA * (1 - dones) * self.target_critics[agent_id](
                next_self_states, next_self_action, next_other_states
            )
        
        # Compute current Q-value
        current_q = self.critics[agent_id](self_states, self_actions, other_states)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.value_losses[agent_id].append(critic_loss.item())
        
        # Optimize critic
        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), max_norm=1.0)
        self.critic_optimizers[agent_id].step()
        
        # Compute actor loss
        # For actor, we use the critic with current other agents' states
        # In training, we use the actual other states from the batch
        actor_loss = -self.critics[agent_id](
            self_states, self.actors[agent_id](self_states), other_states
        ).mean()
        self.policy_losses[agent_id].append(actor_loss.item())
        
        # Optimize actor
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), max_norm=1.0)
        self.actor_optimizers[agent_id].step()
        
        # Soft update target networks
        self.soft_update_agent(agent_id, TAU)

    def soft_update_agent(self, agent_id, tau):
        """Soft update target networks for a specific agent"""
        for target_param, param in zip(self.target_actors[agent_id].parameters(), 
                                       self.actors[agent_id].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critics[agent_id].parameters(), 
                                       self.critics[agent_id].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def soft_update_all(self, tau):
        """Soft update all target networks"""
        for agent_id in range(self.num_agents):
            self.soft_update_agent(agent_id, tau)


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
        # Unity sends: NUM_DRONES * (STATE_DIM + 1) floats
        # +1 is for termination flag per drone
        expected = NUM_DRONES * (STATE_DIM + 1) * 4
        while len(data) < expected:
            packet = self.conn.recv(expected - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data += packet
        states_with_flags = struct.unpack('f' * NUM_DRONES * (STATE_DIM + 1), data)
        
        # Parse states and termination flags
        states = []
        termination_flags = []
        for i in range(NUM_DRONES):
            start_idx = i * (STATE_DIM + 1)
            end_idx = start_idx + STATE_DIM
            states.append(list(states_with_flags[start_idx:end_idx]))
            termination_flags.append(int(states_with_flags[end_idx]))
        
        return states, termination_flags

    def send_action(self, actions):
        # send computed actions for all the drones together to Unity
        # actions is a flat list: [action0_0, action0_1, action1_0, action1_1, ...]
        data = struct.pack('f' * NUM_DRONES * ACTION_DIM, *actions)
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
def plot_losses(agents):
    """Plot losses for all agents"""
    fig, axes = plt.subplots(2, NUM_DRONES, figsize=(5 * NUM_DRONES, 10))
    if NUM_DRONES == 1:
        axes = axes.reshape(2, 1)
    
    for agent_id in range(NUM_DRONES):
        if len(agents.value_losses[agent_id]) > 0:
            axes[0, agent_id].plot(agents.value_losses[agent_id], label=f"Drone {agent_id} Value Loss")
            axes[0, agent_id].set_xlabel("Training Steps")
            axes[0, agent_id].set_ylabel("Loss")
            axes[0, agent_id].set_title(f"Drone {agent_id} - Critic Loss")
            axes[0, agent_id].legend()
            axes[0, agent_id].grid()
        
        if len(agents.policy_losses[agent_id]) > 0:
            axes[1, agent_id].plot(agents.policy_losses[agent_id], label=f"Drone {agent_id} Policy Loss")
            axes[1, agent_id].set_xlabel("Training Steps")
            axes[1, agent_id].set_ylabel("Loss")
            axes[1, agent_id].set_title(f"Drone {agent_id} - Actor Loss")
            axes[1, agent_id].legend()
            axes[1, agent_id].grid()
    
    plt.tight_layout()
    plt.show()


def main():
    agents = FlockingATTDDPG(STATE_DIM, ACTION_DIM, NUM_DRONES)
    unity_socket = UnitySocket()

    for episode in range(NUM_EPISODES):
        print(f"========== Episode {episode} ==========")
        episode_rewards = [0 for _ in range(NUM_DRONES)]
        
        for t in range(MAX_STEPS):
            if t % 50 == 0:
                print(f"  Step: {t}")
            
            # Receive current states from Unity
            states, termination_flags = unity_socket.receive_state()
            
            # Check if episode is already terminated (from previous step)
            if any(flag != 0 for flag in termination_flags):
                print(f"Episode {episode} terminated (received terminal state)")
                break
            
            # Select actions for each drone (decentralized execution)
            actions = []
            actions_list = []
            for agent_id in range(NUM_DRONES):
                action = agents.select_action(agent_id, states[agent_id], add_noise=True)
                actions_list.append(action)
                actions.extend(action)
            
            # Send actions to Unity
            unity_socket.send_action(actions)
            
            # Receive next states (after action application and physics update)
            next_states, next_termination_flags = unity_socket.receive_state()
            
            # Determine if episode ended
            episode_done = any(flag != 0 for flag in next_termination_flags)
            
            # Calculate rewards and store transitions for each agent
            for agent_id in range(NUM_DRONES):
                # Get other agents' states (for centralized training)
                other_states = [states[j] for j in range(NUM_DRONES) if j != agent_id]
                next_other_states = [next_states[j] for j in range(NUM_DRONES) if j != agent_id]
                
                # Calculate reward for this agent
                reward = calculate_reward(states[agent_id], next_states[agent_id])
                episode_rewards[agent_id] += reward
                
                # Determine if this agent's episode is done
                done = 1.0 if next_termination_flags[agent_id] != 0 else 0.0
                
                # Store transition: (self_state, self_action, other_states, reward, next_self_state, next_other_states, done)
                agents.replay_buffers[agent_id].add((
                    states[agent_id],
                    actions_list[agent_id],
                    other_states,
                    reward,
                    next_states[agent_id],
                    next_other_states,
                    done
                ))
                
                # Train the agent (centralized training)
                agents.train(agent_id)
            
            if episode_done:
                # Determine termination status
                if any(flag == 1 for flag in next_termination_flags):
                    status = "✓ TARGET REACHED"
                else:
                    status = "✗ COLLISION"
                print(f"  Episode ended at step {t}: {status}")
                break
        
        total_reward = sum(episode_rewards)
        avg_reward = total_reward / NUM_DRONES
        print(f"Episode {episode} Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Per-drone rewards: {[f'{r:.2f}' for r in episode_rewards]}")
        print("=" * 50)

    unity_socket.close()

    # Plot losses after training
    plot_losses(agents)


if __name__ == "__main__":
    main()