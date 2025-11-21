import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import socket
import struct
import matplotlib.pyplot as plt
import os
import datetime

# Hyperparameters
MAX_MEMORY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01
ACTOR_LR = 0.001
CRITIC_LR = 0.001
NUM_DRONES = 3
STATE_DIM = 17  # Excludes reward and termination flag
FULL_STATE_DIM = 19  # Includes reward and termination flag (17+1+1)
ACTION_DIM = 2
HIDDEN_DIM = 64
NUM_EPISODES = 1000
MAX_STEPS = 500
MODEL_DIR = "saved_models"
LOG_FILE = "training_log.txt"

# Exploration settings
NOISE_START = 0.5
NOISE_END = 0.05
NOISE_DECAY_EPISODES = 700

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=MAX_MEMORY):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            np.array(obs),
            np.array(act),
            np.array(rew),
            np.array(next_obs),
            np.array(done)
        )

    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output in [-1, 1] for both dimensions
        # We interpret act[0] as steering [-1, 1]
        # We interpret act[1] as throttle raw [-1, 1] -> mapped to [0, 1] later
        return torch.tanh(self.fc3(x))

class AttentionCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(AttentionCritic, self).__init__()
        self.ego_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU()
        )
        self.other_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU()
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, ego_state, ego_action, others_states, others_actions):
        batch_size = ego_state.shape[0]
        ego_input = torch.cat([ego_state, ego_action], dim=1)
        ego_encoded = self.ego_encoder(ego_input)
        others_input = torch.cat([others_states, others_actions], dim=2)
        others_input_flat = others_input.view(-1, others_input.shape[2])
        others_encoded_flat = self.other_encoder(others_input_flat)
        others_encoded = others_encoded_flat.view(batch_size, -1, others_encoded_flat.shape[1])
        query = self.query(ego_encoded).unsqueeze(1)
        keys = self.key(others_encoded)
        values = self.value(others_encoded)
        scores = torch.bmm(query, keys.transpose(1, 2))
        weights = F.softmax(scores / np.sqrt(others_encoded.shape[-1]), dim=2)
        attention_output = torch.bmm(weights, values).squeeze(1)
        combined = torch.cat([ego_encoded, attention_output], dim=1)
        x = F.relu(self.fc1(combined))
        q_value = self.fc2(x)
        return q_value

class Agent:
    def __init__(self, state_dim, action_dim, num_drones):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        self.critic = AttentionCritic(state_dim, action_dim).to(device)
        self.target_critic = AttentionCritic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def save(self, path, index):
        torch.save(self.actor.state_dict(), os.path.join(path, f"agent_{index}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, f"agent_{index}_critic.pth"))

    def load(self, path, index):
        self.actor.load_state_dict(torch.load(os.path.join(path, f"agent_{index}_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, f"agent_{index}_critic.pth")))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

class ATT_MADDPG:
    def __init__(self, num_drones, state_dim, action_dim):
        self.num_drones = num_drones
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents = [Agent(state_dim, action_dim, num_drones) for _ in range(num_drones)]
        self.replay_buffer = ReplayBuffer()
        self.value_losses = []
        self.policy_losses = []

    def select_actions(self, states, noise_scale=0.0):
        """
        states: np.array shape (NUM_DRONES, state_dim)
        Returns: numpy array shape (NUM_DRONES, ACTION_DIM) with
        action[:,0] = steering_norm in [-1,1]
        action[:,1] = throttle_norm in [0,1]  (no negatives)
        """
        actions = []
        for i, agent in enumerate(self.agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(device)  # (1, S)
            raw_action = agent.actor(state).detach().cpu().numpy()[0]     # in [-1,1] per component
            
            # add exploration noise to the *raw* output (before throttle remap)
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, size=self.action_dim)
                raw_action += noise
                
            # clip raw to [-1,1] to keep stable
            raw_action = np.clip(raw_action, -1.0, 1.0)
            
            # Map throttle (assume index 1 is throttle) from [-1,1] -> [0,1]
            steering = float(raw_action[0])                  # keep in [-1,1]
            throttle_norm = float((raw_action[1] + 1.0) / 2.0)  # map to [0,1]
            throttle_norm = np.clip(throttle_norm, 0.0, 1.0)    # ensure non-negative
            actions.append([steering, throttle_norm])
        return np.array(actions, dtype=np.float32)

    def update(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        obs, act, rew, next_obs, done = self.replay_buffer.sample(BATCH_SIZE)
        obs = torch.FloatTensor(obs).to(device)
        act = torch.FloatTensor(act).to(device)
        rew = torch.FloatTensor(rew).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        done = torch.FloatTensor(done).to(device)

        for i, agent in enumerate(self.agents):
            next_actions = []
            for j, other_agent in enumerate(self.agents):
                # Get target action
                raw_next_action = other_agent.target_actor(next_obs[:, j, :])
                # We must apply the same transformation to target actions as we do to real actions
                # However, the critic expects the raw output or the transformed one? 
                # Typically DDPG critic takes the output of the actor directly.
                # But if our environment sees [0,1], and actor outputs [-1,1], there's a mismatch if we don't transform.
                # SIMPLIFICATION: Let's feed the raw [-1,1] actions to the critic to avoid complex graph logic,
                # BUT we must remember that 'act' from buffer is ALREADY transformed to [0,1] for throttle by select_actions.
                # This is a problem. The buffer stores what was sent to Unity ([0,1] throttle).
                # The actor outputs [-1,1].
                # We should store the RAW actions in the buffer or transform actor output here.
                # Let's transform the actor output here to match the buffer distribution.
                
                # Transform raw [-1,1] -> steering [-1,1], throttle [0,1]
                # Steering is index 0, Throttle is index 1
                steering = raw_next_action[:, 0:1]
                throttle_raw = raw_next_action[:, 1:2]
                throttle = (throttle_raw + 1.0) / 2.0
                next_actions.append(torch.cat([steering, throttle], dim=1))
                
            next_actions = torch.stack(next_actions, dim=1)
            
            target_ego_state = next_obs[:, i, :]
            target_ego_action = next_actions[:, i, :]
            target_others_states = torch.cat([next_obs[:, :i, :], next_obs[:, i+1:, :]], dim=1)
            target_others_actions = torch.cat([next_actions[:, :i, :], next_actions[:, i+1:, :]], dim=1)
            
            with torch.no_grad():
                target_q = agent.target_critic(target_ego_state, target_ego_action, target_others_states, target_others_actions)
                y = rew[:, i].unsqueeze(1) + GAMMA * target_q * (1 - done[:, i].unsqueeze(1))
            
            ego_state = obs[:, i, :]
            ego_action = act[:, i, :] # This is from buffer, so it is already [steering, throttle_01]
            others_states = torch.cat([obs[:, :i, :], obs[:, i+1:, :]], dim=1)
            others_actions = torch.cat([act[:, :i, :], act[:, i+1:, :]], dim=1)
            
            current_q = agent.critic(ego_state, ego_action, others_states, others_actions)
            critic_loss = F.mse_loss(current_q, y)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()
            self.value_losses.append(critic_loss.item())

            # Actor Update
            raw_curr_pol = agent.actor(ego_state)
            # Apply transformation to match critic expectation
            steering_curr = raw_curr_pol[:, 0:1]
            throttle_curr = (raw_curr_pol[:, 1:2] + 1.0) / 2.0
            curr_pol_out = torch.cat([steering_curr, throttle_curr], dim=1)
            
            actor_loss = -agent.critic(ego_state, curr_pol_out, others_states, others_actions).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_optimizer.step()
            self.policy_losses.append(actor_loss.item())

            for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                target_param.data.copy_(tau_update(target_param.data, param.data, TAU))
            for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                target_param.data.copy_(tau_update(target_param.data, param.data, TAU))

    def save_model(self):
        for i, agent in enumerate(self.agents):
            agent.save(MODEL_DIR, i)
        print("Model saved successfully.")

    def load_model(self):
        for i, agent in enumerate(self.agents):
            try:
                agent.load(MODEL_DIR, i)
            except FileNotFoundError:
                print(f"No saved model found for agent {i}, starting from scratch.")
                return False
        print("Model loaded successfully.")
        return True

def tau_update(target, source, tau):
    return target * (1 - tau) + source * tau

class UnitySocket:
    def __init__(self, host='127.0.0.1', port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print('Waiting for Unity connection...')
        self.conn, _ = self.sock.accept()
        print('Connected to Unity')

    def receive_state(self):
        data = b''
        expected = NUM_DRONES * FULL_STATE_DIM * 4
        while len(data) < expected:
            packet = self.conn.recv(expected - len(data))
            if not packet:
                return None, None, None
            data += packet
        fmt_recv = '<' + 'f' * (NUM_DRONES * FULL_STATE_DIM)
        state_flat = struct.unpack(fmt_recv, data)
        all_data = np.array(state_flat).reshape(NUM_DRONES, FULL_STATE_DIM)
        states = all_data[:, :STATE_DIM]
        rewards = all_data[:, STATE_DIM]
        flags = all_data[:, STATE_DIM+1]
        return states, rewards, flags

    def send_action(self, actions):
        actions_flat = actions.flatten().tolist()
        fmt_send = '<' + 'f' * len(actions_flat)
        data = struct.pack(fmt_send, *actions_flat)
        self.conn.sendall(data)

    def close(self):
        if self.conn:
            self.conn.close()
        self.sock.close()

def log_training(episode, steps, total_reward, status, states_info=None):
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Episode: {episode}, Steps: {steps}, Total Reward: {total_reward:.2f}, Status: {status}\n")
        if states_info:
             f.write(f"  State Snapshot (Agent 0): {states_info}\n")

def format_state(state):
    # state: [17]
    # 0: yaw, 1: vel, 2: angle_target, 3: dist_target, 4: ang_n1, 5: dist_n1, 6: ang_n2, 7: dist_n2, 8-16: sensors
    return (f"Yaw:{state[0]:.1f}, Vel:{state[1]:.2f}, AngTarg:{state[2]:.1f}, DistTarg:{state[3]:.1f}, "
            f"DistN1:{state[5]:.1f}, DistN2:{state[7]:.1f}, MinObs:{np.min(state[8:]):.1f}")

def main():
    maddpg = ATT_MADDPG(NUM_DRONES, STATE_DIM, ACTION_DIM)
    unity = UnitySocket()
    
    # Check if we want to load a model
    load_existing = input("Load existing model? (y/n): ").lower() == 'y'
    if load_existing:
        success = maddpg.load_model()
        if success:
            print("Model loaded. Running in inference mode (no training/noise).")
    
    training_mode = not load_existing

    try:
        states, _, _ = unity.receive_state()
        
        for episode in range(NUM_EPISODES):
            print(f"--- Episode {episode} ---")
            
            if states is None: break
            
            total_reward = 0
            step = 0
            episode_over = False
            status_code = 0
            
            # Calculate noise scale for this episode (linear decay)
            if training_mode:
                noise_scale = NOISE_START - (episode / NOISE_DECAY_EPISODES) * (NOISE_START - NOISE_END)
                noise_scale = max(NOISE_END, noise_scale)
            else:
                noise_scale = 0.0
            
            while step < MAX_STEPS:
                # Select actions with decaying noise
                actions = maddpg.select_actions(states, noise_scale=noise_scale)
                
                unity.send_action(actions)
                next_states, rewards, flags = unity.receive_state()
                
                if next_states is None: 
                    states = None
                    break
                
                dones = [1 if f != 0 else 0 for f in flags]
                if any(dones):
                    episode_over = True
                    status_code = int(max(flags)) # 1=Target, 2=Collision

                # Only train if in training mode
                if training_mode:
                    maddpg.replay_buffer.add(states, actions, rewards, next_states, dones)
                    maddpg.update()
                
                states = next_states
                total_reward += sum(rewards)
                step += 1
                
                # Detailed logging every 50 steps or end of episode
                if step % 50 == 0 or episode_over:
                    print(f"--- Step {step} ---")
                    for d_i in range(NUM_DRONES):
                        s_info = format_state(states[d_i])
                        print(f"  Drone {d_i}: Rew:{rewards[d_i]:.2f}, {s_info}, Act:{actions[d_i]}")
                
                if episode_over:
                    status_str = "Target Reached" if status_code == 1 else "Collision"
                    print(f"Episode ended. Status: {status_str}")
                    # Log detailed state of agent 0 at end
                    log_training(episode, step, total_reward, status_str, format_state(states[0]))
                    states, _, _ = unity.receive_state()
                    break
            
            print(f"Total Reward: {total_reward:.2f}")
            
            # Save model periodically
            if training_mode and episode % 10 == 0:
                maddpg.save_model()
            
    except KeyboardInterrupt:
        print("Interrupted.")
        if training_mode:
             maddpg.save_model()
    finally:
        unity.close()
        
    # Plot losses
    if training_mode and len(maddpg.value_losses) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(maddpg.value_losses, label="Critic Loss")
        plt.plot(maddpg.policy_losses, label="Actor Loss")
        plt.legend()
        plt.savefig('att_maddpg_losses.png')
        print("Saved loss plot to att_maddpg_losses.png")

if __name__ == "__main__":
    main()
