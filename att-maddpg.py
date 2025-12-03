#!/usr/bin/env python3
"""
ATT-MADDPG trainer adapted from test.py for Unity communication.
Added: per-episode reward tracking and plotting (moving average).
Protocol:
  - Unity -> Python: for each drone: [19 floats state] + [1 float reward] + [1 float flag]
    -> total floats per step = NUM_DRONES * 21
  - Python -> Unity: actions flattened: NUM_DRONES * ACTION_DIM floats
  - Reset signal: Python sends float -99.0 as first float of action packet (length NUM_DRONES * ACTION_DIM).
"""
import os
import socket
import struct
import datetime
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# ----------------------------
# Hyperparameters (tuneable)
# ----------------------------
MAX_MEMORY = 100000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.01
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
NUM_DRONES = 3
STATE_DIM = 19   # per test.py / Unity state (excludes reward and flag)
FULL_STATE_DIM = STATE_DIM + 2  # includes reward and flag
ACTION_DIM = 2
HIDDEN_DIM = 64
NUM_EPISODES = 1000
MAX_STEPS = 500
MODEL_DIR = "saved_models_att"
LOG_FILE = "training_log_att.txt"

WARMUP_EPISODES = 50
NOISE_START = 1.0
NOISE_END = 0.1
NOISE_DECAY_EPISODES = 500

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Utility: tau update
# ----------------------------
def tau_update(target, source, tau):
    return target * (1 - tau) + source * tau

# ----------------------------
# Replay buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, max_size=MAX_MEMORY):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, act, rew, next_obs, done):
        # obs: (NUM_DRONES, STATE_DIM)
        self.buffer.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),      # (B, N, S)
            np.array(act, dtype=np.float32),      # (B, N, A)
            np.array(rew, dtype=np.float32),      # (B, N)
            np.array(next_obs, dtype=np.float32), # (B, N, S)
            np.array(done, dtype=np.float32)      # (B, N)
        )

    def size(self):
        return len(self.buffer)

# ----------------------------
# OU Noise for exploration
# ----------------------------
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# ----------------------------
# Networks
# ----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # outputs in [-1,1]

class AttentionCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(AttentionCritic, self).__init__()
        # ego encoder and other encoder operate on (state + action)
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
        # ego_state: (B, S)
        # ego_action: (B, A)
        # others_states: (B, N-1, S)
        # others_actions: (B, N-1, A)
        batch_size = ego_state.shape[0]
        ego_input = torch.cat([ego_state, ego_action], dim=1)  # (B, S+A)
        ego_encoded = self.ego_encoder(ego_input)            # (B, H)

        # flatten others for encoding
        others_input = torch.cat([others_states, others_actions], dim=2)  # (B, N-1, S+A)
        others_flat = others_input.view(-1, others_input.shape[2])       # ((B*(N-1)), S+A)
        others_encoded_flat = self.other_encoder(others_flat)            # ((B*(N-1)), H)
        others_encoded = others_encoded_flat.view(batch_size, -1, others_encoded_flat.shape[1])  # (B, N-1, H)

        query = self.query(ego_encoded).unsqueeze(1)   # (B,1,H)
        keys = self.key(others_encoded)               # (B, N-1, H)
        values = self.value(others_encoded)           # (B, N-1, H)

        # attention scores & output
        scores = torch.bmm(query, keys.transpose(1, 2))  # (B,1,N-1)
        dim_k = others_encoded.shape[-1]
        weights = F.softmax(scores / np.sqrt(dim_k), dim=2)  # (B,1,N-1)
        attention_output = torch.bmm(weights, values).squeeze(1)  # (B,H)

        combined = torch.cat([ego_encoded, attention_output], dim=1)  # (B, 2H)
        x = F.relu(self.fc1(combined))
        q_value = self.fc2(x)
        return q_value

# ----------------------------
# Agent container (actor + critic + target nets + optimizers)
# ----------------------------
class Agent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = AttentionCritic(state_dim, action_dim).to(device)
        self.target_critic = AttentionCritic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.noise = OUNoise(action_dim)

    def reset_noise(self):
        self.noise.reset()

    def save(self, path, index):
        torch.save(self.actor.state_dict(), os.path.join(path, f"agent_{index}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, f"agent_{index}_critic.pth"))

    def load(self, path, index):
        self.actor.load_state_dict(torch.load(os.path.join(path, f"agent_{index}_actor.pth"), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(path, f"agent_{index}_critic.pth"), map_location=device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

# ----------------------------
# ATT-MADDPG manager
# ----------------------------
class ATT_MADDPG:
    def __init__(self, num_drones, state_dim, action_dim):
        self.num_drones = num_drones
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents = [Agent(state_dim, action_dim) for _ in range(num_drones)]
        self.replay_buffer = ReplayBuffer()
        self.value_losses = []
        self.policy_losses = []
        # reward tracking per episode
        self.episode_rewards = []

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def select_actions(self, states, noise_scale=0.0):
        # states: (N, S) numpy
        actions = []
        for i, agent in enumerate(self.agents):
            s = torch.FloatTensor(states[i]).unsqueeze(0).to(device)  # (1,S)
            raw = agent.actor(s).detach().cpu().numpy()[0]  # (A,)
            if noise_scale > 0:
                raw += agent.noise.sample() * noise_scale
            actions.append(np.clip(raw, -1.0, 1.0))
        return np.array(actions, dtype=np.float32)

    def update(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        obs, act, rew, next_obs, done = self.replay_buffer.sample(BATCH_SIZE)
        # Convert to tensors:
        obs = torch.FloatTensor(obs).to(device)           # (B, N, S)
        act = torch.FloatTensor(act).to(device)           # (B, N, A)
        rew = torch.FloatTensor(rew).to(device)           # (B, N)
        next_obs = torch.FloatTensor(next_obs).to(device) # (B, N, S)
        done = torch.FloatTensor(done).to(device)         # (B, N)

        B = obs.shape[0]
        N = self.num_drones

        for i, agent in enumerate(self.agents):
            # Build next actions for all agents using target actors
            next_actions = []
            for j, other_agent in enumerate(self.agents):
                next_a = other_agent.target_actor(next_obs[:, j, :])  # (B, A)
                next_actions.append(next_a)
            next_actions = torch.stack(next_actions, dim=1)  # (B, N, A)

            target_ego_state = next_obs[:, i, :]             # (B, S)
            target_ego_action = next_actions[:, i, :]        # (B, A)
            # others
            target_others_states = torch.cat([next_obs[:, :i, :], next_obs[:, i+1:, :]], dim=1)   # (B, N-1, S)
            target_others_actions = torch.cat([next_actions[:, :i, :], next_actions[:, i+1:, :]], dim=1) # (B, N-1, A)

            with torch.no_grad():
                target_q = agent.target_critic(target_ego_state, target_ego_action, target_others_states, target_others_actions)
                y = rew[:, i].unsqueeze(1) + GAMMA * target_q * (1 - done[:, i].unsqueeze(1))

            # Current Q
            ego_state = obs[:, i, :]
            ego_action = act[:, i, :]
            others_states = torch.cat([obs[:, :i, :], obs[:, i+1:, :]], dim=1)
            others_actions = torch.cat([act[:, :i, :], act[:, i+1:, :]], dim=1)

            current_q = agent.critic(ego_state, ego_action, others_states, others_actions)
            critic_loss = F.mse_loss(current_q, y)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()
            self.value_losses.append(critic_loss.item())

            # Actor update
            raw_act = agent.actor(ego_state)
            actor_loss = -agent.critic(ego_state, raw_act, others_states, others_actions).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_optimizer.step()
            self.policy_losses.append(actor_loss.item())

            # Soft update targets
            for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                target_param.data.copy_(tau_update(target_param.data, param.data, TAU))
            for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                target_param.data.copy_(tau_update(target_param.data, param.data, TAU))

    def save_model(self):
        for i, agent in enumerate(self.agents):
            agent.save(MODEL_DIR, i)
        print("Saved agents to", MODEL_DIR)

    def load_model(self):
        for i, agent in enumerate(self.agents):
            try:
                agent.load(MODEL_DIR, i)
            except Exception as e:
                print(f"Failed loading agent {i}: {e}")
                return False
        print("Loaded models from", MODEL_DIR)
        return True

# ----------------------------
# Unity socket server
# ----------------------------
class UnitySocket:
    def __init__(self, host='127.0.0.1', port=5555, num_drones=NUM_DRONES):
        self.num_drones = num_drones
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # allow quick restart
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print("Waiting for Unity connection on port", port, "...")
        self.conn, _ = self.sock.accept()
        print("Unity connected.")

    def receive_state(self):
        # Expect num_drones * FULL_STATE_DIM floats (each float = 4 bytes)
        expected = self.num_drones * FULL_STATE_DIM * 4
        data = b''
        while len(data) < expected:
            packet = self.conn.recv(expected - len(data))
            if not packet:
                return None, None, None
            data += packet
        # unpack as little-endian floats
        fmt = '<' + 'f' * (self.num_drones * FULL_STATE_DIM)
        flat = struct.unpack(fmt, data)
        arr = np.array(flat, dtype=np.float32).reshape(self.num_drones, FULL_STATE_DIM)
        states = arr[:, :STATE_DIM]      # (N, S)
        rewards = arr[:, STATE_DIM]      # (N,)
        flags = arr[:, STATE_DIM + 1]    # (N,)
        return states, rewards, flags

    def send_action(self, actions):
        # actions shape (N, A)
        flat = actions.flatten().tolist()
        fmt = '<' + 'f' * len(flat)
        data = struct.pack(fmt, *flat)
        self.conn.sendall(data)

    def send_reset(self):
        reset_actions = [-99.0] * (self.num_drones * ACTION_DIM)
        fmt = '<' + 'f' * len(reset_actions)
        data = struct.pack(fmt, *reset_actions)
        self.conn.sendall(data)

    def close(self):
        try:
            self.conn.close()
        except:
            pass
        try:
            self.sock.close()
        except:
            pass

# ----------------------------
# Logging and plotting
# ----------------------------
def log_training(episode, steps, total_reward, status, state_snapshot=None):
    with open(LOG_FILE, "a") as f:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] Episode:{episode}, Steps:{steps}, TotalReward:{total_reward:.2f}, Status:{status}\n")
        if state_snapshot is not None:
            f.write(f"  Snapshot (agent0): {state_snapshot}\n")

def save_loss_plot(maddpg, fname="att_maddpg_losses.png"):
    if len(maddpg.value_losses) == 0:
        return
    plt.figure(figsize=(10,5))
    plt.plot(maddpg.value_losses, label="critic")
    plt.plot(maddpg.policy_losses, label="actor")
    plt.legend()
    plt.xlabel("Training updates")
    plt.ylabel("Loss")
    plt.title("ATT-MADDPG losses")
    plt.savefig(fname)
    plt.close()

def save_reward_plot(episode_rewards, fname="att_maddpg_rewards.png", ma_window=10):
    if len(episode_rewards) == 0:
        return
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, label="Episode Reward")
    if len(episode_rewards) >= ma_window:
        cumsum = np.cumsum(np.insert(np.array(episode_rewards), 0, 0))
        ma = (cumsum[ma_window:] - cumsum[:-ma_window]) / float(ma_window)
        # align moving average to right position
        ma_x = list(range(ma_window - 1, ma_window - 1 + len(ma)))
        plt.plot(ma_x, ma, label=f"{ma_window}-ep MA")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("ATT-MADDPG Episode Rewards")
    plt.savefig(fname)
    plt.close()

def format_state(state):
    # Provide human readable snapshot (same mapping as Unity)
    return (f"Yaw:{state[0]:.1f},Vel:{state[1]:.2f},AngT:{state[2]:.1f},DistT:{state[3]:.1f},"
            f"DistN1:{state[5]:.1f},AlignN1:{state[17]:.2f},MinObs:{np.min(state[8:17]):.1f}")

# ----------------------------
# Main training loop
# ----------------------------
def main():
    maddpg = ATT_MADDPG(NUM_DRONES, STATE_DIM, ACTION_DIM)
    unity = UnitySocket()

    load_existing = input("Load existing model? (y/n): ").strip().lower() == 'y'
    if load_existing:
        success = maddpg.load_model()
        if success:
            print("Loaded models; running in inference (no training).")
    training_mode = not load_existing

    # For reward plotting:
    episode_rewards = maddpg.episode_rewards

    try:
        states, _, _ = unity.receive_state()
        if states is None:
            print("Unity did not send initial state. Exiting.")
            return

        for episode in range(NUM_EPISODES):
            print(f"=== Episode {episode} ===")
            if training_mode:
                maddpg.reset_noise()

            total_reward = 0.0
            step = 0
            episode_over = False
            status_code = 0

            # linear noise schedule
            if training_mode:
                noise_scale = NOISE_START - (episode / float(NOISE_DECAY_EPISODES)) * (NOISE_START - NOISE_END)
                noise_scale = max(NOISE_END, noise_scale)
            else:
                noise_scale = 0.0

            while step < MAX_STEPS:
                current_noise = noise_scale
                if training_mode and episode < WARMUP_EPISODES:
                    current_noise = 1.5

                actions = maddpg.select_actions(states, noise_scale=current_noise)  # (N,A)
                unity.send_action(actions)

                next_states, rewards, flags = unity.receive_state()
                if next_states is None:
                    print("Unity disconnected during episode.")
                    states = None
                    break

                dones = [1 if f != 0 else 0 for f in flags]
                if all(dones):
                    episode_over = True
                    unique_flags = set(flags.tolist())
                    if unique_flags == {1.0}:
                        status_code = 1
                    elif unique_flags == {2.0}:
                        status_code = 2
                    else:
                        status_code = 3

                if training_mode:
                    maddpg.replay_buffer.add(states, actions, rewards, next_states, dones)
                    maddpg.update()

                states = next_states
                total_reward += float(sum(rewards))
                step += 1

                if step % 50 == 0 or episode_over:
                    print(f"Step {step}: total_reward so far {total_reward:.2f}")
                    for d in range(NUM_DRONES):
                        print(f"  Drone {d} reward {rewards[d]:.2f}, state: {format_state(states[d])}, action: {actions[d]}")

                if episode_over:
                    if status_code == 1:
                        status = "All Targets Reached"
                    elif status_code == 2:
                        status = "All Collided"
                    else:
                        status = "Mixed Outcomes"
                    print("Episode ended: ", status)
                    log_training(episode, step, total_reward, status, format_state(states[0]))
                    # Receive next initial state (Unity sends initial after reset)
                    states, _, _ = unity.receive_state()
                    break

            if not episode_over:
                print("Episode ended by timeout.")
                log_training(episode, step, total_reward, "Timeout", format_state(states[0]) if states is not None else None)
                unity.send_reset()
                states, _, _ = unity.receive_state()

            print(f"Episode {episode} total_reward={total_reward:.2f}")
            # record reward
            episode_rewards.append(total_reward)

            if training_mode and episode % 10 == 0:
                maddpg.save_model()
                save_loss_plot(maddpg)
                save_reward_plot(episode_rewards)

    except KeyboardInterrupt:
        print("Interrupted by user.")
        if training_mode:
            maddpg.save_model()
            save_reward_plot(episode_rewards)
    finally:
        unity.close()

    if training_mode:
        save_loss_plot(maddpg)
        save_reward_plot(episode_rewards)
        print("Training finished. Loss & reward plots saved.")

if __name__ == "__main__":
    main()
