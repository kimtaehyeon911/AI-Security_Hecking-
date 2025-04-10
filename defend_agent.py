import random
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# 강화학습 환경 정의 (방어)
# ======================
class PacketDefenseEnv(gym.Env):
    def __init__(self):
        super(PacketDefenseEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: 알림 후 로그 저장, 1: 패킷 드랍, 2: 정상으로 간주

    def reset(self):
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        # 실제로는 패킷 분석 기반이지만 지금은 랜덤 시뮬레이션
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

        # 위협 판단을 확률적으로 하도록 설계
        malicious_score = random.random()  # 0~1 위협 점수 (실제로는 모델 추론값)

        if action == 0:  # 알림+로그 저장
            reward = 1 if malicious_score > 0.7 else -0.5
        elif action == 1:  # 패킷 드랍
            reward = 1.5 if malicious_score > 0.8 else -1
        elif action == 2:  # 정상으로 간주 (패스)
            reward = 1 if malicious_score < 0.3 else -1.5

        obs = self._get_state()
        if self.step_count > 30:
            done = True
        return obs, reward, done, {}

# ======================
# 정책 신경망 정의
# ======================
class DefensePolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DefensePolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# ======================
# 학습 루프 (REINFORCE)
# ======================
def train():
    env = PacketDefenseEnv()
    policy_net = DefensePolicyNet(7, env.action_space.n)
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

        print(f"[Defense Episode {episode}] Total Reward: {sum(rewards):.2f}")

if __name__ == '__main__':
    train()
