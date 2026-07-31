### README.md

# AI Adversarial Cyber Range

> PyTorch 기반 공격/방어 AI 모델이 동일한 K8S 환경 내에서 TCP/UDP 패킷을 통해 상호작용하며 자율적으로 학습하는 사이버 보안 강화 학습 프로젝트입니다.

## 주요 구성

- **강화학습 에이전트**: 공격 AI 및 방어 AI 모두 독립적인 정책 네트워크를 가짐
- **환경**: 실제 TCP/UDP 네트워크 환경을 Scapy로 시뮬레이션
- **상태(State)**: 실시간 패킷 피쳐 추출 (포트, 프로토콜, 길이, 플래그, 엔트로피 등)
- **행동(Action)**: 랜덤/학습된 방식으로 공격 또는 방어 행동 결정
- **보상(Reward)**: 탐지/비탐지/성공 여부에 따라 리워드 부여

## 시스템 구성도
```
[강화학습 에이전트]
     ↑            ↓
[Packet Featurizer (state)] ← scapy, tcpdump
     ↓
[PolicyNet (PyTorch)] → 선택된 action
     ↓
[Packet Generator (action 실행기)]
     ↓
[보안 VM 응답 → reward 계산기 → 다음 상태]
```

## 설치
```bash
pip install torch torchvision torchaudio
pip install scapy gym
```

## 사용법
```bash
python attack_agent.py  # 공격 에이전트 실행
python defend_agent.py  # 방어 에이전트 실행 (개발 중)
```

## 향후 개발
- [ ] 방어 AI PolicyNet 개발
- [ ] 패킷 로그 기반 학습 강화
- [ ] K8S 상에서 자가 재학습 자동화

## 라이선스
MIT

---

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*.pyo

# Jupyter
.ipynb_checkpoints

# 환경설정
.env
.venv

# 시스템 파일
.DS_Store
Thumbs.db

# 모델 파일
*.pt
*.pth

# 로그 및 캐시
*.log
*.tmp
.cache/
```

---

## 📈 Verifiable TradingAgents System (branch: `claude/verifiable-trading-system-dmsmz9`)

This branch hosts a separate work stream: a **verifiable automated-trading system** built on a fork of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) (Apache-2.0), adopting
Trading-R1's evidence-based thesis → 5-tier rating (`Buy/Overweight/Hold/Underweight/Sell`).

Priority order: **(1) backtest harness → (2) paper trading → (3) live trading (kill-switch mandatory, last)**.

- Step 0 code-reading analysis: [`docs/step0_code_reading.md`](docs/step0_code_reading.md)
- Running decision log: [`docs/DECISIONS.md`](docs/DECISIONS.md)

Progress: **Step 0** (code map), **Step 1** (`vts/pit`, `vts/sources` — point-in-time data layer),
**Step 2** (`vts/backtest` — walk-forward engine: cost model, cutoff gate, N=3 vote, no-lookahead
execution). Run tests with `.venv/bin/python -m pytest`.
