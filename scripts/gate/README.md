# 실거래 전 게이트 실행 runbook

HANDOVER §3의 "실거래 전 필수 관문"을 **그대로 실행 가능한 스크립트**로 옮긴 것입니다. 각 스크립트는
통과/실패를 종료 코드로 판정하고, 실패하면 이유를 출력하고 즉시 멈춥니다(`set -euo pipefail`).

> ⚠️ **이 스크립트들은 이 CI 샌드박스에서 돌지 않습니다.** 조직 egress 프록시가 바이낸스
> (`api.binance.com` · `testnet.binance.vision`)를 403으로 막기 때문입니다. **바이낸스 egress가
> 허용된 로컬/서버에서 실행하세요.** 게이트 로직·코드·안전장치는 모두 준비돼 있고, 남은 것은 실데이터
> 실행뿐입니다.

## 한눈에

```
scripts/gate/
├── config.env                 # 편집 지점: 종목·윈도우·모델·데이터 경로 (전부 env로 덮어쓰기 가능)
├── lib.sh                     # 공통 헬퍼 (직접 실행 X)
├── check_window.py            # 백테스트 윈도우가 cutoff-clean인지 사전 검사 (엔진과 같은 레지스트리)
├── 00_setup.sh                # 게이트 0: venv + 설치 + 전체 테스트 green
├── 10_ingest.sh               # 게이트 1: PIT 데이터 적재 (+ 최소 바 수 검증)
├── 20_backtest_validation.sh  # 게이트 2: cutoff-clean 윈도우에서 두 벤치마크 초과 + clean 인증
├── 30_paper.sh                # 게이트 3: 페이퍼 루프 (매일·8주), shortfall gap 감시
├── 40_testnet_smoke.sh        # 게이트 4: 테스트넷 주문 왕복 (수동 실행)
└── run_all.sh                 # 0→3 순차 실행, 첫 실패에서 정지 (게이트 4는 별도)
```

## 실행

```bash
# 0) 값 확인/수정
$EDITOR scripts/gate/config.env        # 또는 셸에서 export INGEST_START=... 등

# 1) 키 (필요할 때만): 공개 klines는 키 불필요. LLM/라이브만 .env 필요.
cp .env.example .env && $EDITOR .env   # 절대 커밋 금지 (.gitignore가 방어)

# 2) 오프라인-안전 게이트 0~3 순차 실행
bash scripts/gate/run_all.sh

# 또는 하나씩
bash scripts/gate/00_setup.sh
bash scripts/gate/10_ingest.sh
bash scripts/gate/20_backtest_validation.sh
bash scripts/gate/30_paper.sh

# 3) 테스트넷 스모크 (테스트넷 키 필요, 실자금 없음)
bash scripts/gate/40_testnet_smoke.sh                 # 드라이런만
SMOKE_GO_LIVE=1 VTS_LIVE_TRADING_ARMED=I_UNDERSTAND_THE_RISK \
  bash scripts/gate/40_testnet_smoke.sh               # 실제 테스트넷 주문
```

## 각 게이트가 막는 것

| 게이트 | 통과 조건 | 실패 시 |
|---|---|---|
| 0 setup | 전체 pytest green | 테스트가 빨간 채로는 어떤 실거래 게이트도 신뢰 불가 → 정지 |
| 1 ingest | 종목별 최소 바 수 확보 | egress가 조용히 빈 데이터를 줬는지 여기서 잡음 |
| 2a window | 윈도우가 effective cutoff **이후** | 오염 구간이면 정지 (`check_window.py`, 엔진과 동일 레지스트리) |
| 2b backtest | 비용차감 후 **두 벤치마크 모두 초과** + `Contamination: **clean**` | 못 이기면 FAIL. **같은 윈도우 재튜닝 후 재평가 금지** |
| 3 paper | 루프 정상 + shortfall gap 비확대 | gap이 계속 벌어지면 모델이 아니라 실행 마찰 → 라이브 금지 |
| 4 testnet | 주문 왕복·취소·flat 검증 | 어댑터 기본이 테스트넷이라 실자금 없음 |

## 이중 트랙 (V3 검증 · V4 운영)

DeepSeek는 공식 knowledge cutoff를 발표하지 않아 운영자 채택값(`max(sources)+buffer`)을 씁니다.
방향 규칙: **오염은 cutoff 이전**이므로 보수적 = 늦은 날짜.

- **검증 트랙 (기본, `VTS_DEEP_THINK_LLM=deepseek-v3`)**: effective cutoff `2024-09-29` → 약 23개월
  clean. 전략·하네스·게이트를 여기서 검증합니다. `config.env`의 `BACKTEST_START`는 이 날짜 이후여야
  하며 `check_window.py`가 강제합니다.
- **운영 트랙 (`VTS_DEEP_THINK_LLM=deepseek-v4-pro`)**: effective cutoff ≈ `2026-06-29`이라 현재
  clean 백테스트 구간이 짧습니다. **페이퍼는 미래 데이터라 정의상 항상 clean** → V4는 게이트 3(페이퍼)로
  즉시 검증하고, clean 백테스트 구간이 자란 뒤(수개월 후) 게이트 2를 재실행합니다.
- 두 트랙 결론이 갈리면(V3 백테스트 PASS인데 V4 페이퍼가 계속 뒤처짐) **라이브 금지**.

> 게이트가 `unknown`을 내면 해당 모델의 cutoff가 레지스트리에 없다는 뜻입니다.
> `vts/backtest/model_cutoffs.json`에 `max(sources)+buffer_days`로 채우세요. 게이트는 절대 추측으로
> clean 처리하지 않습니다. 결정론적 `momentum` 모델(`MODEL=momentum`)은 학습 파라미터가 없어
> **오염 면제(clean)** 이며, LLM/키/네트워크 없이 하네스만 스모크할 때 씁니다.

## 안전장치 (게이트 통과와 무관하게 항상 작동)

| 장치 | 조작 | 효과 |
|---|---|---|
| 킬스위치 | `export VTS_KILL_SWITCH=1` | 라이브: 미체결 취소→sleeve 전량청산→flat 검증→래치. 페이퍼: flat. 백테스트: 실행 거부 |
| 무장 토큰 | `export VTS_LIVE_TRADING_ARMED=I_UNDERSTAND_THE_RISK` | 이것 없이는 `--go-live`도 실주문 불가 |
| 자본 하드캡 | 코드 상수 `MAX_INITIAL_CAPITAL_FRACTION = 0.01` | 총자산 1% 초과 배분 매 사이클 거부. env로 못 올림 |
| halt 래치 | 자동(일일손실/낙폭) | 자동 해제 없음. `vts reset-halt --operator 이름`만 (감사기록) |
| 리스크 한도 | `VTS_RISK_*` env | 종목당 비중·총노출·일일손실·낙폭·확신도 하한. **LLM은 이 값을 못 바꿈** |

## Go-live (게이트 0~4 모두 통과 후에만)

1. 프로덕션 어댑터를 **코드에서 명시**: `BinanceBroker(base_url=PROD_BASE_URL)` (기본은 테스트넷).
2. 무장: `export VTS_LIVE_TRADING_ARMED=I_UNDERSTAND_THE_RISK` (주문마다 재확인됨).
3. 초기 자본은 총자산 1% 이하 (하드캡이 강제하지만 `--capital`도 작게).
4. `--go-live`로 실행하되 첫 사이클 후 `vts status`로 sleeve·halt 확인, 킬스위치 1회 리허설.
5. 이 전체 라이브 개시는 원 지시사항상 **운영자(당신)의 별도 명시 승인**이 필요한 단계입니다.

## 성적이 좋게 나오면

**먼저 데이터 누수를 의심하십시오.** 반증 테스트가 이미 있습니다
(`tests/test_no_lookahead_falsification.py`, `test_engine_no_lookahead.py`, `test_reflection_gate.py`).
새 소스/모델을 붙였다면 같은 패턴의 반증 테스트를 **먼저** 쓰고 성과를 믿으세요.
