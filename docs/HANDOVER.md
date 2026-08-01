# 운영 인수인계 (HANDOVER)

검증 가능한 자동매매 시스템 — TauricResearch/TradingAgents(Apache-2.0) fork 기반, Trading-R1의
"증거 기반 thesis → 5단계 등급" 결정 스키마 채택. 브로커/데이터: **바이낸스 현물**.
전체 결정 기록: [`DECISIONS.md`](./DECISIONS.md) · Step 0 코드 분석: [`step0_code_reading.md`](./step0_code_reading.md)

---

## 1. 아키텍처 한 장 요약

```
Binance klines (공개 API)
   └─ vts/sources/binance_data.py     ← 종가 경계 stamping, 진행중 캔들 차단
        └─ vts/pit/ (PointInTimeStore) ← event_time/knowledge_time 이중 타임스탬프,
             │                            assert_no_lookahead 강제 관문
             ▼
     결정 파이프라인 (백테스트·페이퍼·라이브 모두 동일 코드)
     sample_decisions (N표 다수결·디스크 캐시)
        └─ RiskEngine.apply            ← vts/risk/: 한도는 frozen 상수(LLM 불가침),
             │                            저확신→Hold, 일일손실·낙폭 halt 래치, 킬스위치
             ▼
     ┌─ vts/backtest/  워크포워드 + 비용모델 + cutoff 오염 게이트 + 벤치마크 게이트
     ├─ vts/paper/     동일 경로 페이퍼 루프, 일별 implementation shortfall
     └─ vts/live/      안전 봉투: 1% 하드캡, 무장 토큰, 전량청산 킬스위치, sleeve 원장
                        └─ vts/live/binance_broker.py (테스트넷 기본)
```

## 2. 설치와 명령어

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest            # 232개 전부 green이어야 정상
cp .env.example .env                  # 키 기입 (절대 커밋 금지)

.venv/bin/python -m vts ingest   --start 2025-01-01 --end 2025-06-30
.venv/bin/python -m vts backtest --start 2025-02-01 --end 2025-06-30 --report bt.md
.venv/bin/python -m vts paper    --start 2025-07-01 --end 2025-08-31
.venv/bin/python -m vts live     --capital 100      # 드라이런 (기본)
.venv/bin/python -m vts status
.venv/bin/python -m vts reset-halt --scope paper --operator "이름"
```

- `backtest`의 **종료 코드가 게이트 판정**입니다: 0 = PASS(비용차감 후 buy&hold와 60/40을 모두
  초과), 1 = FAIL. CI에 그대로 물릴 수 있습니다.
- 기본 모델은 결정론적 `momentum`(오프라인 검증용). LLM 그래프는 fork 설치 + `.env`의 LLM 키 후
  `--model tradingagents --samples 3`.
- 프로필 변경은 `VTS_*` 환경변수(`.env.example` 참조). 주식으로 돌아가려면
  `VTS_ASSET_CLASS=us_equity VTS_DATA_SOURCE=alpha_vantage`.

## 3. 실거래 전 필수 관문 (순서 고정 — 건너뛰지 말 것)

1. **모델 knowledge cutoff 기입**: `vts/backtest/model_cutoffs.json`에 사용 모델의 cutoff 날짜를
   공식 문서에서 확인해 넣고 `verified: true`. 미기입 시 모든 결과는 `unknown`(오염 미판정)으로
   표기되며 게이트가 clean을 인증하지 않습니다. **날짜를 추측으로 넣지 마세요.**
2. **백테스트 구간은 cutoff 이후로만**. cutoff 이전 성과는 "참고용(오염 가능)"입니다.
3. **게이트 PASS**: 비용차감 후 두 벤치마크 모두 초과 + cutoff-clean 구간에서 `certifiable`.
4. **페이퍼 8주**: `vts paper`를 매일 실행(재시작 안전·멱등). shortfall `gap`이 지속 확대되면
   모델이 아니라 실행 마찰이 원인이므로 라이브 진행 금지.
5. **테스트넷 스모크**: testnet.binance.vision 키로 `vts live --capital 100 --go-live`
   (어댑터 기본이 테스트넷이므로 실자금 없음). 주문 왕복·취소·청산 확인.
6. 그 후에만 프로덕션: 코드에서 `BinanceBroker(base_url=PROD_BASE_URL)` 명시 + 아래 무장 절차.

## 4. 안전장치 조작법

| 장치 | 조작 | 효과 |
|---|---|---|
| 킬스위치 | `export VTS_KILL_SWITCH=1` | 라이브: 미체결 취소 → sleeve 전량 청산 → flat 검증 → 래치. 페이퍼: flat. 백테스트: 실행 거부(조용한 오염 방지). 비어있지 않은 값은 전부 발동(fail-safe) |
| 무장 토큰 | `export VTS_LIVE_TRADING_ARMED=I_UNDERSTAND_THE_RISK` | 이것 없이는 `--go-live`도 실주문 불가. 주문마다 재확인 |
| 자본 하드캡 | 코드 상수 `MAX_INITIAL_CAPITAL_FRACTION = 0.01` | 총자산 1% 초과 배분은 매 사이클 거부. env로 올릴 수 없음(코드 리뷰 필요) |
| halt 래치 | 자동(일일손실/낙폭) | 좋은 날이 와도 자동 해제 없음. `vts reset-halt --operator 이름`만 해제(감사기록) |
| 리스크 한도 | `VTS_RISK_*` env | 종목당 비중·총노출·일일손실·낙폭·확신도 하한. **LLM은 절대 이 값을 못 바꿈** |

## 5. 남은 리스크 / 미완 항목 (정직하게)

- **LLM 그래프 미검증**: TradingAgents fork + LLM 키가 이 환경에 없어 `TradingAgentsDecisionModel`은
  파싱·게이트 로직만 오프라인 검증됨. 첫 LLM 백테스트에서 응답 캐시(`decision_cache.sqlite`)와
  등급 분산도(`dispersion`)를 반드시 확인할 것.
- **뉴스/센티먼트 소스 없음(크립토 프로필)**: 바이낸스는 과거 뉴스를 제공하지 않아 현재 크립토
  thesis는 가격 기반. 뉴스 벤더를 붙이려면 `DataSource.fetch_news` 구현 + `knowledge_time=발행시각`.
- **심볼별 venue 규칙**: CLI는 보수적 기본값(`BINANCE_SPOT_DEFAULT`) 사용. 라이브 전에
  `venue_from_exchange_filters`로 심볼별 tick/step/minNotional을 만들어 쓸 것.
- **1× 지정가 체결 가정**: 페이퍼는 결정 시점 종가 체결을 가정. 실제 괴리는 shortfall 로그가
  일별로 잡지만, 미체결 지정가 재시도 정책은 운영하며 조정 필요.
- 과거 리뷰에서 확정·수정된 84건의 결함 목록은 `DECISIONS.md`의 각 라운드 요약 참조 — 재발
  방지용 회귀 테스트가 전부 스위트에 있음.

## 6. 성적이 좋게 나올 때 (규칙 재확인)

**먼저 데이터 누수를 의심하십시오.** 반증 테스트가 이미 있습니다:
`tests/test_no_lookahead_falsification.py`, `test_engine_no_lookahead.py`,
`test_reflection_gate.py`. 새 데이터 소스나 모델을 붙였다면 같은 패턴의 반증 테스트를 먼저
쓰고 나서 성과를 믿으세요. 같은 구간 파라미터 재튜닝 후 재평가는 금지(원 지시사항)입니다.
