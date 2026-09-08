# AI 셀프 인테리어 시각화 데모

방 사진 + 가구 사진 + 배치 위치를 받아, 그 가구가 그 자리에 놓인 것처럼 합성한 이미지를 만든다.

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env      # API_KEY 입력
python app.py             # http://127.0.0.1:7860
```

기본은 **mock**이다. `REAL_API=0` 이면 API를 호출하지 않고 마커가 그려진 방 사진을 그대로 돌려준다.
실제로 합성하려면 `.env`에 `API_KEY`를 넣고 `REAL_API=1`로 바꾼다. 호출마다 요금이 발생한다.

> **이미지 모델은 무료 등급이 없다.** Google AI Studio 무료 키로 호출하면 `429 ... free_tier_requests, limit: 0`
> 이 뜬다. 키가 잘못된 게 아니라 할당량이 0이라서 그렇다. Google Cloud 프로젝트에 결제 수단을 등록해
> Tier 1로 올려야 호출된다. 이미지 1장당 `gemini-3.1-flash-image` $0.067, `gemini-2.5-flash-image` $0.039.

## 사용법

1. 방 사진을 올린다.
2. 가구를 놓을 자리를 **브러시로 칠하거나**, 칠하지 않으면 **슬라이더 좌표**를 쓴다.
3. 가구 사진을 올리고 이름을 적는다 (선택).
4. `가구 배치 생성` → Before / 배치 위치 표시 / After 가 나란히 나온다.

하단 `샘플`을 누르면 방·가구 사진이 한 번에 채워진다. 시연할 때 쓴다.

## 실험 (보고서용)

위치 지정 방식 3가지 — 반투명 사각형 마커 / 마스크 / 텍스트 지시 — 를 같은 세트에 대해 돌려 비교한다.
세 프롬프트는 **위치를 알려주는 문장만 다르고 나머지 문구는 동일하다.**

```bash
python eval/run_eval.py                          # mock, 10조합 (요금 없음)
REAL_API=1 python eval/run_eval.py --yes         # 실제 호출 30회
REAL_API=1 python eval/run_eval.py --all --yes   # 방10 x 가구10 = 300회
```

결과는 `eval/results/`에 쌓인다.

- `marker/`, `mask/`, `text/` — 방식별 입력 이미지, 결과 이미지, 사용된 프롬프트
- `comparison_grid.png` — 방식별 결과 비교 한 장
- `scoresheet.csv` — 위치정확도 / 합성자연스러움 / 가구보존 점수 기입용

자동 채점은 하지 않는다. 사람이 눈으로 보고 설문으로 점수를 매긴다.

## samples/

지금 들어 있는 이미지는 **PIL로 그린 임시 placeholder**다. 파이프라인 확인용이므로,
심사 전에 실제 방 사진 10장과 가구 사진 10장으로 교체할 것.
파일명은 `rooms/room_01.png … room_10.png`, `items/item_01.png … item_10.png` 순서를 유지한다.
가구 이름은 `eval/run_eval.py`의 `ITEM_NAMES`와 순서를 맞춘다.

## 구조

```
app.py           Gradio 데모 전체 (draw_marker / build_prompt / generate)
eval/run_eval.py 위치 지정 방식 비교 실험
samples/         방 사진, 가구 사진
.env             API_KEY, REAL_API, MODEL  (커밋하지 않는다)
```
