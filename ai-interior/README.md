# AI 셀프 인테리어 시각화 — 로컬 파이프라인

방 사진 + 가구 사진 + 놓을 자리를 받아, 그 가구가 그 자리에 놓인 합성 이미지를 만든다.
**모든 처리는 로컬에서 돈다.** Gemini 이미지 편집 API는 비교 기준선으로만 남아 있고 기본값은 호출하지 않는다.

```
방 사진 ─ Depth Anything V2 ─ 바닥 평면(RANSAC) ─ 지평선 ─┬─ 크기 자동 보정
                                                         ├─ 원근 배치
가구 사진 ─ SAM ViT-B (박스 프롬프트) ─ 누끼 ─────────────┤
                                                         └─ 조명 정합 · 접지 그림자 · 깊이 가림 ─ 합성
```

## 설치와 실행

검증한 환경은 Python 3.13, macOS Apple Silicon(M5 16GB, MPS)이다. CUDA가 있으면 CUDA, 없으면 CPU로 돈다(CPU는 느리다).

```bash
pip install -r requirements.txt
python app.py             # http://127.0.0.1:7860
```

첫 실행 때 모델을 자동으로 받는다. 앱은 Depth Anything V2 Small(95MB)과 SAM ViT-B(358MB)를 쓰고,
평가 스크립트는 CLIP ViT-B/32(577MB)를 추가로 쓴다. 모델 로드에 20초쯤 걸리고 이후 한 장에 몇 초다.
**AI 다듬기**를 처음 켜면 Stable Diffusion 1.5 인페인팅과 ControlNet 깊이(fp16 약 2.9GB)를 받는다.
켜면 한 장에 10초 안팎이 더 걸리고 메모리를 약 7GB 더 쓴다.

포트를 고정하려면 `GRADIO_SERVER_PORT=7860 python app.py`. 7860이 사용 중이면 gradio가 다른 포트로 넘어가니
실행 직후 터미널의 `Running on local URL:` 줄을 확인한다.

## 사용법

1. **방 사진**을 올리고, 가구를 놓을 자리를 **가구가 차지할 크기만큼** 칠한다. 칠한 영역의 아랫변이 가구가 바닥에 닿는 선이다.
   칠하지 않으면 아래 슬라이더 좌표를 쓴다.
2. **가구 사진**을 올리고 **가구를 감싸게** 칠한다. 샘플 10개 기준으로 칠하지 않으면 4개, 칠하면 8개가 제대로 분리됐다(DEVLOG §19).
   하단 **샘플**을 누르면 미리 정해 둔 가구 박스가 자동으로 쓰인다.
3. **가구 이름**을 샘플과 같게 적으면 실제 높이로 크기를 자동 계산한다. 모르는 이름이면 칠한 박스 크기를 쓴다.
4. `가구 배치 생성` → Before / 추정한 바닥(초록)·지평선(주황)·배치 위치(빨강) / After.

5. (선택) **AI 다듬기**를 켜면 가구 주변의 경계·색·그림자를 로컬 Stable Diffusion이 다시 그린다.
   강도가 높을수록 자연스럽지만 **가구 모양과 색이 바뀔 수 있다**(0.35에서도 의자 좌판 천 색이 바뀌었다). 보는 각도는 바뀌지 않는다. DEVLOG §26, §27.

처리 내역 창에 판단 근거와 경고가 나온다.

| 경고 | 뜻과 대처 |
|---|---|
| 칠한 영역의 N% 높이만 가구로 잡혔습니다 (확인) | 넉넉히 칠했다면 괜찮다. 가구 일부만 잡혔다면(조명의 갓만 등) SAM 후보를 바꾼다. 크기 계산은 바꾸지 않는다 |
| 배경이 잡혔을 수 있습니다 | SAM이 가구 대신 배경을 잡았다. SAM 후보를 0/1/2로 바꾸거나 가구에 더 딱 맞게 칠한다 |
| 실제 높이를 모릅니다 | 가구 이름이 샘플에 없다. 박스 크기를 쓴다 |
| 가구가 화면 끝에서 잘렸습니다 / 높이가 N%뿐입니다 | 칠한 영역을 옮기거나 크기 배율을 조정한다 |

가구 사진이 **투명 배경 PNG**(쇼핑몰 제품 컷)면 SAM을 건너뛰고 그 알파를 그대로 쓴다. 가장 정확하다.

## 파이프라인

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 깊이 | `pipeline/depth.py` | Depth Anything V2 Small, float32 상대 역깊이 (0~1, 클수록 가깝다) |
| 바닥 평면 | `pipeline/geometry.py` `floor_plane` | 역깊이가 이미지 좌표의 1차식이 되는 영역을 RANSAC으로 찾는다. 지평선이 함께 나온다 |
| 누끼 | `pipeline/segment.py` `cutout` | SAM ViT-B 박스 프롬프트. 박스는 가장자리에서 8% 안쪽으로 물린다 |
| 크기 | `geometry.auto_height_px` | `화면 높이 / (접지 y − 지평선 y) = 실제 높이 / 카메라 높이(1.4m)`. 지평선은 사진 높이 41%로 가정(깊이로 찾은 지평선보다 정확했다 — ADE20K 문 검증, DEVLOG §30) |
| 배치 | `geometry.place_transform` | 세우는 가구는 크기만, 까는 가구(러그)는 원근 사다리꼴 |
| 합성 | `pipeline/compose.py` | 조명 정합(방의 밝은 띠 기준 백색점), 원근 반영 접지 그림자, 깊이 기반 가림, 알파 합성 |
| 다듬기 (선택) | `pipeline/refine.py` | 가구 주변만 잘라 SD 1.5 인페인팅 + ControlNet 깊이로 다시 그린다. 기본 꺼짐 |

각 파일은 단독 실행된다. 설계 근거와 실패한 시도는 [DEVLOG.md](DEVLOG.md)에 있다.

## 수치 재현

모두 로컬이고 요금이 없다.

```bash
python pipeline/geometry.py                          # 바닥 IoU, 튜닝셋 10장 → 평균 0.864
python pipeline/geometry.py samples/holdout/*.png    # 튜닝에 안 쓴 10장 → 평균 0.784
python eval/run_eval.py --local                      # 로컬 두 조건(가구 자동/칠함) 10쌍, 약 45초
python eval/test_harmonize.py                        # 조명 정합 계약 테스트 (중성색 갈변, 색상 유지)
python eval/metrics.py                               # CLIP 지표 자기 검증
python eval/size_check.py                            # 크기 공식을 ADE20K 문(2.03m)으로 검증, 약 5분 (ADE20K 필요)
```

`run_eval --local` 결과는 `eval/results/local-NNN/`에 쌓인다. 수치(`scoresheet.csv`)만 커밋하고 이미지는 다시 만든다.
CLIP 점수는 **같은 방·같은 가구에서 조건끼리 비교할 때만** 의미가 있다. 서로 다른 가구 사이의 품질이나 크기의 사실감은 구별하지 못한다(DEVLOG §23).

## 사람 평가 설문

```bash
python survey.py              # http://127.0.0.1:7861, 같은 와이파이의 휴대폰은 --host 0.0.0.0
python survey.py --share      # 계정 없이 접속하는 임시 공개 링크
python eval/survey/summarize.py   # 조건별 평균, 95% 구간, 같은 사람 안의 짝 비교
```

19문항(방식 비교 9 + 다듬기 효과 10), 문항마다 위치·자연스러움·가구 보존을 1~5점으로 받는다.
방식은 표시하지 않고 순서는 사람마다 섞는다. 응답은 `eval/survey/responses.csv`에만 쌓인다. DEVLOG §31.

## Gemini API 비교 경로 (선택, 유료)

```bash
cp .env.example .env      # API_KEY 입력, REAL_API=1
REAL_API=1 python eval/run_eval.py --yes          # 10쌍 × API 3조건 = 30회, 약 $2
```

- 이미지 모델은 **무료 등급이 없다.** 무료 키로 호출하면 `429 ... limit: 0`이 뜬다. 결제 등록이 필요하고 `gemini-3.1-flash-image`는 장당 $0.067이다.
- 앱에서는 "유료 호출에 동의합니다"를 켜야 Gemini 경로가 실행된다. 상단 배너가 현재 상태(mock/유료)를 보여준다.
- `run_eval`은 `REAL_API=1`인데 `--yes`가 없으면 금액을 알려주고 멈춘다. 결과는 실행마다 `eval/results/real-NNN/`에 따로 쌓인다.

## 폴더

```
app.py                Gradio 데모
api_baseline.py       Gemini 편집 API 경로 (비교 기준선, Gradio 비의존)
pipeline/             depth · segment · geometry · compose
eval/                 run_eval(비교 실험) · metrics(CLIP, SSIM) · test_harmonize
samples/rooms/        평가용 방 10장 (ADE20K) + rooms_gt/ 정답 바닥·벽 마스크
samples/holdout/      튜닝에 쓰지 않은 방 10장 + holdout_gt/
samples/items/        가구 10점 (Wikimedia Commons) + items.json (이름·높이·가구 박스)
samples/SOURCES.md    출처와 라이선스
docs/evidence/        DEVLOG가 참조하는 측정 결과 이미지
```

`samples/items_cutout/`(`python pipeline/segment.py`), `samples/rooms_depth/`(`python pipeline/depth.py`),
`samples/*_floor/`(`python pipeline/geometry.py`)는 다시 만들 수 있는 산출물이라 커밋하지 않는다.
