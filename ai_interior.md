---
name: ai-interior
description: AI 셀프 인테리어 시각화 졸업과제 개발. 방 사진과 가구 사진을 받아 지정한 위치에 가구를 합성하는 로컬 파이프라인(Depth/SAM/OpenCV)과 Gradio 데모를 만들거나 수정할 때 사용. pipeline/, app.py, eval/, 원근 보정, 누끼, 깊이 추정, 비교 실험 관련 작업이면 이 스킬을 따를 것.
---

# AI 셀프 인테리어 시각화

## 프로젝트란

대학교 졸업과제. **개발자 1명, 심사까지 3~4주.**

방 사진 + 가구 사진 + 배치 위치를 받아, 그 가구가 그 자리에 놓인 것처럼 합성한 이미지를 만든다.

**핵심: 합성 파이프라인을 직접 구현한다.** 상용 API 호출만으로 결과를 만드는 것은
이 과제의 목표가 아니다. 기획서(`AI_셀프_인테리어_시각화_서비스_앱.pdf`, 2026.05.05)가
PyTorch / SAM / Depth Estimation / OpenCV / CLIP을 핵심 기술 스택으로 명시했고,
심사는 그 기획서를 기준으로 이루어진다.

## 기획서 약속 ↔ 구현 매핑

이 표가 이 프로젝트의 정의다. 작업이 어느 칸을 채우는지 항상 확인할 것.

| 기획서 약속 (p7~p8) | 구현 | 상태 |
|---|---|---|
| Depth Estimation, "Depth Map 추출" | `pipeline/depth.py` — Depth Anything V2 추론 | 미구현 |
| SAM, "객체 분리 / 배경 제거 + 마스킹" | `pipeline/segment.py` — SAM 누끼 | 미구현 |
| OpenCV, "원근 변환 행렬 계산" | `pipeline/geometry.py` — 바닥 평면·소실점·호모그래피 | 미구현 |
| "배치 합성 — 원근·크기 자동 보정" | `pipeline/compose.py` — 알파 합성 + 접지 그림자 | 미구현 |
| CLIP | `eval/metrics.py` — CLIP score | 미구현 |
| PyTorch | 위 추론 전부의 실행 프레임워크 | 미구현 |
| — (기획서 외, 비교용) | `api_baseline.py` — Gemini 편집 API | 구현됨 |

## 절대 규칙

1. **모델을 학습하거나 파인튜닝하지 말 것.** 추론만 쓴다. 3~4주에 학습은 불가능하고
   심사도 요구하지 않는다. ADE20K / LSUN Bedroom은 학습용이 아니라 **평가용 방 사진 소스**로만 쓴다.
2. **로컬 모델 추론은 이 프로젝트의 본체다.** Depth Anything V2, SAM을 실제로 돌린다.
   (이전 판에서 이걸 금지했던 것은 잘못이었다. 기획서와 정면으로 충돌했다.)
3. **로컬 파이프라인이 1순위.** Gemini API 경로는 삭제하지 않고 **비교 대상(baseline)**으로 유지한다.
   보고서의 핵심 주장은 "직접 만든 파이프라인 vs 상용 생성 API" 비교다.
4. **이미지 생성 API는 비용이 발생한다.** 기본은 mock. 환경변수 `REAL_API=1`일 때만 실제 호출한다.
   무료 등급에 이미지 모델 할당량이 없으므로 결제 등록이 필요하다(장당 $0.039~0.067).
5. **웹앱으로 만들지 말 것.** Gradio 데모 하나다. React, Next.js, FastAPI, DB, 로그인, 배포 설정 금지.
   (기획서의 FastAPI는 Gradio 서빙으로 대체한다. 심사에서 문제되지 않는다.)
6. **테스트 프레임워크, 린터, CI, 타입 체커 세팅하지 말 것.**
7. **요청하지 않은 리팩터링 제안 금지.**
8. **모델 가중치를 커밋하지 말 것.** HuggingFace 캐시에서 받아 쓴다.

## 구조

```
ai-interior/
├── app.py              # Gradio UI + 이벤트 핸들러
├── pipeline/
│   ├── depth.py        # Depth Anything V2 추론 → 깊이맵
│   ├── segment.py      # SAM → 가구 알파 컷아웃
│   ├── geometry.py     # OpenCV: 바닥 평면 추정, 소실점, 호모그래피
│   └── compose.py      # 원근 보정 배치 + 접지 그림자 + 알파 블렌딩
├── api_baseline.py     # Gemini 편집 API 경로 (비교 대상)
├── eval/
│   ├── run_eval.py     # 로컬 파이프라인 vs API × 위치 지정 방식 비교
│   ├── metrics.py      # CLIP score, 마스크 밖 SSIM(구조 보존)
│   └── results/
├── samples/rooms/, samples/items/
├── .env.example        # API_KEY=, REAL_API=0, MODEL=
└── requirements.txt
```

`app.py` 한 파일 규칙은 폐기한다. 파이프라인 단계별로 파일을 나눈다 —
각 단계가 독립적으로 실행·검증 가능해야 실험을 돌릴 수 있다.

## 의존성

```
gradio, pillow, requests, python-dotenv        # 기존
torch, torchvision                             # 추론 프레임워크
transformers                                   # Depth Anything V2, SAM, CLIP 로딩
opencv-python                                  # 기하 처리
numpy, scikit-image                            # 배열 연산, SSIM
```

이 외를 추가할 때는 먼저 물어볼 것. Mac M5 / 16GB / MPS 환경이다. CUDA는 없다.

## 파이프라인 계약

```python
# pipeline/depth.py
def estimate_depth(room: Image) -> np.ndarray:
    """방 사진의 상대 깊이맵 (H, W) float32, 0~1 정규화."""

# pipeline/segment.py
def cutout(item: Image, box: tuple[int, int, int, int] | None = None) -> Image:
    """가구 사진에서 객체만 분리한 RGBA. box는 사용자가 가구 주위에 그린 사각형.
    미지정 시 이미지 중앙 80% 박스를 쓴다.
    (실측 결과 박스 프롬프트 7/10 > 중앙점 5/10 > 전체박스 2/10 이라 박스로 확정)"""

# pipeline/geometry.py
def floor_plane(room: Image, depth: np.ndarray) -> dict:
    """바닥 평면 추정. 소실점, 지평선 y, 바닥 마스크를 담은 dict."""

def place_transform(plane: dict, box: tuple, item_size: tuple) -> np.ndarray:
    """배치 박스를 바닥 평면 위 원근에 맞춘 3x3 호모그래피."""

# pipeline/compose.py
def compose(room: Image, item_rgba: Image, H: np.ndarray, plane: dict) -> Image:
    """원근 보정된 가구를 방에 합성. 접지 그림자 포함."""
```

## 주차별 계획 (3~4주)

- **1주차** — `geometry.py` + `segment.py`. 원근 보정과 누끼가 이 과제의 자작 핵심이다.
  SAM 컷아웃과 호모그래피 결과를 눈으로 확인할 수 있는 상태까지.
- **2주차** — `depth.py` + `compose.py`. 앱에 연결해 로컬 경로만으로 결과가 나오게 한다.
- **3주차** — `eval` 확장. 로컬 파이프라인 vs Gemini API × 위치 지정 방식 3종.
  `metrics.py`로 CLIP score와 마스크 밖 SSIM 산출. **보고서의 핵심 근거.**
- **4주차** — 실제 방/가구 사진으로 교체, 보고서·발표 자료.

## 실험 (보고서 핵심)

두 축으로 비교한다.

1. **합성 방식**: 로컬 파이프라인 / Gemini API
2. **위치 지정 방식**: 반투명 사각형 마커 / 마스크 / 텍스트 지시

정성 평가(설문)와 정량 지표를 함께 낸다.

- CLIP score — 결과 이미지가 "가구가 놓인 방"에 얼마나 부합하는가
- 마스크 밖 SSIM — 배치 영역 밖의 방 구조가 얼마나 보존됐는가

이전 판은 "자동 채점을 만들지 말 것"이라고 했으나, 로컬 파이프라인을 직접 구현하는 이상
정량 지표는 보고서에서 방어 가능한 근거가 된다. 사람 설문은 그대로 병행한다.

## 작업 방식

- **한 단계가 끝날 때마다 `DEVLOG.md`에 기록한다. 이것은 선택이 아니라 필수다.**
  발표·보고서가 이 파일에서 나온다. `상황 → 측정/근거 → 결론` 세 줄 구조를 지키고,
  **실패한 시도와 뒤집은 판단을 반드시 남긴다** — 측정값이 남은 실패가 가장 강한 발표 재료다.
  결과 이미지는 `docs/evidence/`에 `<절번호>_<내용>.png`로 저장하고 일지에서 참조한다.
  임시 폴더에 두지 말 것 (세션이 끝나면 사라진다).
- 숫자 없는 주장은 일지에 쓰지 않는다. 측정하지 않았으면 "측정하지 않음"이라고 적는다.
- 한 번에 하나의 단계만. 끝나면 실제로 실행해서 결과 이미지를 확인하고 보고할 것.
- 기능 하나당 커밋 하나. 커밋 메시지는 한국어.
- API 키는 `.env`에서 읽는다. 절대 커밋하지 않는다.
- 모델을 처음 쓸 때는 반드시 작은 입력으로 스모크 테스트를 먼저 돌려 MPS에서 동작과
  소요 시간을 확인할 것. 파이프라인에 엮은 뒤에 실패를 발견하지 말 것.
- 실제 API 응답 형식이 불확실하면 추측해서 파싱하지 말고 응답 전문을 출력해서 확인할 것.

## 하지 않는 기능

요청받아도 먼저 "3~4주 스코프 밖인데 정말 추가할까요?"라고 확인할 것.

- 모델 학습 / 파인튜닝
- ControlNet, Stable Diffusion 로컬 실행 (여유가 생기면 재검토)
- 스타일 변환 (모던/북유럽 등 전체 리스타일), 벽지·조명 변경
- 쇼핑몰 연동, 가격 비교, 브랜드 추천
- 3D 뷰, AR
- 결과 히스토리 저장, 사용자 계정
