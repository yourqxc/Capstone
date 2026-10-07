# RoomFit · 방에 놓아 보기

**방 사진에 원하는 자리를 표시하면, 가구 사진을 참고해 그 방에 놓인 새로운 장면을 생성하는 대학 졸업과제입니다.**

공개 이미지 편집 모델 **FLUX.2 klein 4B**를 직접 실행합니다. 가구의 촬영 각도·조명·접지 그림자를 모델이 다시 그리며, 선택한 편집 영역 밖의 방 사진은 코드로 복원합니다. 외부 이미지 생성 API와 API 키는 사용하지 않습니다.

## 지금 사용할 코드

```text
app.py                    Gradio 화면: 사진 입력, 위치 표시, 합성, 다운로드
generate.py               같은 실험을 UI 없이 실행하는 명령
evaluate.py               전체 실행 기록과 빈 사람 평가지 내보내기
roomfit/placement.py       좌표 변환, 위치 지시문, 편집 마스크, 원본 보존
roomfit/engine.py          Mac MLX / Colab CUDA 로컬 모델 실행
roomfit/workflow.py        입력·모델 설정·원시 출력·최종 결과·실패 기록
notebooks/colab.ipynb      현재 로컬 프로젝트를 Colab GPU에서 실행
tests/test_core.py        좌표·픽셀 보존·실패 기록 검증
docs/                     개발 방향, 모델 선택 근거, 평가 계획, 실제 진행 기록
presentation-v2/          12장 발표 자료와 발표 대본
archive/legacy/           이전 코드·개발 일지·평가·발표 자료 보존
```

현재 방향은 [프로젝트 설명](docs/PROJECT.md), 변경 과정은 [진행 기록](docs/PROGRESS.md)을 보시면 됩니다. 이전 버전의 성능 수치는 새 모델의 성능으로 사용하지 않습니다.

## Mac 실행

Python 3.12와 Apple Silicon Mac을 사용합니다. 현재 M5·메모리 16GB에서 실제 추론을 확인했습니다. 사진·해상도·동시에 실행 중인 앱에 따라 메모리와 속도는 달라집니다. 처음에는 **512px**를 선택해 주세요.

이 폴더에는 실행 환경을 설치해 두었습니다.

```bash
.venv/bin/python app.py
```

브라우저에서 <http://127.0.0.1:7860>을 여세요. Finder에서 `실행.command`를 열어도 됩니다. 포트를 사용 중이면 `--port 7861`을 지정합니다. 종료는 실행한 터미널에서 `Ctrl+C`입니다.

다른 Mac에서 처음 설치할 때:

```bash
zsh scripts/setup_mac.sh
.venv/bin/python app.py
```

첫 합성 때 공개 4비트 가중치를 다운로드합니다(약 4.6GB, 디스크 공간에 여유가 필요합니다). 프로젝트의 `.cache/`를 사용하며, 이후 캐시가 완성되어 있으면 모델을 다시 내려받지 않습니다. 다운로드에는 인터넷이 필요하지만 **사진을 외부 추론 서버로 보내지 않습니다**. 자세한 버전·라이선스·메모리 조건은 [모델 실행 환경](docs/MODELS.md)에 기록했습니다.

1. 방 사진을 올리고 **가구 전체가 차지할 영역**을 칠합니다. 칠한 자국을 감싸는 사각형이 목표 영역이며, 아래쪽이 바닥 접지점입니다. 제공된 예제도 사진을 불러온 뒤 직접 칠합니다.
2. 가구가 전체로 보이는 사진을 올립니다. 필요하면 원본의 특징을 짧게 적습니다.
3. `배치 미리보기`로 범위를 확인한 뒤 `자연스럽게 합성하기`를 누릅니다.
4. 원래 방과 가구를 비교하고, 결과 PNG와 실행 증거 ZIP을 내려받습니다.

앞에 남겨야 하는 기존 물체가 있으면 `배치 확인과 앞쪽 물체 보호`에서 그 물체를 칠해 주세요. 이 영역은 원본 픽셀로 보존합니다. 생성 마스크는 그림자용 여백을 포함하므로 칠한 사각형 바로 주변도 바뀔 수 있습니다.

가구 특징과 위치는 모델이 따르는 **조건**이며 정확한 제품 동일성·실측 크기의 보증은 아닙니다. 안내선이 남거나 위치가 벗어나는 경우는 실패 사례로 평가합니다. 현재 모델 품질은 사례 실험 단계입니다.

## Colab GPU에서 실행

Mac의 메모리가 부족하거나 더 높은 해상도로 실험하려면 [Colab 노트북](notebooks/colab.ipynb)을 Colab에서 엽니다.

현재 코드를 담은 [업로드용 ZIP](roomfit-colab.zip)도 준비했습니다. 샘플·발표·문서를 포함하며 실행 환경·모델 캐시·이전 자료·개별 생성 결과는 제외했습니다. 코드가 바뀌면 노트북 첫 안내의 명령으로 ZIP을 새로 만드세요.

노트북에 프로젝트 ZIP 만들기, 업로드, GPU 확인, 설치, 앱 실행 셀이 있습니다. 현재 로컬 파일을 ZIP으로 사용하므로 미완성 코드를 GitHub에 공개할 필요가 없습니다. `.venv`, 모델 캐시, 비밀값, 이전 자료는 ZIP에서 제외합니다.

Colab에서는 직접 내려받은 모델을 GPU에서 추론합니다. Gradio가 만드는 임시 링크로 Mac 브라우저에서 이용할 수 있습니다. 사진은 Colab에 올라가며 링크를 아는 사람이 앱을 사용할 수 있습니다. 생성 이미지와 ZIP을 내려받은 뒤 런타임을 종료해 주세요. **이 작업 환경에서는 Colab의 실제 CUDA 추론을 실행하지 않았습니다.**

## 발표 자료

- [발표 슬라이드](presentation-v2/index.html): 브라우저에서 열어 `← / →`, `Home / End`로 이동합니다. 인쇄하면 슬라이드별 페이지로 출력됩니다.
- [발표 대본](presentation-v2/SPEAKER_SCRIPT.md)
- [개발 방향과 범위](docs/PROJECT.md)
- [평가 계획](docs/EVALUATION.md)
- [진행 기록](docs/PROGRESS.md)

## 실험 재현

```bash
.venv/bin/python generate.py \
  --room 'Sample/방.jpg' --furniture 'Sample/침대.jpg' \
  --box 0.22 0.38 0.78 0.93 --resolution 512 --seed 42 \
  --description 'beige upholstered double bed with taupe bedding'

.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluate.py
```

`--box`는 0~1 범위의 **왼쪽·위·오른쪽·아래 좌표**입니다. 크기와 위치는 픽셀에 대한 비율이며 실제 미터 치수가 아닙니다. 기본은 깨끗한 방·가구 **두 사진**과 텍스트 좌표입니다. `--guide marker`와 `--guide map`은 위치 안내 방법 비교를 위한 실험 옵션입니다. 첫 marker 실험에서는 빨간 상자가 결과에 남아 기본 경로에서 제외했습니다.

각 실행은 `outputs/runs/<UTC시각>-<식별자>/`에 저장됩니다.

- `room.png`, `furniture.png`, `placement.png`: 실제 입력과 사용자에게 보이는 배치 안내
- `reference_*.png`, `prompt.txt`: 모델에 실제 전달한 입력
- `edit_mask.png`: 생성 결과를 방 사진에 반영할 영역
- `raw.png`, `result.png`: 모델 원시 출력과 원본 보존 후 출력
- `run.json`: 모델·버전·시드·해상도·시간·입력 해시·성공/실패
- `evidence.zip`: 위 파일을 묶은 발표·평가용 기록

입력 사진과 모델 결과는 개인 실행 산출물이므로 Git에서 제외합니다. 발표에 쓸 사례는 출처와 성공/실패 설명을 붙여 선별합니다. 시드가 같아도 MLX와 CUDA의 출력이 동일하다고 보장하지 않습니다.
