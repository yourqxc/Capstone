# 캡스톤 중간 발표 자료

기준일: 2026-10-01. 슬라이드 12장과 한국어 발표 대본을 제공합니다. 외부 라이브러리·CDN·웹 폰트 없이 실행하는 HTML 발표입니다. HTML과 CSS의 텍스트·표·영역 표시를 직접 수정할 수 있습니다.

## 열기와 발표

`index.html`을 브라우저에서 여시면 됩니다. 인터넷 연결이나 웹 서버가 필요하지 않습니다. `presentation-v2/` 폴더 전체를 함께 옮기십시오. 이미지와 CSS·JS는 상대 경로로 읽습니다.

| 조작 | 기능 |
| --- | --- |
| 방향키, Page Up/Down, Space | 이전·다음 슬라이드 |
| Home / End | 첫·마지막 슬라이드 |
| N 또는 발표 메모 버튼 | 현재 슬라이드의 설명과 출처 |
| F 또는 전체 화면 버튼 | 지원하는 브라우저에서 전체 화면 |
| 인쇄 버튼 | 모든 슬라이드를 한 장씩 인쇄 |
| 주소 끝 `#slide-5` | 해당 슬라이드로 바로 열기 |

인쇄 대화상자에서 배경 그래픽을 켜고 머리글·바닥글을 끄십시오. CSS는 가로 16:9 페이지와 슬라이드별 페이지 나눔을 지정합니다. 실제 PDF 저장 결과의 여백과 페이지 수는 사용하는 브라우저에서 확인하십시오.

## 파일 구성

- `index.html`: 슬라이드 본문과 각 슬라이드의 발표 메모
- `slides.css`: 16:9 레이아웃, 화면 크기 대응과 인쇄 스타일
- `slides.js`: 키보드·버튼 이동, 슬라이드 수와 발표 메모 표시
- `SPEAKER_SCRIPT.md`: 약 7~9분을 목표로 한 읽기용 발표 대본
- `assets/`: 발표에 사용한 입력과 과거 증거의 사본

발표자·팀명은 확인된 정보가 없어 넣지 않았습니다. 필요한 경우 표지의 날짜 주변에 실제 정보를 추가하십시오.

## 결과를 반영할 때

현재 11번 슬라이드는 첫 Mac 실제 생성의 **품질 실패**와 두 참조 수정 후 출력을 보여 줍니다. 첫 출력의 빨간 안내선은 수정 결과에서 사라졌지만 가구 색·형태 변화가 남습니다. 12번은 동일 입력·설명·시드의 좌·우 배치 응답과 다음 검증을 보여 줍니다. 제품 형태와 기존 탁자 가림의 한계를 표시했으며 사람 평가는 아직 실시하지 않았습니다.

실제 결과가 생기면 다음 근거를 함께 반영합니다.

1. 생성 후보와 최종 PNG, 사용한 방·가구·목표 박스
2. 실행 ID, 백엔드, 모델 버전, 양자화, 해상도, 스텝과 시드
3. 처리 시간·메모리의 실측 값과 측정 방식
4. 영역 밖 및 전경 보호 픽셀 검사 결과
5. 실패와 재시도 횟수, 품질 평가 실시 여부

새 결과를 추가할 경우 11·12번 슬라이드의 이미지와 설명을 근거에 맞게 바꾸십시오. 자동 검사와 실행 상태는 8·9번에도 표시합니다. 발표 메모, 대본, `docs/PROGRESS.md`의 상태를 같은 근거로 갱신하십시오. 첫 실패 기록은 삭제하지 않고 이전 실험 자료로 유지합니다. 기존 구현의 수치를 새 모델 성능으로 복사하지 마십시오.

## 이미지 출처

| 발표 파일 | 원본 | 설명 |
| --- | --- | --- |
| `assets/room.jpg` | `Sample/방.jpg` | 사용자가 제공한 실제 입력의 사본 |
| `assets/furniture.jpg` | `Sample/침대.jpg` | 사용자가 제공한 실제 입력의 사본 |
| `assets/archive-angle.png` | `archive/legacy/ai-interior/docs/evidence/14_촬영각도_불일치.png` | 이전 누끼 방식의 관찰 기록 |
| `assets/first-marker-failure.png` | `outputs/runs/20261001T065558Z-8e858ddc/result.png` | 첫 실제 Mac 생성의 품질 실패 출력 |
| `assets/two-reference-result.png` | `outputs/runs/20261001T070310Z-e056ad81/result.png` | 안내선은 없어졌으나 색·형태 변화가 남은 두 참조 출력 |
| `assets/placement-left.png` | `outputs/runs/20261001T071447Z-b79be4b0/result.png` | 왼쪽 목표 박스의 실제 출력 |
| `assets/placement-right.png` | `outputs/runs/20261001T071530Z-b9f75c4a/result.png` | 오른쪽 목표 박스의 실제 출력 |

Sample 원본의 실제 파일 형식은 AVIF입니다. 브라우저 호환성을 위해 발표용 사본만 JPEG로 변환했으며 원본 파일은 수정하지 않았습니다. 가구의 체크무늬는 원본에 포함된 배경입니다. 방 사진은 입력 설명과 편집 범위 설명에 반복 사용했습니다. 편집 범위의 색 박스는 실제 출력 증거가 아닌 설명용 표시입니다.

이전 자료는 [DEVLOG §21·§26](../archive/legacy/ai-interior/DEVLOG.md)의 관찰과 함께 읽어야 합니다. 첫 실행의 321.749초는 최초 모델 다운로드를 포함합니다. 이후 두 참조 12.332초, LEFT13.625초, RIGHT11.624초는 캐시 사용 단일 실행의 모델 호출·후처리 기록이며 평균이나 순수 추론 시간으로 해석하지 않습니다. LEFT의 MLX 피크4.25GB는 전체 프로세스·시스템 메모리 값이 아닙니다. 발표는 새 모델에 대한 성능 수치나 성공 이미지를 임의로 만들지 않습니다.

2026-10-01에 발표 전용 로컬 HTTP 서버에서 Chrome으로 슬라이드 12장, 키보드 이동과 메모를 검토했습니다. 인쇄 미리보기에서 12쪽과 시트당 1페이지를 확인했습니다. PDF 파일을 저장해 별도로 검토하지는 않았습니다.

## 기술 자료

확인일: 2026-10-01.

- [FLUX.2-klein-4B 공식 모델 카드](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): 다중 참조 편집, Apache 2.0, CPU offload 예제와 모델 한계. 5·8번 슬라이드 참고.
- [mflux 공식 FLUX.2 문서](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux2/README.md): Apple Silicon용 MLX 구현, 다중 참조 편집 CLI와 양자화 지원. 5·8번 슬라이드 참고.
- [Diffusers 공식 Flux2 문서](https://huggingface.co/docs/diffusers/api/pipelines/flux2): `Flux2KleinPipeline`과 참조 이미지 목록 입력. 8번 슬라이드 참고.

공식 문서가 설명하는 모델 기능과 일반 예제는 이번 입력·하드웨어에서의 실측 성능이 아닙니다. 프로젝트의 좌표 안내, 참조 구성과 픽셀 보존은 자체 구현과 검사로 확인해야 합니다.

프로젝트 범위는 [PROJECT.md](../docs/PROJECT.md), 평가 프로토콜은 [EVALUATION.md](../docs/EVALUATION.md), 진행 상태는 [PROGRESS.md](../docs/PROGRESS.md)에 있습니다.

`.venv/bin/python evaluate.py`는 모든 실행 상태와 픽셀 보존을 `runtime.csv`, 빈 사람 평가 양식을 `quality_review.csv`로 내보냅니다. 양식 생성은 평가 완료가 아닙니다. 자세한 절차는 평가 문서에 있습니다. 추가 입력의 출처는 [samples/SOURCES.md](../samples/SOURCES.md), 기존 자료는 `archive/legacy/`에서 확인합니다.
