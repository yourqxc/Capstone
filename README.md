# AI 셀프 인테리어 시각화 서비스

방 사진과 가구 사진을 올리고 배치 위치를 지정하면, 그 자리에 가구가 놓인 모습을 합성해 보여주는 시스템.
2026학년도 캡스톤디자인 졸업과제.

## 접근

가구 합성을 **직접 구현한 로컬 파이프라인**으로 수행하고, 상용 생성 API(Gemini)를 비교 기준선으로 둔다.
로컬 경로는 외부 호출 없이 이 컴퓨터에서 전부 돈다.

| 단계 | 기술 | 위치 |
|---|---|---|
| 방 깊이 추정 | Depth Anything V2 Small (PyTorch) | `ai-interior/pipeline/depth.py` |
| 바닥 평면 · 지평선 | 역깊이 1차식 RANSAC (NumPy · OpenCV) | `ai-interior/pipeline/geometry.py` |
| 가구 분할 (누끼) | SAM ViT-B, 박스 프롬프트 | `ai-interior/pipeline/segment.py` |
| 크기 자동 보정 · 원근 배치 | 지평선 기반 크기 공식, 호모그래피 (OpenCV) | `ai-interior/pipeline/geometry.py` |
| 합성 | 조명 정합, 접지 그림자, 깊이 기반 가림, 알파 합성 | `ai-interior/pipeline/compose.py` |
| 평가 | 바닥 IoU(ADE20K 정답), 배치 영역 CLIP, 마스크 밖 SSIM | `ai-interior/eval/` |
| 비교 기준선 | Gemini 이미지 편집 API (유료, 기본 꺼짐) | `ai-interior/api_baseline.py` |

기획서 대비 무엇을 구현·대체·제외했는지와 그 이유는 [DEVLOG §22](ai-interior/DEVLOG.md#22-범위-결정--기획서-대비-무엇을-했고-무엇을-왜-뺐는가).

## 주요 수치

| 항목 | 값 | 근거 |
|---|---|---|
| 바닥 평면 IoU (튜닝셋 10장) | 0.864 | DEVLOG §11 |
| 바닥 평면 IoU (튜닝에 안 쓴 10장) | 0.784 | DEVLOG §18 |
| 누끼 정상 (가구 칠함 / 안 칠함) | 8/10 / 4/10 | DEVLOG §19 |
| 조명 정합 후 순백 가구 채도 | 3.77 → 1.90 | DEVLOG §14 |

## 문서

- [개발 일지](ai-interior/DEVLOG.md) — 발표·보고서용. 실패한 시도와 측정값 포함
- [실행 방법](ai-interior/README.md)
- [샘플 출처·라이선스](ai-interior/samples/SOURCES.md)
- [작업 지침](ai_interior.md) — 프로젝트 규칙과 기획서 대비 매핑

## 실행

```bash
cd ai-interior
pip install -r requirements.txt
python app.py                    # 데모, http://127.0.0.1:7860
```

기본값으로는 유료 API를 호출하지 않는다. 자세한 사용법과 수치 재현 명령은 [ai-interior/README.md](ai-interior/README.md).
