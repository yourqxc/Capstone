# AI 셀프 인테리어 시각화 서비스

방 사진과 가구 사진을 올리고 배치 위치를 지정하면, 그 자리에 가구가 놓인 모습을 합성해 보여주는 시스템.
2026학년도 캡스톤디자인 졸업과제.

## 접근

가구 합성을 **직접 구현한 로컬 파이프라인**과 **상용 생성 API** 두 가지로 수행하고 비교 평가한다.

| 단계 | 기술 | 위치 |
|---|---|---|
| 방 깊이 추정 | Depth Anything V2 (PyTorch) | `ai-interior/pipeline/depth.py` |
| 바닥 평면 · 원근 변환 | OpenCV | `ai-interior/pipeline/geometry.py` |
| 가구 분할 (누끼) | SAM ViT-B | `ai-interior/pipeline/segment.py` ✅ |
| 합성 · 접지 그림자 | OpenCV | `ai-interior/pipeline/compose.py` |
| 비교 대상 | Gemini 이미지 편집 API | `ai-interior/app.py` |
| 정량 평가 | CLIP score, SSIM, 바닥 추정 IoU | `ai-interior/eval/` |

## 문서

- [개발 일지](ai-interior/DEVLOG.md) — 발표·보고서용. 실패한 시도와 측정값 포함
- [실행 방법](ai-interior/README.md)
- [샘플 출처·라이선스](ai-interior/samples/SOURCES.md)
- [작업 지침](ai_interior.md) — 프로젝트 규칙과 주차별 계획

## 실행

```bash
cd ai-interior
pip install -r requirements.txt
python app.py                    # 데모
python pipeline/segment.py       # 누끼 단계 단독 실행
```

이미지 생성 API는 기본적으로 호출하지 않는다(mock). 실제 호출은 `.env`에 `REAL_API=1`이 필요하며 요금이 발생한다.
