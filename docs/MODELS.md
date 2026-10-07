# 로컬 모델 실행 근거와 제한

2026-10-01에 모델 제작사와 런타임 개발자의 공개 문서 및 소스를 확인했습니다. 이 프로젝트는 **FLUX.2 klein 4B distilled**를 사용합니다. 사진을 외부 추론 API로 전송하는 코드는 없습니다. 처음 실행할 때 Hugging Face에서 모델 가중치를 다운로드합니다.

## 모델과 입력

- 원본 모델: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Apache 2.0이며 다중 참고 이미지 편집을 지원합니다. [공개 Hub 메타데이터](https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-4B)의 `gated:false`를 확인했으므로 필수 API 키나 HF 토큰은 없습니다.
- 기본 입력은 두 장입니다. 1번은 깨끗한 방 사진, 2번은 가구 참고 사진입니다. 사용자가 칠한 영역은 0~1 정규화 좌표로 바꿔 프롬프트의 백분율 범위와 왼쪽·중앙·오른쪽 설명으로 전달합니다. 화면의 빨간 배치 미리보기는 기본 모델 입력에 포함하지 않습니다.
- 모델에 직접 마스크나 구조화된 좌표를 전달하는 방식은 아닙니다. 두 참고 이미지와 자연어 지시문으로 위치를 유도합니다. 제품 세부 형태, 정확한 크기, 선택 영역 준수, 그림자의 자연스러움은 보장되지 않습니다.
- 모델이 방 전체를 재생성하므로 주변 물체나 색이 바뀔 수 있습니다. 생성 후 로컬 합성으로 그림자 여백을 포함한 편집 영역 밖과 사용자가 보호한 영역의 원본 픽셀을 유지합니다. 합성 영역과 가구·그림자 경계가 어긋날 수 있어 결과를 직접 확인해야 합니다. 픽셀 보존은 코드의 동작이며 생성 품질 점수가 아닙니다.

세 번째 참고 이미지는 `generate.py --guide marker` 또는 `--guide map`으로 선택하는 실험 기능입니다. `marker`는 빨간 사각형을 그린 방 사진, `map`은 검정 바탕에 흰 배치 사각형을 그린 별도 위치 지도를 추가합니다. 첫 `marker` 실제 생성에서는 제거 지시에도 [빨간 선이 결과에 남았습니다](../outputs/runs/20261001T065558Z-8e858ddc/raw.png). 이 때문에 기본값은 두 장과 텍스트 위치 안내인 `--guide text`입니다. `map`의 품질은 실제 생성으로 검증하지 않았습니다.

모델이 약 13GB VRAM에 들어간다는 제작사 설명은 NVIDIA GPU 기준입니다. 이 숫자를 macOS가 함께 쓰는 16GB 통합 메모리에 그대로 적용할 수 없습니다. 이미지와 참고 이미지 수에 따라 활성값 메모리도 증가합니다. [모델 카드](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

## Apple Silicon Mac

[`mflux==0.20.0`](https://pypi.org/project/mflux/)을 사용합니다. [MFLUX FLUX.2 문서](https://github.com/mflux-community/mflux/blob/ada53237b2ac865bc649ef1d06ac1b19b6983865/src/mflux/models/flux2/README.md)와 [해당 릴리스의 편집 CLI 소스](https://github.com/mflux-community/mflux/blob/ada53237b2ac865bc649ef1d06ac1b19b6983865/src/mflux/models/flux2/cli/flux2_edit_generate.py)에서 다중 이미지, 모델 경로, 양자화 인자를 확인했습니다.

기본 가중치는 [`mlx-community/flux2-klein-4b-4bit`](https://huggingface.co/mlx-community/flux2-klein-4b-4bit)입니다. 모델 제작사의 원본을 MFLUX로 변환한 **커뮤니티 배포본**이며 원본 제작사의 직접 배포본은 아닙니다. 모델 카드가 원본 모델과 변환 도구를 명시합니다. 저장소 파일은 약 4.62GB이고 이 크기는 최대 실행 메모리 사용량을 뜻하지 않습니다. 원본 약 15GB를 내려받아 실행 중 양자화하는 경로 대신 미리 양자화된 가중치를 사용합니다.

```bash
python -m pip install -r requirements-mac.txt
python -m mflux.models.flux2.cli.flux2_edit_generate \
  --model mlx-community/flux2-klein-4b-4bit \
  --base-model flux2-klein-4b \
  --image-paths room.png furniture.png \
  --prompt "Use image 1 as the room. Place ONE furniture item from image 2 at the right side, within x=50% to 94%, y=38% to 93%. Preserve its design and color. Match perspective, scale, lighting and contact shadows. Preserve the room camera and layout." \
  --width 512 --height 512 --seed 42 \
  --steps 4 --guidance 1.0 --quantize 4 --low-ram --output result.png
```

실제 Python API는 다음과 같습니다. 앱은 생성 후 모델 메모리를 확실히 해제하기 위해 위 CLI를 별도 프로세스로 실행합니다.

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2KleinEdit

model = Flux2KleinEdit(
    model_config=ModelConfig.flux2_klein_4b(),
    model_path="mlx-community/flux2-klein-4b-4bit",
    quantize=4,
)
result = model.generate_image(
    image_paths=["room.png", "furniture.png"],
    prompt="Place ONE furniture item from image 2 at the right side of the room in image 1, within x=50% to 94%, y=38% to 93%. Preserve its design and color. Match perspective, lighting and contact shadows.",
    seed=42, num_inference_steps=4, width=512, height=512, guidance=1.0,
)
result.save("result.png")
```

[`Flux2KleinEdit` 소스](https://github.com/mflux-community/mflux/blob/ada53237b2ac865bc649ef1d06ac1b19b6983865/src/mflux/models/flux2/variants/edit/flux2_klein_edit.py)에서 `model_path`, `quantize`, `image_paths` 인자를 확인했습니다. `--low-ram`은 CLI 콜백으로 적용됩니다. Python 예제에 동일한 메모리 절감 콜백이 자동 적용된다고 가정하지 마십시오.

### 실제 Mac 실행 기록

2026-10-01에 Apple M5, 16GB 통합 메모리에서 두 장 입력으로 실행했습니다. 아래 세 실행은 모두 512×352, seed 42, 4단계, 4비트 가중치와 `--low-ram` 설정입니다. 시간은 가중치가 이미 다운로드된 상태에서 별도 프로세스의 모델 초기화·추론·결과 저장을 포함한 `run.json`의 경과 시간입니다. 다운로드 시간은 포함하지 않습니다.

| 실행 | 경과 시간 | MFLUX가 기록한 Peak MLX memory | 근거 |
| --- | ---: | ---: | --- |
| 중앙, 두 장 입력 | 12.332초 | 기록 없음 | [실행 기록](../outputs/runs/20261001T070310Z-e056ad81/run.json), [원시 결과](../outputs/runs/20261001T070310Z-e056ad81/raw.png) |
| 왼쪽, 두 장 입력 | 13.625초 | 4.25GB | [실행 기록](../outputs/runs/20261001T071447Z-b79be4b0/run.json), [로그](../outputs/runs/20261001T071447Z-b79be4b0/runtime.log), [원시 결과](../outputs/runs/20261001T071447Z-b79be4b0/raw.png) |
| 오른쪽, 두 장 입력 | 11.624초 | 4.20GB | [실행 기록](../outputs/runs/20261001T071530Z-b9f75c4a/run.json), [로그](../outputs/runs/20261001T071530Z-b9f75c4a/runtime.log), [원시 결과](../outputs/runs/20261001T071530Z-b9f75c4a/raw.png) |

좌우 두 실행은 같은 방·가구·제품 설명·시드를 사용하고 배치 좌표만 바꿨습니다. 원시 결과에서 침대가 좌우 지시를 따랐고 빨간 표식은 없었습니다. 침구와 세부 형태는 달라졌으므로 제품 동일성과 엄격한 배치 경계 준수를 검증한 결과는 아닙니다. 중앙 실행은 제품 설명이 달라 좌우 실행과의 품질 비교에 사용하지 않습니다. 별도 사람 평가는 아직 하지 않았으며 각 기록의 `quality_verified`도 `false`입니다.

`Peak MLX memory`는 MLX가 보고한 할당 메모리입니다. 전체 프로세스 RSS, macOS 전체 사용량, 이 앱의 최소 RAM 요구량을 뜻하지 않습니다. M5 16GB에서 512px 실행이 성공했다는 근거이며 768~1024px 또는 다른 사진의 성공을 보장하지 않습니다. MFLUX의 [메모리 옵션 문서](https://github.com/mflux-community/mflux/blob/ada53237b2ac865bc649ef1d06ac1b19b6983865/src/mflux/models/common/README.md)에서 low-RAM 모드의 성능 비용과 VAE 타일링을 설명합니다.

## Colab / NVIDIA CUDA

`notebooks/colab.ipynb`를 Colab에 업로드하고 GPU 런타임을 선택해 주세요. 노트북이 현재 프로젝트 ZIP을 받아 `/content/roomfit`에 풀기 때문에 아직 원격 저장소에 커밋하지 않은 변경도 사용할 수 있습니다.

CUDA 런타임은 [`diffusers==0.40.0`](https://pypi.org/project/diffusers/0.40.0/), [`transformers==5.18.0`](https://pypi.org/project/transformers/5.18.0/), [`accelerate==1.15.0`](https://pypi.org/project/accelerate/1.15.0/), [`bitsandbytes==0.50.2`](https://pypi.org/project/bitsandbytes/0.50.2/)를 고정합니다. 공통 의존성인 NumPy 2.5.3 때문에 Python 3.12 이상이 필요합니다. Transformers 5.18.0은 PyTorch 2.5 이상을 요구합니다. CUDA 드라이버와 맞춰 설치되어 있는 Colab의 PyTorch를 유지하며 Python 버전, PyTorch 2.5 이상, CUDA 사용 가능 여부를 노트북에서 확인합니다. 패키지 설치 후에는 런타임을 재시작하고, 별도 확인 셀에서 `/content/roomfit` 경로와 모델 런타임 import를 다시 준비합니다. 이미 메모리에 불러온 NumPy 등과 새 설치 파일이 섞이지 않게 하기 위한 순서입니다.

핵심 실행 방식은 다음과 같습니다.

```python
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quantization = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    },
    components_to_quantize=["transformer", "text_encoder"],
)
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", dtype=torch.float16,
    quantization_config=quantization,
)
pipe.enable_model_cpu_offload(device="cuda")
pipe.vae.enable_tiling()
image = pipe(
    image=[Image.open(path).convert("RGB") for path in ["room.png", "furniture.png"]],
    prompt="Place ONE furniture item from image 2 at the right side of the room in image 1, within x=50% to 94%, y=38% to 93%. Preserve its design and color. Match perspective, lighting and contact shadows.",
    width=512, height=512, num_inference_steps=4, guidance_scale=1.0,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("result.png")
```

[`Flux2KleinPipeline` 릴리스 소스](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)는 `image`에 PIL 이미지 목록을 받고 Qwen3 텍스트 인코더를 사용합니다. [`PipelineQuantizationConfig` 릴리스 소스](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/quantizers/pipe_quant_config.py)와 [Diffusers 공식 메모리 최적화 문서](https://huggingface.co/docs/diffusers/main/en/optimization/speed-memory-optims)에서 두 컴포넌트를 양자화하고 CPU 오프로딩하는 구성을 확인했습니다. [`Diffusers 0.40.0의 from_pretrained 구현`](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/pipeline_utils.py#L724)은 `dtype` 인자를 처리하므로 이 버전에서는 `dtype=torch.float16`이 맞습니다. 별도 원격 텍스트 인코더는 사용하지 않습니다.

T4에서는 float16을 사용합니다. [bitsandbytes 공식 하드웨어 문서](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)는 NF4/FP4가 Pascal 이후 GPU에서 지원된다고 설명합니다. 이것은 T4에서 양자화 연산을 사용할 수 있다는 근거이며, 이 앱의 두 장 입력 편집이 특정 해상도에서 반드시 성공한다는 측정 결과는 아닙니다. **Colab/CUDA 실제 생성은 이 작업 환경에서 실행하지 않았습니다.** 모델 다운로드 공간과 CPU RAM도 필요하며, 4비트 로딩이어도 원본 CUDA 가중치를 다운로드합니다. 메모리 부족 시 512px로 낮추거나 런타임을 재시작해 주세요.

노트북의 `share=True`는 Gradio의 임시 공개 접속 링크를 만듭니다. 추론은 Colab GPU에서 실행되지만 업로드한 방 사진과 가구 사진은 Colab으로 전송됩니다. 링크를 아는 사람이 앱에 접근할 수 있고 Colab 런타임 종료 시 앱도 종료됩니다. 개인 사진을 다룰 때는 링크를 공유하는 대상을 확인해 주세요. Mac의 기본 실행은 공유 링크를 만들지 않습니다.

## 대안

Qwen Image Edit도 편집 모델이지만 MFLUX의 [모델 목록](https://github.com/mflux-community/mflux/blob/ada53237b2ac865bc649ef1d06ac1b19b6983865/README.md)은 기존 Qwen Image를 20B의 더 큰 모델로 설명합니다. 이번 M5 16GB 첫 구현에는 더 작은 FLUX.2 klein 4B를 선택했습니다. 정확한 배치와 가구 보존이 충분한지는 제공하신 실제 샘플을 사용해 별도로 평가해야 합니다.
