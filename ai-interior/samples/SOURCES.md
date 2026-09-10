# 샘플 이미지 출처

모든 이미지는 재배포·연구 이용이 허용된 소스에서 가져왔다. 보고서·발표에 이 표를 그대로 인용할 것.


## 방 사진 — ADE20K (MIT Scene Parsing Benchmark)

기획서 7쪽 '데이터 수집 전략'에 명시한 데이터셋이다. 학습에는 쓰지 않고 **평가용 실내 사진 소스**로만 사용한다.

- 출처: <http://sceneparsing.csail.mit.edu/> (ADEChallengeData2016)
- 인용: Zhou et al., *Scene Parsing through ADE20K Dataset*, CVPR 2017
- 선정 기준: 침실/거실 씬 중 **정답 마스크 기준 바닥 ≥10%, 벽 ≥12%**, 가로 방향, 가로 600px 이상
- `rooms_gt/`에 바닥(255)·벽(128) 정답 마스크를 함께 보관 — `pipeline/geometry.py`의 바닥 평면 추정 정확도 평가에 쓴다.

| 파일 | ADE20K ID | 씬 | 바닥 비율 | 벽 비율 |
|---|---|---|---|---|
| `rooms/room_01.png` | ADE_val_00001180 | bedroom | 25.9% | 37.9% |
| `rooms/room_02.png` | ADE_val_00000118 | bedroom | 26.2% | 31.7% |
| `rooms/room_03.png` | ADE_val_00001517 | living_room | 30.3% | 18.5% |
| `rooms/room_04.png` | ADE_val_00000141 | bedroom | 20.6% | 36.8% |
| `rooms/room_05.png` | ADE_val_00001150 | bedroom | 20.3% | 32.8% |
| `rooms/room_06.png` | ADE_val_00001130 | bedroom | 21.9% | 27.7% |
| `rooms/room_07.png` | ADE_val_00000512 | living_room | 23.8% | 22.5% |
| `rooms/room_08.png` | ADE_val_00001526 | living_room | 18.2% | 30.2% |
| `rooms/room_09.png` | ADE_val_00001505 | living_room | 10.9% | 44.3% |
| `rooms/room_10.png` | ADE_val_00000156 | bedroom | 12.2% | 38.8% |

## 가구 사진 — Wikimedia Commons

SAM 누끼 난이도가 섞이도록 스튜디오 컷과 배경 있는 실사를 함께 골랐다.

| 파일 | 가구 | SAM 난이도 | 라이선스 | 저작자 | 원본 |
|---|---|---|---|---|---|
| `items/item_01.png` | 장식 의자 | 쉬움(스튜디오) | CC BY-SA 4.0 | Didier Descouens | [Commons](https://commons.wikimedia.org/wiki/File:(Barcelona)_Butaca_del_salo_del_pis_principal_de_la_Casa_Calvet_-_Museu_Nacional_d%27Art_de_Catalunya.jpg) |
| `items/item_02.png` | 원목 의자 | 쉬움(스튜디오) | CC BY-SA 4.0 | Didier Descouens | [Commons](https://commons.wikimedia.org/wiki/File:(Barcelona)_Chair_Carved_wood_by_Josep_Maria_Jujol_-_Museu_Nacional_d%27Art_de_Catalunya.jpg) |
| `items/item_03.png` | 마호가니 의자 | 쉬움(스튜디오) | CC0 | Daderot | [Commons](https://commons.wikimedia.org/wiki/File:Armchair,_Mexico,_1750-1800,_mahogany,_other_woods,_modern_upholstery_-_Brooklyn_Museum_-_DSC09266.JPG) |
| `items/item_04.png` | 조각 암체어 | 쉬움(스튜디오) | CC0 | Rijksmuseum | [Commons](https://commons.wikimedia.org/wiki/File:Armstoel_van_tropisch_hout_met_los_zitraam_met_riet_bespannen,_de_stoel_heeft_o.m._geslingerde_colonetten_en_is_versierd_met_gestoken_lotusranken_en_kabelranden,_BK-1994-36.jpg) |
| `items/item_05.png` | 커피 테이블 | 보통(평면 배경) | CC BY-SA 3.0 | Pierre gencey | [Commons](https://commons.wikimedia.org/wiki/File:Hitier_01.JPG) |
| `items/item_06.png` | 유리문 책장 | 보통(평면 배경) | CC BY-SA 4.0 | Scsi44 | [Commons](https://commons.wikimedia.org/wiki/File:Aarau_herzogplatz.jpg) |
| `items/item_07.png` | 플로어 램프 | 보통(벽 배경) | CC BY 2.0 | Erich Ferdinand from germany | [Commons](https://commons.wikimedia.org/wiki/File:Eyecatcher_-_Flickr_-_erix.jpg) |
| `items/item_08.png` | 모던 라운지체어 | 어려움(실사) | CC BY-SA 4.0 | Mirliak | [Commons](https://commons.wikimedia.org/wiki/File:5_-_Kopia.jpg) |
| `items/item_09.png` | 타일 테이블 | 어려움(실사) | CC BY-SA 4.0 | Jörg Blobelt | [Commons](https://commons.wikimedia.org/wiki/File:20101024200DR_Beistelltisch_mit_Delfter_Fliesen.jpg) |
| `items/item_10.png` | 스탠드 조명 | 어려움(실사) | CC0 | Tomwsulcer | [Commons](https://commons.wikimedia.org/wiki/File:Lamp_in_early_morning_through_blinds_with_camera_flash_on_(2_of_2).JPG) |

## 라이선스 준수

- CC0 / Public domain: 조건 없음
- CC BY / CC BY-SA: **저작자 표시 필수** — 위 표를 보고서 부록에 포함할 것
- ADE20K: 연구·교육 목적 이용. 논문 인용 필요


## 이전 placeholder

`_placeholder_rooms/`, `_placeholder_items/`는 개발 초기에 PIL로 그린 임시 도형이다. 참고용으로만 남겨둔다.

## 홀드아웃 방 사진 10장 — `holdout/`, `holdout_gt/`

바닥 추정 파라미터(tol)를 고르는 데 **한 번도 쓰지 않은** 방들이다.
`rooms/`와 같은 선정 조건을 적용하고 기존 10장을 제외하면 55장이 통과하며, 그중 상위 10장이다.
IoU 0.864가 일반화 성능인지 확인하기 위한 것이고, `eval/holdout.py`로 재현한다.

| 파일 | ADE20K ID | 씬 |
|---|---|---|
| `holdout/hold_01.png` | ADE_val_00001512 | living_room |
| `holdout/hold_02.png` | ADE_val_00000158 | bedroom |
| `holdout/hold_03.png` | ADE_val_00000506 | living_room |
| `holdout/hold_04.png` | ADE_val_00001172 | bedroom |
| `holdout/hold_05.png` | ADE_val_00000519 | living_room |
| `holdout/hold_06.png` | ADE_val_00000532 | living_room |
| `holdout/hold_07.png` | ADE_val_00000522 | living_room |
| `holdout/hold_08.png` | ADE_val_00000176 | bedroom |
| `holdout/hold_09.png` | ADE_val_00001506 | living_room |
| `holdout/hold_10.png` | ADE_val_00000132 | bedroom |
