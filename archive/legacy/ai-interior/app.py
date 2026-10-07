"""AI 셀프 인테리어 시각화 데모.

방 사진 + 가구 사진 + 배치 위치 -> 그 자리에 가구가 놓인 합성 이미지.

두 가지 방식을 고를 수 있다.
  로컬 파이프라인 : Depth Anything -> 바닥 평면(OpenCV RANSAC) -> SAM 누끼 -> 원근 배치 -> 합성
  Gemini API      : 마커를 그린 방 사진과 가구 사진을 편집 모델에 넘긴다 (비교 대상)
"""

import json
import math
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import api_baseline as api
from pipeline import refine as sd_refine
from pipeline.compose import compose, occluder_mask
from pipeline.depth import estimate_depth
from pipeline.geometry import auto_height_px, floor_plane, place_transform, placement_check, reference_height_px, floor_placement
from pipeline.segment import (MIN_COVER, background_risk, box_coverage, clamp_box, cutout,
                              default_box, has_alpha, pct_box, segment)

ROOT = Path(__file__).parent
SAMPLE_ROOMS = ROOT / "samples" / "rooms"
SAMPLE_ITEMS = ROOT / "samples" / "items"


def load_items():
    """가구 목록의 단일 출처. samples/items.json 하나만 고치면 앱과 실험이 함께 따라온다.

    이전에는 app.py와 eval/run_eval.py가 각자 이름 목록을 들고 있었고 10개가 전부 어긋나
    있었다. 이 이름은 API 프롬프트의 "배치할 가구: OO"에 그대로 들어가므로,
    어긋나면 사진과 지시문이 불일치한 조건에서 실험이 돌아간다.
    """
    with open(ROOT / "samples" / "items.json", encoding="utf-8") as f:
        return json.load(f)["items"]


ITEMS = load_items()


def _thumb(img):
    return np.asarray(img.convert("L").resize((16, 16)), dtype=np.int16)


# 샘플 가구 사진 -> items.json 항목. 샘플을 클릭하면 가구 칠하기 레이어가 비어 있어서
# 중앙 80% 기본 박스로 누끼가 돌고, 그러면 10개 중 6개가 깨진다(좌판만, 두 동강, 벽 조각).
# 샘플이면 미리 정해 둔 가구 박스(사진만 보고 정한 것)를 쓴다.
_SAMPLE_THUMBS = [(Image.open(SAMPLE_ITEMS / it["file"]).size,
                   _thumb(Image.open(SAMPLE_ITEMS / it["file"])), it)
                  for it in ITEMS if (SAMPLE_ITEMS / it["file"]).exists()]


def sample_item(item):
    """올라온 가구 사진이 샘플이면 그 items.json 항목, 아니면 None."""
    t = _thumb(item)
    for size, ref, it in _SAMPLE_THUMBS:
        if size == item.size and np.abs(t - ref).mean() < 2:
            return it
    return None


# --- 입력 해석 -------------------------------------------------------------

def brush_box(editor_value):
    """ImageEditor에서 브러시로 칠한 영역의 사각형 범위. 안 칠했으면 None."""
    box = None
    for layer in (editor_value or {}).get("layers") or []:
        if layer is None:
            continue
        found = layer.convert("RGBA").getchannel("A").getbbox()
        if found is None:
            continue
        box = found if box is None else (min(box[0], found[0]), min(box[1], found[1]),
                                         max(box[2], found[2]), max(box[3], found[3]))
    return box


def percent_box(room, x_pct, y_pct, w_pct, h_pct):
    """슬라이더 백분율을 픽셀 좌표로 (x0, y0, x1, y1)."""
    W, H = room.size
    x, y = int(W * x_pct / 100), int(H * y_pct / 100)
    return x, y, min(x + int(W * w_pct / 100), W - 1), min(y + int(H * h_pct / 100), H - 1)


def _background(editor_value, keep_alpha: bool = False):
    """ImageEditor의 배경 이미지. keep_alpha=True면 투명도를 살린다 (누끼된 PNG용)."""
    if not editor_value or editor_value.get("background") is None:
        return None
    img = editor_value["background"]
    return img.convert("RGBA") if keep_alpha else img.convert("RGB")


def preview(room, box, plane=None):
    """배치 위치와 추정된 바닥을 눈으로 확인하는 그림."""
    arr = np.array(room).astype(np.int16)
    if plane is not None and plane.get("mask") is not None:
        m = plane["mask"]
        arr[m, 1] = np.minimum(255, arr[m, 1] + 70)
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline=(255, 40, 40), width=max(2, room.size[0] // 200))
    if plane is not None and plane.get("horizon_y") is not None:
        hy = plane["horizon_y"]
        if 0 <= hy < room.size[1]:
            d.line([(0, hy), (room.size[0], hy)], fill=(255, 140, 0), width=2)
    return img


def checked_editor(editor, room):
    """다른 방에 그린 보정이 새 사진에 재사용되지 않게 한다."""
    if brush_box(editor) is None:
        return None
    background = _background(editor)
    if background is None or not np.array_equal(np.asarray(background), np.asarray(room)):
        raise gr.Error("보정 사진이 현재 방과 다릅니다. 현재 방에서 보정 영역을 다시 칠해 주세요.")
    return editor


def reset_room_corrections(editor, reference=None, occlusion=None):
    room = _background(editor)
    value = {"background": room, "layers": [], "composite": room} if room else None
    # 배치 위치만 다시 칠할 때는 같은 방의 보정 표시를 지우지 않는다.
    return tuple(gr.skip() if old is not None and room is not None
                 and _background(old) is not None
                 and np.array_equal(np.asarray(_background(old)), np.asarray(room)) else value
                 for old in (reference, occlusion))


def corrected_cutout(rgba, editor, item):
    """파랑은 원본 픽셀 복구, 빨강은 배경 제거. 보이지 않는 부분을 생성하지 않는다."""
    checked_editor(editor, item.convert('RGB'))
    alpha = np.array(rgba.getchannel('A'))
    for layer in (editor or {}).get('layers') or []:
        if layer is None:
            continue
        if layer.size != item.size:
            raise gr.Error('누끼 수정 영역의 크기가 가구 사진과 다릅니다.')
        paint = np.asarray(layer.convert('RGBA'))
        active = paint[..., 3] > 0
        alpha[active & (paint[..., 2] > paint[..., 0])] = 255
        alpha[active & (paint[..., 0] > paint[..., 2])] = 0
    rgba.putalpha(Image.fromarray(alpha))
    bounds = rgba.getchannel('A').getbbox()
    if bounds is None:
        raise gr.Error('가구를 모두 지웠습니다. 남길 부분을 파랑으로 칠해 주세요.')
    return rgba.crop(bounds)


def reset_item_correction(editor, previous=None):
    item = _background(editor)
    if item is not None and previous is not None and _background(previous) is not None \
            and np.array_equal(np.asarray(item), np.asarray(_background(previous))):
        return gr.skip()
    return {'background': item, 'layers': [], 'composite': item} if item else None


def make_cutout(item, box, candidate, fine_edges, editor=None, mask=None):
    edited = brush_box(editor) is not None
    rgba = cutout(item, box=box, candidate=candidate, crop=not edited,
                  refine_edges=fine_edges, mask=mask)
    return corrected_cutout(rgba, editor, item) if edited else rgba


def preview_cutout(item_ed, candidate, fine_edges, cutout_ed):
    item = _background(item_ed, keep_alpha=True)
    if item is None:
        raise gr.Error('가구 사진을 올려주세요.')
    sample = sample_item(item)
    box = brush_box(item_ed) or (pct_box(item.size, sample['box']) if sample else default_box(item.size))
    try:
        return make_cutout(item, box, None if candidate == '자동' else int(candidate), fine_edges, cutout_ed)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except OSError as e:
        raise gr.Error('가구 분리 모델을 불러오지 못했습니다. 첫 실행에는 인터넷 연결이 필요합니다.') from e


# --- 이벤트 핸들러 ---------------------------------------------------------

def run(room_ed, item_ed, item_name, engine, mode, candidate, size_mode, size_scale, pay_ok,
        x_pct, y_pct, w_pct, h_pct, refine_on=False, refine_strength=sd_refine.DEFAULT_STRENGTH,
        item_height_m=None, fine_edges=True, reference_ed=None, reference_m=None,
        horizon_pct=41, flip_item=False, occlusion_ed=None, occlusion_mode="자동",
        color_amount=.35, snap_floor=True, cutout_ed=None,
        progress=gr.Progress()):
    room = _background(room_ed)
    item = _background(item_ed, keep_alpha=True)
    if room is None:
        raise gr.Error("방 사진을 올려주세요.")
    if item is None:
        raise gr.Error("가구 사진을 올려주세요.")

    painted = brush_box(room_ed)
    if painted:
        box, how = painted, "브러시로 칠한 영역"
    else:
        box, how = percent_box(room, x_pct, y_pct, w_pct, h_pct), "슬라이더 좌표"

    item_box = brush_box(item_ed)
    log = [f"배치 위치: {how} → {box}"]

    if engine.startswith("로컬"):
        flat = mode.startswith("바닥")
        calibrated = size_mode.startswith("기준")
        if calibrated and flat:
            raise gr.Error("높이 기준 보정은 세워 놓는 가구용입니다. 러그는 수동 크기를 사용하세요.")
        sample = sample_item(item)
        if sample is not None and not sample.get("full_object", True):
            raise gr.Error("이 샘플은 조명 하단과 받침이 사진 밖으로 잘려 있습니다. "
                           "전등갓만 전체 높이로 확대할 수 없으므로 가구 전체가 나온 사진을 사용해 주세요.")
        needs_height = not flat and (size_mode.startswith("자동") or calibrated)
        height_m = item_height_m if needs_height else None
        if height_m is not None:
            if not math.isfinite(height_m) or height_m <= 0:
                raise gr.Error("자동 크기에는 가구 실제 높이를 0보다 큰 숫자(m)로 입력해 주세요. "
                               "높이를 모르면 '수동 (칠한 박스 크기)'을 선택하세요.")
        elif sample:
            height_m = sample["height_m"]
        if needs_height and height_m is None:
            raise gr.Error("가구 실제 높이(m)를 입력하거나 '수동' 크기 모드를 선택해 주세요. "
                           "가구 이름만으로 실제 치수를 정하지 않습니다.")
        ref_box = None
        if calibrated:
            ref_box = brush_box(checked_editor(reference_ed, room))
            if ref_box is None or reference_m is None:
                raise gr.Error("기준 물체의 바닥부터 꼭대기까지 칠하고 실제 높이(m)를 입력하세요.")
            if not math.isfinite(horizon_pct) or not 0 <= horizon_pct <= 95:
                raise gr.Error("크기 기준선은 0~95% 범위로 입력하세요.")
            try:
                reference_height_px(ref_box[1], ref_box[3], reference_m, height_m,
                                    box[3], room.height * horizon_pct / 100)
            except ValueError as e:
                raise gr.Error(str(e)) from e
        if not math.isfinite(color_amount) or not 0 <= color_amount <= .6:
            raise gr.Error("조명 보정 강도는 0~0.6 범위로 입력하세요.")
        manual_keep = np.zeros((room.height, room.width), bool)
        if occlusion_mode != "끔":
            editor = checked_editor(occlusion_ed, room)
            for layer in (editor or {}).get("layers") or []:
                if layer is not None:
                    if layer.size != room.size:
                        raise gr.Error("가림 영역 크기가 현재 방 사진과 다릅니다.")
                    manual_keep |= np.asarray(layer.convert("RGBA").getchannel("A")) > 0
            if occlusion_mode == "칠한 영역만" and not manual_keep.any():
                raise gr.Error("가구보다 앞에 남길 물체를 가림 보정 사진에 칠해 주세요.")
        progress(0.1, desc="깊이 추정")
        depth = estimate_depth(room)
        progress(0.4, desc="바닥 평면 추정")
        try:
            plane = floor_plane(room, depth)
        except OSError as e:
            raise gr.Error('바닥 구분 모델을 불러오지 못했습니다. 첫 실행에는 인터넷 연결이 필요합니다.') from e
        if plane["coef"] is None:
            if flat:
                raise gr.Error("러그의 원근을 계산할 바닥을 찾지 못했습니다. "
                               "바닥이 더 보이는 사진을 써 주세요.")
            plane = None
            log.append(("경고: 바닥을 찾지 못했습니다. 기준 물체로 크기만 보정합니다. " if calibrated else
                        "경고: 바닥을 찾지 못해 칠한 박스 크기로 수동 합성합니다. ") +
                       "자동 크기·깊이 가림·바닥 기준 그림자는 적용하지 않습니다.")
        if calibrated and plane is not None:
            plane = {**plane, "horizon_y": room.height * horizon_pct / 100}

        progress(0.6, desc="가구 분리 (SAM)")
        item_how = "브러시"
        if item_box is None and sample is not None:
            item_box, item_how = pct_box(item.size, sample["box"]), "샘플에 미리 정해 둔 박스"
            log.append(f"샘플 가구({sample['name']})라 미리 정해 둔 가구 박스를 씁니다. "
                       "가구 사진을 직접 칠하면 그 박스가 우선합니다.")
        elif item_box is None:
            item_how = "기본(중앙 80%)"
            log.append("안내: 가구 사진을 칠하지 않아 중앙 80% 기본 박스로 가구를 분리합니다. "
                       "샘플 10개에서 기본 박스는 4개만 제대로 분리됐고, "
                       "가구를 감싸게 칠하면 8개가 제대로 분리됐습니다.")
        cand = None if candidate == "자동" else int(candidate)
        raw_box = item_box or default_box(item.size)
        used_box = clamp_box(raw_box, item.size)
        mask = None
        if has_alpha(item):
            risk = 0.0
            log.append("가구 사진에 이미 투명 배경이 있어 SAM을 건너뛰고 그 알파를 씁니다.")
        else:
            mask = segment(item, raw_box, cand)
            risk = background_risk(mask, used_box)
        if risk > 0.55:
            log.append(f"경고: 가구 대신 배경이 잡혔을 수 있습니다 (배경 위험도 {risk:.2f}). "
                       "SAM 후보를 0/1/2로 바꾸거나 가구에 더 딱 맞게 칠해보세요.")
        try:
            manual_cutout = brush_box(cutout_ed) is not None
            if fine_edges and mask is not None:
                progress(0.75, desc="가구 틈과 경계 정리")
            rgba = make_cutout(item, raw_box, cand, fine_edges, cutout_ed, mask)
            if manual_cutout:
                log.append('누끼 직접 수정 적용 (파랑 복구·빨강 제거)')
        except ValueError as e:
            raise gr.Error(str(e)) from e
        except OSError as e:
            raise gr.Error("가구 분리 모델을 불러오지 못했습니다. 첫 실행에는 인터넷 연결이 필요합니다. "
                           "기존 모델만 받았다면 '가구 틈·경계 정리'를 끄고 다시 실행해 주세요.") from e
        # 칠한 박스보다 누끼가 훨씬 작으면 가구 일부만 잡혔을 수 있다(조명의 갓만 등).
        # 경고만 한다. 넉넉히 칠해도 비율이 내려가서(침대 0.78) 크기를 바꾸면 오작동한다.
        cover = None if (item_box is None or has_alpha(item)) \
            else box_coverage(rgba, raw_box, item.size)
        if cover is not None and cover < MIN_COVER:
            log.append(f"확인: 칠한 영역의 {cover * 100:.0f}% 높이만 가구로 잡혔습니다. "
                       "넉넉히 칠해서라면 괜찮습니다. 가구 일부만 잡혔다면(예: 조명의 갓만) "
                       "SAM 후보를 0/1/2로 바꿔보거나 가구 전체가 찍힌 사진을 쓰세요.")

        progress(0.85, desc="원근 배치 · 합성")
        # 사용자가 입력한 실제 높이를 우선한다. 저장 높이는 샘플 사진에만 쓴다.
        height_px = None
        if calibrated:
            height_px = reference_height_px(ref_box[1], ref_box[3], reference_m, height_m,
                                             box[3], room.height * horizon_pct / 100)
            log.append(f"기준 물체 {reference_m:g}m / 화면 {ref_box[3]-ref_box[1]}px로 높이를 보정했습니다. "
                       "같은 바닥·평행한 수직선 근사이며 거리 차이는 크기 기준선에 의존합니다.")
        elif not flat and plane is not None and size_mode.startswith("자동"):
            height_px = auto_height_px(plane, box[3], room.size, height_m)
            if height_px is None:
                raise gr.Error("가구의 접지점을 주황색 크기 기준선 아래로 옮기거나 수동 모드를 선택해 주세요.")
            source = "입력값" if item_height_m is not None else "샘플 사진의 저장값"
            log.append(f"가구 높이: {height_m:g}m ({source}). 카메라 높이는 1.4m로 가정합니다.")
        if flip_item:
            rgba = ImageOps.mirror(rgba)
            log.append("가구 좌우 반전 적용: 글자와 비대칭 구조도 반전됩니다. 촬영 각도의 3D 회전은 아닙니다.")
        if snap_floor and not flat:
            box, height_px, placement_note = floor_placement(plane, box, rgba.size, room.size,
                                                            size_scale, height_px, manual_keep)
            log.append("바닥 배치 보정: " + placement_note)
        M = place_transform(plane, box, rgba.size, room.size,
                            mode="flat" if flat else "upright",
                            scale=size_scale, height_px=height_px)
        result = compose(room, rgba, M, plane, depth=depth, shadow=plane is not None,
                         harmonize_amount=color_amount, occlude=occlusion_mode.startswith("자동"),
                         protected_mask=manual_keep)
        from pipeline.compose import _warp_rgba
        placed_alpha = _warp_rgba(rgba, M, room.size)[1]
        protected = manual_keep.astype(np.float32)
        if occlusion_mode.startswith("자동"):
            auto_keep = occluder_mask(placed_alpha, plane, depth)
            if auto_keep is not None:
                protected = np.maximum(protected, auto_keep)
        log += ["경고: " + w for w in placement_check(placed_alpha, plane, room.size,
                                                    check_size=height_px is None)]

        # 선택 단계: 가구 주변만 로컬 SD로 다시 그린다 (DEVLOG §26, §27)
        if refine_on:
            if not sd_refine.available():
                log.append("AI 다듬기: diffusers가 설치되지 않아 건너뜁니다 (pip install diffusers).")
            else:
                progress(0.92, desc="AI 다듬기 (Stable Diffusion)")
                meta = next((it for it in ITEMS if it["name"] == (item_name or "").strip()), sample)
                t0 = time.time()
                result = sd_refine.refine(result, placed_alpha, box, meta["en"] if meta else None,
                                          refine_strength,
                                          protected_mask=protected)
                log.append(f"AI 다듬기: 강도 {refine_strength:.2f}, {time.time() - t0:.1f}초. "
                           "가구 픽셀은 유지하고 주변 배경·그림자만 다듬었습니다.")

        log += [
            f"가구 박스: {item_how} → {used_box}"
            + ("  (가장자리에서 8% 안쪽으로 조정됨)" if used_box != raw_box else ""),
            f"배경 위험도: {risk:.2f} (0.55 넘으면 배경일 수 있음)",
            f"SAM 후보: {candidate}",
            f"틈·경계 보정: {rgba.info.get('edge_refinement', '입력 투명도 유지')}",
            f"가림: {occlusion_mode}, 직접 보호 {int(manual_keep.sum())}픽셀 · 조명 보정 {color_amount:.2f}",
            (f"크기 기준선 y: {plane['horizon_y']:.0f} ({'직접 지정' if calibrated else '사진 높이 41% 가정'})"
             if plane is not None else "크기 기준선: 바닥 추정 실패로 사용하지 않음"),
            (f"바닥 추정 면적: {plane['mask'].mean() * 100:.1f}%"
             if plane is not None else "바닥 추정 면적: 없음"),
            f"배치 방식: {mode}   크기 배율: {size_scale:.2f}",
            (f"크기: 자동 (배율 적용 후 화면 {height_px * size_scale:.0f}px)" if height_px
             else "크기: 박스 기준 (수동)"),
        ]
        guide = preview(room, box, plane if plane is not None else
                        ({"horizon_y": room.height * horizon_pct / 100} if calibrated else None))
        ay, ax = np.where(placed_alpha > .5)
        if len(ay):
            ImageDraw.Draw(guide).rectangle((int(ax.min()), int(ay.min()), int(ax.max()), int(ay.max())),
                                           outline=(0, 220, 230), width=2)
        if ref_box is not None:
            ImageDraw.Draw(guide).rectangle(ref_box, outline=(40, 120, 255), width=3)
        return room, guide, result, "\n".join(log), rgba

    # --- Gemini API 경로 (비교 대상) ---
    if api.REAL_API and not pay_ok:
        raise gr.Error(f"유료 호출입니다 (약 ${api.COST_PER_CALL:.3f}). "
                       "'유료 호출에 동의합니다'를 켜거나 로컬 파이프라인을 쓰세요.")
    x0, y0, x1, y1 = box
    marked = api.draw_marker(room, box)
    item = item.convert("RGB")
    prompt = api.build_prompt(item_name)
    progress(0.5, desc="API 호출")
    try:
        result = api.generate(marked, item, prompt)
    except RuntimeError as e:
        raise gr.Error(str(e))
    log += [f"모드: {'실제 API' if api.REAL_API else 'mock (REAL_API=0 이라 원본 반환)'}",
            f"모델: {api.MODEL}",
            f"이번 세션 유료 호출 {api.call_count()}회 · 누적 약 ${api.spent():.2f}",
            "", prompt]
    return room, marked, result, "\n".join(log), None


def _examples():
    rooms = sorted(SAMPLE_ROOMS.glob("*.png"))
    return [[str(room), str(SAMPLE_ITEMS / item["file"]), item["name"], item["height_m"],
             *item["demo_box"], "자동", "자동 (입력 높이·촬영 높이 가정)", 1.0, False, True]
            for room, item in zip(rooms, ITEMS) if item.get("full_object", True)]


with gr.Blocks(title="AI 셀프 인테리어 시각화") as demo:
    gr.Markdown(
        "# AI 셀프 인테리어 시각화\n"
        "방 사진에는 **가구를 놓을 자리**를 칠하세요. 칠한 영역의 아랫변이 접지점입니다. "
        "자동 크기는 입력한 실제 높이로, 수동 크기와 러그는 칠한 박스로 정합니다. "
        "가구 사진에는 **가구를 감싸는 영역**을 칠하세요. "
        "가구 전체와 발끝이 보여야 하며 다른 물체에 가려진 부분은 복원되지 않습니다."
    )
    if api.REAL_API:
        gr.Markdown(
            f"### ⚠️ 유료 모드입니다 — Gemini 경로는 호출 1회당 약 ${api.COST_PER_CALL:.3f}\n"
            "아래 **확인란을 켜야** Gemini 경로가 실행됩니다. 로컬 파이프라인은 무료라 제한이 없습니다."
        )
    else:
        gr.Markdown("Gemini API는 현재 mock입니다 (`.env`의 `REAL_API=1`로 실제 호출). "
                    "로컬 파이프라인은 mock 없이 항상 실제로 동작합니다.")

    with gr.Row():
        with gr.Column():
            room_ed = gr.ImageEditor(label="방 사진 — 놓을 자리를 칠하세요", type="pil", format="png",
                                     layers=False, brush=gr.Brush(colors=["#ff4d4d"], default_size=40))
            gr.Markdown("칠하지 않으면 아래 슬라이더를 씁니다.")
            with gr.Row():
                x_pct = gr.Slider(0, 95, value=35, step=1, label="가로 %")
                y_pct = gr.Slider(0, 95, value=55, step=1, label="세로 %")
            with gr.Row():
                w_pct = gr.Slider(5, 100, value=30, step=1, label="너비 %")
                h_pct = gr.Slider(5, 100, value=30, step=1, label="높이 %")
        with gr.Column():
            item_ed = gr.ImageEditor(label="가구 사진 — 가구를 감싸게 칠하세요", type="pil", format="png",
                                     layers=False, brush=gr.Brush(colors=["#4d9dff"], default_size=60))
            item_name = gr.Textbox(label="가구 이름 (선택, 생성 모델 설명용)",
                                   placeholder="예: 원목 의자", lines=1, max_lines=1)
            item_height_m = gr.Number(value=None, precision=3,
                                      label="가구 실제 높이 (m, 자동 크기용)",
                                      info="자동·기준 물체 모드에서 필요합니다. 높이를 모르면 수동 크기를 선택하세요. 비워두면 샘플 사진에만 저장 높이를 씁니다.")
            engine = gr.Radio(["로컬 파이프라인", "Gemini API (비교)"],
                              value="로컬 파이프라인", label="합성 방식")
            pay_ok = gr.Checkbox(value=False, visible=api.REAL_API,
                                 label=f"유료 호출에 동의합니다 (1회 약 ${api.COST_PER_CALL:.3f})")
            with gr.Row():
                mode = gr.Radio(["세워놓기 (의자·책장)", "바닥에 깔기 (러그)"],
                                value="세워놓기 (의자·책장)", label="배치 방식")
                candidate = gr.Radio(["자동", "0", "1", "2"], value="자동",
                                     label="SAM 후보 (누끼가 이상하면 바꿔보세요)")
            size_mode = gr.Radio(["자동 (입력 높이·촬영 높이 가정)", "기준 물체 (높이 비교)", "수동 (칠한 박스 크기)"],
                                 value="자동 (입력 높이·촬영 높이 가정)", label="크기 결정 방식")
            size_scale = gr.Slider(0.4, 2.5, value=1.0, step=0.05,
                                   label="크기 미세조정 배율")
            fine_edges = gr.Checkbox(value=True, label="가구 틈·경계 정리",
                                     info="다리 사이의 원래 배경을 제거합니다. 가구 일부가 지워지면 꺼보세요.")
            cutout_btn = gr.Button('가구 분리 먼저 확인')
            snap_floor = gr.Checkbox(value=True, label="가까운 바닥 자리로 위치 보정",
                                     info="가구 밑면이 들어가는 가까운 바닥을 찾습니다. 정확히 지정한 위치를 유지하려면 끄세요.")
            with gr.Accordion('분리된 가구 수정', open=False):
                gr.Markdown('생성 후 아래 **분리된 가구**에서 상태를 확인하세요. 원래 사진에 보이는데 지워진 부분은 '
                            '**파랑으로 복구**, 남은 배경은 **빨강으로 제거**한 뒤 다시 생성하세요.')
                cutout_ed = gr.ImageEditor(label='파랑: 가구 복구 / 빨강: 배경 제거', type='pil', format='png', layers=False,
                                           brush=gr.Brush(colors=['#2878ff', '#ff4d4d'], default_size=8))
            with gr.Accordion("크기·방향·가림 보정", open=False):
                gr.Markdown("**크기:** '기준 물체' 모드를 고른 뒤, 아래 방 사진에서 높이를 아는 물체의 "
                            "바닥부터 꼭대기까지 세로로 칠하세요. 수직 모서리를 따라 표시하면 됩니다. "
                            "기준 물체와 가구는 같은 바닥에 있어야 합니다.")
                reference_ed = gr.ImageEditor(label="크기 기준 물체 표시", type="pil", format="png", layers=False,
                                               brush=gr.Brush(colors=["#2878ff"], default_size=4))
                reference_m = gr.Number(value=None, label="기준 물체의 실제 높이 (m)",
                                        info="기준 물체 모드에서만 사용합니다.")
                horizon_pct = gr.Slider(0, 95, value=41, step=.5, label="크기 기준선 위치 (사진 위에서 %)",
                                        info="거리 차이를 보정할 기준선입니다. 모르면 41% 가정이 남습니다. 처리 결과의 주황선을 확인하세요.")
                gr.Markdown("**방향:** 가구 사진은 방과 비슷한 높이·방향에서 찍은 것을 사용하세요. "
                            "좌우 반전은 가능하지만 보이지 않는 옆면을 새로 만들지는 않습니다.")
                flip_item = gr.Checkbox(value=False, label="가구 좌우 반전 (글자·비대칭 구조도 반전)")
                gr.Markdown("**가림:** 아래에서 새 가구보다 앞에 있어야 할 기존 물체를 칠하세요. "
                            "자동 가림이 잘못되면 '칠한 영역만'을 선택하세요.")
                occlusion_ed = gr.ImageEditor(label="앞에 남길 물체 표시", type="pil", format="png", layers=False,
                                               brush=gr.Brush(colors=["#ff4d4d"], default_size=20))
                occlusion_mode = gr.Radio(["자동", "칠한 영역만", "끔"], value="자동", label="앞뒤 가림")
                color_amount = gr.Slider(0, .6, value=.35, step=.05, label="방 조명에 맞추는 강도",
                                         info="0이면 원래 가구 색을 유지합니다. 유리에 비친 원래 풍경은 제거하지 못합니다.")
            with gr.Row():
                refine_on = gr.Checkbox(
                    value=False, interactive=sd_refine.available(),
                    label="주변 AI 다듬기 (가구 유지, 실험)",
                    info=("가구는 유지하고 주변 배경·그림자만 다시 그립니다. 기기에 따라 수십 초 더 걸리며, "
                          "처음 켤 때 모델 약 2.9GB를 받습니다." if sd_refine.available()
                          else "diffusers가 설치되지 않아 쓸 수 없습니다 (pip install diffusers)."))
                refine_strength = gr.Slider(0.2, 0.8, value=sd_refine.DEFAULT_STRENGTH, step=0.05,
                                            label="주변 수정 강도 (높을수록 배경 변화가 큼)")
            run_btn = gr.Button("가구 배치 생성", variant="primary")

    with gr.Row():
        before = gr.Image(label="Before", type="pil")
        middle = gr.Image(label="가구 크기(하늘색) · 바닥(초록) · 기준선(주황) · 기준 물체(파랑)", type="pil")
        after = gr.Image(label="After", type="pil")

    log_box = gr.Textbox(label="처리 내역", lines=9)
    cutout_preview = gr.Image(label='분리된 가구 — 다리·틈·잘린 부분 확인', type='pil', format='png', image_mode='RGBA')

    run_btn.click(run,
                  inputs=[room_ed, item_ed, item_name, engine, mode, candidate,
                          size_mode, size_scale, pay_ok, x_pct, y_pct, w_pct, h_pct,
                          refine_on, refine_strength, item_height_m, fine_edges,
                          reference_ed, reference_m, horizon_pct, flip_item, occlusion_ed, occlusion_mode, color_amount, snap_floor, cutout_ed],
                  outputs=[before, middle, after, log_box, cutout_preview])
    room_ed.change(reset_room_corrections, inputs=[room_ed, reference_ed, occlusion_ed],
                   outputs=[reference_ed, occlusion_ed])
    item_ed.change(reset_item_correction, inputs=[item_ed, cutout_ed], outputs=[cutout_ed])
    cutout_btn.click(preview_cutout, inputs=[item_ed, candidate, fine_edges, cutout_ed], outputs=[cutout_preview])

    if (ex := _examples()):
        gr.Examples(examples=ex,
                    inputs=[room_ed, item_ed, item_name, item_height_m,
                            x_pct, y_pct, w_pct, h_pct, candidate, size_mode, size_scale, refine_on, fine_edges],
                    example_labels=[row[2] for row in ex],
                    label="샘플 — 참고 높이와 배치 위치를 함께 불러옵니다")


if __name__ == "__main__":
    print(f"Gemini API REAL_API={'1' if api.REAL_API else '0'}  MODEL={api.MODEL}")
    print("로컬 파이프라인 첫 실행 시 모델 로드에 20초 정도 걸립니다.")
    demo.launch()
