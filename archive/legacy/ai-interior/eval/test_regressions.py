"""리뷰 회귀 검사. 모델 다운로드·추론·API 호출 없이 실행: python eval/test_regressions.py"""
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['REAL_API'] = '0'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from pipeline.geometry import place_transform, reference_height_px, floor_placement, floor_plane
from pipeline.segment import cutout, has_alpha, _clean, _refined_alpha
from pipeline.compose import _warp_rgba, compose, occluder_mask, contact_shadow
from pipeline import refine
from eval.size_check import evaluation_split


def check_geometry():
    floor = np.ones((200, 300), bool)
    floor[:80] = False
    floor[110:175, 115:185] = False
    plane = {'mask': floor, 'placement_mask': floor, 'horizon_y': 70, 'coef': (0., 1., 0.)}
    box = (125, 100, 175, 170)
    moved, height, note = floor_placement(plane, box, (40, 70), (300, 200))
    assert moved != box and height is None and '이동' in note
    unchanged, _, _ = floor_placement(plane, moved, (40, 70), (300, 200))
    assert unchanged == moved
    moved, height, _ = floor_placement(plane, box, (40, 70), (300, 200), height_px=70.)
    assert np.isclose(height/70, (moved[3]-70)/(box[3]-70))
    assert floor_placement(None, box, (40, 70), (300, 200))[0] == box
    assert floor_placement(plane, box, (400, 700), (300, 200), height_px=700)[0] == box
    protected = np.ones((200, 300), bool)
    assert floor_placement(plane, box, (40, 70), (300, 200), protected_mask=protected)[0] == box
    y, x = np.mgrid[:200, :300]
    depth = (.2*x/300 + .5*y/200 + .1).astype(np.float32)
    with patch('pipeline.floor.floor_regions', return_value=(floor, floor)):
        fitted = floor_plane(Image.new('RGB', (300, 200)), depth)
        assert np.allclose(fitted['coef'], (.2, .5, .1), atol=1e-5)
    # A reference recovers known scale despite different camera heights / image crops.
    for horizon in (20, 110, 240):
        for camera in (1., 1.4, 2.):
            bottom = 400.
            ref_pixels = (bottom - horizon) * 1.8 / camera
            for target_y in (300., 500.):
                predicted = reference_height_px(bottom-ref_pixels, bottom, 1.8, .9, target_y, horizon)
                assert np.isclose(predicted, (target_y-horizon)*.9/camera)
    for values in ((0, 100, 0, 1, 200, 40), (0, 5, 1, 1, 200, 40),
                   (0, 100, 1, 1, 20, 40), (0, 100, float('nan'), 1, 200, 40)):
        try:
            reference_height_px(*values)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid reference accepted')
    corners = np.float32([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    for width in (124, 125):
        box = (200 - width / 2, 100, 200 + width / 2, 400)
        out = cv2.perspectiveTransform(corners, place_transform(None, box, (100, 100), (500, 500), height_px=200))[0]
        assert np.isclose(out[3, 1] - out[0, 1], 200)
    plane = {'coef': (0, 1, 0)}
    for scale in (.4, 1, 2.5):
        out = cv2.perspectiveTransform(corners, place_transform(plane, (100, 200, 300, 400), (100, 100), (500, 500), mode='flat', scale=scale))[0]
        assert np.allclose((out[2] + out[3]) / 2, [200, 400])
        assert np.isclose(out[2, 0] - out[3, 0], 200 * scale)
    # A manual upright placement remains possible with no floor, preserving aspect ratio.
    out = cv2.perspectiveTransform(corners, place_transform(None, (100, 200, 200, 400), (100, 100), (500, 500)))[0]
    assert np.isclose(out[2, 0] - out[3, 0], out[3, 1] - out[0, 1])


def check_alpha():
    arr = np.full((8, 8, 4), [180, 90, 30, 0], dtype=np.uint8)
    arr[2:6, 2:6, 3] = np.array([1, 64, 192, 255], dtype=np.uint8)
    rgba = Image.fromarray(arr)
    with patch('pipeline.segment.segment', side_effect=AssertionError('PNG must bypass SAM')):
        assert np.array_equal(np.asarray(cutout(rgba, crop=False)), arr)
        assert np.array_equal(np.asarray(cutout(rgba)), arr[2:6, 2:6])
        la = rgba.convert('LA')
        assert np.array_equal(np.asarray(cutout(la, crop=False))[:, :, 3], arr[:, :, 3])
        pal = Image.new('P', (8, 8), 0)
        pal.putpixel((4, 4), 1)
        pal.info['transparency'] = bytes([0, 192] + [255] * 254)
        assert has_alpha(pal) and cutout(pal).getpixel((0, 0))[3] == 192
        try:
            cutout(Image.new('RGBA', (8, 8), (0, 0, 0, 0)))
        except ValueError:
            pass
        else:
            raise AssertionError('Empty cutout must fail explicitly')
    edge = Image.fromarray(np.array([[[255, 0, 0, 255], [0, 0, 0, 0]]] * 2, dtype=np.uint8))
    rgb, alpha = _warp_rgba(edge, np.array([[1, 0, .5], [0, 1, 0], [0, 0, 1]], float), (3, 2))
    blended = 255 * (1 - alpha[0, 1]) + rgb[0, 1] * alpha[0, 1]
    assert np.isclose(blended[0], 255) and np.isclose(alpha[0, 1], .5)

    # Fine source texture must average to gray, not become black dots when shrunk.
    checker = (np.indices((128, 128)).sum(axis=0) % 2 * 255).astype(np.uint8)
    texture = Image.fromarray(np.dstack([checker, checker, checker, np.full_like(checker, 255)]))
    rgb, alpha = _warp_rgba(texture, np.diag([.125, .125, 1]), (16, 16))
    assert np.allclose(rgb[2:-2, 2:-2], 127.5, atol=1)
    assert np.allclose(alpha[2:-2, 2:-2], 1)
    # A real opening smaller than the old hole threshold must remain transparent.
    mask = np.zeros((200, 200), bool)
    mask[20:180, 20:180] = True
    mask[60:65, 60:65] = False
    mask[1:3, 1:3] = True
    clean = _clean(mask)
    assert not clean[60:65, 60:65].any() and not clean[1:3, 1:3].any()
    assert clean[20:60, 20:60].all()
    # Remove enclosed background but keep thin legs, and never admit another object.
    mask = np.zeros((100, 100), bool)
    mask[10:90, 20:80] = True
    matte = mask.astype(np.float32)
    matte[55:85, 25:75] = 0
    matte[:5, :5] = 1
    refined = _refined_alpha(mask, matte)
    assert refined is not None and not refined[55:85, 25:75].any()
    assert refined[55:90, 20:25].min() == 255 and not refined[:5, :5].any()
    # A model that erases the tabletop or legs must fall back to SAM.
    assert _refined_alpha(mask, np.zeros_like(matte)) is None
    bad = mask.astype(np.float32); bad[50:] = 0
    assert _refined_alpha(mask, bad) is None
    with patch('pipeline.segment._foreground_alpha', return_value=matte), patch('pipeline.segment.segment', side_effect=AssertionError('Reuse the selected mask')):
        result = cutout(Image.new('RGB', (100, 100)), mask=mask, crop=False)
        assert np.array_equal(np.asarray(result.getchannel('A')), refined)
    # Strong directional light must not detach contact shadows from either foot.
    legs = np.zeros((140, 140), np.float32)
    legs[20:60, 30:110] = 1
    legs[60:110, 30:35] = 1
    legs[60:120, 105:110] = 1
    for light in (-1, 0, 1):
        shadow = contact_shadow(legs, light=light)
        assert shadow[109, 32] > .2 and shadow[119, 107] > .2
        assert not shadow[130:, :].any()
    assert not contact_shadow(legs, strength=0).any()


def check_refinement():
    room_array = np.full((96, 128, 3), 180, np.uint8)
    room_array[30:65, 64:110] = (20, 40, 60)
    room = Image.fromarray(room_array)
    item = Image.new('RGBA', (50, 50), (200, 100, 20, 255))
    matrix = np.array([[1, 0, 40], [0, 1, 20], [0, 0, 1]], float)
    _, alpha = _warp_rgba(item, matrix, room.size)
    depth = np.full((96, 128), .2, np.float32)
    depth[30:65, 64:110] = .95
    floor = np.zeros((96, 128), bool)
    floor[75:] = True
    plane = {'coef': (0., 1., 0.), 'mask': floor, 'horizon_y': 0.}
    comp = compose(room, item, matrix, plane, shadow=False, harmonize_amount=0, depth=depth)
    protected = occluder_mask(alpha, plane, depth)
    captured = []

    def fake(**kw):
        captured.append(kw)
        return SimpleNamespace(images=[Image.new('RGB', kw['image'].size, (255, 0, 255))])

    with patch.object(refine, 'estimate_depth', return_value=np.zeros((96, 128), np.float32)), patch.object(refine, '_load', return_value=fake):
        for mode in ('whole', 'crop'):
            out = np.asarray(refine.refine(comp, alpha, (40, 20, 90, 70), mode=mode, protected_mask=protected))
            keep = (protected > 0) | (alpha > 0)
            assert keep.any() and np.array_equal(out[keep], np.asarray(comp)[keep])
            assert np.array_equal(out[protected >= 1], room_array[protected >= 1])
            assert np.any(out[~keep] != np.asarray(comp)[~keep])
            mask = captured[-1]['mask_image']
            guard = Image.fromarray(keep.astype(np.float32)).resize(mask.size, Image.Resampling.BOX)
            assert np.all(np.asarray(mask)[np.asarray(guard) > 0] == 0)

    # A tiny location brush must not restrict refinement to a tall item's feet.
    room = Image.new('RGB', (600, 600), (180, 180, 180))
    item = Image.new('RGBA', (60, 100), (200, 100, 20, 255))
    box = (290, 520, 310, 540)
    matrix = place_transform(None, box, item.size, room.size, height_px=378)
    _, alpha = _warp_rgba(item, matrix, room.size)
    comp = compose(room, item, matrix, None, shadow=False, harmonize_amount=0)
    with patch.object(refine, 'estimate_depth', return_value=np.zeros((600, 600), np.float32)), patch.object(refine, '_load', return_value=fake):
        out = np.asarray(refine.refine(comp, alpha, box))
        visible = alpha > .9
        changed = np.any(out != np.asarray(comp), axis=2)
        assert not changed[alpha > 0].any()
        upper_surroundings = (refine.redraw_mask(alpha) > 0) & (alpha == 0)
        upper_surroundings[200:] = False
        assert upper_surroundings.any() and changed[upper_surroundings].mean() > .99
        legacy = np.asarray(refine.refine(comp, alpha, box, preserve_item=False))
        assert np.any(legacy != np.asarray(comp), axis=2)[visible].mean() > .99


def check_app():
    import app
    room = Image.new('RGB', (200, 200), (180, 180, 180))
    item = Image.new('RGBA', (60, 80), (180, 40, 20, 192))
    assert app.preview_cutout({'background': item}, '자동', True, None).tobytes() == item.tobytes()
    plane = {'coef': (0, 1, 0), 'mask': np.ones((200, 200), bool), 'horizon_y': 82.}
    depth = np.tile(np.linspace(0, 1, 200, dtype=np.float32)[:, None], (1, 200))
    args = dict(room_ed={'background': room}, item_ed={'background': item}, item_name='원목 의자',
                engine='로컬', mode='세워놓기', candidate='자동', size_mode='자동', size_scale=1.,
                pay_ok=False, x_pct=35, y_pct=55, w_pct=30, h_pct=30, progress=lambda *a, **k: None)
    old = {'background': room, 'layers': [Image.new('RGBA', room.size)]}
    assert app.reset_room_corrections({'background': room}, old, old) == (app.gr.skip(), app.gr.skip())
    changed = Image.new('RGB', room.size, 'white')
    assert app.checked_editor({'background': changed, 'layers': []}, room) is None
    # Browser transport must preserve pixels used to match correction backgrounds.
    transport = Image.fromarray(np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8))
    for component in (app.room_ed, app.item_ed, app.reference_ed, app.occlusion_ed, app.cutout_ed):
        restored = component.preprocess(component.postprocess(transport))
        assert np.array_equal(np.asarray(app._background(restored)), np.asarray(transport))
    reset = app.reset_room_corrections({'background': changed}, old, old)
    assert all(not v['layers'] and np.array_equal(np.asarray(v['background']), np.asarray(changed)) for v in reset)
    # Restore a thin source leg and remove a residual background patch without changing RGB.
    raw = Image.new('RGB', (30, 40), (80, 60, 40))
    separated = raw.convert('RGBA'); separated.putalpha(0)
    paint = Image.new('RGBA', raw.size)
    from PIL import ImageDraw
    ImageDraw.Draw(paint).rectangle((5, 5, 20, 35), fill=(40, 120, 255, 255))
    ImageDraw.Draw(paint).rectangle((10, 10, 15, 15), fill=(255, 40, 40, 255))
    repaired = app.corrected_cutout(separated, {'background': raw, 'layers': [paint]}, raw)
    assert repaired.size == (16, 31) and repaired.getpixel((0, 0)) == (80, 60, 40, 255)
    assert repaired.getpixel((5, 5))[3] == 0
    with patch.object(app, 'estimate_depth', return_value=depth), patch.object(app, 'floor_plane', return_value=plane), patch.object(app, 'sample_item', return_value=None):
        try:
            app.run(**args)
        except app.gr.Error as error:
            assert '높이' in str(error)
        else:
            raise AssertionError('A catalog name alone must not supply dimensions')
        for value in (0, -1, float('nan'), float('inf')):
            try:
                app.run(**args, item_height_m=value)
            except app.gr.Error:
                pass
            else:
                raise AssertionError('Invalid real height was accepted')
        result = app.run(**args, item_height_m=1.2)
        assert result[2].size == room.size and '1.2m (입력값)' in result[3]
        # Gradio preprocesses even collapsed/unused controls before calling run.
        zero_height = app.item_height_m.preprocess(0)
        zero_reference = app.reference_m.preprocess(0)
        assert app.run(**{**args, 'size_mode': '수동'}, item_height_m=zero_height,
                       reference_m=zero_reference)[2].size == room.size
        assert app.run(**args, item_height_m=1.2, reference_m=zero_reference)[2].size == room.size
        assert app.run(**args, item_height_m=1.2,
                       occlusion_ed={'background': changed, 'layers': []})[2].size == room.size
        reference = Image.new('RGBA', room.size)
        from PIL import ImageDraw
        ImageDraw.Draw(reference).rectangle((15, 30, 18, 169), fill=(0, 0, 255, 255))
        calibrated = {**args, 'size_mode': '기준 물체', 'item_height_m': 1.2,
                      'reference_ed': {'background': room, 'layers': [reference]}, 'reference_m': 2.0}
        assert '기준 물체 2m' in app.run(**calibrated)[3]
        with patch.object(app, 'floor_plane', return_value={'coef': None}):
            assert '기준 물체로 크기만' in app.run(**calibrated)[3]
        keep = Image.new('RGBA', room.size)
        ImageDraw.Draw(keep).rectangle((60, 90, 140, 180), fill=(255, 0, 0, 255))
        mask = np.asarray(keep.getchannel('A')) > 0
        protected_args = {**args, 'item_height_m': 1.2, 'occlusion_mode': '칠한 영역만',
                          'occlusion_ed': {'background': room, 'layers': [keep]}}
        protected_result = app.run(**protected_args)
        assert np.array_equal(np.asarray(protected_result[2])[mask], np.asarray(room)[mask])
        with patch.object(app.sd_refine, 'refine', return_value=room) as operation, patch.object(app.sd_refine, 'available', return_value=True):
            app.run(**protected_args, refine_on=True)
            assert operation.call_args.kwargs['protected_mask'][mask].all()
        try:
            app.run(**{**calibrated, 'reference_ed': {'background': Image.new('RGB', room.size), 'layers': [reference]}})
        except app.gr.Error:
            pass
        else:
            raise AssertionError('A reference from another room must be rejected')
        with patch.object(app, 'sample_item', return_value=app.ITEMS[0]):
            painted = Image.new('RGBA', item.size, (0, 0, 255, 255))
            sample_args = {**args, 'item_name': '', 'item_ed': {'background': item, 'layers': [painted]}}
            assert '샘플 사진의 저장값' in app.run(**sample_args)[3]
            assert '1.2m (입력값)' in app.run(**sample_args, item_height_m=1.2)[3]
        for incomplete in (app.ITEMS[6], app.ITEMS[9]):
            with patch.object(app, 'sample_item', return_value=incomplete):
                try:
                    app.run(**args)
                except app.gr.Error as error:
                    assert '사진 밖' in str(error)
                else:
                    raise AssertionError('A cropped lamp must not become a giant lampshade')
        with patch.object(app, 'floor_plane', return_value={'coef': None}):
            result = app.run(**{**args, 'size_mode': '수동'})
            assert result[2].size == room.size and '수동 합성' in result[3]
            assert '수동 합성' in app.run(**args, item_height_m=1.2)[3]
            try:
                app.run(**{**args, 'mode': '바닥에 깔기'})
            except app.gr.Error as error:
                assert '러그' in str(error)
            else:
                raise AssertionError('Flat placement requires floor geometry')
        with patch.object(app.sd_refine, 'refine', return_value=room) as operation, patch.object(app.sd_refine, 'available', return_value=True), patch.object(app, 'occluder_mask', return_value=np.ones((200, 200))):
            app.run(**args, item_height_m=1.2, refine_on=True)
            assert operation.call_args.kwargs['protected_mask'].all()
    examples = app._examples()
    assert len(examples) == 8
    for example in examples:
        assert example[3] > 0 and len(example) == 13
        assert example[8:] == ['자동', '자동 (입력 높이·촬영 높이 가정)', 1.0, False, True]


def check_room_split():
    with open(ROOT / 'eval/results/size_check/doors.csv', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    groups = {key: {r['image'] for r in rows if evaluation_split(r['image']) == key} for key in ('dev', 'test')}
    assert groups['dev'] and groups['test'] and groups['dev'].isdisjoint(groups['test'])
    assert groups['dev'] | groups['test'] == {r['image'] for r in rows}


if __name__ == '__main__':
    for check in (check_geometry, check_alpha, check_refinement, check_app, check_room_split):
        check()
        print(f'{check.__name__}: 통과')
