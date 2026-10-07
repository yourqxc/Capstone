"""네트워크/모델 없이 입력·좌표·원본 보존·실패 기록 계약을 확인합니다."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from roomfit import engine
from roomfit.placement import (Box, brush_mask, clean_image, editable_mask, generation_size,
                              placement_guide, preserve_room, selection)
from roomfit.workflow import run


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.room = Image.fromarray(np.random.default_rng(12).integers(0, 255, (192, 256, 3), dtype=np.uint8))
        self.item = Image.new("RGB", (80, 120), "white")
        self.box = Box(.3, .3, .65, .75)

    def test_coordinates_and_brush_use_alpha(self):
        for values in ((0, 0, 0, 1), (-.1, 0, 1, 1), (0, 0, 2, 1), (0, 0, float("nan"), 1)):
            with self.assertRaises(ValueError):
                Box(*values)
        layer = Image.new("RGBA", self.room.size, (255, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle((40, 60, 99, 139), fill=(255, 0, 0, 255))
        room, box = selection({"background": self.room, "layers": [None, layer]})
        self.assertEqual(box.pixels(room.size), (40, 60, 100, 140))
        with self.assertRaises(ValueError):
            selection({"background": self.room, "layers": []})
        with self.assertRaises(ValueError):
            brush_mask({"layers": [Image.new("RGBA", (32, 32))]}, self.room.size)

    def test_exact_outside_and_foreground_preservation(self):
        protected = Image.new("L", self.room.size)
        ImageDraw.Draw(protected).rectangle((110, 70, 120, 120), fill=255)
        mask = editable_mask(self.room.size, self.box, protected)
        generated = Image.new("RGB", (512, 384), "red")
        result = np.asarray(preserve_room(self.room, generated, mask))
        locked = np.asarray(mask) == 0
        self.assertTrue(np.array_equal(result[locked], np.asarray(self.room)[locked]))
        self.assertTrue(np.array_equal(result[70:121, 110:121], np.asarray(self.room)[70:121, 110:121]))
        self.assertTrue(np.all(result[np.asarray(mask) == 255] == [255, 0, 0]))
        with self.assertRaises(ValueError):
            preserve_room(self.room, generated, Image.new("L", (32, 32)))

    def test_alpha_orientation_and_aspect_ratio(self):
        transparent = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
        self.assertEqual(clean_image(transparent).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(generation_size((740, 518), 512), (512, 352))
        self.assertEqual(generation_size((518, 740), 512), (352, 512))
        with self.assertRaises(ValueError):
            generation_size((740, 518), 500)
        before = self.room.tobytes()
        placement_guide(self.room, self.box)
        self.assertEqual(self.room.tobytes(), before)

    def test_success_records_actual_inputs_and_raw_output(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(engine, "resolve_backend", return_value="mlx"), \
                patch.object(engine, "generate", return_value=Image.new("RGB", (512, 384), "red")) as call:
            result = run(self.room, self.item, self.box, output_root=Path(directory))
            meta = json.loads((result["folder"] / "run.json").read_text())
            self.assertEqual(meta["status"], "succeeded")
            self.assertTrue(meta["protected_pixels_unchanged"])
            self.assertFalse(meta["quality_verified"])
            self.assertEqual(len(call.call_args.args[0]), 2)
            self.assertTrue((result["folder"] / "raw.png").is_file())
            self.assertTrue(result["zip"].is_file())

    def test_failure_does_not_make_fake_result(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(engine, "resolve_backend", return_value="mlx"), \
                patch.object(engine, "generate", side_effect=RuntimeError("test: 모델 실패")):
            with self.assertRaisesRegex(RuntimeError, "모델 실패"):
                run(self.room, self.item, self.box, output_root=Path(directory))
            folders = list(Path(directory).iterdir())
            self.assertEqual(len(folders), 1)
            self.assertEqual(json.loads((folders[0] / "run.json").read_text())["status"], "failed")
            self.assertFalse((folders[0] / "result.png").exists())

    def test_no_cpu_fallback_or_invalid_device(self):
        with self.assertRaises(ValueError):
            engine.resolve_backend("api")
        with self.assertRaises(ValueError):
            engine.generate([Path("missing.png")], "prompt", 512, 512, 42)


if __name__ == "__main__":
    unittest.main()
