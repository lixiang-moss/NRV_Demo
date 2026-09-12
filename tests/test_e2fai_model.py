from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
import torch
from nrv_e2fai import load_recurrent_model


class ModelParityTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "checkpoints/e2fai_backbone.ckpt").is_file()
                         and (ROOT / "checkpoints/image_residual_epoch043.pt").is_file(),
                         "Local paired checkpoints are required")
    def test_complete_checkpoint_model_matches_research_source_for_recurrent_steps(self):
        # Small native synthetic sensor keeps correctness testing inexpensive;
        # full 960x720 camera/GPU measurements are a separate performance run.
        torch.set_num_threads(2)
        backbone = ROOT / "checkpoints/e2fai_backbone.ckpt"
        adapter = ROOT / "checkpoints/image_residual_epoch043.pt"
        actual, metadata = load_recurrent_model(adapter, backbone_checkpoint=backbone,
                                                device="cpu", sensor_height=32, sensor_width=48)
        source = subprocess.check_output(["git", "show", "7d2fb86:runtime/nrv_e2fai/model.py"], cwd=ROOT, text=True)
        module = types.ModuleType("reference_e2fai_model")
        exec(compile(source, "7d2fb86_model", "exec"), module.__dict__)
        payload = torch.load(adapter, map_location="cpu")
        expected = module.FrozenFlowRecurrentImageE2FAI(backbone, 32, 48,
                    payload["image_adapter"]["projection.weight"].shape[0])
        expected.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
        expected.eval()
        self.assertEqual(metadata["backbone_sha256"], payload["pretrained_sha256"])
        self.assertFalse(any(parameter.requires_grad for parameter in actual.backbone.parameters()))
        rng = torch.Generator().manual_seed(913)
        expected_state = actual_state = None
        with torch.inference_mode():
            for step in range(3):
                voxel = torch.randn(1, 15, 32, 48, generator=rng)
                expected_output, expected_state = expected.forward_step(voxel, expected_state)
                actual_output, actual_state = actual.forward_step(voxel, actual_state)
                for key in ("flow", "log_image", "image"):
                    torch.testing.assert_close(actual_output[key], expected_output[key], rtol=0, atol=0)
                torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
                self.assertEqual(actual_output["flow"].shape, (1, 2, 32, 48))


if __name__ == "__main__":
    unittest.main()
