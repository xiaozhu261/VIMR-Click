import json
from contextlib import contextmanager
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from efficientvit.models.process_sam import ProcessSAMConfig
from tools.run_process_sam_validation import train_group


@contextmanager
def test_workspace():
    # tempfile's restrictive directory mode is unusable under this Windows sandbox.
    workspace = Path(__file__).resolve().parents[1]
    folder = workspace / "outputs/process_sam_resume_tests" / str(uuid.uuid4())
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        if not folder.resolve().is_relative_to(workspace / "outputs/process_sam_resume_tests"):
            raise ValueError("Test cleanup escaped its workspace")
        shutil.rmtree(folder)


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.process_config = config
        self.process_branch = torch.nn.Linear(2, 2)
        self.mask_decoder = torch.nn.Linear(2, 1)

    def forward(self, value):
        return self.mask_decoder(self.process_branch(value))

    def parameter_summary(self):
        return {"trainable": sum(p.numel() for p in self.parameters())}


def tiny_epoch(model, dataset, indices, cache, optimizer, scaler, device, amp, epoch, seed, accumulation, **kwargs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    features = torch.randn(3, 2) + random.random() + np.random.random()
    loss = (model(features) - torch.randn(3, 1)).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def tiny_evaluate(model, *args):
    model.eval()
    with torch.no_grad():
        return {"mean_iou": float(model(torch.ones(1, 2)).sigmoid().mean())}, []


class ResumeTests(unittest.TestCase):
    def run_group(self, output, resume=False, epoch_function=tiny_epoch):
        args = SimpleNamespace(output_dir=output, epochs=2, resume=resume)
        with patch(
            "tools.run_process_sam_validation.make_model", side_effect=lambda _, config, device: TinyModel(config)
        ), patch("tools.run_process_sam_validation.train_epoch", side_effect=epoch_function), patch(
            "tools.run_process_sam_validation.evaluate", side_effect=tiny_evaluate
        ):
            return train_group(
                args,
                "B1",
                9527,
                Path("unused.pt"),
                ProcessSAMConfig(use_event_history=True).to_dict(),
                {"train": [], "val": []},
                {"train": [0], "val": [0]},
                SimpleNamespace(model=None),
                torch.device("cpu"),
                False,
            )

    def assert_checkpoint_equal(self, left, right):
        a = torch.load(left / "seed_9527/B1/last.pt", weights_only=False)
        b = torch.load(right / "seed_9527/B1/last.pt", weights_only=False)
        self.assertEqual(a["history"], b["history"])
        for name in a["model"]:
            self.assertTrue(torch.equal(a["model"][name], b["model"][name]))
        for index, values in a["optimizer"]["state"].items():
            for name, tensor in values.items():
                self.assertTrue(torch.equal(tensor, b["optimizer"]["state"][index][name]))

    def test_epoch_resume_matches_uninterrupted_optimizer_and_random_inputs(self):
        with test_workspace() as root:
            self.run_group(root / "reference")

            def interrupted(*args, **kwargs):
                if args[8] == 2:
                    raise RuntimeError("simulated worker exit")
                return tiny_epoch(*args, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "worker exit"):
                self.run_group(root / "resumed", epoch_function=interrupted)
            folder = root / "resumed/seed_9527/B1"
            # A partial best/history update must be repaired from the committed epoch.
            torch.save({"partial": True}, folder / "best.pt")
            (folder / "history.json").write_text("[]", encoding="utf-8")
            self.run_group(root / "resumed", resume=True)
            self.assert_checkpoint_equal(root / "reference", root / "resumed")
            self.assertFalse((folder / "last.pt.tmp").exists())

    def test_legacy_partial_run_is_archived_and_replayed_with_original_seed(self):
        with test_workspace() as root:
            self.run_group(root / "reference")
            folder = root / "legacy/seed_9527/B1"
            folder.mkdir(parents=True)
            history = json.loads((root / "reference/seed_9527/B1/history.json").read_text())[:1]
            (folder / "history.json").write_text(json.dumps(history), encoding="utf-8")
            torch.save({"legacy": True}, folder / "best.pt")
            self.run_group(root / "legacy", resume=True)
            self.assert_checkpoint_equal(root / "reference", root / "legacy")
            recovery = json.loads((folder / "recovery.json").read_text())
            self.assertTrue((Path(recovery["archive"]) / "best.pt").exists())
            self.assertEqual(recovery["replayed_epoch_deltas"][0]["train_loss"], 0.0)
            self.assertEqual(recovery["replayed_epoch_deltas"][0]["val_mean_iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
