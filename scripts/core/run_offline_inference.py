"""Offline policy inference on a recorded LeRobot dataset.

Runs a trained policy (act / diffusion / pi0 / pi05 / ...) on frames from an
existing LeRobot dataset, without requiring a real robot or cameras. Useful for
sanity-checking a checkpoint when no hardware is available.

Reads its parameters from `scripts/config/record_cfg.yaml` under the
`inference:` section.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device

logging.basicConfig(level=logging.INFO, format="%(message)s")


class OfflineInferenceConfig:
    """Configuration for offline inference on a recorded dataset."""

    def __init__(self, cfg: Dict[str, Any]):
        self.pretrained_path: str = cfg["pretrained_path"]
        self.dataset_repo_id: str = cfg["dataset_repo_id"]
        self.episode_idx: int = cfg.get("episode_idx", 0)
        self.task_description: str = cfg.get("task_description", "")
        self.device: str = cfg.get("device", "cuda")
        # Optional cap on number of frames; None means run through the whole episode.
        self.max_frames: Optional[int] = cfg.get("max_frames")
        self.print_every: int = cfg.get("print_every", 1)


def _to_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Convert a single dataset sample into a batched dict on `device`.

    - Keeps only `observation.*` keys (policies do not need `action` at inference).
    - Adds a leading batch dimension for tensors.
    - Strings / metadata pass through untouched.
    """
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if not key.startswith("observation."):
            continue
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value
    return batch


def run_offline_inference(cfg: OfflineInferenceConfig) -> None:
    logging.info("====== [START] Offline inference ======")
    logging.info(f"  pretrained_path : {cfg.pretrained_path}")
    logging.info(f"  dataset_repo_id : {cfg.dataset_repo_id}")
    logging.info(f"  episode_idx     : {cfg.episode_idx}")
    logging.info(f"  device          : {cfg.device}")

    if not Path(cfg.pretrained_path).exists():
        raise FileNotFoundError(
            f"pretrained_path does not exist: {cfg.pretrained_path}. "
            "Make sure it points to a directory containing `config.json`."
        )

    # Load the policy config that was saved at training time. Works for any
    # policy type registered in lerobot (act / diffusion / pi0 / pi05 / ...).
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_path)
    policy_cfg.pretrained_path = cfg.pretrained_path
    policy_cfg.device = cfg.device

    dataset = LeRobotDataset(cfg.dataset_repo_id, episodes=[cfg.episode_idx])

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    policy.eval()
    policy.reset()
    device = get_safe_torch_device(cfg.device)

    n_frames = dataset.num_frames
    if cfg.max_frames is not None:
        n_frames = min(n_frames, cfg.max_frames)
    logging.info(f"  episode_frames  : {n_frames}\n")

    errors = []
    with torch.no_grad():
        for idx in range(n_frames):
            sample = dataset[idx]
            batch = _to_batch(sample, device)
            batch["task"] = cfg.task_description

            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action).squeeze(0).cpu().numpy()

            gt_action = sample["action"].numpy()
            err = float(np.sqrt(np.mean((action - gt_action) ** 2)))
            errors.append(err)

            if cfg.print_every > 0 and idx % cfg.print_every == 0:
                logging.info(
                    f"[{idx:04d}] pred={np.round(action, 3)} "
                    f"gt={np.round(gt_action, 3)} |rmse|={err:.4f}"
                )

    if errors:
        logging.info(
            f"\n====== [DONE] frames={len(errors)} "
            f"mean_rmse={np.mean(errors):.4f} max_rmse={np.max(errors):.4f} ======"
        )


def main() -> None:
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "inference" not in cfg:
        raise KeyError(
            f"Section `inference:` not found in {cfg_path}. "
            "Please add it (see README §10.3)."
        )

    inference_cfg = OfflineInferenceConfig(cfg["inference"])
    run_offline_inference(inference_cfg)


if __name__ == "__main__":
    main()
