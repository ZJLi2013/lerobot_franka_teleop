"""Offline policy inference on a recorded LeRobot dataset.

Runs a trained policy (act / diffusion / pi0 / pi05 / ...) on frames from an
existing LeRobot dataset, without requiring a real robot or cameras. Useful for
sanity-checking a checkpoint when no hardware is available.

Reads its parameters from `scripts/config/record_cfg.yaml` under the
`inference:` section.
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device

# Force INFO level even if some upstream module already configured root logger.
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger().setLevel(logging.INFO)


def _say(msg: str = "") -> None:
    """Print a progress line with explicit flushing.

    We use plain print (not logging) so messages always appear regardless of
    how upstream libraries have (re)configured the root logger.
    """
    print(msg, flush=True)


class OfflineInferenceConfig:
    """Configuration for offline inference on a recorded dataset."""

    def __init__(self, cfg: Dict[str, Any]):
        self.pretrained_path: str = cfg["pretrained_path"]
        self.dataset_repo_id: str = cfg["dataset_repo_id"]
        self.episode_idx: int = cfg.get("episode_idx", 0)
        self.task_description: str = cfg.get("task_description", "")
        self.device: str = cfg.get("device", "cuda")
        self.max_frames: Optional[int] = cfg.get("max_frames")
        self.print_every: int = cfg.get("print_every", 1)
        # Optional remap from dataset image-key -> model's expected image-key.
        # Useful when the dataset was recorded with cameras named
        # `observation.images.front` etc. but the policy was trained with
        # different names (e.g. pi0_base expects
        # `observation.images.base_0_rgb` / `left_wrist_0_rgb` / ...).
        self.image_key_map: Dict[str, str] = cfg.get("image_key_map") or {}


def _to_batch(
    sample: Dict[str, Any],
    device: torch.device,
    image_key_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert a single dataset sample into a batched dict on `device`.

    - Keeps only `observation.*` keys (policies do not need `action` at inference).
    - Adds a leading batch dimension for tensors.
    - Optionally renames keys via `image_key_map` (dataset_key -> model_key).
    - Strings / metadata pass through untouched.
    """
    image_key_map = image_key_map or {}
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if not key.startswith("observation."):
            continue
        out_key = image_key_map.get(key, key)
        if torch.is_tensor(value):
            batch[out_key] = value.unsqueeze(0).to(device)
        else:
            batch[out_key] = value
    return batch


def run_offline_inference(cfg: OfflineInferenceConfig) -> None:
    _say("====== [START] Offline inference ======")
    _say(f"  pretrained_path : {cfg.pretrained_path}")
    _say(f"  dataset_repo_id : {cfg.dataset_repo_id}")
    _say(f"  episode_idx     : {cfg.episode_idx}")
    _say(f"  device          : {cfg.device}")
    _say(f"  max_frames      : {cfg.max_frames}")

    if not Path(cfg.pretrained_path).exists():
        raise FileNotFoundError(
            f"pretrained_path does not exist: {cfg.pretrained_path}. "
            "Make sure it points to a directory containing `config.json`."
        )

    t0 = time.perf_counter()
    _say("[1/5] Loading PreTrainedConfig ...")
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_path)
    policy_cfg.pretrained_path = cfg.pretrained_path
    policy_cfg.device = cfg.device
    _say(f"      done in {time.perf_counter() - t0:.1f}s "
         f"(policy type = {policy_cfg.type})")

    t0 = time.perf_counter()
    _say(f"[2/5] Loading dataset {cfg.dataset_repo_id} (episode {cfg.episode_idx}) ...")
    dataset = LeRobotDataset(cfg.dataset_repo_id, episodes=[cfg.episode_idx])
    _say(f"      done in {time.perf_counter() - t0:.1f}s "
         f"(num_frames in episode = {dataset.num_frames})")

    t0 = time.perf_counter()
    _say("[3/5] Building policy (this loads weights into GPU; may take a while for VLA) ...")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    _say(f"      done in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    _say("[4/5] Building pre/post processors ...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    _say(f"      done in {time.perf_counter() - t0:.1f}s")

    policy.eval()
    policy.reset()
    device = get_safe_torch_device(cfg.device)

    n_frames = dataset.num_frames
    if cfg.max_frames is not None:
        n_frames = min(n_frames, cfg.max_frames)
    _say(f"[5/5] Running inference on {n_frames} frame(s)\n")

    errors = []
    last_t = time.perf_counter()
    with torch.no_grad():
        for idx in range(n_frames):
            t_step = time.perf_counter()

            sample = dataset[idx]
            batch = _to_batch(sample, device, cfg.image_key_map)
            batch["task"] = cfg.task_description

            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action).squeeze(0).cpu().numpy()

            gt_action = sample["action"].numpy()

            # VLA policies (pi0 / pi05 / ...) pad action to a fixed
            # `max_action_dim` (default 32) so the same model can drive
            # robots with different joint counts. Truncate the prediction
            # to the dataset's real action dimension before comparing.
            if action.shape[0] != gt_action.shape[0]:
                if action.shape[0] > gt_action.shape[0]:
                    action_cmp = action[: gt_action.shape[0]]
                    if idx == 0:
                        _say(
                            f"[INFO] policy outputs action of shape {action.shape}; "
                            f"truncating to dataset action shape {gt_action.shape} "
                            f"for RMSE comparison."
                        )
                else:
                    raise ValueError(
                        f"Predicted action shape {action.shape} is smaller than "
                        f"dataset action shape {gt_action.shape}; cannot compare."
                    )
            else:
                action_cmp = action

            err = float(np.sqrt(np.mean((action_cmp - gt_action) ** 2)))
            errors.append(err)

            step_dt = time.perf_counter() - t_step

            if cfg.print_every > 0 and idx % cfg.print_every == 0:
                _say(
                    f"[{idx:04d}] {step_dt*1000:6.0f}ms  "
                    f"pred={np.round(action_cmp, 3)} "
                    f"gt={np.round(gt_action, 3)} "
                    f"|rmse|={err:.4f}"
                )
            last_t = time.perf_counter()

    if errors:
        _say(
            f"\n====== [DONE] frames={len(errors)} "
            f"mean_rmse={np.mean(errors):.4f} max_rmse={np.max(errors):.4f} ======"
        )
    else:
        _say("\n====== [DONE] No frames were processed (n_frames=0) ======")


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

    try:
        run_offline_inference(inference_cfg)
    except BaseException:
        # Make absolutely sure any failure is visible (and not swallowed by
        # logging buffering or upstream `except Exception: pass` patterns).
        _say("\n====== [ERROR] Inference crashed; full traceback below ======")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
