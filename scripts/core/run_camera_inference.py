"""Live camera -> policy inference, with **no robot connection**.

Reads frames from one or more RealSense cameras, packs them into the same
batch format as `run_offline_inference.py`, runs `policy.select_action()`,
and prints the predicted action. Useful for validating, *before* moving the
robot:

  - camera serial / key mapping (e.g. `observation.images.front` is correctly
    remapped to the model's expected key like `observation.images.base_0_rgb`)
  - image resolution and color space match what the policy was trained on
  - live inference rate vs. camera FPS
  - that the policy produces sane action values (sanity-check magnitude /
    direction before allowing the robot to execute them)

The script never moves the robot. The `observation.state` slot is filled
either with the first frame of a reference dataset (default) or with zeros,
so the only "live" signal is the camera feed.

Reads its parameters from `scripts/config/record_cfg.yaml` under the
`inference_camera:` section.
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import (
    RealSenseCamera,
    RealSenseCameraConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger().setLevel(logging.INFO)


def _say(msg: str = "") -> None:
    """Print a progress line with explicit flushing.

    We use plain print (not logging) so messages always appear regardless of
    how upstream libraries have (re)configured the root logger.
    """
    print(msg, flush=True)


class CameraInferenceConfig:
    """Configuration for live-camera inference (no robot)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.pretrained_path: str = cfg["pretrained_path"]
        self.dataset_repo_id: str = cfg["dataset_repo_id"]
        self.episode_idx: int = cfg.get("episode_idx", 0)
        self.task_description: str = cfg.get("task_description", "")
        self.device: str = cfg.get("device", "cuda")
        self.fps: float = float(cfg.get("fps", 15))
        self.max_steps: Optional[int] = cfg.get("max_steps")
        self.print_every: int = cfg.get("print_every", 1)

        # `cameras` maps a *dataset-style* image key to the physical camera
        # to read it from. Example:
        #   cameras:
        #     observation.images.front:
        #       serial: "401622070138"
        #       width: 424
        #       height: 240
        cams = cfg.get("cameras") or {}
        if not cams:
            raise ValueError(
                "`inference_camera.cameras` must define at least one camera, "
                "keyed by the dataset image key (e.g. `observation.images.front`)."
            )
        self.cameras: Dict[str, Dict[str, Any]] = cams

        # `image_key_map` has the same semantics as in `run_offline_inference`:
        # remap dataset_key -> model_key. Useful for VLA policies that expect
        # `base_0_rgb` / `left_wrist_0_rgb` etc.
        self.image_key_map: Dict[str, str] = cfg.get("image_key_map") or {}

        # Whether to take `observation.state` from the reference dataset
        # (default) or feed zeros. With `True` we replicate the start of an
        # episode and stay close to training distribution.
        self.use_dataset_state: bool = cfg.get("use_dataset_state", True)

        # If True, call `policy.reset()` every step to clear any internal
        # action chunk queue. Useful for measuring *steady-state* inference
        # latency on policies that chunk multiple future actions (pi0 / pi05 /
        # ACT). Without this the 2nd..n_action_steps frames are just queue
        # pops (~ms), which is misleading.
        self.force_fresh_action: bool = cfg.get("force_fresh_action", False)

        # How many leading action dimensions to print. VLA policies (pi0)
        # pad action to `max_action_dim=32`; for franka only the first ~8
        # dims are meaningful. Set to None / 0 to print all dims.
        self.print_action_dims: Optional[int] = cfg.get("print_action_dims", 8)

        # Optional override for the policy's `num_inference_steps` field
        # (currently used by pi0 / pi05 flow-matching). The default in
        # pi0 is 10; lowering to 5 typically halves inference time with
        # negligible action quality loss. Set to None to keep the
        # checkpoint default.
        self.num_inference_steps: Optional[int] = cfg.get("num_inference_steps")


def _make_camera(serial: str, width: int, height: int, fps: float) -> RealSenseCamera:
    """Construct + connect a RealSense camera with sane defaults."""
    cam_cfg = RealSenseCameraConfig(
        serial_number_or_name=serial,
        fps=int(fps),
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    cam = RealSenseCamera(cam_cfg)
    cam.connect()
    return cam


def _frame_to_chw_tensor(
    frame_hwc: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Convert an (H, W, 3) RGB uint8 frame -> (1, 3, H, W) float in [0, 1].

    Mirrors what `LeRobotDataset.__getitem__` produces for image features so
    that downstream `preprocessor(...)` sees the same shape/dtype as during
    offline inference.
    """
    t = torch.from_numpy(frame_hwc).to(device=device)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    elif t.dtype != torch.float32:
        t = t.float()
    t = t.permute(2, 0, 1).contiguous().unsqueeze(0)
    return t


def run_camera_inference(cfg: CameraInferenceConfig) -> None:
    _say("====== [START] Camera inference (no robot) ======")
    _say(f"  pretrained_path : {cfg.pretrained_path}")
    _say(f"  dataset_repo_id : {cfg.dataset_repo_id} (used for stats / state)")
    _say(f"  device          : {cfg.device}")
    _say(f"  fps target      : {cfg.fps}")
    _say(f"  cameras         :")
    for k, v in cfg.cameras.items():
        _say(f"    {k:40s} -> serial={v['serial']} {v['width']}x{v['height']}")
    if cfg.image_key_map:
        _say(f"  image_key_map   :")
        for k, v in cfg.image_key_map.items():
            _say(f"    {k:40s} -> {v}")

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
    if cfg.num_inference_steps is not None and hasattr(
        policy_cfg, "num_inference_steps"
    ):
        old = getattr(policy_cfg, "num_inference_steps")
        policy_cfg.num_inference_steps = int(cfg.num_inference_steps)
        _say(f"      override num_inference_steps: {old} -> "
             f"{policy_cfg.num_inference_steps}")
    _say(f"      done in {time.perf_counter() - t0:.1f}s "
         f"(policy type = {policy_cfg.type})")

    t0 = time.perf_counter()
    _say(f"[2/5] Loading reference dataset {cfg.dataset_repo_id} "
         f"(episode {cfg.episode_idx}; only meta + frame 0 are used) ...")
    dataset = LeRobotDataset(cfg.dataset_repo_id, episodes=[cfg.episode_idx])
    _say(f"      done in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    _say("[3/5] Building policy (loads weights into GPU; may take a while for VLA) ...")
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

    # Pull a reference state from the dataset (frame 0). Without a robot we
    # have no real joint state; using the dataset's first state mimics the
    # start of an episode and stays in-distribution for normalization.
    ref_state: Optional[torch.Tensor] = None
    if cfg.use_dataset_state:
        ref_sample = dataset[0]
        if "observation.state" in ref_sample:
            ref_state = ref_sample["observation.state"]
            _say(
                f"      using dataset[0]['observation.state'] as fixed state, "
                f"shape={tuple(ref_state.shape)}"
            )
        else:
            _say("      [WARN] dataset has no `observation.state`; "
                 "no state will be sent.")
    else:
        _say("      use_dataset_state=False; no `observation.state` will be sent.")

    _say("[5/5] Connecting cameras ...")
    cameras: Dict[str, RealSenseCamera] = {}
    try:
        for dataset_key, ccfg in cfg.cameras.items():
            _say(f"      connecting {dataset_key} (serial={ccfg['serial']}) ...")
            cameras[dataset_key] = _make_camera(
                serial=str(ccfg["serial"]),
                width=int(ccfg["width"]),
                height=int(ccfg["height"]),
                fps=cfg.fps,
            )
        _say("      all cameras connected.\n")

        if cfg.force_fresh_action:
            _say("      force_fresh_action=True: clearing the policy's action "
                 "queue every step (measures *steady-state* inference latency)")

        period = 1.0 / cfg.fps if cfg.fps > 0 else 0.0
        step = 0
        with torch.no_grad():
            while cfg.max_steps is None or step < cfg.max_steps:
                t_step = time.perf_counter()

                if cfg.force_fresh_action:
                    policy.reset()

                batch: Dict[str, Any] = {}
                t_cam = time.perf_counter()
                for dataset_key, cam in cameras.items():
                    frame = cam.async_read(timeout_ms=2000)
                    out_key = cfg.image_key_map.get(dataset_key, dataset_key)
                    batch[out_key] = _frame_to_chw_tensor(frame, device)
                cam_dt_ms = (time.perf_counter() - t_cam) * 1000.0

                if ref_state is not None:
                    batch["observation.state"] = ref_state.unsqueeze(0).to(device)
                batch["task"] = cfg.task_description

                # Detect whether this select_action call will be a "fresh"
                # forward pass or a queue pop. We probe the queue length
                # *before* the call; this works for pi0 / pi05 / ACT.
                queue_len_before = len(getattr(policy, "_action_queue", []))
                fresh_call = queue_len_before == 0

                t_inf = time.perf_counter()
                batch = preprocessor(batch)
                action = policy.select_action(batch)
                action_np = postprocessor(action).squeeze(0).cpu().numpy()
                inf_dt_ms = (time.perf_counter() - t_inf) * 1000.0

                step_dt_ms = (time.perf_counter() - t_step) * 1000.0

                if cfg.print_every > 0 and step % cfg.print_every == 0:
                    n = cfg.print_action_dims
                    show = action_np if not n else action_np[:n]
                    fresh_marker = "FRESH" if fresh_call else "queue"
                    _say(
                        f"[{step:05d}] total={step_dt_ms:6.0f}ms "
                        f"cam={cam_dt_ms:5.0f}ms infer={inf_dt_ms:5.0f}ms "
                        f"({fresh_marker})  "
                        f"action[:{len(show)}]={np.round(show, 3)}"
                    )

                step += 1
                if period > 0:
                    elapsed = time.perf_counter() - t_step
                    if elapsed < period:
                        time.sleep(period - elapsed)

    except KeyboardInterrupt:
        _say("\n[INFO] Ctrl+C received, shutting down ...")
    finally:
        for k, cam in cameras.items():
            try:
                cam.disconnect()
                _say(f"      disconnected {k}")
            except Exception:
                pass
        _say("====== [DONE] Camera inference finished ======")


def main() -> None:
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "inference_camera" not in cfg:
        raise KeyError(
            f"Section `inference_camera:` not found in {cfg_path}. "
            "Please add it (see README §10.3)."
        )

    cam_cfg = CameraInferenceConfig(cfg["inference_camera"])

    try:
        run_camera_inference(cam_cfg)
    except BaseException:
        _say("\n====== [ERROR] Camera inference crashed; full traceback below ======")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
