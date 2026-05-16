from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

src = LeRobotDataset(
    repo_id='milk_power_2_20260515_v01',
    root='/home/xunwang2/project/lerobot_franka_teleop/babycare/milk_power_2_20260515_v01',
)
delete_episodes(
    dataset=src,
    episode_indices=list(range(32, 52)),
    output_dir='/home/xunwang2/project/lerobot_franka_teleop/babycare/milk_power_2_20260515_v01_clean',
    repo_id='milk_power_2_20260515_v01_clean',
)