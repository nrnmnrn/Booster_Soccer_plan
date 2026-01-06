"""MJX 足球訓練環境

提供 GPU 加速的批量模擬環境，用於 PPO/SAC 訓練。

Features:
- 支持批量並行環境 (num_envs)
- 輸出 87 維 observation（與官方一致）
- 使用 mj_name2id 獲取 body ID（禁止硬編碼）

Usage:
    env = MJXSoccerEnv(num_envs=1024)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    state, obs, reward, done, info = env.step(state, action)
"""

from pathlib import Path
from typing import NamedTuple, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from . import ASSETS_DIR
from .preprocessor_jax import (
    EnvInfo,
    preprocess_observation,
    quat_wxyz_to_xyzw,
)


class EnvState(NamedTuple):
    """環境狀態（用於 JAX functional style）"""
    mjx_data: mjx.Data
    step_count: jnp.ndarray
    key: jnp.ndarray


class MJXSoccerEnv:
    """MJX 足球環境

    Attributes:
        mj_model: MuJoCo model
        mjx_model: MJX compiled model
        num_envs: 並行環境數量
        max_steps: 每 episode 最大步數
    """

    def __init__(
        self,
        xml_path: str = None,
        num_envs: int = 1024,
        max_steps: int = 1000,
        player_team: int = 0,
        task_index: int = 0,
    ):
        """初始化環境

        Args:
            xml_path: XML 文件路徑，默認使用 soccer_env.xml
            num_envs: 並行環境數量
            max_steps: 每 episode 最大步數
            player_team: 球員隊伍 (0 或 1)
            task_index: 任務索引 (0, 1, 2)
        """
        if xml_path is None:
            xml_path = str(ASSETS_DIR / "soccer_env.xml")

        # 載入 MuJoCo 模型
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # 編譯為 MJX
        self.mjx_model = mjx.put_model(self.mj_model)

        self.num_envs = num_envs
        self.max_steps = max_steps

        # 使用 mj_name2id 獲取 body ID（禁止硬編碼！）
        self._init_body_ids()

        # 任務和隊伍設定
        self.player_team = jnp.array(
            [1, 0] if player_team == 0 else [0, 1],
            dtype=jnp.float32
        )
        self.task_one_hot = jnp.zeros(3, dtype=jnp.float32).at[task_index].set(1.0)

        # Action 和 Observation 維度
        self.action_dim = self.mj_model.nu  # 12
        self.obs_dim = 87

    def _init_body_ids(self):
        """使用 mj_name2id 初始化 body ID"""
        model = self.mj_model

        # Body IDs
        self.trunk_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        self.ball_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.goal_0_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "goal_team_0"
        )
        self.goal_1_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "goal_team_1"
        )

        # Site IDs（用於 sensor）
        self.imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )

        # Joint addresses（用於從 qpos/qvel 提取關節數據）
        # freejoint 佔 7 DOF (qpos) / 6 DOF (qvel)
        self.robot_qpos_start = 7  # 跳過 root freejoint
        self.robot_qpos_end = 7 + 12

        # Ball freejoint
        self.ball_qpos_start = 19  # robot (7+12=19) 之後
        self.ball_qvel_start = 18  # robot (6+12=18) 之後

    # =========================================================================
    # Core Environment Methods
    # =========================================================================

    def reset(self, key: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """重置環境

        Args:
            key: JAX random key

        Returns:
            (state, observation)
        """
        key, subkey = jax.random.split(key)

        # 創建初始 mjx_data
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # 批量化
        if self.num_envs > 1:
            mjx_data = jax.tree.map(
                lambda x: jnp.stack([x] * self.num_envs),
                mjx_data
            )

        # 隨機化初始狀態
        mjx_data = self._randomize_state(mjx_data, subkey)

        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            key=key,
        )

        # 計算初始 observation
        obs = self._get_obs(mjx_data)

        return state, obs

    def step(
        self,
        state: EnvState,
        action: jnp.ndarray
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """執行一步模擬

        Args:
            state: 當前環境狀態
            action: 動作，shape (num_envs, 12)

        Returns:
            (new_state, observation, reward, done, info)
        """
        mjx_data = state.mjx_data

        # 設定控制輸入
        mjx_data = mjx_data.replace(ctrl=action)

        # MJX step
        mjx_data = self._mjx_step(mjx_data)

        # 計算 observation
        obs = self._get_obs(mjx_data)

        # 計算 reward
        reward = self._compute_reward(mjx_data, action)

        # 檢查終止條件
        done = self._check_termination(mjx_data, state.step_count)

        # 更新狀態
        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=state.step_count + 1,
            key=state.key,
        )

        info = {
            "step_count": new_state.step_count,
        }

        return new_state, obs, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _mjx_step(self, mjx_data: mjx.Data) -> mjx.Data:
        """JIT 編譯的 MJX step（支援批量環境）"""
        if self.num_envs > 1:
            # 批量執行：model 不變，data 按 batch 軸映射
            batched_step = jax.vmap(mjx.step, in_axes=(None, 0))
            return batched_step(self.mjx_model, mjx_data)
        else:
            return mjx.step(self.mjx_model, mjx_data)

    # =========================================================================
    # Observation
    # =========================================================================

    def _get_obs(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """從 mjx_data 提取 87 維 observation"""
        # 提取機器人關節狀態
        robot_qpos = mjx_data.qpos[..., self.robot_qpos_start:self.robot_qpos_end]
        robot_qvel = mjx_data.qvel[..., 6:18]  # 跳過 root 的 6 DOF

        # 提取環境資訊
        info = self._extract_env_info(mjx_data)

        # 調用 preprocessor
        if self.num_envs > 1:
            # 批量處理
            obs = jax.vmap(preprocess_observation)(robot_qpos, robot_qvel, info)
        else:
            obs = preprocess_observation(robot_qpos, robot_qvel, info)

        return obs

    def _extract_env_info(self, mjx_data: mjx.Data) -> EnvInfo:
        """從 mjx_data 提取 EnvInfo 結構"""
        # 機器人軀幹狀態
        robot_xpos = mjx_data.xpos[..., self.trunk_body_id, :]

        # 四元數：MuJoCo 返回 [w,x,y,z]，需要轉換為 [x,y,z,w]
        robot_quat_wxyz = mjx_data.xquat[..., self.trunk_body_id, :]
        robot_quat = quat_wxyz_to_xyzw(robot_quat_wxyz)

        # Sensor 數據（需要根據 sensor 順序提取）
        # 假設 sensor 順序：gyro(3), vel(3), accel(3), quat(4), pos(3), linvel(3)
        robot_gyro = mjx_data.sensordata[..., 0:3]
        robot_velocimeter = mjx_data.sensordata[..., 3:6]
        robot_accelerometer = mjx_data.sensordata[..., 6:9]

        # 球狀態
        ball_xpos = mjx_data.xpos[..., self.ball_body_id, :]
        ball_velp = mjx_data.cvel[..., self.ball_body_id, 3:6]  # 平移速度
        ball_velr = mjx_data.cvel[..., self.ball_body_id, 0:3]  # 旋轉速度

        # 球門位置
        goal_0_xpos = mjx_data.xpos[..., self.goal_0_body_id, :]
        goal_1_xpos = mjx_data.xpos[..., self.goal_1_body_id, :]

        # 門將/目標/防守者（MJX 預訓練階段可用零值）
        zeros_3 = jnp.zeros_like(robot_xpos)

        # 廣播 player_team 和 task_one_hot 到 batch 維度
        # 原始 shape: (2,) 和 (3,)，需要變成 (num_envs, 2) 和 (num_envs, 3)
        if self.num_envs > 1:
            player_team_batched = jnp.tile(
                self.player_team[None, :], (self.num_envs, 1)
            )
            task_one_hot_batched = jnp.tile(
                self.task_one_hot[None, :], (self.num_envs, 1)
            )
        else:
            player_team_batched = self.player_team
            task_one_hot_batched = self.task_one_hot

        return EnvInfo(
            robot_quat=robot_quat,
            robot_gyro=robot_gyro,
            robot_accelerometer=robot_accelerometer,
            robot_velocimeter=robot_velocimeter,
            robot_xpos=robot_xpos,
            ball_xpos=ball_xpos,
            ball_velp=ball_velp,
            ball_velr=ball_velr,
            goal_0_xpos=goal_0_xpos,
            goal_1_xpos=goal_1_xpos,
            goalkeeper_0_xpos=zeros_3,
            goalkeeper_0_vel=zeros_3,
            goalkeeper_1_xpos=zeros_3,
            goalkeeper_1_vel=zeros_3,
            target_xpos=ball_xpos,  # 預訓練階段：目標 = 球
            target_vel=ball_velp,
            defender_xpos=zeros_3,
            player_team=player_team_batched,
            task_one_hot=task_one_hot_batched,
        )

    # =========================================================================
    # Reward (TODO: Human collaboration point)
    # =========================================================================

    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """計算獎勵

        TODO(human): 這裡需要設計獎勵函數
        """
        return jnp.zeros(self.num_envs)

    # =========================================================================
    # Termination
    # =========================================================================

    def _check_termination(
        self,
        mjx_data: mjx.Data,
        step_count: jnp.ndarray
    ) -> jnp.ndarray:
        """檢查終止條件

        Returns:
            done: bool array, shape (num_envs,)
        """
        # 1. 達到最大步數
        timeout = step_count >= self.max_steps

        # 2. 機器人倒下（軀幹高度過低）
        trunk_height = mjx_data.xpos[..., self.trunk_body_id, 2]
        fallen = trunk_height < 0.3

        return timeout | fallen

    # =========================================================================
    # State Randomization
    # =========================================================================

    def _randomize_state(
        self,
        mjx_data: mjx.Data,
        key: jnp.ndarray
    ) -> mjx.Data:
        """隨機化初始狀態

        TODO: 實現更豐富的隨機化
        - 機器人初始位置/姿態
        - 球初始位置/速度
        - 環境擾動
        """
        return mjx_data


# =============================================================================
# Factory function
# =============================================================================

def create_soccer_env(
    num_envs: int = 1024,
    player_team: int = 0,
    task_index: int = 0,
) -> MJXSoccerEnv:
    """創建 MJX 足球環境

    Args:
        num_envs: 並行環境數量
        player_team: 球員隊伍 (0 或 1)
        task_index: 任務索引 (0=GoaliePenaltyKick, 1=DrillPassing, 2=DrillShooting)

    Returns:
        MJXSoccerEnv instance
    """
    return MJXSoccerEnv(
        num_envs=num_envs,
        player_team=player_team,
        task_index=task_index,
    )


if __name__ == "__main__":
    # 簡單測試
    import os
    os.environ["MUJOCO_GL"] = "disabled"

    print("Testing MJXSoccerEnv...")

    env = MJXSoccerEnv(num_envs=4)
    print(f"Action dim: {env.action_dim}")
    print(f"Obs dim: {env.obs_dim}")

    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)

    print(f"Initial obs shape: {obs.shape}")
    assert obs.shape == (4, 87), f"Expected (4, 87), got {obs.shape}"

    # Test step
    action = jnp.zeros((4, 12))
    state, obs, reward, done, info = env.step(state, action)

    print(f"After step obs shape: {obs.shape}")
    print(f"Reward shape: {reward.shape}")
    print(f"Done shape: {done.shape}")

    print("✅ MJXSoccerEnv test passed!")
