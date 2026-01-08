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
    """環境狀態（用於 JAX functional style）

    Attributes:
        mjx_data: MJX 物理引擎狀態
        step_count: 每個環境的當前步數，shape (num_envs,)
        key: JAX random key
        task_one_hot: 任務 one-hot 編碼，shape (num_envs, 3)
    """
    mjx_data: mjx.Data
    step_count: jnp.ndarray
    key: jnp.ndarray
    task_one_hot: jnp.ndarray  # 添加 task_one_hot 以支持隨機化


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
        random_task_index: bool = False,
        auto_reset: bool = True,
    ):
        """初始化環境

        Args:
            xml_path: XML 文件路徑，默認使用 soccer_env.xml
            num_envs: 並行環境數量
            max_steps: 每 episode 最大步數
            player_team: 球員隊伍 (0 或 1)
            task_index: 任務索引 (0, 1, 2)，當 random_task_index=False 時使用
            random_task_index: 是否在每次 reset 時隨機選擇任務
            auto_reset: 是否在 done 時自動重置環境
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
        self.random_task_index = random_task_index
        self.auto_reset = auto_reset
        self._fixed_task_index = task_index

        # 使用 mj_name2id 獲取 body ID（禁止硬編碼！）
        self._init_body_ids()

        # 任務和隊伍設定
        self.player_team = jnp.array(
            [1, 0] if player_team == 0 else [0, 1],
            dtype=jnp.float32
        )
        # task_one_hot 將在 reset 中設置（支持隨機化）
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

        # Foot Body IDs（用於 reward 計算）
        self.left_foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link"
        )
        self.right_foot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link"
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
        key, subkey, task_key = jax.random.split(key, 3)

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

        # 生成 task_one_hot（支持隨機化）
        if self.random_task_index:
            # 每個環境隨機選擇任務
            task_indices = jax.random.randint(task_key, (self.num_envs,), 0, 3)
            task_one_hot = jax.nn.one_hot(task_indices, 3)
        else:
            # 使用固定任務
            task_one_hot = jnp.tile(
                self.task_one_hot[None, :], (self.num_envs, 1)
            )

        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            key=key,
            task_one_hot=task_one_hot,
        )

        # 計算初始 observation（使用 state 中的 task_one_hot）
        obs = self._get_obs_with_task(mjx_data, task_one_hot)

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

        # 計算 reward（在 reset 之前）
        reward = self._compute_reward(mjx_data, action)

        # 檢查終止條件（使用 step_count + 1 避免 off-by-one 錯誤）
        # 這樣確保 episode 精確運行 max_steps 步
        done = self._check_termination(mjx_data, state.step_count + 1)

        # AutoReset：對已終止的環境執行重置
        if self.auto_reset:
            mjx_data, task_one_hot, step_count, key = self._auto_reset(
                mjx_data,
                state.task_one_hot,
                state.step_count,
                state.key,
                done,
            )
        else:
            task_one_hot = state.task_one_hot
            step_count = state.step_count + 1
            key = state.key

        # 計算 observation（使用可能已更新的 task_one_hot）
        obs = self._get_obs_with_task(mjx_data, task_one_hot)

        # 更新狀態
        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=step_count,
            key=key,
            task_one_hot=task_one_hot,
        )

        info = {
            "step_count": new_state.step_count,
        }

        return new_state, obs, reward, done, info

    def _auto_reset(
        self,
        mjx_data: mjx.Data,
        task_one_hot: jnp.ndarray,
        step_count: jnp.ndarray,
        key: jnp.ndarray,
        done: jnp.ndarray,
    ) -> Tuple[mjx.Data, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """對已終止的環境執行自動重置

        Args:
            mjx_data: 當前 MJX 數據
            task_one_hot: 當前任務 one-hot
            step_count: 當前步數
            key: JAX random key
            done: 終止標誌

        Returns:
            (new_mjx_data, new_task_one_hot, new_step_count, new_key)
        """
        key, reset_key, task_key = jax.random.split(key, 3)

        # 創建初始 mjx_data（用於重置）
        init_mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # 批量化初始數據
        if self.num_envs > 1:
            init_mjx_data = jax.tree.map(
                lambda x: jnp.stack([x] * self.num_envs),
                init_mjx_data
            )

        # 隨機化初始狀態（與 reset() 保持一致）
        init_mjx_data = self._randomize_state(init_mjx_data, reset_key)

        # 使用 where 選擇：done 的環境用初始數據，否則用當前數據
        new_mjx_data = jax.tree.map(
            lambda init, curr: jnp.where(
                done[:, None] if init.ndim > 1 else done,
                init,
                curr
            ),
            init_mjx_data,
            mjx_data,
        )

        # 更新 task_one_hot（如果啟用隨機化）
        if self.random_task_index:
            new_task_indices = jax.random.randint(task_key, (self.num_envs,), 0, 3)
            new_task_one_hot_rand = jax.nn.one_hot(new_task_indices, 3)
            new_task_one_hot = jnp.where(
                done[:, None],
                new_task_one_hot_rand,
                task_one_hot,
            )
        else:
            new_task_one_hot = task_one_hot

        # 重置步數
        new_step_count = jnp.where(done, 0, step_count + 1)

        return new_mjx_data, new_task_one_hot, new_step_count, key

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
        """從 mjx_data 提取 87 維 observation（使用實例的 task_one_hot）

        注意：此方法使用 self.task_one_hot，不支持隨機 task_index。
        對於訓練，請使用 _get_obs_with_task。
        """
        task_one_hot = jnp.tile(
            self.task_one_hot[None, :], (self.num_envs, 1)
        )
        return self._get_obs_with_task(mjx_data, task_one_hot)

    def _get_obs_with_task(
        self,
        mjx_data: mjx.Data,
        task_one_hot: jnp.ndarray
    ) -> jnp.ndarray:
        """從 mjx_data 提取 87 維 observation（指定 task_one_hot）

        Args:
            mjx_data: MJX 數據
            task_one_hot: 任務 one-hot，shape (num_envs, 3)

        Returns:
            observation，shape (num_envs, 87)
        """
        # 提取機器人關節狀態
        robot_qpos = mjx_data.qpos[..., self.robot_qpos_start:self.robot_qpos_end]
        robot_qvel = mjx_data.qvel[..., 6:18]  # 跳過 root 的 6 DOF

        # 提取環境資訊（傳入 task_one_hot）
        info = self._extract_env_info(mjx_data, task_one_hot)

        # 調用 preprocessor
        if self.num_envs > 1:
            # 批量處理
            obs = jax.vmap(preprocess_observation)(robot_qpos, robot_qvel, info)
        else:
            obs = preprocess_observation(robot_qpos, robot_qvel, info)

        return obs

    def _extract_env_info(
        self,
        mjx_data: mjx.Data,
        task_one_hot: jnp.ndarray = None
    ) -> EnvInfo:
        """從 mjx_data 提取 EnvInfo 結構

        Args:
            mjx_data: MJX 數據
            task_one_hot: 任務 one-hot，shape (num_envs, 3)。
                          如果為 None，使用 self.task_one_hot。

        Returns:
            EnvInfo 結構
        """
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

        # 廣播 player_team 到 batch 維度
        if self.num_envs > 1:
            player_team_batched = jnp.tile(
                self.player_team[None, :], (self.num_envs, 1)
            )
        else:
            player_team_batched = self.player_team

        # 使用傳入的 task_one_hot 或默認值
        if task_one_hot is None:
            if self.num_envs > 1:
                task_one_hot_batched = jnp.tile(
                    self.task_one_hot[None, :], (self.num_envs, 1)
                )
            else:
                task_one_hot_batched = self.task_one_hot
        else:
            task_one_hot_batched = task_one_hot

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
    # Reward
    # =========================================================================

    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """計算 MJX 預訓練獎勵

        獎勵組件（對齊官方評估指標）：
        - r_stand:    站立獎勵（軀幹高度）    | 官方: robot_fallen -1.5
        - r_approach: 靠近球獎勵              | 官方: robot_distance_ball +0.25
        - r_ball_vel: 球朝向球門速度          | 官方: ball_vel_twd_goal +1.5
        - r_kick:     腳接觸球獎勵            | goal_scored proxy
        - r_energy:   能量懲罰                | 穩定性
        - r_time:     時間懲罰                | 官方: steps -1.0

        Returns:
            reward: shape (num_envs,)
        """
        # === R1: Standing Reward ===
        trunk_height = mjx_data.xpos[..., self.trunk_body_id, 2]
        r_stand = jnp.where(trunk_height > 0.35, 0.5, -1.5)

        # === R2: Approach Ball ===
        robot_xy = mjx_data.xpos[..., self.trunk_body_id, :2]
        ball_xy = mjx_data.xpos[..., self.ball_body_id, :2]
        ball_dist = jnp.linalg.norm(robot_xy - ball_xy, axis=-1)
        r_approach = 0.25 * jnp.exp(-ball_dist)

        # === R3: Ball Velocity Toward Goal ===
        # TODO(human): 決定攻擊方向邏輯
        # 目前假設 player_team=0 攻擊 goal_0（正 X 方向）
        # 如需修改，請實現 _get_attack_goal_id() 方法
        attack_goal_xy = mjx_data.xpos[..., self.goal_0_body_id, :2]

        ball_to_goal = attack_goal_xy - ball_xy
        ball_to_goal_norm = ball_to_goal / (
            jnp.linalg.norm(ball_to_goal, axis=-1, keepdims=True) + 1e-6
        )

        ball_vel_xy = mjx_data.cvel[..., self.ball_body_id, 3:5]
        vel_toward_goal = jnp.sum(ball_vel_xy * ball_to_goal_norm, axis=-1)
        r_ball_vel = 1.5 * jnp.clip(vel_toward_goal, 0.0, 2.0) / 2.0

        # === R4: Foot-Ball Contact ===
        left_foot_xy = mjx_data.xpos[..., self.left_foot_body_id, :2]
        right_foot_xy = mjx_data.xpos[..., self.right_foot_body_id, :2]
        left_dist = jnp.linalg.norm(left_foot_xy - ball_xy, axis=-1)
        right_dist = jnp.linalg.norm(right_foot_xy - ball_xy, axis=-1)
        foot_ball_dist = jnp.minimum(left_dist, right_dist)
        r_kick = jnp.where(foot_ball_dist < 0.2, 0.2, 0.0)

        # === R5: Energy Penalty ===
        r_energy = -0.01 * jnp.sum(action ** 2, axis=-1)

        # === R6: Time Penalty ===
        r_time = jnp.full((self.num_envs,), -0.01)

        # === Combine Rewards ===
        # 權重設計：站立 > 靠近球 > 踢球方向 > 接觸 > 懲罰
        reward = (
            r_stand * 0.4 +
            r_approach * 0.3 +
            r_ball_vel * 0.2 +
            r_kick * 0.05 +
            r_energy +
            r_time
        )

        return reward

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
    random_task_index: bool = False,
    auto_reset: bool = True,
) -> MJXSoccerEnv:
    """創建 MJX 足球環境

    Args:
        num_envs: 並行環境數量
        player_team: 球員隊伍 (0 或 1)
        task_index: 任務索引 (0=GoaliePenaltyKick, 1=ObstaclePK, 2=KickToTarget)
        random_task_index: 是否隨機化任務索引
        auto_reset: 是否自動重置終止的環境

    Returns:
        MJXSoccerEnv instance
    """
    return MJXSoccerEnv(
        num_envs=num_envs,
        player_team=player_team,
        task_index=task_index,
        random_task_index=random_task_index,
        auto_reset=auto_reset,
    )


if __name__ == "__main__":
    # 簡單測試
    import os
    os.environ["MUJOCO_GL"] = "disabled"

    print("Testing MJXSoccerEnv...")
    print("=" * 50)

    # 測試 1: 基本功能
    print("\n[1/3] Testing basic functionality...")
    env = MJXSoccerEnv(num_envs=4)
    print(f"  Action dim: {env.action_dim}")
    print(f"  Obs dim: {env.obs_dim}")

    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)

    print(f"  Initial obs shape: {obs.shape}")
    assert obs.shape == (4, 87), f"Expected (4, 87), got {obs.shape}"
    assert state.task_one_hot.shape == (4, 3), f"Expected task_one_hot (4, 3), got {state.task_one_hot.shape}"
    print(f"  Task one-hot shape: {state.task_one_hot.shape}")

    # Test step
    action = jnp.zeros((4, 12))
    state, obs, reward, done, info = env.step(state, action)

    print(f"  After step obs shape: {obs.shape}")
    print(f"  Reward shape: {reward.shape}")
    print(f"  Done shape: {done.shape}")
    print("  ✓ Basic functionality passed!")

    # 測試 2: 隨機 task_index
    print("\n[2/3] Testing random task index...")
    env_rand = MJXSoccerEnv(num_envs=100, random_task_index=True)
    key = jax.random.PRNGKey(42)
    state, obs = env_rand.reset(key)

    # 檢查 task_one_hot 分佈
    task_sums = jnp.sum(state.task_one_hot, axis=0)
    print(f"  Task distribution: {task_sums}")
    assert jnp.all(task_sums > 0), "Each task should appear at least once with 100 envs"
    print("  ✓ Random task index passed!")

    # 測試 3: AutoReset
    print("\n[3/3] Testing auto reset...")
    env_auto = MJXSoccerEnv(num_envs=4, auto_reset=True, max_steps=5)
    key = jax.random.PRNGKey(123)
    state, obs = env_auto.reset(key)

    # 運行幾步直到超時
    for i in range(10):
        action = jax.random.uniform(key, (4, 12), minval=-1, maxval=1)
        state, obs, reward, done, info = env_auto.step(state, action)
        key = jax.random.split(key)[0]

        if jnp.any(done):
            print(f"  Step {i}: done={done}, step_count={state.step_count}")

    # 檢查 step_count 是否被重置（不應該超過 max_steps）
    assert jnp.all(state.step_count <= env_auto.max_steps), \
        f"Step count should be reset, but got {state.step_count}"
    print("  ✓ Auto reset passed!")

    print("\n" + "=" * 50)
    print("✅ All MJXSoccerEnv tests passed!")
