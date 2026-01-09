"""JAX 版本的 87 維 Observation Preprocessor

與官方 Preprocessor (imitation_learning/scripts/preprocessor.py) 保持一致。
支持 GPU 批量處理，用於 MJX 訓練。

87 維組成：
    - robot_qpos: 12 (關節位置)
    - robot_qvel: 12 (關節速度)
    - project_gravity: 3 (重力投影到機器人座標系)
    - robot_gyro: 3 (角速度)
    - robot_accelerometer: 3
    - robot_velocimeter: 3
    - goal_team_0_rel_robot: 3
    - goal_team_1_rel_robot: 3
    - goal_team_0_rel_ball: 3
    - goal_team_1_rel_ball: 3
    - ball_xpos_rel_robot: 3
    - ball_velp_rel_robot: 3
    - ball_velr_rel_robot: 3
    - player_team: 2 (one-hot)
    - goalkeeper_team_0_xpos_rel_robot: 3
    - goalkeeper_team_0_velp_rel_robot: 3
    - goalkeeper_team_1_xpos_rel_robot: 3
    - goalkeeper_team_1_velp_rel_robot: 3
    - target_xpos_rel_robot: 3
    - target_velp_rel_robot: 3
    - defender_xpos: 3
    - task_one_hot: 7 (擴展至 7 維以達到官方 87 維)
    總計: 80 + 7 = 87 ✓

    技術決策（ADR-004）：
    - 官方 main.py 使用 n_features=87
    - 固定部分 80 維 + task_one_hot 7 維 = 87 維
    - 前 3 維保持原語義（GoaliePK, ObstaclePK, KickToTarget）
    - 後 4 維為預留空間（填 0）
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional


class EnvInfo(NamedTuple):
    """環境資訊結構，用於 Preprocessor 輸入"""
    robot_quat: jnp.ndarray          # (4,) 四元數 - 注意格式見下方說明
    robot_gyro: jnp.ndarray          # (3,) 角速度
    robot_accelerometer: jnp.ndarray # (3,) 加速度
    robot_velocimeter: jnp.ndarray   # (3,) 線速度
    robot_xpos: jnp.ndarray          # (3,) 機器人位置（世界座標）
    ball_xpos: jnp.ndarray           # (3,) 球位置
    ball_velp: jnp.ndarray           # (3,) 球平移速度
    ball_velr: jnp.ndarray           # (3,) 球旋轉速度
    goal_0_xpos: jnp.ndarray         # (3,) 球門 0 位置
    goal_1_xpos: jnp.ndarray         # (3,) 球門 1 位置
    goalkeeper_0_xpos: jnp.ndarray   # (3,) 門將 0 位置
    goalkeeper_0_vel: jnp.ndarray    # (3,) 門將 0 速度
    goalkeeper_1_xpos: jnp.ndarray   # (3,) 門將 1 位置
    goalkeeper_1_vel: jnp.ndarray    # (3,) 門將 1 速度
    target_xpos: jnp.ndarray         # (3,) 目標位置
    target_vel: jnp.ndarray          # (3,) 目標速度
    defender_xpos: jnp.ndarray       # (3,) 防守者位置
    player_team: jnp.ndarray         # (2,) 球員隊伍 one-hot
    task_one_hot: jnp.ndarray        # (7,) 任務編碼（擴展至 7 維以達到 87 維總輸出）


# =============================================================================
# 四元數工具函數
# =============================================================================

def quat_rotate_inverse(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """四元數反向旋轉：將世界座標系向量轉換到機器人座標系

    ⚠️ 重要：此函數假設四元數格式為 [x, y, z, w]（標量在最後）
    與官方 Preprocessor 保持一致。

    如果 MuJoCo 返回 [w, x, y, z] 格式，需要先轉換：
        quat_xyzw = jnp.roll(quat_wxyz, -1)

    Args:
        q: 四元數 [x, y, z, w]，shape (4,)
        v: 3D 向量，shape (3,)

    Returns:
        旋轉後的向量，shape (3,)
    """
    q_w = q[3]  # 標量部分（最後一個）
    q_vec = q[:3]  # 向量部分（前三個）

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = jnp.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (jnp.dot(q_vec, v) * 2.0)

    return a - b + c


def quat_wxyz_to_xyzw(q_wxyz: jnp.ndarray) -> jnp.ndarray:
    """將 MuJoCo [w,x,y,z] 格式轉換為 [x,y,z,w] 格式"""
    return jnp.roll(q_wxyz, -1)


def compute_project_gravity(quat: jnp.ndarray) -> jnp.ndarray:
    """計算重力在機器人座標系下的投影

    Args:
        quat: 四元數 [x, y, z, w]，shape (4,)

    Returns:
        重力投影，shape (3,)
    """
    gravity_world = jnp.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_world)


# =============================================================================
# 相對位置計算
# =============================================================================

def world_to_robot_frame(
    pos_world: jnp.ndarray,
    robot_pos: jnp.ndarray,
    robot_quat: jnp.ndarray
) -> jnp.ndarray:
    """將世界座標系位置轉換到機器人座標系

    Args:
        pos_world: 世界座標系位置，shape (3,)
        robot_pos: 機器人位置（世界座標），shape (3,)
        robot_quat: 機器人四元數 [x,y,z,w]，shape (4,)

    Returns:
        機器人座標系下的位置，shape (3,)
    """
    rel_pos_world = pos_world - robot_pos
    return quat_rotate_inverse(robot_quat, rel_pos_world)


def compute_relative_velocity(
    vel_world: jnp.ndarray,
    robot_quat: jnp.ndarray
) -> jnp.ndarray:
    """將世界座標系速度轉換到機器人座標系

    Args:
        vel_world: 世界座標系速度，shape (3,)
        robot_quat: 機器人四元數 [x,y,z,w]，shape (4,)

    Returns:
        機器人座標系下的速度，shape (3,)
    """
    return quat_rotate_inverse(robot_quat, vel_world)


# =============================================================================
# 主 Preprocessor
# =============================================================================

def preprocess_observation(
    robot_qpos: jnp.ndarray,
    robot_qvel: jnp.ndarray,
    info: EnvInfo
) -> jnp.ndarray:
    """將原始觀察轉換為 87 維特徵向量

    Args:
        robot_qpos: 關節位置，shape (12,)
        robot_qvel: 關節速度，shape (12,)
        info: EnvInfo 結構

    Returns:
        87 維觀察向量
    """
    # 基礎機器人狀態
    project_gravity = compute_project_gravity(info.robot_quat)

    # 球相對於機器人
    ball_xpos_rel_robot = world_to_robot_frame(
        info.ball_xpos, info.robot_xpos, info.robot_quat
    )
    ball_velp_rel_robot = compute_relative_velocity(
        info.ball_velp, info.robot_quat
    )
    ball_velr_rel_robot = compute_relative_velocity(
        info.ball_velr, info.robot_quat
    )

    # 球門相對於機器人
    goal_0_rel_robot = world_to_robot_frame(
        info.goal_0_xpos, info.robot_xpos, info.robot_quat
    )
    goal_1_rel_robot = world_to_robot_frame(
        info.goal_1_xpos, info.robot_xpos, info.robot_quat
    )

    # 球門相對於球
    goal_0_rel_ball = world_to_robot_frame(
        info.goal_0_xpos, info.ball_xpos, info.robot_quat
    )
    goal_1_rel_ball = world_to_robot_frame(
        info.goal_1_xpos, info.ball_xpos, info.robot_quat
    )

    # 門將相對於機器人
    gk_0_xpos_rel = world_to_robot_frame(
        info.goalkeeper_0_xpos, info.robot_xpos, info.robot_quat
    )
    gk_0_vel_rel = compute_relative_velocity(
        info.goalkeeper_0_vel, info.robot_quat
    )
    gk_1_xpos_rel = world_to_robot_frame(
        info.goalkeeper_1_xpos, info.robot_xpos, info.robot_quat
    )
    gk_1_vel_rel = compute_relative_velocity(
        info.goalkeeper_1_vel, info.robot_quat
    )

    # 目標相對於機器人
    target_xpos_rel = world_to_robot_frame(
        info.target_xpos, info.robot_xpos, info.robot_quat
    )
    target_vel_rel = compute_relative_velocity(
        info.target_vel, info.robot_quat
    )

    # 組裝 87 維向量（順序必須與官方一致！）
    obs = jnp.concatenate([
        robot_qpos,                     # 12
        robot_qvel,                     # 12
        project_gravity,                # 3
        info.robot_gyro,                # 3
        info.robot_accelerometer,       # 3
        info.robot_velocimeter,         # 3
        goal_0_rel_robot,               # 3
        goal_1_rel_robot,               # 3
        goal_0_rel_ball,                # 3
        goal_1_rel_ball,                # 3
        ball_xpos_rel_robot,            # 3
        ball_velp_rel_robot,            # 3
        ball_velr_rel_robot,            # 3
        info.player_team,               # 2
        gk_0_xpos_rel,                  # 3
        gk_0_vel_rel,                   # 3
        gk_1_xpos_rel,                  # 3
        gk_1_vel_rel,                   # 3
        target_xpos_rel,                # 3
        target_vel_rel,                 # 3
        info.defender_xpos,             # 3  TODO: 應該也要轉換到機器人座標系？
        info.task_one_hot,              # 7 (擴展以達到 87 維總輸出)
    ])

    return obs  # shape: (87,)


# 批量版本（用於 vmap）
preprocess_observation_batched = jax.vmap(
    preprocess_observation,
    in_axes=(0, 0, EnvInfo(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
)


# =============================================================================
# 測試工具
# =============================================================================

def create_dummy_env_info(batch_size: Optional[int] = None) -> EnvInfo:
    """創建測試用的 EnvInfo

    Args:
        batch_size: 如果提供，創建批量數據；否則創建單個樣本

    Returns:
        EnvInfo 結構
    """
    shape = () if batch_size is None else (batch_size,)

    return EnvInfo(
        robot_quat=jnp.array([0, 0, 0, 1] if batch_size is None
                            else [[0, 0, 0, 1]] * batch_size, dtype=jnp.float32),
        robot_gyro=jnp.zeros((*shape, 3)),
        robot_accelerometer=jnp.zeros((*shape, 3)),
        robot_velocimeter=jnp.zeros((*shape, 3)),
        robot_xpos=jnp.zeros((*shape, 3)),
        ball_xpos=jnp.ones((*shape, 3)),
        ball_velp=jnp.zeros((*shape, 3)),
        ball_velr=jnp.zeros((*shape, 3)),
        goal_0_xpos=jnp.array([4.5, 0, 0] if batch_size is None
                              else [[4.5, 0, 0]] * batch_size, dtype=jnp.float32),
        goal_1_xpos=jnp.array([-4.5, 0, 0] if batch_size is None
                              else [[-4.5, 0, 0]] * batch_size, dtype=jnp.float32),
        goalkeeper_0_xpos=jnp.array([4.0, 0, 0.5] if batch_size is None
                                    else [[4.0, 0, 0.5]] * batch_size, dtype=jnp.float32),
        goalkeeper_0_vel=jnp.zeros((*shape, 3)),
        goalkeeper_1_xpos=jnp.array([-4.0, 0, 0.5] if batch_size is None
                                    else [[-4.0, 0, 0.5]] * batch_size, dtype=jnp.float32),
        goalkeeper_1_vel=jnp.zeros((*shape, 3)),
        target_xpos=jnp.ones((*shape, 3)),
        target_vel=jnp.zeros((*shape, 3)),
        defender_xpos=jnp.zeros((*shape, 3)),
        player_team=jnp.array([1, 0] if batch_size is None
                              else [[1, 0]] * batch_size, dtype=jnp.float32),
        # 7 維 task_one_hot：前 3 維為任務 one-hot，後 4 維填 0
        task_one_hot=jnp.array([1, 0, 0, 0, 0, 0, 0] if batch_size is None
                               else [[1, 0, 0, 0, 0, 0, 0]] * batch_size, dtype=jnp.float32),
    )


def verify_output_dim():
    """驗證輸出維度是否為 87"""
    robot_qpos = jnp.zeros(12)
    robot_qvel = jnp.zeros(12)
    info = create_dummy_env_info()

    obs = preprocess_observation(robot_qpos, robot_qvel, info)

    assert obs.shape == (87,), f"Expected (87,), got {obs.shape}"
    print(f"✅ Output dimension verified: {obs.shape}")
    return obs


if __name__ == "__main__":
    # 簡單測試
    print("Testing Preprocessor JAX...")
    obs = verify_output_dim()
    print(f"Sample observation:\n{obs}")
