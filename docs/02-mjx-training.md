# MJX 訓練流程

## 概述

本文件說明如何建立 MJX 環境並進行 JAX SAC 預訓練。

**重要：** 開始訓練前必須通過 [05-verification-gates.md](./05-verification-gates.md) 中的 Gate 1 和 Gate 2。

---

## MJX 環境建立

### 方案選擇：Brax Wrapper（推薦）

使用 Brax 作為介面層，底層仍使用 MJX 物理引擎。Brax 提供：
- `AutoResetWrapper`：自動處理 episode 結束時的 reset
- `VmapWrapper`：簡化批次環境邏輯
- 與 JAX 生態系統良好整合

```bash
pip install brax
```

### 建立 MJX Scene XML（Critical Fix）

**問題：** `booster_lower_t1.xml` 只包含機器人定義，缺少地板和球。

**解決方案：** 建立 `mimic/assets/mjx_scene.xml`：

```xml
<mujoco model="mjx_soccer_scene">
  <!-- meshdir 指定 mesh 檔案的搜尋路徑（相對於此 XML） -->
  <compiler angle="radian" autolimits="true" meshdir="booster_t1/"/>
  <option timestep="0.002" integrator="RK4"/>

  <default>
    <geom condim="3" friction="1 0.5 0.5"/>
  </default>

  <worldbody>
    <!-- 地板 -->
    <geom name="ground" type="plane" size="15 10 0.1" rgba="0.3 0.6 0.3 1"/>

    <!-- 包含機器人（相對於 mjx_scene.xml 的路徑） -->
    <include file="booster_t1/booster_lower_t1.xml"/>

    <!-- 球 -->
    <body name="ball" pos="1 0 0.11">
      <joint name="ball_freejoint" type="free"/>
      <geom name="ball_geom" type="sphere" size="0.11" mass="0.43"
            rgba="1 0.5 0 1" friction="0.8 0.02 0.01"/>
    </body>

    <!-- 球門（簡化版，用於計算相對位置） -->
    <site name="goal_team_0" pos="7 0 0" size="0.1"/>
    <site name="goal_team_1" pos="-7 0 0" size="0.1"/>
  </worldbody>
</mujoco>
```

### ⚠️ XML 相對路徑注意事項

**常見錯誤：** `XML Error: File not found` 或 `Mesh not found`

**路徑結構必須如下：**
```
mimic/assets/
├── mjx_scene.xml           ← 主場景 XML
└── booster_t1/
    ├── booster_lower_t1.xml ← 機器人定義
    └── *.stl                ← Mesh 檔案
```

**Troubleshooting：**
1. `<include file="...">` 路徑是相對於**包含它的 XML 檔案**
2. Mesh 檔案路徑由 `<compiler meshdir="..."/>` 控制
3. 如果 `booster_lower_t1.xml` 內部使用絕對路徑引用 mesh，需要修改為相對路徑
4. **Day 1 第一個測試：** 執行 `mujoco.MjModel.from_xml_path("mimic/assets/mjx_scene.xml")`，確認無報錯

**🔧 暴力解法（Gemini 建議）：**
如果路徑問題搞不定，直接把所有 `.stl` 和 `.xml` 檔案**全部丟到同一個資料夾**（Flatten 結構）：
```
mimic/assets/flat/
├── mjx_scene.xml
├── booster_lower_t1.xml
├── *.stl  (所有 mesh 檔案)
```
比賽初期先能跑再說，後續再整理。

### 載入 Scene XML

```python
import mujoco
from mujoco import mjx

# 使用完整場景 XML（包含地板 + 球）
xml_path = "mimic/assets/mjx_scene.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)

# 轉換為 MJX model
try:
    mjx_model = mjx.put_model(mj_model)
    print("MJX model 建立成功！")
except Exception as e:
    print(f"MJX 兼容性問題: {e}")
    # 可能需要修改 XML 移除不支援的功能（如 user sensors）
```

### MJX 環境封裝（Brax 介面層）

建立文件 `training_scripts/mjx_env.py`：

```python
import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
from brax.envs.base import Env, State
from typing import Tuple

class MJXSoccerEnv(Env):
    """
    使用 Brax 作為介面層的 MJX 環境
    底層物理引擎是 MJX，Brax 只提供 Env 介面和 Wrappers
    """

    def __init__(self, config: dict = None):
        # 載入完整場景 XML（包含地板 + 球）
        xml_path = "mimic/assets/mjx_scene.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # === 使用 mj_name2id 獲取 Body ID（禁止硬編碼！）===
        self.torso_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.ball_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        # 腳部 ID（用於踢球獎勵）
        self.foot_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"),
        ]
        # 球門 Site ID
        self.goal_0_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_team_0"
        )
        self.goal_1_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_team_1"
        )

        # Domain Randomization 參數（見下方章節）
        self.config = config or {}
        self.mass_range = self.config.get('mass_range', (0.9, 1.1))
        self.friction_range = self.config.get('friction_range', (0.8, 1.2))

        # 訓練進度相關（用於動態權重）
        self.total_steps = self.config.get('total_steps', 10_000_000)

    @property
    def observation_size(self) -> int:
        return 87  # 與官方環境一致

    @property
    def action_size(self) -> int:
        return 12  # 12 個關節

    def reset(self, rng: jax.Array) -> State:
        """重置環境，返回 Brax State"""
        rng, ball_rng, domain_rng, task_rng = jax.random.split(rng, 4)

        # 初始化 MJX data
        data = mjx.make_data(self.mjx_model)

        # 隨機初始化球位置
        ball_pos = jax.random.uniform(ball_rng, (2,), minval=-1.0, maxval=1.0)

        # Domain Randomization（可選）
        data = self._apply_domain_randomization(data, domain_rng)

        # === Task Index 隨機化（Gemini 最佳建議）===
        # 即使物理場景相同，也隨機注入 task_index
        # 讓網路保持對 task_index 的敏感度
        task_id = jax.random.randint(task_rng, (), 0, 3)
        task_onehot = jax.nn.one_hot(task_id, 3)

        obs = self._get_obs(data, ball_pos, task_onehot)

        return State(
            pipeline_state=data,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={'ball_pos': ball_pos, 'task_id': task_id, 'prev_action': jnp.zeros(12), 'step_count': 0}
        )

    def step(self, state: State, action: jax.Array) -> State:
        """執行一步，返回新的 State"""
        data = state.pipeline_state
        task_id = state.info['task_id']
        prev_action = state.info['prev_action']
        step_count = state.info.get('step_count', 0)

        # MJX 物理 step
        data = data.replace(ctrl=action)
        data = mjx.step(self.mjx_model, data)

        # === 修正：傳入 task_onehot 給 _get_obs ===
        task_onehot = jax.nn.one_hot(task_id, 3)
        obs = self._get_obs(data, task_onehot)
        reward = self._compute_reward(data, prev_action, action, step_count, self.total_steps)
        done = self._check_done(data)

        # 更新 prev_action 和 step_count 用於下一步
        new_info = state.info.copy()
        new_info['prev_action'] = action
        new_info['step_count'] = step_count + 1

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info
        )

    def _get_obs(self, data, task_onehot):
        """提取觀測 - 使用 JAX Preprocessor"""
        # 1. 從 MJX data 建構 info dict
        info = self._build_info_from_mjx_data(data, task_onehot)

        # 2. 建構基礎 obs（qpos, qvel）
        robot_qpos = data.qpos[7:19]  # 跳過 root freejoint (7 DOF)
        robot_qvel = data.qvel[6:18]  # 跳過 root (6 DOF)
        base_obs = jnp.concatenate([robot_qpos, robot_qvel])

        # 3. 使用 Preprocessor 產生 87 維觀測
        from training_scripts.preprocessor_jax import PreprocessorJAX
        preprocessor = PreprocessorJAX()
        return preprocessor.modify_state(base_obs[None, :], info, task_onehot[None, :])[0]

    def _build_info_from_mjx_data(self, data, task_onehot):
        """
        從 MJX data 重建官方環境的 info dict

        **Critical:** MJX 不會自動提供 info，必須手動計算！
        只建構 Preprocessor 實際使用的 keys。
        """
        # 機器人位置和朝向
        robot_pos = data.xpos[self.torso_id]
        robot_quat = data.qpos[3:7]  # MuJoCo: [w, x, y, z]
        robot_vel = data.qvel[:3]
        robot_ang_vel = data.qvel[3:6]

        # 球位置和速度
        ball_pos = data.xpos[self.ball_id]
        ball_vel = data.qvel[-6:-3]  # ball freejoint 的線速度
        ball_ang_vel = data.qvel[-3:]

        # 球門位置（從 site 獲取）
        goal_0_pos = data.site_xpos[self.goal_0_id]
        goal_1_pos = data.site_xpos[self.goal_1_id]

        # 計算相對位置
        ball_rel_robot = ball_pos - robot_pos
        goal_0_rel_robot = goal_0_pos - robot_pos
        goal_1_rel_robot = goal_1_pos - robot_pos
        goal_0_rel_ball = goal_0_pos - ball_pos
        goal_1_rel_ball = goal_1_pos - ball_pos

        return {
            "robot_quat": robot_quat[None, :],
            "robot_gyro": robot_ang_vel[None, :],
            "robot_accelerometer": jnp.zeros((1, 3)),  # 簡化：MJX 預訓練不需要精確值
            "robot_velocimeter": robot_vel[None, :],
            "goal_team_0_rel_robot": goal_0_rel_robot[None, :],
            "goal_team_1_rel_robot": goal_1_rel_robot[None, :],
            "goal_team_0_rel_ball": goal_0_rel_ball[None, :],
            "goal_team_1_rel_ball": goal_1_rel_ball[None, :],
            "ball_xpos_rel_robot": ball_rel_robot[None, :],
            "ball_velp_rel_robot": ball_vel[None, :],
            "ball_velr_rel_robot": ball_ang_vel[None, :],
            "player_team": jnp.array([[1.0, 0.0]]),  # 固定為 team 0
            # === 以下 keys 在 MJX 預訓練設為 zeros ===
            "goalkeeper_team_0_xpos_rel_robot": jnp.zeros((1, 3)),
            "goalkeeper_team_0_velp_rel_robot": jnp.zeros((1, 3)),
            "goalkeeper_team_1_xpos_rel_robot": jnp.zeros((1, 3)),
            "goalkeeper_team_1_velp_rel_robot": jnp.zeros((1, 3)),
            "target_xpos_rel_robot": jnp.zeros((1, 3)),
            "target_velp_rel_robot": jnp.zeros((1, 3)),
            "defender_xpos": jnp.zeros((1, 3)),
        }

    def _compute_reward(self, data, prev_action, action, step, total_steps):
        """簡化獎勵函數 - 見下方章節"""
        from training_scripts.rewards import compute_locomotion_reward

        # 獲取球位置和攻擊方球門位置
        ball_pos = data.xpos[self.ball_id]
        goal_pos = data.site_xpos[self.goal_0_id]  # 假設 team 0 攻擊 goal 0

        return compute_locomotion_reward(
            data, ball_pos, goal_pos, self.torso_id, self.foot_ids,
            prev_action, action, step, total_steps
        )

    def _check_done(self, data):
        """檢查是否結束"""
        torso_height = data.xpos[self.torso_id, 2]
        return torso_height < 0.25  # 修正：閾值從 0.2 改為 0.25

    def _apply_domain_randomization(self, data, rng):
        """Domain Randomization - 見下方章節"""
        # 可選：在 reset 時隨機化物理參數
        return data
```

### 使用 Brax Wrappers

```python
from brax.envs.wrappers.training import AutoResetWrapper, VmapWrapper

# 建立基礎環境
base_env = MJXSoccerEnv()

# 添加 Auto Reset（episode 結束自動 reset）
env = AutoResetWrapper(base_env)

# 添加 Vmap（並行多個環境）
env = VmapWrapper(env, batch_size=2048)

# 現在可以批次操作
rng = jax.random.PRNGKey(0)
state = env.reset(rng)  # state.obs.shape = (2048, 87)
action = jnp.zeros((2048, 12))
state = env.step(state, action)
```

---

## Preprocessor JAX 翻譯

原始 NumPy 版本在 `training_scripts/main.py`。需要翻譯成 JAX 版本。

### ⚠️ Day 1 必做：先驗證再實作

**Critical:** 在實作 JAX Preprocessor 前，**必須先執行 `verify_info_dimensions.py`**！

目前的 Preprocessor 是**佔位符版本**，維度假設可能不正確。流程：

1. Day 1 第一步：執行 `verify_info_dimensions.py` 確認實際維度
2. 根據驗證結果更新下方的 `PreprocessorJAX`
3. 執行 Gate 1 驗證確保 JAX/NumPy 輸出一致
4. 通過 Gate 1 後才能開始 MJX 訓練

建立文件 `training_scripts/preprocessor_jax.py`：

```python
import jax.numpy as jnp

class PreprocessorJAX:
    """
    ⚠️ 佔位符版本 - Gate 1 驗證後更新！

    目前假設的維度可能不正確。Day 1 必須：
    1. 執行 verify_info_dimensions.py
    2. 根據實際結果更新此類
    3. 重新通過 Gate 1 驗證

    已知不確定性：
    - 4 維差距來源未確認（可能是 Quaternion 或接觸力）
    - Quaternion 順序需要物理驗證 (w,x,y,z vs x,y,z,w)

    更新記錄：
    - v0.1: 初始佔位符版本
    - v0.2: [Day 1 驗證後更新]
    """

    def quat_rotate_inverse(self, q, v):
        """四元數旋轉逆運算 - JAX 版本"""
        # ⚠️ 假設 MuJoCo 標準順序 [w, x, y, z]
        # 如果 Gate 1 失敗，檢查這裡的順序！
        q_w = q[:, -1:]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = jnp.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (jnp.dot(q_vec, v).reshape(-1, 1) * 2.0)
        return a - b + c

    def modify_state(self, obs, info, task_onehot):
        """
        翻譯自 training_scripts/main.py 的 Preprocessor
        確保輸出維度與順序完全一致！

        🚨 維度不匹配警告！
        官方 n_features=87，但基於假設的計算 = 83
        差距 4 維 → Day 1 第一步用 verify_info_dimensions.py 驗證！

        === 缺失 4 維的嫌疑犯（Gemini 分析）===

        嫌疑犯 A：Root Quaternion (4維) ★極可能★
          - 很多環境保留原始 robot_quat (4維) 在觀測中
          - project_gravity (3維) 是從 quat 計算出來的，但可能兩者都用

        嫌疑犯 B：腳部接觸感測器 (4維) ★可能★
          - 雙足機器人常有 4 個接觸點 (左前/左後/右前/右後)
          - 對 Locomotion 非常重要

        嫌疑犯 C：時間相關 (4維)
          - sin(time), cos(time), phase 等

        嫌疑犯 D：Task ID 擴展 (4維)
          - task_index 可能是 7 維 [task_onehot(3) + params(4)]

        === 當前假設的維度 ===
        robot_qpos:           12
        robot_qvel:           12
        project_gravity:       3
        base_ang_vel:          3
        robot_accelerometer:   3
        robot_velocimeter:     3
        goal_team_0_rel_robot: 3
        goal_team_1_rel_robot: 3
        goal_team_0_rel_ball:  3
        goal_team_1_rel_ball:  3
        ball_xpos_rel_robot:   3
        ball_velp_rel_robot:   3
        ball_velr_rel_robot:   3
        player_team:           2
        goalkeeper_0_xpos:     3
        goalkeeper_0_velp:     3
        goalkeeper_1_xpos:     3
        goalkeeper_1_velp:     3
        target_xpos:           3
        target_velp:           3
        defender_xpos:         3
        task_onehot:           3
        ─────────────────────────
        假設 Total:           83
        目標 Total:           87
        缺失:                  4 ← 用腳本驗證！
        """
        robot_qpos = obs[:, :12]
        robot_qvel = obs[:, 12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(
            quat, jnp.array([0.0, 0.0, -1.0])
        )

        obs = jnp.concatenate([
            robot_qpos,                              # 12
            robot_qvel,                              # 12
            project_gravity,                         # 3
            base_ang_vel,                            # 3
            info["robot_accelerometer"],             # 3
            info["robot_velocimeter"],               # 3
            info["goal_team_0_rel_robot"],           # 3
            info["goal_team_1_rel_robot"],           # 3
            info["goal_team_0_rel_ball"],            # 3
            info["goal_team_1_rel_ball"],            # 3
            info["ball_xpos_rel_robot"],             # 3
            info["ball_velp_rel_robot"],             # 3
            info["ball_velr_rel_robot"],             # 3
            info["player_team"],                     # 2
            info["goalkeeper_team_0_xpos_rel_robot"],# 3
            info["goalkeeper_team_0_velp_rel_robot"],# 3
            info["goalkeeper_team_1_xpos_rel_robot"],# 3
            info["goalkeeper_team_1_velp_rel_robot"],# 3
            info["target_xpos_rel_robot"],           # 3
            info["target_velp_rel_robot"],           # 3
            info["defender_xpos"],                   # 3
            task_onehot                              # 3
        ], axis=-1)  # Total: 87

        return obs
```

### Day 1 必做：維度驗證

⚠️ **Critical:** 在實作 JAX Preprocessor 前，必須先確認各欄位的實際維度！

```python
# scripts/verify_info_dimensions.py
import gymnasium as gym
import sai_mujoco  # noqa: F401
import numpy as np

def verify_dimensions():
    """Day 1 第一步：驗證 info dict 各欄位維度"""
    env = gym.make("LowerT1GoaliePenaltyKick-v0")
    obs, info = env.reset()

    print("=== Observation ===")
    print(f"obs.shape: {obs.shape}")  # 預期: (24,) 或類似

    print("\n=== Info Dict 維度 ===")
    total_dim = 0
    fields = [
        "robot_quat", "robot_gyro", "robot_accelerometer", "robot_velocimeter",
        "goal_team_0_rel_robot", "goal_team_1_rel_robot",
        "goal_team_0_rel_ball", "goal_team_1_rel_ball",
        "ball_xpos_rel_robot", "ball_velp_rel_robot", "ball_velr_rel_robot",
        "player_team",
        "goalkeeper_team_0_xpos_rel_robot", "goalkeeper_team_0_velp_rel_robot",
        "goalkeeper_team_1_xpos_rel_robot", "goalkeeper_team_1_velp_rel_robot",
        "target_xpos_rel_robot", "target_velp_rel_robot",
        "defender_xpos", "task_index"
    ]

    for key in fields:
        if key in info:
            val = np.array(info[key])
            dim = val.shape[-1] if len(val.shape) > 0 else 1
            print(f"  {key}: {val.shape} → dim={dim}")
            total_dim += dim
        else:
            print(f"  {key}: NOT FOUND")

    # 加上 robot_qpos(12) + robot_qvel(12) + project_gravity(3)
    print(f"\n=== 總維度計算 ===")
    print(f"Info 欄位總和: {total_dim}")
    print(f"+ robot_qpos(12) + robot_qvel(12) + project_gravity(3)")
    print(f"= {total_dim + 12 + 12 + 3}")
    print(f"目標: 87")

    # === 新增：物理參數驗證（對齊 MJX 場景） ===
    print("\n=== Physics Parameters ===")
    import mujoco
    model = env.unwrapped.model  # 獲取底層 MuJoCo model

    # 嘗試獲取地面摩擦力
    try:
        ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if ground_id >= 0:
            friction = model.geom_friction[ground_id]
            print(f"Ground friction: {friction}")
        else:
            # 嘗試其他可能的名稱
            for name in ["floor", "plane", "field"]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    friction = model.geom_friction[gid]
                    print(f"{name} friction: {friction}")
                    break
    except Exception as e:
        print(f"Failed to get ground friction: {e}")

    # 嘗試獲取球摩擦力
    try:
        for ball_name in ["ball_geom", "ball", "soccer_ball"]:
            ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ball_name)
            if ball_id >= 0:
                friction = model.geom_friction[ball_id]
                print(f"Ball ({ball_name}) friction: {friction}")
                break
    except Exception as e:
        print(f"Failed to get ball friction: {e}")

    # 嘗試獲取機器人腳摩擦力
    try:
        for foot_name in ["left_foot_geom", "right_foot_geom", "left_foot", "right_foot"]:
            foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            if foot_id >= 0:
                friction = model.geom_friction[foot_id]
                print(f"{foot_name} friction: {friction}")
    except Exception as e:
        print(f"Failed to get foot friction: {e}")

    print("\n>>> 請將以上 friction 值更新到 mimic/assets/mjx_scene.xml <<<")

    env.close()

if __name__ == "__main__":
    verify_dimensions()
```

**執行後更新文檔中的維度假設！**

### 驗證翻譯正確性

```python
import numpy as np
import jax.numpy as jnp

# 準備測試數據
test_obs = np.random.randn(1, 87).astype(np.float32)
test_info = {...}  # 準備 info dict

# NumPy 版本
from training_scripts.main import Preprocessor
np_preprocessor = Preprocessor()
np_result = np_preprocessor.modify_state(test_obs[0], test_info)

# JAX 版本
from training_scripts.preprocessor_jax import PreprocessorJAX
jax_preprocessor = PreprocessorJAX()
jax_result = jax_preprocessor.modify_state(jnp.array(test_obs), test_info, task_onehot)

# 驗證
assert np.allclose(np_result, np.array(jax_result[0])), "Preprocessor 翻譯不一致！"
print("Preprocessor 翻譯驗證通過！")
```

---

## 獎勵函數設計

### 官方評估 vs MJX 預訓練

官方評估使用**稀疏獎勵**（只在 episode 結束時計算），我們的 MJX 預訓練使用**密集獎勵**。

| 獎勵類型 | 計算時機 | 用途 |
|----------|----------|------|
| **官方評估** | Episode 結束 | 最終排名分數 |
| **MJX 預訓練** | 每一步 | 加速學習基礎技能 |

### 官方評估獎勵結構

**GoaliePenaltyKick & ObstaclePenaltyKick（權重相同）：**

```python
reward_config = {
    "robot_distance_ball": 0.25,   # 接近球
    "ball_vel_twd_goal": 1.5,      # 球朝目標速度
    "goal_scored": 2.50,           # 進球
    "offside": -3.0,               # 越位
    "ball_hits": -0.2,             # 球碰撞
    "robot_fallen": -1.5,          # 倒下（重懲罰！）
    "ball_blocked": -0.5,          # 球被阻擋
    "steps": -1.0,                 # 時間懲罰
}
```

**KickToTarget：**

```python
reward_config = {
    "offside": -1.0,
    "success": 2.0,
    "distance": 0.5,
    "steps": -0.3,  # 較輕的時間懲罰
}
```

### 設計洞察

1. **`robot_fallen` 懲罰很重 (-1.5)**：確認「站立」是核心技能
2. **`steps` 懲罰存在**：Agent 需要快速完成任務
3. **進球獎勵 (+2.5) 高於接近球 (+0.25)**：最終目標是進球，不是靠近球

### MJX 預訓練獎勵（密集版本）

建立文件 `training_scripts/rewards.py`：

```python
import jax.numpy as jnp

def compute_locomotion_reward(data, ball_pos, goal_pos, torso_id, foot_ids, prev_action, action, step, total_steps):
    """
    MJX 預訓練獎勵 - 密集版本（整合 Gemini 審查建議）

    設計原則：
    1. 對齊官方指標（但每步計算）
    2. 強調站立（官方 robot_fallen = -1.5）
    3. 鼓勵快速行動（對應 steps 懲罰）
    4. Action Smoothness（減少 Sim-to-Sim Gap）
    5. 動態權重調整（訓練進度）
    """
    # 訓練進度（用於動態權重）
    progress = step / total_steps

    # R1: 站立獎勵（對應官方 robot_fallen）
    torso_height = data.xpos[:, torso_id, 2]
    r_stand = jnp.where(torso_height > 0.3, 0.5, -1.5)  # 倒下給重懲罰

    # R2: 接近球（對應官方 robot_distance_ball）
    robot_xy = data.xpos[:, 0, :2]
    ball_dist = jnp.linalg.norm(robot_xy - ball_pos[:, :2], axis=-1)
    r_approach = 0.25 * jnp.exp(-ball_dist)  # 權重對齊官方 0.25

    # R3: 腳接觸球（僅作為引導信號，降低權重防止「貼球站著」）
    foot_pos = data.xpos[:, foot_ids[0], :2]
    foot_ball_dist = jnp.linalg.norm(foot_pos - ball_pos[:, :2], axis=-1)
    r_kick = jnp.where(foot_ball_dist < 0.1, 0.1, 0.0)  # 降低: 2.5 → 0.1

    # R4: 球朝向球門的速度（主要踢球獎勵，對應官方 ball_vel_twd_goal）
    # 這是真正的踢球信號：鼓勵把球踢向球門，而不只是接觸球
    ball_vel_xy = data.qvel[-6:-4]  # 球的 XY 速度（假設球是最後一個 freejoint）
    goal_pos_xy = goal_pos[:, :2]   # 需要從外部傳入攻擊方球門位置
    ball_to_goal = goal_pos_xy - ball_pos[:, :2]
    ball_to_goal_dir = ball_to_goal / (jnp.linalg.norm(ball_to_goal, axis=-1, keepdims=True) + 1e-6)
    vel_towards_goal = jnp.sum(ball_vel_xy * ball_to_goal_dir, axis=-1)
    r_ball_vel = 1.5 * jnp.clip(vel_towards_goal, 0.0, 2.0)  # 權重對齊官方 1.5，速度上限 2 m/s

    # R5: 時間懲罰（對應官方 steps = -1.0）
    r_time = -0.01  # 每步小懲罰，鼓勵快速完成

    # R6: 能量懲罰
    r_energy = -0.01 * jnp.sum(data.ctrl ** 2, axis=-1)

    # === R7: Action Smoothness（Gemini 建議）===
    # 減少高頻震盪，提高 Sim-to-Sim 遷移穩定性
    delta = action - prev_action
    r_smoothness = -jnp.sum(delta ** 2, axis=-1)

    # === 動態權重（根據訓練進度調整）===
    # 早期：重視站立 + smoothness
    # 後期：重視任務完成（ball_vel 成為主力）
    w_stand = 0.4 - 0.2 * progress      # 0.4 → 0.2
    w_approach = 0.3 - 0.1 * progress   # 0.3 → 0.2（後期降低，讓 ball_vel 主導）
    w_kick = 0.05                        # 固定小權重（僅引導）
    w_ball_vel = 0.1 + 0.3 * progress   # 0.1 → 0.4（後期成為主要踢球獎勵）
    w_smooth = 0.1 - 0.05 * progress    # 0.1 → 0.05（後期降低）

    reward = (
        r_stand * w_stand +
        r_approach * w_approach +
        r_kick * w_kick +
        r_ball_vel * w_ball_vel +       # 新增：主要踢球獎勵
        r_time * 0.1 +
        r_energy * 0.05 +
        r_smoothness * w_smooth
    )

    return reward
```

### 獎勵權重對齊表

| 官方指標 | 官方權重 | MJX 對應 | 調整說明 |
|----------|----------|----------|----------|
| `robot_fallen` | -1.5 | `r_stand` | 每步檢查，倒下立即懲罰 |
| `robot_distance_ball` | +0.25 | `r_approach` | 距離越近獎勵越高 |
| `goal_scored` | +2.5 | `r_kick` | **降為 0.1（僅引導信號）** |
| `ball_vel_twd_goal` | +1.5 | `r_ball_vel` | **新增：主要踢球獎勵** |
| `steps` | -1.0 | `r_time` | 每步小懲罰累積 |

**設計理念：**
- 讓機器人學會「不倒 + 走向球 + 踢球 + 快速行動」
- 複雜策略（射門角度、閃避守門員）由官方環境微調學習

---

## Domain Randomization

為減少 Sim-to-Sim Gap（MJX → 官方環境），在預訓練階段加入物理參數隨機化。

### 為什麼需要 Domain Randomization？

| 差異來源 | MJX | 官方 MuJoCo | 影響 |
|----------|-----|-------------|------|
| 浮點精度 | float32 (GPU) | float64 (CPU) | 累積誤差 |
| 時間步長 | 可能不同 | 固定 | 控制頻率差異 |
| 接觸模型 | MJX 實作 | MuJoCo 原生 | 碰撞行為差異 |

### 實作

在 `training_scripts/mjx_env.py` 中添加：

```python
def _apply_domain_randomization(self, data, rng):
    """
    在每個 episode 開始時隨機化物理參數
    讓模型學習更魯棒的策略
    """
    rng, mass_rng, friction_rng, damping_rng = jax.random.split(rng, 4)

    # 1. 質量隨機化 (±10%)
    mass_scale = jax.random.uniform(
        mass_rng, (), minval=self.mass_range[0], maxval=self.mass_range[1]
    )
    # 注意：MJX 中修改質量需要重新計算慣性矩陣
    # 這裡是概念代碼，實際實作需要參考 MJX API

    # 2. 摩擦力隨機化 (±20%)
    friction_scale = jax.random.uniform(
        friction_rng, (), minval=self.friction_range[0], maxval=self.friction_range[1]
    )

    # 3. 關節阻尼隨機化 (±15%)
    damping_scale = jax.random.uniform(
        damping_rng, (), minval=0.85, maxval=1.15
    )

    # 返回修改後的 data（或 model）
    return data

def _apply_observation_noise(self, obs, rng):
    """
    可選：添加感測器噪聲
    增強對感測器誤差的魯棒性
    """
    noise = jax.random.normal(rng, obs.shape) * 0.01
    return obs + noise
```

### 隨機化參數配置（三級分層）

根據 Gemini 審查建議，DR 分為三個級別：

```python
# Level 1: 基礎 DR（默認使用）
level1_config = {
    'mass_range': (0.95, 1.05),       # ±5%
    'friction_range': (0.9, 1.1),     # ±10%
    'damping_range': (0.95, 1.05),    # ±5%
    'obs_noise_std': 0.005,           # 小噪聲
}

# Level 2: 進階 DR（如果 Gate 3 失敗）
level2_config = {
    'mass_range': (0.9, 1.1),         # ±10%
    'friction_range': (0.7, 1.3),     # ±30%
    'damping_range': (0.85, 1.15),    # ±15%
    'obs_noise_std': 0.02,            # 較大噪聲
}

# Level 3: 激進 DR + 球/腳強化（最後手段）
level3_config = {
    'mass_range': (0.8, 1.2),         # ±20%
    'friction_range': (0.5, 1.5),     # ±50%
    'damping_range': (0.8, 1.2),      # ±20%
    'obs_noise_std': 0.03,            # 大噪聲
    # 球/腳特別強化（Gemini 建議）
    'ball_mass_range': (0.8, 1.2),    # ±20%
    'ball_friction_range': (0.7, 1.3), # ±30%
    'foot_friction_range': (0.5, 1.5), # ±50%（重點）
}
```

**使用策略：**
1. 先用 Level 1 完成基礎訓練
2. 如果 Gate 3（行為一致性）失敗，升級到 Level 2
3. 如果仍有問題，使用 Level 3 + 更長微調時間

### 注意事項

1. **不要過度隨機化：** 過大的隨機範圍會讓學習變困難
2. **漸進式增加：** 可以先在小範圍訓練，再逐步增加隨機程度
3. **監控訓練曲線：** 如果 reward 突然下降，可能是隨機化太激進

---

## JAX SAC 訓練腳本

建立文件 `training_scripts/train_mjx_sac.py`：

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
from flashbax import make_item_buffer

from mjx_env import MJXSoccerEnv
from rewards import compute_locomotion_reward

# 重用現有網路架構
from imitation_learning.utils.networks import MLP

class SACAgent:
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        self.actor = MLP(obs_dim, action_dim * 2, hidden_dims)  # mean + log_std
        self.critic1 = MLP(obs_dim + action_dim, 1, hidden_dims)
        self.critic2 = MLP(obs_dim + action_dim, 1, hidden_dims)
        # Target networks
        self.target_critic1 = MLP(obs_dim + action_dim, 1, hidden_dims)
        self.target_critic2 = MLP(obs_dim + action_dim, 1, hidden_dims)

    def init_params(self, rng, obs_dim, action_dim):
        # 初始化網路參數
        pass

    @jax.jit
    def select_action(self, params, obs, rng):
        mean, log_std = jnp.split(self.actor.apply(params['actor'], obs), 2, axis=-1)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, mean.shape)
        action = jnp.tanh(mean + std * noise)
        return action

    @jax.jit
    def update(self, params, batch, rng):
        # SAC 更新邏輯
        # 1. 更新 Critic
        # 2. 更新 Actor
        # 3. 更新 entropy coefficient
        # 4. 軟更新 target networks
        pass

def main():
    # 初始化 W&B
    wandb.init(project="booster_soccer_mjx", config={
        "batch_size": 2048,
        "total_timesteps": 10_000_000,
        "learning_rate": 3e-4,
    })

    # 建立環境和 agent
    env = MJXSoccerEnv(batch_size=2048)
    agent = SACAgent(obs_dim=87, action_dim=12)

    # 建立 Replay Buffer
    buffer = make_item_buffer(
        max_length=1_000_000,
        min_length=10_000,
        sample_batch_size=256,
    )

    # 訓練循環
    rng = jax.random.PRNGKey(0)
    for step in range(10_000_000):
        # 1. 收集數據
        rng, action_rng = jax.random.split(rng)
        actions = agent.select_action(params, obs, action_rng)
        data, obs, reward, done, ball_pos = env.step(data, actions, ball_pos)

        # 2. 存入 buffer
        buffer.add(...)

        # 3. 更新 agent
        if step > 10_000:
            batch = buffer.sample()
            rng, update_rng = jax.random.split(rng)
            params, info = agent.update(params, batch, update_rng)

        # 4. 記錄到 W&B
        if step % 10_000 == 0:
            wandb.log({
                "reward": float(reward.mean()),
                "step": step
            })

    # 保存 checkpoint
    import pickle
    with open("exp/mjx_sac/checkpoint.pkl", "wb") as f:
        pickle.dump(params, f)

if __name__ == "__main__":
    main()
```

---

## 執行訓練

```bash
# 在 Databricks 執行
python training_scripts/train_mjx_sac.py
```

**預期結果：**
- 10M 步 ≈ 2-4 小時（L4 GPU）
- 機器人應能：穩定站立、走向球、嘗試踢球
- W&B 監控：reward 曲線上升並穩定

---

## 下一步

訓練完成後，前往 [03-finetuning-submission.md](./03-finetuning-submission.md) 進行模型轉換和微調。
