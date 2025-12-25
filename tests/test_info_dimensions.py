#!/usr/bin/env python3
"""
Day 1 第一步：驗證 info dict 各欄位維度
解開 83 vs 87 之謎！

執行方式：
    python scripts/verify_info_dimensions.py

需要在有 sai_mujoco 的環境中執行。
"""

import sys
import numpy as np

try:
    import gymnasium as gym
    import sai_mujoco  # noqa: F401 - 註冊環境
except ImportError as e:
    print(f"[ERROR] 缺少必要套件: {e}")
    print("請確保在有 sai_mujoco 的環境中執行")
    sys.exit(1)


def verify_dimensions():
    """驗證 info dict 各欄位維度"""

    print("=" * 60)
    print("維度驗證腳本 - 解開 83 vs 87 之謎")
    print("=" * 60)

    # 測試所有三個環境
    envs = [
        "LowerT1GoaliePenaltyKick-v0",
        "LowerT1ObstaclePenaltyKick-v0",
        "LowerT1KickToTarget-v0",
    ]

    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"環境: {env_name}")
        print("=" * 60)

        try:
            env = gym.make(env_name)
            obs, info = env.reset()

            print(f"\n=== Observation ===")
            print(f"obs.shape: {obs.shape}")
            print(f"obs.dtype: {obs.dtype}")

            print(f"\n=== Info Dict Keys ===")
            for key in sorted(info.keys()):
                val = info[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {key}: type={type(val).__name__}, value={val}")

            # 計算維度
            print(f"\n=== 維度計算 ===")

            # 預期的 info 欄位（基於 Preprocessor）
            expected_fields = [
                ("robot_quat", 4),
                ("robot_gyro", 3),
                ("robot_accelerometer", 3),
                ("robot_velocimeter", 3),
                ("goal_team_0_rel_robot", 3),
                ("goal_team_1_rel_robot", 3),
                ("goal_team_0_rel_ball", 3),
                ("goal_team_1_rel_ball", 3),
                ("ball_xpos_rel_robot", 3),
                ("ball_velp_rel_robot", 3),
                ("ball_velr_rel_robot", 3),
                ("player_team", 2),
                ("goalkeeper_team_0_xpos_rel_robot", 3),
                ("goalkeeper_team_0_velp_rel_robot", 3),
                ("goalkeeper_team_1_xpos_rel_robot", 3),
                ("goalkeeper_team_1_velp_rel_robot", 3),
                ("target_xpos_rel_robot", 3),
                ("target_velp_rel_robot", 3),
                ("defender_xpos", 3),
                ("task_index", "?"),  # 這是關鍵！可能是 3 或 7
            ]

            total_info_dim = 0
            print("\nInfo 欄位維度檢查:")
            for field, expected in expected_fields:
                if field in info:
                    val = np.array(info[field])
                    actual_dim = val.shape[-1] if len(val.shape) > 0 else 1
                    total_info_dim += actual_dim

                    # 標記關鍵欄位
                    marker = ""
                    if field == "task_index":
                        marker = " <-- 關鍵嫌疑犯 D！"
                    elif field == "robot_quat":
                        marker = " <-- 關鍵嫌疑犯 A！"

                    if expected == "?":
                        print(f"  {field}: dim={actual_dim}{marker}")
                    elif actual_dim != expected:
                        print(f"  {field}: dim={actual_dim} (預期: {expected}) ⚠️ 不匹配!{marker}")
                    else:
                        print(f"  {field}: dim={actual_dim} ✓{marker}")
                else:
                    print(f"  {field}: NOT FOUND ❌")

            # 檢查是否有額外的欄位
            known_fields = set(f[0] for f in expected_fields)
            extra_fields = set(info.keys()) - known_fields
            if extra_fields:
                print("\n額外發現的欄位（可能是缺失的 4 維！）:")
                for field in sorted(extra_fields):
                    val = info[field]
                    if isinstance(val, np.ndarray):
                        dim = val.shape[-1] if len(val.shape) > 0 else 1
                        total_info_dim += dim
                        print(f"  {field}: dim={dim} <-- 可能是缺失的維度！")
                    else:
                        print(f"  {field}: type={type(val).__name__}")

            # 總結
            print(f"\n=== 總結 ===")
            print(f"Info 欄位總維度: {total_info_dim}")
            print(f"+ robot_qpos (12) + robot_qvel (12) + project_gravity (3)")
            calculated_total = total_info_dim + 12 + 12 + 3
            print(f"= 計算總維度: {calculated_total}")
            print(f"目標維度: 87")

            diff = 87 - calculated_total
            if diff == 0:
                print("✅ 維度匹配！")
            elif diff > 0:
                print(f"❌ 差距: {diff} 維（缺少）")
            else:
                print(f"⚠️ 差距: {abs(diff)} 維（多出）")

            env.close()

        except Exception as e:
            print(f"[ERROR] 無法建立環境 {env_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("驗證完成！")
    print("=" * 60)


def examine_preprocessor_usage():
    """額外檢查：看看官方 Preprocessor 實際使用哪些欄位"""
    print("\n" + "=" * 60)
    print("額外檢查：Preprocessor 使用的欄位")
    print("=" * 60)

    try:
        # 嘗試導入官方 Preprocessor
        sys.path.insert(0, "/home/uk67h/Project/Booster_Soccer-plan/booster_soccer_showdown")
        from training_scripts.main import Preprocessor

        env = gym.make("LowerT1GoaliePenaltyKick-v0")
        obs, info = env.reset()

        preprocessor = Preprocessor()
        processed = preprocessor.modify_state(obs, info)

        print(f"\n原始 obs 維度: {obs.shape}")
        print(f"處理後維度: {processed.shape}")
        print(f"✅ 這就是官方 n_features 應該等於的值！")

        env.close()

    except Exception as e:
        print(f"[WARNING] 無法測試官方 Preprocessor: {e}")


if __name__ == "__main__":
    verify_dimensions()
    examine_preprocessor_usage()
