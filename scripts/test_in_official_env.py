"""
官方環境測試腳本

使用方式：
    python scripts/test_in_official_env.py --api-key YOUR_API_KEY

前置條件：
    1. model.pt 已放置在 submission/ 目錄
    2. 已安裝 sai_rl 套件
"""
import argparse
import numpy as np
import sys
from pathlib import Path

# 添加專案路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from submission.model import BoosterModel
from submission.preprocessor import Preprocessor


# Task one-hot 編碼（根據 CLAUDE.md）
TASK_ONE_HOT = {
    "GoaliePenaltyKick": np.array([1.0, 0.0, 0.0]),
    "ObstaclePenaltyKick": np.array([0.0, 1.0, 0.0]),
    "KickToTarget": np.array([0.0, 0.0, 1.0]),
}

# 競賽 ID 對應
COMP_IDS = {
    "GoaliePenaltyKick": "lower-t1-penalty-kick-goalie",
    "ObstaclePenaltyKick": "lower-t1-penalty-kick-obstacle",
    "KickToTarget": "lower-t1-kick-to-target",
}


def test_single_task(
    model: BoosterModel,
    preprocessor: Preprocessor,
    api_key: str,
    task_name: str,
    max_steps: int = 1000,
    num_episodes: int = 3,
):
    """測試單一任務"""
    from sai_rl import SAIClient

    print(f"\n{'='*50}")
    print(f"測試任務: {task_name}")
    print(f"{'='*50}")

    comp_id = COMP_IDS[task_name]
    task_one_hot = TASK_ONE_HOT[task_name]

    sai = SAIClient(comp_id=comp_id, api_key=api_key)
    env = sai.make_env()

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0

        for step in range(max_steps):
            # 預處理觀察
            processed_obs = preprocessor.modify_state(obs, info, task_one_hot)

            # 模型推理
            action = model(processed_obs[np.newaxis, :]).numpy().squeeze()

            # 執行動作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, steps={step_count}")

    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\n  平均 reward: {avg_reward:.2f}")
    print(f"  平均 steps:  {avg_length:.1f}")

    # 基本成功標準
    success = avg_length > 50  # 至少存活 50 步
    print(f"  基本測試: {'✅ 通過' if success else '❌ 失敗'}（存活 > 50 步）")

    return {
        "task": task_name,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "success": success,
    }


def main():
    parser = argparse.ArgumentParser(description="官方環境測試")
    parser.add_argument("--api-key", type=str, required=True, help="SAI API Key")
    parser.add_argument("--model-path", type=str, default="submission/model.pt", help="模型路徑")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "GoaliePenaltyKick", "ObstaclePenaltyKick", "KickToTarget"],
                        help="要測試的任務")
    parser.add_argument("--episodes", type=int, default=3, help="每個任務測試的 episode 數")
    args = parser.parse_args()

    # 載入模型
    model_path = PROJECT_ROOT / args.model_path
    if not model_path.exists():
        print(f"❌ 找不到模型: {model_path}")
        print("請先執行 jax2torch 轉換並將 model.pt 放到 submission/ 目錄")
        return 1

    print(f"載入模型: {model_path}")
    model = BoosterModel(str(model_path))
    preprocessor = Preprocessor()

    # 測試任務
    tasks = list(TASK_ONE_HOT.keys()) if args.task == "all" else [args.task]
    results = []

    for task in tasks:
        try:
            result = test_single_task(
                model=model,
                preprocessor=preprocessor,
                api_key=args.api_key,
                task_name=task,
                num_episodes=args.episodes,
            )
            results.append(result)
        except Exception as e:
            print(f"❌ 測試 {task} 時發生錯誤: {e}")
            results.append({"task": task, "success": False, "error": str(e)})

    # 總結
    print(f"\n{'='*50}")
    print("測試總結")
    print(f"{'='*50}")
    for r in results:
        status = "✅" if r.get("success") else "❌"
        if "error" in r:
            print(f"  {status} {r['task']}: 錯誤 - {r['error']}")
        else:
            print(f"  {status} {r['task']}: reward={r['avg_reward']:.2f}, steps={r['avg_length']:.1f}")

    all_passed = all(r.get("success", False) for r in results)
    print(f"\n總體結果: {'✅ 全部通過' if all_passed else '⚠️ 部分失敗'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
