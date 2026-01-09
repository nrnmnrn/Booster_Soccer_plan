import numpy as np

# class Preprocessor():

#     def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
#         q_w = q[-1]
#         q_vec = q[:3]
#         a = v * (2.0 * q_w**2 - 1.0)
#         b = np.cross(q_vec, v) * (q_w * 2.0)
#         c = q_vec * (np.dot(q_vec, v) * 2.0)  
#         return a - b + c 

#     def modify_state(self, obs, info):
        
#         quat = info["robot_quat"]
#         base_ang_vel = info["robot_gyro"]

#         project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
#         obs = np.hstack((obs,
#                          project_gravity,
#                          base_ang_vel))

#         return obs

class Preprocessor():

    def get_task_onehot(self, info):
        """從 info dict 獲取 task one-hot 編碼"""
        if 'task_index' in info:
            return info['task_index']
        else:
            # 返回 3 維 zeros 作為默認值，保證維度一致
            return np.zeros(3)

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        """將 raw obs 轉換為 87 維（符合 SAI 框架 API 契約）"""
        # 從 info 內部獲取 task_one_hot
        task_one_hot = self.get_task_onehot(info)

        robot_qpos = obs[:12]
        robot_qvel = obs[12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.hstack((robot_qpos, 
                         robot_qvel,
                         project_gravity,
                         base_ang_vel,
                         info["robot_accelerometer"],
                         info["robot_velocimeter"],
                         info["goal_team_0_rel_robot"], 
                         info["goal_team_1_rel_robot"], 
                         info["goal_team_0_rel_ball"], 
                         info["goal_team_1_rel_ball"], 
                         info["ball_xpos_rel_robot"], 
                         info["ball_velp_rel_robot"], 
                         info["ball_velr_rel_robot"], 
                         info["player_team"], 
                         info["goalkeeper_team_0_xpos_rel_robot"], 
                         info["goalkeeper_team_0_velp_rel_robot"], 
                         info["goalkeeper_team_1_xpos_rel_robot"], 
                         info["goalkeeper_team_1_velp_rel_robot"], 
                         info["target_xpos_rel_robot"], 
                         info["target_velp_rel_robot"], 
                         info["defender_xpos"],
                         task_one_hot))

        return obs