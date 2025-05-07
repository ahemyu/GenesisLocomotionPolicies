from simple_go2_env import Go2Env
import torch 

# yaw is from left to right, pitch is from up to down, roll is rotating 
class WalkFlat(Go2Env):

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    # def _reward_sideway_movement(self):
    #     # Penalize sideway movement away from the starting point
    #     return torch.abs(self.base_pos[:, 1] - self.base_init_pos[1])
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
class RunOnFlatGround(Go2Env):

    def _reward_lin_vel_x(self):
        # reward lin_velocity in the forward direction
        forward_velocity = self.base_lin_vel[:, 0]
        return forward_velocity
    
    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity
        return torch.abs(self.base_lin_vel[:, 1])
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        base_height_target = self.reward_cfg['base_height_target']
        return torch.square(base_height - base_height_target)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
    
            dim=1,
        )
class WalkUneven(Go2Env):

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])


    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)


    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    #TODO: think about penalty for premature termination (only if it didn't reach the goal)
    