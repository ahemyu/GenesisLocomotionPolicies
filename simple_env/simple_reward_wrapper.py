from simple_go2_env import Go2Env
import torch 


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
        # reward tracking lin_velocity in the forward direction
        lin_vel_target = self.reward_cfg["lin_vel_target"]
        lin_vel_x = self.base_lin_vel[:, 0]
        squared_error = torch.square(lin_vel_x - lin_vel_target)

        return torch.exp(-squared_error / self.reward_cfg["tracking_sigma"]) #between [0,1]

    def _reward_forward_progress_x(self):
        # reward forward progress in x direction from the starting point
        starting_point_x = self.base_init_pos[0].item() # get the x position of the starting point  
        return self.base_pos[:,0] - starting_point_x # calculate progress from starting point
    
    def _reward_sideway_movement(self):
        # Penalize sideway movement away from the starting point
        return torch.abs(self.base_pos[:, 1] - self.base_init_pos[1])
    
    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_yaw_deviation(self):
        # Penalize yaw deviation from the target (so the robot should point straight)
        return torch.square(self.base_euler[:, 2]) 
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
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
    
    # def _reward_foot_clearance(self) -> torch.Tensor:
    #     """
    #     Reward each swing leg for clearing at least 5 cm above the local ground height.

    #     Returns
    #     -------
    #     torch.Tensor  # shape: (num_envs,)
    #         Mean per-foot reward per environment. 1.0 is full score, 0.0 means all
    #         swing feet are below the clearance threshold.
    #     """
    #     clearance_thresh = 0.05          #  desired clearance
    #     contact_thresh   = 1.0           #  ≈ zero-contact cutoff

    #     # Ground height directly under each foot
    #     hscale = self.terrain_cfg['horizontal_scale']
    #     # (num_envs, 4)  world X/Y of every foot, clamped to terrain borders
    #     px = self.foot_positions[:, :, 0].clamp_(0.0, self.terrain_margin[0])
    #     py = self.foot_positions[:, :, 1].clamp_(0.0, self.terrain_margin[1])

    #     # Convert world coords → discrete height-field indices
    #     ix = ((px / hscale) - 0.5).floor().long()
    #     iy = ((py / hscale) - 0.5).floor().long()

    #     # Guard against out-of-range indices
    #     ix.clamp_(0, self.height_field.shape[0] - 1)
    #     iy.clamp_(0, self.height_field.shape[1] - 1)

    #     ground_height = self.height_field[ix, iy]          # (num_envs, 4)

    #     # True clearance of every foot tip
    #     clearance = self.foot_positions[:, :, 2] - ground_height    # (env, foot)

    #     #  Reward only for swing legs (≈ no contact force)
    #     contact_force = torch.norm(
    #         self.link_contact_forces[:, self.feet_link_indices, :], dim=-1
    #     )                                                            # (env, foot)
    #     swing_mask = (contact_force < contact_thresh).float()        # 1 if swing

    #     per_foot_reward = torch.clamp(clearance / clearance_thresh, 0.0, 1.0)
    #     per_foot_reward = per_foot_reward * swing_mask              # ignore stance feet

    #     # Return mean across the four feet
    #     return torch.mean(per_foot_reward, dim=1)   

