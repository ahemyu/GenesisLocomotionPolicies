from simple_go2_env import Go2Env
import torch 

class RunOnFlatGround(Go2Env):
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(
    #         torch.square(
    #             self.commands[:, :2] - self.base_lin_vel[:, :2]
    #         ),
    #         dim=1,
    #     )
    #     return torch.exp(-lin_vel_error / self.reward_cfg['tracking_sigma'])

    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(
    #         self.commands[:, 2] - self.base_ang_vel[:, 2]
    #     )
    #     return torch.exp(-ang_vel_error / self.reward_cfg['tracking_sigma'])

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
    def _reward_absolute_lin_vel(self):
        # Reward absolute linear velocity (encourages speed in any direction)
        # We take the norm of the horizontal velocity (x and y components)
        absolute_velocity = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return absolute_velocity


class WalkUneven(Go2Env):
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_orientation(self): # 
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

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
    #####TODO: include reward that encourages the robot to walk straight
    ####TODO: foot slip penalty 
    #TODO: include reward related to current height of the robot and the height of the terrain in front of it, maybe smth with the feet, e.g. take bigger steps if terrain is high in front
    #TODO: add feet heih