from simple_go2_env import Go2Env
import torch 

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
    
    def _reward_foot_clearance(self):
        """
        Give positive reward when a swing‑foot rises above the clearance
        target during the swing phase.
        """
        # Contact mask: 1 = foot in contact, 0 = swing
        contact_mask = (
            torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1) > 0.10
        ).float()                                    # shape (E, 4)

        swing_mask = 1.0 - contact_mask             # 1 when in swing

        foot_height = self.foot_positions[:, :, 2]   # world‑frame z in metres
        target = self.reward_cfg.get("feet_height_target", 0.06)
        # Positive clearance only when higher than target
        clearance = torch.clamp(foot_height - target, min=0.0)

        # Mean over the four feet, scaled by swing_mask
        return torch.mean(clearance * swing_mask, dim=1)
    
    def _reward_foot_phase_symmetry(self):
        """
        Encourage diagonal pairs (FL–RR, FR–RL) to share contact phase,
        which produces a clean trot duty cycle.
        """
        contact = (
            torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1) > 0.10
        ).float()

        # Indices match links_to_keep order ➜ [FL, FR, RL, RR]
        FL, FR, RL, RR = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]

        # 1 when phases match, 0 when opposite
        diag_sym_1 = 1.0 - torch.abs(FL - RR)
        diag_sym_2 = 1.0 - torch.abs(FR - RL)

        return 0.5 * (diag_sym_1 + diag_sym_2)

class WalkUneven(Go2Env):

    # def _reward_lin_vel_x(self):
    #     # reward lin_velocity in the forward direction
    #     forward_velocity = self.base_lin_vel[:, 0]
    #     return forward_velocity

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
    
    # TODO: If it ain't working try adding a penalty for deviation from y starting point
    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

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