import torch
from locomotion_env import *

class Go2(LocoEnv):
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, :2] - self.base_lin_vel[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

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

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

# These are the different tasks that the go2 robot learned to perform 
class Backflip(Go2):
    """Perform a backflip on flat ground."""

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset root states - position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, 2] = 0.32
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_base_quat
        )

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        base_vel = torch.concat(
            [self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1
        )
        self.robot.set_dofs_velocity(
            velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx
        )

        self._resample_commands(envs_idx)

        # reset buffers
        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        # fill extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        # send timeout info to the algorithm
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf

    def compute_observations(self):

        phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 10
                self.dof_vel * self.obs_scales['dof_vel'],                          # 10
                self.actions,                                                       # 10
                self.last_actions,                                                  # 10
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )

        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs:], self.obs_buf.detach()], dim=1
        )

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.base_pos[:, 2:3],                                              # 1
                    self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                    self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                    self.projected_gravity,                                             # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 10
                    self.dof_vel * self.obs_scales['dof_vel'],                          # 10
                    self.actions,                                                       # 10
                    self.last_actions,                                                  # 10
                    torch.sin(phase),
                    torch.cos(phase),
                    torch.sin(phase / 2),
                    torch.cos(phase / 2),
                    torch.sin(phase / 4),
                    torch.cos(phase / 4),
                ],
                axis=-1,
            )

    def check_termination(self):
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = gs_quat_from_angle_axis(4 * phase * torch.pi,
                                             torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = gs_quat_mul(quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)

        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff
    
    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2) #negative because backflip
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, 2].clamp(max=3)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_height_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        target_height = 0.3
        height_diff = torch.square(target_height - self.base_pos[:, 2]) * torch.logical_or(current_time < 0.4, current_time > 1.4)
        return height_diff

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff
    
    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        current_time = self.episode_length_buf * self.dt
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply(gs_quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.zeros([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        
        return stance_diff

    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        current_time = self.episode_length_buf * self.dt
        return (1.0 * (torch.norm(self.link_contact_forces[:, self.penalized_contact_link_indices, :], dim=-1) > 0.1)).sum(dim=1)

class FrontFlip(Go2):
    """Perform a frontflip on flat ground."""

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, 2] = 0.32
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_base_quat
        )

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        base_vel = torch.concat(
            [self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1
        )
        self.robot.set_dofs_velocity(
            velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx
        )

        self._resample_commands(envs_idx)

        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf

    def compute_observations(self):
        phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
                self.last_actions,
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )

        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs:], self.obs_buf.detach()], dim=1
        )

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.base_pos[:, 2:3],                                              # 1
                    self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                    self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                    self.projected_gravity,                                             # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 10
                    self.dof_vel * self.obs_scales['dof_vel'],                          # 10
                    self.actions,                                                       # 10
                    self.last_actions,                                                  # 10
                    torch.sin(phase),
                    torch.cos(phase),
                    torch.sin(phase / 2),
                    torch.cos(phase / 2),
                    torch.sin(phase / 4),
                    torch.cos(phase / 4),
                ],
                axis=-1,
            )

    def check_termination(self):
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )

    def _reward_orientation_control(self):
        """Penalizes deviation from desired orientation during and after the frontflip.
        
        Encourages the robot to follow a pitch rotation profile (0 to -360 degrees between 0.5 and 1.0 seconds)
        and return to the initial upright orientation, ensuring a controlled flip and stable landing preparation."""
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = gs_quat_from_angle_axis(4 * phase * torch.pi,
                                             torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = gs_quat_mul(quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)

        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff

    def _reward_ang_vel_y(self):
        """Rewards forward angular velocity around the y-axis during the flip (0.5 to 1.0 seconds).
        
        Promotes the rotational motion necessary for the frontflip by rewarding positive y-axis angular velocity,
        capped to prevent excessive spinning."""
        current_time = self.episode_length_buf * self.dt
        ang_vel = self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2) #positive because frontflip
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        """Penalizes angular velocity around the z-axis throughout the episode.
        
        Discourages unwanted yaw rotation to keep the frontflip aligned and prevent twisting, aiding a straight landing."""
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        """Rewards upward linear velocity during the takeoff phase (0.5 to 0.75 seconds).
        
        Encourages the robot to gain sufficient height during the initial jump to complete the flip successfully."""
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, 2].clamp(max=3)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_height_control(self):
        """Penalizes base height deviation from target (0.3m) before takeoff and after landing (before 0.4 and after 1.0 seconds).
        
        Maintains a stable standing height before the flip and ensures the robot lands upright without collapsing."""

        current_time = self.episode_length_buf * self.dt
        target_height = 0.3
        height_diff = torch.square(target_height - self.base_pos[:, 2]) * torch.logical_or(current_time < 0.4, current_time > 1.4)
        return height_diff

    def _reward_actions_symmetry(self):
        """Penalizes asymmetry in actions between symmetric joints (e.g., left vs. right legs).
        
        Encourages coordinated and symmetric leg movements to maintain balance and consistency during the flip and landing."""
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff

    def _reward_gravity_y(self):
        """Penalizes roll (gravity deviation in the y-direction).
        
        Prevents the robot from tilting sideways, ensuring a straight frontflip and stable landing orientation."""
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        """Penalizes deviation of feet y-positions from desired stance width (0.3m).
        
        Maintains a stable and wide stance before and after the flip, aiding balance and proper leg positioning."""
        current_time = self.episode_length_buf * self.dt
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply(gs_quat_conjugate(self.base_quat),
                                                             cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.zeros([self.num_envs, 1], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)

        return stance_diff

    def _reward_feet_height_before_frontflip(self):
        """Penalizes feet lifting off the ground before the flip (before 0.5 seconds).
        
        Ensures the robot remains grounded and stable during the preparation phase before initiating the frontflip."""
        current_time = self.episode_length_buf * self.dt
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)

    def _reward_collision(self):
        """Penalizes collisions of non-feet links with the ground throughout the episode.
        
        Prevents the robot from crashing or landing on its base/body, encouraging a clean flip and feet-first landing."""
        current_time = self.episode_length_buf * self.dt
        return (1.0 * (torch.norm(self.link_contact_forces[:, self.penalized_contact_link_indices, :], dim=-1) > 0.1)).sum(dim=1) 

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class WalkUneven(Go2):
    """
    Environment for training the Go2 robot to walk on uneven pyramid stairs terrain.
    Focuses on teaching the robot to elegantly navigate both ascending and descending stairs.
    """
    
    def reset_idx(self, envs_idx):
        """
        Resets the environments with the given indices.
        Places the robot at the start of the terrain facing the direction of the stairs.
        """
        logger.info(f"Resetting environments for indices: {envs_idx}")
        if len(envs_idx) == 0:
            logger.info("No environments to reset")
            return

        # reset dofs to default positions with slight randomization
        self.dof_pos[envs_idx] = self.default_dof_pos + gs_rand_float(-0.2, 0.2, (len(envs_idx), self.num_dof), self.device)
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        logger.info("Setting initial positions and orientations")
        # reset root states - position
        # Place at beginning of terrain with some randomization in y-direction
        terrain_start_x = 2.0  # Start slightly into the terrain
        self.base_pos[envs_idx] = self.base_init_pos.clone()
        self.base_pos[envs_idx, 0] = terrain_start_x
        self.base_pos[envs_idx, 1] += gs_rand_float(-1.0, 1.0, (len(envs_idx), 1), self.device).squeeze(1)
        
        # Face directly along x-axis (direction of the terrain)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        
        # Apply slight random rotation for robustness
        base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)
        base_euler[:, 2] = gs_rand_float(-0.2, 0.2, (len(envs_idx),), self.device) # yaw variation
        self.base_quat[envs_idx] = gs_quat_mul(
            gs_euler2quat(base_euler),
            self.base_quat[envs_idx],
        )
        
        logger.info("Updating robot state")
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_base_quat
        )

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        base_vel = torch.concat(
            [self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1
        )
        self.robot.set_dofs_velocity(
            velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx
        )

        logger.info("Resetting commands and buffers")
        # Set commands - primarily forward movement with slight turning ability
        self._resample_commands(envs_idx)
        
        # Constrain the commands to focus on forward motion
        self.commands[envs_idx, 0] = torch.clamp(self.commands[envs_idx, 0], 0.5, 1.0)  # Forward velocity
        self.commands[envs_idx, 1] *= 0.3 # Reduce lateral velocity
        self.commands[envs_idx, 2] *= 0.3 # Reduce turning

        # reset buffers
        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1
        self.progress_buf[envs_idx] = 0  # Track progress along terrain

        # fill extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        
        # send timeout info to the algorithm
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf

        logger.info("Reset completed")

    def compute_observations(self):
        """
        Compute observations including terrain height information to help the robot
        understand the terrain ahead.
        """
        logger.info("Computing observations")
        # Get terrain height information ahead of the robot
        terrain_heights_ahead = self._get_terrain_heights_ahead()
        
        # Base observation components similar to Go2
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                self.commands[:, :3] * self.commands_scale,                         # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
                self.dof_vel * self.obs_scales['dof_vel'],                          # 12
                self.actions,                                                       # 12
                terrain_heights_ahead,                                             # terrain heights ahead
            ],
            dim=-1,
        )

        # add noise in non-eval mode
        if not self.eval:
            self.obs_buf += gs_rand_float(
                -1.0, 1.0, (self.obs_buf.shape[1],), self.device
            ) * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs:], self.obs_buf.detach()], dim=1
        )

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                    self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                    self.projected_gravity,                                             # 3
                    self.commands[:, :3] * self.commands_scale,                         # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
                    self.dof_vel * self.obs_scales['dof_vel'],                          # 12
                    self.actions,                                                       # 12
                    self.last_actions,                                                  # 12
                    terrain_heights_ahead,                                             # terrain heights ahead
                    self.base_pos,                                                     # global position
                ],
                dim=-1,
            )
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        logger.info("Observations computed")

    def _get_terrain_heights_ahead(self):
        """
        Sample terrain heights in front of the robot to help it anticipate the terrain.
        """
        logger.info("Getting terrain heights ahead")
        # Number of samples ahead to look
        n_samples = 5
        
        # Initialize tensor for heights
        terrain_heights = torch.zeros((self.num_envs, n_samples), device=self.device, dtype=gs.tc_float)
        
        # Get robot's current position and orientation
        robot_pos = self.base_pos
        robot_quat = self.base_quat
        
        # Get forward direction in world frame
        forward_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        forward_vec[:, 0] = 1.0  # Forward is along x-axis in robot frame
        forward_world = gs_quat_apply(robot_quat, forward_vec)
        
        # Sample points ahead
        for i in range(n_samples):
            # Distance ahead (increasing as we look further)
            distance = 0.5 + i * 0.5  # 0.5m, 1.0m, 1.5m, ...
            
            # Calculate sample position
            sample_pos = robot_pos + forward_world * distance
            
            # Get terrain height at this position
            clipped_pos = sample_pos[:, :2].clamp(min=torch.zeros(2, device=self.device), max=self.terrain_margin)
            height_field_ids = (clipped_pos / self.terrain_cfg['horizontal_scale'] - 0.5).floor().int()
            height_field_ids.clamp(min=0)
            terrain_heights[:, i] = self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]]
        
        logger.info("Terrain heights calculated")
        return terrain_heights

    def check_termination(self):
        """
        Check if episodes should terminate.
        Additional termination conditions for uneven terrain navigation.
        """
        logger.info("Checking termination conditions")
        # Basic termination conditions from Go2
        self.reset_buf = torch.any(
            torch.norm(
                self.link_contact_forces[:, self.termination_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        
        # Terminate if the robot tips over too much
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg['termination_if_pitch_greater_than'],
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg['termination_if_roll_greater_than'],
        )
        
        # Terminate if robot falls off the terrain
        if self.env_cfg['use_terrain']:
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] > self.terrain_margin[0],
                self.base_pos[:, 1] > self.terrain_margin[1],
            )
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] < 1,
                self.base_pos[:, 1] < 1,
            )
            
        # Terminate if robot gets too low (e.g., fell through terrain or fell down)
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg['termination_if_height_lower_than']
        
        # Success termination - reached the end of terrain
        self.success_buf = (self.base_pos[:, 0] > self.terrain_margin[0] - 3.0)
        
        # Apply termination
        self.reset_buf |= self.time_out_buf
        
        logger.info("Termination check completed")

    def post_physics_step(self):
        """
        Update buffers and track progress after each physics step.
        """
        logger.info("Running post physics step")
        # Update basic buffers from parent class
        super().post_physics_step()
        
        # Update progress tracking
        self.progress_buf = self.base_pos[:, 0] / self.terrain_margin[0]
        
        logger.info("Post physics step completed")

    # REWARD FUNCTIONS
    def _reward_forward_progress(self):
        """
        Reward forward progress along the x-axis (direction of the terrain).
        """
        logger.info("Calculating forward progress reward")
        reward = self.base_lin_vel[:, 0]
        logger.info("Forward progress reward calculated")
        return reward
    
    def _reward_terrain_adaptation(self):
        """
        Reward the robot for adapting its gait to the terrain height.
        Measured by how well the base height follows the terrain height.
        """
        logger.info("Calculating terrain adaptation reward")
        local_terrain_height = self.terrain_heights
        target_height_offset = 0.4  # Desired height above terrain
        
        # Calculate how well the robot maintains proper height above terrain
        height_adaptation = torch.square(
            (self.base_pos[:, 2] - local_terrain_height) - target_height_offset
        )
        
        logger.info("Terrain adaptation reward calculated")
        return height_adaptation
    
    def _reward_balance_on_slope(self):
        """
        Reward maintaining proper orientation relative to the terrain slope.
        """
        logger.info("Calculating balance on slope reward")
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        logger.info("Balance on slope reward calculated")
        return reward
    
    def _reward_foot_clearance(self):
        """
        Reward lifting feet high enough to clear stairs during swing phase,
        but not excessively high (energy efficiency).
        """
        logger.info("Calculating foot clearance reward")
        # Calculate foot heights during swing phase
        foot_heights = 0.0
        foot_contacts = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        
        # For feet not in contact (swing phase)
        swing_mask = ~foot_contacts
        if swing_mask.any():
            # Relative height of feet to terrain
            rel_foot_heights = self.foot_positions[:, :, 2] - self.terrain_heights.unsqueeze(1)
            
            # Target height range for swing phase
            min_clearance = 0.05
            max_clearance = 0.2
            
            # Penalize if too low (might hit stairs) or too high (inefficient)
            too_low = torch.relu(min_clearance - rel_foot_heights) * swing_mask
            too_high = torch.relu(rel_foot_heights - max_clearance) * swing_mask
            
            foot_heights = torch.sum(too_low, dim=1) + torch.sum(too_high, dim=1)
        
        logger.info("Foot clearance reward calculated")
        return foot_heights
    
    def _reward_gait_stability(self):
        """
        Reward stable gait patterns especially when on stairs.
        Measured by consistency of foot contact timing and base velocity.
        """
        logger.info("Calculating gait stability reward")
        # Penalize velocity fluctuations
        vel_consistency = torch.sum(torch.square(self.base_lin_vel - self.last_root_vel), dim=1)
        
        logger.info("Gait stability reward calculated")
        return vel_consistency
    
    def _reward_energy_efficiency(self):
        """
        Reward energy-efficient locomotion.
        Measured by minimizing joint torques and jerky movements.
        """
        logger.info("Calculating energy efficiency reward")
        # Sum of squared torques (power usage)
        torque_penalty = torch.sum(torch.square(self.torques), dim=1)
        
        # Penalize jerky acceleration changes
        jerk_penalty = torch.sum(
            torch.square((self.dof_vel - 2*self.last_dof_vel)), dim=1
        )
        
        logger.info("Energy efficiency reward calculated")
        return torque_penalty + 0.1 * jerk_penalty
    
    def _reward_foot_slip(self):
        """
        Penalize foot slippage, which is especially important on stairs.
        Measured by horizontal movement of feet during stance phase.
        """
        logger.info("Calculating foot slip reward")
        # Identify feet in contact with ground (stance phase)
        foot_contacts = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        
        # Calculate foot slip for feet in contact
        foot_slip = 0.0
        if foot_contacts.any():
            # Get horizontal velocity of feet
            foot_vel_horizontal = torch.norm(self.foot_velocities[:, :, :2], dim=2)
            
            # Penalize horizontal velocity for feet in contact
            foot_slip = torch.sum(foot_vel_horizontal * foot_contacts, dim=1)
        
        logger.info("Foot slip reward calculated")
        return foot_slip