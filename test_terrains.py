import genesis as gs
import numpy as np

# Initialize genesis with CPU backend
gs.init(backend=gs.cpu)

# Create a scene without a viewer, but with camera for recording
scene = gs.Scene(
    show_viewer=False,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 720),
        camera_pos=(5.0, 0.0, 3.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.3, 0.3, 0.3),
    ),
    renderer=gs.renderers.Rasterizer(),
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.81),
    ),
)

# Create a pyramid_stairs_terrain
terrain_length = 24.0
terrain_width = 8.0

# Add terrain - using only pyramid_stairs_terrain
terrain = scene.add_entity(
    gs.morphs.Terrain(
        n_subterrains=(3, 1),  # 3 sections in length, 1 in width
        subterrain_size=(terrain_length/3, terrain_width),
        horizontal_scale=0.25,  # Size of each cell in terrain
        vertical_scale=0.005,  # Height scale factor
        subterrain_types="pyramid_stairs_terrain",
    ),
)

# Place the robot at the boundary between first and second subterrain
# The terrain has 3 sections so boundaries are at terrain_length/3 and 2*terrain_length/3
boundary_x = terrain_length/3  # Position at first boundary
start_y = terrain_width / 2    # Middle of width
start_z = 0.42                 # Base height of robot above ground

# Add Go2 robot at the boundary position
go2 = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/go2/urdf/go2.urdf',
        pos=(boundary_x, start_y, start_z),
        quat=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
        fixed=False,  # Allow the robot to settle on the ground
    ),
)

# Add a camera that will orbit around
cam_orbit = scene.add_camera(
    res=(1920, 1080),
    pos=(boundary_x, start_y - 8.0, 4.0),
    lookat=(boundary_x, start_y, start_z),
    fov=45,
    GUI=False,
)

# Add a second camera that shows the terrain from the side
cam_side = scene.add_camera(
    res=(1920, 1080),
    pos=(terrain_length/2, terrain_width + 6.0, 3.0),
    lookat=(terrain_length/2, terrain_width/2, 0.5),
    fov=50,
    GUI=False,
)

# Build the scene
scene.build()

# Set initial joint positions for a standing pose
joint_names = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]
joint_indices = [go2.get_joint(name).dof_idx_local for name in joint_names]

initial_positions = [
    0.0, 0.8, -1.5,  # FR leg
    0.0, 0.8, -1.5,  # FL leg
    0.0, 1.0, -1.5,  # RR leg
    0.0, 1.0, -1.5,  # RL leg
]
go2.set_dofs_position(initial_positions, joint_indices)

# Set up basic PD controller gains so the robot can stand
kp = np.array([30.0] * 12)  # Position gains
kd = np.array([1.5] * 12)   # Velocity gains
go2.set_dofs_kp(kp, joint_indices)
go2.set_dofs_kv(kd, joint_indices)

# Start recording with both cameras
cam_orbit.start_recording()
cam_side.start_recording()

# First, let the robot settle on the ground for 100 steps
print("Letting robot settle on the ground...")
for i in range(100):
    scene.step()
    # Continuously control the robot to maintain standing pose
    go2.control_dofs_position(initial_positions, joint_indices)
    
    # Just render from side camera during settling
    cam_side.render()

# Now orbit the camera around the settled robot
print("Recording orbit around the terrain...")
total_steps = 600
robot_pos = go2.get_pos()  # Get updated position after settling

for i in range(total_steps):
    # Continue controlling the robot to maintain its position
    go2.control_dofs_position(initial_positions, joint_indices)
    
    # Simulate physics
    scene.step()
    
    # Update robot_pos occasionally to account for any drift
    if i % 60 == 0:
        robot_pos = go2.get_pos()
    
    # Orbit camera around
    orbit_angle = i * 2 * np.pi / total_steps
    
    # Vary orbit parameters
    orbit_radius = 8.0
    orbit_height = 3.0 + 1.5 * np.sin(orbit_angle)
    
    # Calculate camera position
    cam_x = robot_pos[0] + orbit_radius * np.sin(orbit_angle)
    cam_y = robot_pos[1] + orbit_radius * np.cos(orbit_angle)
    cam_z = robot_pos[2] + orbit_height
    
    # Update orbit camera
    cam_orbit.set_pose(
        pos=(cam_x, cam_y, cam_z),
        lookat=robot_pos,
    )
    
    # Render from both cameras
    cam_orbit.render()
    cam_side.render()

# Stop recording and save videos
cam_orbit.stop_recording(save_to_filename='pyramid_stairs_orbit_view.mp4', fps=60)
cam_side.stop_recording(save_to_filename='pyramid_stairs_side_view.mp4', fps=60)

print("Terrain visualization complete. Videos saved.")