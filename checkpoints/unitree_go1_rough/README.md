- ok walking gaits
- still kind of limping and weird walking gait on horizontal plane
- should reduce terrain. New one:
```python
"horizontal_rails": terrain_gen.HfHorizontalRailsTerrainCfg(
    proportion=0.2, 
    rail_height_range=(0.05, 0.05), 
    rail_thickness=0.2, 
    num_rails=3,
    horizontal_scale=0.005,

),
# other things
"random_rough": terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.2, noise_range=(0.02, 0.06), noise_step=0.01, border_width=0.25
),
```
- current parameters:
```python
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp, weight=4.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)
track_ang_vel_z_exp = RewTerm(
    func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)
# -- penalties
lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
termination = RewTerm(
    func=mdp.is_terminated,
    weight = -1.0
)
orientation = RewTerm(
    func = mdp.flat_orientation_l2,
    weight = -1.0
)

smoothness = RewTerm(
    func = mdp.action_smoothness_penalty,
    weight = -0.05
)

base_height = RewTerm(
    func=mdp.base_height_l2,
    weight=-0.001,
    params={"target_height": 29.0},
)
```
- tried `velx =  [-1.5, 1.5]`. works well.

