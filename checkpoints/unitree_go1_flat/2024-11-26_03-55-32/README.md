- good walking gaits
- good result received at iter 4000
- will add rougher terrains
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
- next attempt will try to change `velx =  [-1.5, 1.5]`
