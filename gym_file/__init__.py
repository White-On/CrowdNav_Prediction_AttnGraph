from gym.envs.registration import register

register(
    id='CrowdSimCar-v0',
    entry_point='gym_file.envs.crowd_sim_car:CrowdSimCar',
)