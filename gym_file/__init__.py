from gym.envs.registration import register

register(
    id="CrowdSimCar-v0",
    entry_point="gym_file.envs.crowd_sim_car:CrowdSimCar",
)

register(
    id="CrowdSimCar-v1",
    entry_point="gym_file.envs.crowd_sim_car_simple_obs:CrowdSimCarSimpleObs",
)


from gymnasium.envs.registration import register

register(
    id="CrowdSimCar-v0",
    entry_point="gym_file.envs.crowd_sim_car:CrowdSimCar",
)

register(
    id="CrowdSimCar-v1",
    entry_point="gym_file.envs.crowd_sim_car_simple_obs:CrowdSimCarSimpleObs",
)