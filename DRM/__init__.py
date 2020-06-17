from gym.envs.registration import register

register(
    id='DRM-v0',
    entry_point='gym_DRM.envs:DRMEnv',
)
register(
    id='DRM-extrahard-v0',
    entry_point='gym_DRM.envs:DRMExtraHardEnv',
)
