from gym.envs.registration import register

register(
    id='merge-v0',
    entry_point='gym_merge.envs:MergeEnv',
)
#register(
#    id='foo-extrahard-v0',
#    entry_point='gym_foo.envs:FooExtraHardEnv',
#)