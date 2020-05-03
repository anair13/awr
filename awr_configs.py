AWR_CONFIGS = {
    "Ant-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.2,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "offpolicy_data_kwargs": dict(
            demo_path="/home/ashvin/data/s3doodad/demos/icml2020/mujoco/ant.npy",
        )
    },

    "HalfCheetah-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
        "offpolicy_data_kwargs": dict(
            demo_path="/home/ashvin/data/s3doodad/demos/icml2020/mujoco/half-cheetah.npy",
        )
    },

    "Hopper-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.0001,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "Humanoid-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00001,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "LunarLander-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.0005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_l2_weight": 0.001,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 100000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "Walker2d-v2":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.000025,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 1.0,
    },

    "rlbench":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 10.0,
    },

    "pen-v0":
    {
        "actor_net_layers": [128, 64],
        "actor_stepsize": 0.00005,
        "actor_momentum": 0.9,
        "actor_init_output_scale": 0.01,
        "actor_batch_size": 256,
        "actor_steps": 1000,
        "action_std": 0.4,

        "critic_net_layers": [128, 64],
        "critic_stepsize": 0.01,
        "critic_momentum": 0.9,
        "critic_batch_size": 256,
        "critic_steps": 200,

        "discount": 0.99,
        "samples_per_iter": 2048,
        "replay_buffer_size": 50000,
        "normalizer_samples": 300000,

        "weight_clip": 20,
        "td_lambda": 0.95,
        "temp": 100.0,

        "load_offpolicy_data": True,
        "offpolicy_data_sources": [
            dict(
                path="/home/ashvin/data/s3doodad/demos/icml2020/hand/pen2_sparse.npy",
                obs_dict=True,
                is_demo=True,
            ),
            dict(
                path="/home/ashvin/data/s3doodad/demos/icml2020/hand/pen_bc_sparse4.npy",
                obs_dict=False,
                is_demo=False,
                train_split=0.9,
            ),
        ],
    },
}
