import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf

import awr_configs
import learning.awr_agent as awr_agent

arg_parser = None

from multiworld.envs.rlbench.rlbench_env import RLBenchEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
from rlbench.tasks.open_drawer import OpenDrawer

from multiworld.core.image_env import ImageEnv
from railrl.launchers.experiments.ashvin.rfeatures.encoder_wrapped_env import EncoderWrappedEnv
from railrl.launchers.experiments.ashvin.rfeatures.rfeatures_model import TimestepPredictionModel
import torch
import railrl.torch.pytorch_util as ptu

def parse_args(args):
    parser = argparse.ArgumentParser(description="Train or test control policies.")

    parser.add_argument("--env", dest="env", default="")

    parser.add_argument("--train", dest="train", action="store_true", default=True)
    parser.add_argument("--test", dest="train", action="store_false", default=True)

    parser.add_argument("--max_iter", dest="max_iter", type=int, default=np.inf)
    parser.add_argument("--test_episodes", dest="test_episodes", type=int, default=32)
    parser.add_argument("--output_dir", dest="output_dir", default="output")
    parser.add_argument("--output_iters", dest="output_iters", type=int, default=50)
    parser.add_argument("--model_file", dest="model_file", default="")

    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--gpu", dest="gpu", default="")

    arg_parser = parser.parse_args()

    return arg_parser

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

def build_env(env_id):
    ptu.set_gpu_mode(True)

    env = RLBenchEnv(
        task_class=OpenDrawer,
        fixed_goal=(),
        headless=False,
        camera=(500, 300),
        state_observation_type="task",
        stub=False,
    )

    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=450000,
        reward_type="image_distance",
        # init_camera=sawyer_pusher_camera_upright_v2,
    )

    variant = dict(
        model_path="/home/ashvin/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id2/itr_4000.pt",
        desired_trajectory="/home/ashvin/code/railrl-private/gitignore/rlbench/demo_door_fixed2/demos5b_10_dict.npy",
        model_kwargs=dict(
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            imsize=224,
            architecture=dict(
                hidden_sizes=[200, 200],
            ),
            delta_features=True,
            pretrained_features=False,
        ),
        reward_params_type="regression_distance",
    )
    model_class = variant.get('model_class', TimestepPredictionModel)
    representation_size = 128
    output_classes = 20
    model = model_class(
        representation_size,
        # decoder_output_activation=decoder_activation,
        output_classes=output_classes,
        **variant['model_kwargs'],
    )
    # model = torch.nn.DataParallel(model)

    model_path = variant.get("model_path")
    # model = load_local_or_remote_file(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(ptu.device)
    model.eval()

    traj = np.load(variant.get("desired_trajectory"), allow_pickle=True)[0]

    goal_image = traj["observations"][-1]["image_observation"]
    goal_image = goal_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # goal_image = goal_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0 # BECAUSE RLBENCH DEMOS ARENT IMAGE_ENV WRAPPED
    # goal_image = goal_image[:, :, :240, 60:500]
    goal_image = goal_image[:, :, 60:, 60:500]
    goal_image_pt = ptu.from_numpy(goal_image)
    # save_image(goal_image_pt.data.cpu(), 'demos/goal.png', nrow=1)
    goal_latent = model.encode(goal_image_pt).detach().cpu().numpy().flatten()

    initial_image = traj["observations"][0]["image_observation"]
    initial_image = initial_image.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
    # initial_image = initial_image.reshape(1, 300, 500, 3).transpose([0, 3, 1, 2]) / 255.0
    # initial_image = initial_image[:, :, :240, 60:500]
    initial_image = initial_image[:, :, 60:, 60:500]
    initial_image_pt = ptu.from_numpy(initial_image)
    # save_image(initial_image_pt.data.cpu(), 'demos/initial.png', nrow=1)
    initial_latent = model.encode(initial_image_pt).detach().cpu().numpy().flatten()

    # Move these to td3_bc and bc_v3 (or at least type for reward_params)
    reward_params = dict(
        goal_latent=goal_latent,
        initial_latent=initial_latent,
        type=variant["reward_params_type"],
    )

    config_params = variant.get("config_params")
    env = EncoderWrappedEnv(
        env,
        model,
        reward_params,
        config_params,
        **variant.get("encoder_wrapped_env_kwargs", dict())
    )
    env = FlatGoalEnv(env, obs_keys=["state_observation", ])
    return env

def build_agent(env):
    env_id = "rlbench"
    agent_configs = {}
    if (env_id in awr_configs.AWR_CONFIGS):
        agent_configs = awr_configs.AWR_CONFIGS[env_id]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

    return agent

def main(args):
    global arg_parser
    arg_parser = parse_args(args)
    enable_gpus(arg_parser.gpu)

    env = build_env(arg_parser.env)

    agent = build_agent(env)
    agent.visualize = arg_parser.visualize
    if (arg_parser.model_file is not ""):
        agent.load_model(arg_parser.model_file)

    if (arg_parser.train):
        agent.train(max_iter=arg_parser.max_iter,
                    test_episodes=arg_parser.test_episodes,
                    output_dir=arg_parser.output_dir,
                    output_iters=arg_parser.output_iters)
    else:
        agent.eval(num_episodes=arg_parser.test_episodes)

    return

if __name__ == "__main__":
    main(sys.argv)
