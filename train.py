import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils
import sys

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SAC", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    
    args = parser.parse_args()

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments

    if domain == 'antmaze':
        src_env_name = src_env_name.split('-')[0]
        src_env_name += '-' + args.srctype + '-v0'
    elif domain == 'adroit':
        src_env_name = src_env_name.split('-')[0]
        src_env_name += '-' + args.srctype + '-v0'
    else:
        src_env_name += '-' + args.srctype + '-v2'
    src_env = None
    src_eval_env = gym.make(src_env_name)
    src_eval_env.seed(args.seed)
    
    tar_env = None
    tar_env_config = {
        'env_name': tar_env_name,
        'shift_level': args.shift_level,
    }
    tar_eval_env = call_env[domain](tar_env_config)
    tar_eval_env.seed(args.seed + 100)
    
    policy_config_name = args.policy.lower()

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")


    # log path, we use logging with tensorboard
    outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    os.makedirs(outdir, exist_ok=True)
    done_file_path = os.path.join(outdir, 'DONE')
    if os.path.exists(done_file_path):
        print(f"[{time.ctime()}] Experiment already finished (found DONE file). Skipping: {outdir}")
        sys.exit(0) # 直接退出，Shell 脚本会继续执行下一条命令
    
    # --- [新增代码] 准备断点续训变量 ---
    state_file_path = os.path.join(outdir, 'training_state.json')
    start_step = 0
    resume_mode = False

    if os.path.exists(state_file_path) and args.save_model:
        try:
            with open(state_file_path, 'r') as f:
                training_state = json.load(f)
                start_step = training_state['step']
                resume_mode = True
            print(f"[{time.ctime()}] Found checkpoint. Resuming from step {start_step}...")
        except Exception as e:
            print(f"Error loading checkpoint state: {e}. Starting from scratch.")

    log_dir = '{}/tb'.format(outdir)
    writer = SummaryWriter(log_dir=log_dir, purge_step=start_step if resume_mode else None)
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    src_env.action_space.seed(args.seed) if src_env is not None else None
    tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
    })

    policy = call_algo(args.policy, config, device)
    
    if resume_mode:
        model_path = '{}/models/model'.format(outdir)
        try:
            policy.load(model_path) 
            # [新增] 非常关键：同步算法内部的计数器，否则学习率衰减和日志记录频率会错乱
            if hasattr(policy, 'total_it'):
                policy.total_it = start_step
            print(f"Successfully loaded model weights. Resuming from step {start_step}...")
        except Exception as e:
            print(f"Warning: Could not load model weights ({e}). Training from scratch.")
            start_step = 0 
            resume_mode = False # 加载失败则重置为从头开始

    ## write logs to record training parameters
    with open(os.path.join(outdir, 'log.txt'), 'w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    policy_name = args.policy.upper()
    if policy_name in ["OTDF", "ROOT"]:
        src_replay_buffer = utils.OTReplayBuffer(state_dim, action_dim, device)
        tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    else:
        src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
        tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # in case that the domain is offline, we directly load its offline data
    src_dataset= d4rl.qlearning_dataset(src_eval_env)
    src_replay_buffer.convert_D4RL(src_dataset)
    if 'antmaze' in args.env:
        src_replay_buffer.reward -= 1.0
    
    tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
    tar_replay_buffer.convert_D4RL(tar_dataset)
    if 'antmaze' in args.env:
        tar_replay_buffer.reward -= 1.0

    # offline-offline learning
    if policy_name== "ROOT":
        print(">>> Initializing OT Logic...")
        from robust_ot_solver import ensure_ot_weights
        # 1. 确保权重文件存在 (自动调用 run_ot 逻辑)
        # 注意：这里需要根据 args.env, args.shift 等拼凑出 run_ot 需要的参数
        weight_path = ensure_ot_weights(
            env_name=tar_env_name, 
            src_type=args.srctype,
            tar_type=args.tartype,
            shift_level= shift_level,
            src_replay_buffer= src_replay_buffer,
            tar_replay_buffer= tar_replay_buffer,
            config=config
        )
        # 2. 加载权重
        ot_weights = utils.load_robust_weights(weight_path)
        global_mean = ot_weights.mean() + 1e-12
        normalized_weights = ot_weights / global_mean
        src_replay_buffer.set_weights(normalized_weights)
        print(f">>> OT Weights injected. Buffer size: {src_replay_buffer.size}")

    elif policy_name== "OTDF":
        print(">>> Initializing OT Logic...")
        ot_cost = utils.call_otdf_cost(args.env, args.srctype, args.tartype)

        if isinstance(ot_cost, np.ndarray):
            ot_cost = torch.from_numpy(ot_cost).float().to(device)
        elif isinstance(ot_cost, torch.Tensor):
            ot_cost = ot_cost.float().to(device)
        else:
            raise TypeError("Unsupported OT cost type.")

        src_replay_buffer.cost[:src_replay_buffer.size] = ot_cost.reshape(-1, 1)
        print(f">>> OTDF cost injected. Buffer size: {src_replay_buffer.size}")
    
    eval_cnt = start_step // config['eval_freq']

    if policy_name in ["OTDF", "ROOT"] and not config.get('noreg', False):
        if not resume_mode:
            print(">>> Starting VAE pretraining for 10000 steps...")
            policy.train_vae(tar_replay_buffer, config['batch_size'], writer)
        else:
            print(">>> Skipping VAE pretraining (Resuming mode).")

    start_time = time.time()
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1
    # [修改] 训练主循环：从 start_step 开始，而不是永远从 0 开始
    for t in range(start_step, int(config['max_step'])):
        policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

        if (t + 1) % config['eval_freq'] == 0:
            # save_dir = f"{outdir}/checkpoints/"
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # policy.save(f'{save_dir}/checkpoint_{t+1}')

            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\nTotal training time: {total_time:.2f} seconds, current epoch is {(t + 1) // config['eval_freq']}")
            print("==========================================================")
            
            src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar('test/source return', src_eval_return, global_step=t+1)
            writer.add_scalar('test/target return', tar_eval_return, global_step=t+1)
            eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
            writer.add_scalar('test/target normalized score', eval_normalized_score, global_step=t+1)
            
            eval_cnt += 1
            start_time = time.time() # 重置计时器
            
            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
                # [新增] 每次保存模型时，将当前步数写入 training_state.json，用于后续断点续训
                with open(state_file_path, 'w') as f:
                    json.dump({'step': t + 1}, f)

    writer.close()
    
    # [新增] 训练完全结束后，生成 DONE 文件，防止被重复运行
    with open(done_file_path, 'w') as f:
        f.write(f"Finished at {time.ctime()}")
    print(f"Experiment completed. DONE file created at {done_file_path}")