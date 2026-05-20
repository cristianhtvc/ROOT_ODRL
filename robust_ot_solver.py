import numpy as np
import torch
import torch.utils.dlpack # 必须引入
import gym
import d4rl
import argparse
import os
import random
import copy
from pathlib import Path
import yaml
import h5py
import jax
import jax.numpy as jnp
import jax.dlpack # 必须引入
from jax import device_put
import ott
import ott.geometry.pointcloud
import ott.problems.linear.linear_problem
import ott.solvers.linear.sinkhorn

import algo.utils as utils

def solve_robust_ot(
    src_embs,  # 接收 JAX Array
    tgt_embs,  # 接收 JAX Array
    cost_type='cosine',
    epsilon=0.05,
    lambda_src=5.0,
    lambda_tar=0.1
):
    # 注意：输入已经是 JAX Array，无需再做 array转换，但需要确保维度正确
    # src_embs shape: [Batch, Dim]

    cost_fn = ott.geometry.costs.Euclidean()

    scale_cost = 'max_cost'
    # 构建几何对象
    geom = ott.geometry.pointcloud.PointCloud(
        src_embs, 
        tgt_embs, 
        cost_fn=cost_fn, 
        epsilon=epsilon, 
        scale_cost=scale_cost
    )

    # 计算松弛系数
    tau_a = lambda_src / (lambda_src + epsilon)
    tau_b = lambda_tar / (lambda_tar + epsilon)

    # 定义双边非平衡问题
    prob = ott.problems.linear.linear_problem.LinearProblem(
        geom,
        tau_a=tau_a,
        tau_b=tau_b
    )

    # 求解 Sinkhorn
    solver = ott.solvers.linear.sinkhorn.Sinkhorn(
        threshold=1e-5, 
        max_iterations=1000
    )
    sinkhorn_output = solver(prob)
    
    # 获取边缘分布权重
    source_weights = sinkhorn_output.marginal(1)
    source_weights = jnp.nan_to_num(source_weights)
    
    return source_weights

def torch_to_jax(torch_tensor):
    """
    辅助函数：通过 DLPack 将 PyTorch Tensor 转换为 JAX Array (零拷贝)
    """
    # 确保 Tensor 连续，否则 DLPack 可能报错
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))

def filter_dataset(src_replay_buffer, tar_replay_buffer, args):
    print("Preparing data for OT (PyTorch GPU -> JAX GPU)...")
    
    # 1. 直接在 GPU 上进行拼接 (使用 torch.cat)
    # 假设 buffer.state 等已经在 GPU 上了
    srcdata = torch.cat([src_replay_buffer.state, src_replay_buffer.action, src_replay_buffer.next_state], dim=1)
    tardata = torch.cat([tar_replay_buffer.state, tar_replay_buffer.action, tar_replay_buffer.next_state], dim=1)
    
    src_num = srcdata.shape[0]
    
    # 2. 计算统计量 (使用 PyTorch 计算，无需转 CPU)
    # 拼接两部分数据以计算统一的统计量
    all_data = torch.cat([srcdata, tardata], dim=0)
    
    mean = torch.mean(all_data, dim=0)
    std = torch.std(all_data, dim=0) + 1e-6
    
    # 3. 标准化 (在 PyTorch GPU 上完成)
    srcdata_norm = (srcdata - mean) / std
    tardata_norm = (tardata - mean) / std
    
    # 释放显存
    del all_data, srcdata, tardata
    torch.cuda.empty_cache()

    # 4. 特征 Masking (PyTorch 切片)
    src_for_ot_torch = srcdata_norm
    tar_for_ot_torch = tardata_norm
    # 5. 准备 Target 数据 (转换一次即可)
    # 将 Target 数据转为 JAX Array (Zero-Copy)
    tar_for_ot_jax = torch_to_jax(tar_for_ot_torch)

    weights_result = []
    
    # 编译 JAX 函数
    from functools import partial
    import gc
    
    batch_solve = jax.jit(partial(
        solve_robust_ot, 
        cost_type=args["metric"],
        epsilon=args["epsilon"],
        lambda_src=args["lambda_src"],
        lambda_tar=args["lambda_tar"]
    ))
    
    # 分批处理源数据
    # 注意：JAX 倾向于预分配显存，如果显存紧张，可以减小 batch_size
    batch_size = 10000 
    iter_time = int(np.ceil(src_num / batch_size))
    
    print(f"Starting OT calculation in batches (Total batches: {iter_time})...")

    for i in range(iter_time):
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), src_num)
        
        # A. 在 PyTorch 中切片 (GPU)
        src_batch_torch = src_for_ot_torch[start_idx:end_idx]
        
        # B. 转换为 JAX (Zero-Copy)
        src_batch_jax = torch_to_jax(src_batch_torch)
        
        # C. 运行 JAX 求解器
        # block_until_ready() 确保异步计算完成，方便计时或内存管理
        weights_jax = batch_solve(src_batch_jax, tar_for_ot_jax).block_until_ready()
        
        # D. 将结果转回 CPU Numpy (用于存储或后续处理)
        # 这里必须用 device_get 转回 CPU，因为我们要把结果存成列表
        part_res = jax.device_get(weights_jax).tolist()
        weights_result.extend(part_res)
        
        # E. 显式删除当前批次的 JAX 引用，辅助 JAX 内存管理
        del src_batch_jax
        del weights_jax
        
        if (i + 1) % 10 == 0:
            print(f'Processed {end_idx} / {src_num} transitions...')

    weights_result = np.array(weights_result)
    
    print("Cleaning up JAX memory...")
    # 清理引用
    del batch_solve
    del tar_for_ot_jax
    del src_for_ot_torch
    del tar_for_ot_torch
    
    # 清理后端缓存
    jax.clear_backends()
    gc.collect()
    torch.cuda.empty_cache()
    
    return weights_result

# 其余 helper 函数 (compute_and_save_weights, ensure_ot_weights) 保持不变
# 因为它们只是调用 filter_dataset 并处理文件 IO
def compute_and_save_weights(src_replay_buffer, tar_replay_buffer, save_path, args):
    import h5py
    weights = filter_dataset(src_replay_buffer, tar_replay_buffer, args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    replay_dataset = dict(cost=weights)
    with h5py.File(save_path, 'w') as hfile:
        for k in replay_dataset:
            hfile.create_dataset(k, data=replay_dataset[k], compression='gzip')
        hfile.attrs['epsilon'] = args['epsilon']
        hfile.attrs['lambda_src'] = args['lambda_src']
        hfile.attrs['lambda_tar'] = args['lambda_tar']
    return weights

def ensure_ot_weights(env_name, src_type, tar_type, shift_level, src_replay_buffer, tar_replay_buffer, config):
    save_dir = Path("weights")
    save_dir.mkdir(exist_ok=True)
    file_name = f"{env_name}-{src_type}-{tar_type}-{shift_level}.hdf5"
    save_path = save_dir / file_name

    if save_path.exists():
        print(f"OT weights found at {save_path}, skipping calculation.")
    else: 
        print(f"OT weights not found. Calculating and saving to {file_name}...")
        robust_weights = compute_and_save_weights(
                src_replay_buffer, 
                tar_replay_buffer, 
                save_path,
                config
            )
    return save_path