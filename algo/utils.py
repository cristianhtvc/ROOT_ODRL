import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional
import h5py
import os
from torch.nn.modules.dropout import Dropout
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # [优化] 直接在 GPU 上初始化 Tensor
        self.state = torch.zeros((max_size, state_dim), device=device)
        self.action = torch.zeros((max_size, action_dim), device=device)
        self.next_state = torch.zeros((max_size, state_dim), device=device)
        self.reward = torch.zeros((max_size, 1), device=device)
        self.not_done = torch.zeros((max_size, 1), device=device)

    def add(self, state, action, next_state, reward, done):
        # 如果输入是 numpy，先转为 tensor 并移动到 GPU
        # 注意：Online 交互时通常输入是 numpy
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            reward = torch.tensor([reward], device=self.device) # 标量转 tensor
            done = torch.tensor([done], device=self.device)     # 标量转 tensor
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # [优化] 使用 torch.randint 在 GPU 上直接生成索引
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)

        # [优化] 直接在 GPU 上切片，无需 .to(device)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )
    
    def convert_D4RL(self, dataset):
        # D4RL 数据集通常是 numpy，加载时一次性搬运到 GPU
        print("Moving D4RL dataset to GPU memory...")
        self.state = torch.from_numpy(dataset['observations']).float().to(self.device)
        self.action = torch.from_numpy(dataset['actions']).float().to(self.device)
        self.next_state = torch.from_numpy(dataset['next_observations']).float().to(self.device)
        self.reward = torch.from_numpy(dataset['rewards'].reshape(-1,1)).float().to(self.device)
        self.not_done = 1. - torch.from_numpy(dataset['terminals'].reshape(-1,1)).float().to(self.device)
        self.size = self.state.shape[0]
        # 重置 ptr，防止后续 add 覆盖数据 (如果是纯 offline 这步无所谓，但为了鲁棒性)
        self.ptr = self.size % self.max_size
        print(f"Dataset moved to GPU. Size: {self.size}")

class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affines.append(nn.Linear(hidden_dim, out_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)
            self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
                    # x = self.norm_layer(x)
        return x

def identity(x):
    return x

def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def orthogonal_init(tensor, gain=0.01):
    torch.nn.init.orthogonal_(tensor, gain=gain)

class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            final_init_scale=None,
            dropout_rate=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]
        
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            # add dropout
            if self.dropout_rate:
                h = self.dropout(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output
    
    def sample(self, *inputs):
        preds = self.forward(*inputs)

        sample_idxs = np.random.choice(self.ensemble_size, 2, replace=False)
        preds_sample = preds[sample_idxs]
        
        return torch.min(preds_sample, dim=0)[0], sample_idxs
    

class OTReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # [优化] GPU Tensor 初始化
        self.state = torch.zeros((max_size, state_dim), device=device)
        self.action = torch.zeros((max_size, action_dim), device=device)
        self.next_state = torch.zeros((max_size, state_dim), device=device)
        self.reward = torch.zeros((max_size, 1), device=device)
        self.not_done = torch.zeros((max_size, 1), device=device)
        self.cost = torch.zeros((max_size, 1), device=device)

    def add(self, state, action, next_state, reward, done):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            reward = torch.tensor([reward], device=self.device)
            done = torch.tensor([done], device=self.device)

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # [优化] GPU 采样
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind],
            self.cost[ind]
        )
    
    def convert_D4RL(self, dataset):
        print("Moving D4RL dataset to GPU memory...")
        self.state = torch.from_numpy(dataset['observations']).float().to(self.device)
        self.action = torch.from_numpy(dataset['actions']).float().to(self.device)
        self.next_state = torch.from_numpy(dataset['next_observations']).float().to(self.device)
        self.reward = torch.from_numpy(dataset['rewards'].reshape(-1,1)).float().to(self.device)
        self.not_done = 1. - torch.from_numpy(dataset['terminals'].reshape(-1,1)).float().to(self.device)
        self.size = self.state.shape[0]
        self.ptr = self.size % self.max_size
        print(f"Dataset moved to GPU. Size: {self.size}")
    
    def set_weights(self, weights):
        """
        weights: numpy array or torch tensor
        """
        assert weights.shape[0] == self.size
        print(f"Injecting weights into ReplayBuffer. Shape: {weights.shape}")
        
        # 确保 weights 是 tensor 并在 GPU 上
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float().to(self.device)
        elif isinstance(weights, torch.Tensor):
            weights = weights.float().to(self.device)
            
        self.cost[:self.size] = weights.reshape(-1, 1)

    def preprocess(self, filter_num):
        pass


def load_robust_weights(weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    print(f"Loading robust weights from: {weight_path}")
    
    data_dict = {}
    with h5py.File(weight_path, 'r') as dataset_file:
        for k in dataset_file.keys():
            try:
                data_dict[k] = dataset_file[k][:]
            except ValueError:
                data_dict[k] = dataset_file[k][()]
    
    # 兼容你的生成脚本，key 可能是 'cost' 或 'weights'
    if 'cost' in data_dict:
        return data_dict['cost']
    elif 'weights' in data_dict:
        return data_dict['weights']
    else:
        # 如果 key 不确定，返回第一个数据集
        first_key = list(data_dict.keys())[0]
        print(f"Warning: 'cost' key not found. Using first key: {first_key}")
        return data_dict[first_key]

from pathlib import Path
from tqdm import tqdm

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def call_otdf_cost(env_name, src_datatype, tar_datatype):
    if '_' in env_name:
        env_name = env_name.replace('_', '-')
    
    cost_path = str(Path(__file__).parent.parent.absolute()) + '/otdf_costlogs/' + env_name + '-srcdatatype-' + str(src_datatype) + '-tardatatype-' + str(tar_datatype) + '.hdf5'

    data_dict = {}
    with h5py.File(cost_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
        
    dataset = data_dict

    return dataset['cost']