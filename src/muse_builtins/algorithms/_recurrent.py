"""RSL RNN 保留完整历史计算，只在 latent 上投影当前 loss 的有效样本。"""
import torch
from rsl_rl.models import RNNModel
from rsl_rl.utils import unpad_trajectories


class MaskedRNNModel(RNNModel):
    """构造和 rollout 接口继承 RNNModel；训练 mask 由 MaskedStorage 提供。"""

    def get_latent(self, obs, masks=None, hidden_state=None):
        if masks is not None and hidden_state is None:
            # 第一个 rollout 的初态可能尚未物化；不能借用持续推理状态。
            rnn = self.rnn.rnn
            zero = next(self.parameters()).new_zeros(rnn.num_layers, masks.shape[1], rnn.hidden_size)
            hidden_state = (zero, torch.zeros_like(zero)) if isinstance(rnn, torch.nn.LSTM) else zero
        latent = super().get_latent(obs, masks, hidden_state)
        if masks is not None:
            valid = unpad_trajectories(obs['_loss_mask'], masks).flatten().bool()
            # RNNModel 在 H=1 时 squeeze(0)，展平后两种形状的顺序一致。
            # 保留 recurrent 的时间轴。原 PPO 沿第 0 轴截取 entropy，
            # [1, samples, features] 可让任意样本数完整进入原 loss。
            latent = latent.reshape(-1, self.latent_dim)[valid].unsqueeze(0)
        return latent
