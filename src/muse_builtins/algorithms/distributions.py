"""RSL 5.4.2 混合动作分布；仅最后一个动作轴参与联合概率求和。"""
import torch
from torch import nn
from torch.distributions import Beta, Categorical, kl_divergence
from torch.nn.functional import softplus
from rsl_rl.modules.distribution import Distribution


class HybridDistribution(Distribution):
    """Beta motion/power + Categorical channel，输出 [..., motion_dim+2]。

    MLP 最后轴依次为 raw alpha[C]、raw beta[C]、channel logits[K]，
    C=motion_dim+1。连续动作在 [0,1]，物理单位仿射属于调用者；channel
    始终是实际 ID。params=(alpha,beta,normalized_logits) 供 RSL storage/KL 使用。
    mean/std 的 channel 是 ID 数值的统计矩；确定性输出则使用合法的众数 ID。
    """

    def __init__(self, output_dim: int, motion_dim: int, channel_ids: list[int]) -> None:
        if (type(motion_dim) is not int or motion_dim < 1
                or type(output_dim) is not int or output_dim != motion_dim + 2):
            raise ValueError('output_dim must equal positive motion_dim + 2')
        if (not isinstance(channel_ids, list) or not channel_ids
                or any(type(item) is not int or abs(item) > 2**24 for item in channel_ids)
                or len(set(channel_ids)) != len(channel_ids)):
            raise ValueError('channel_ids must be unique exact integers within ±2**24')
        super().__init__(output_dim)
        self.continuous_dim = motion_dim + 1
        self.register_buffer('channel_ids', torch.tensor(channel_ids, dtype=torch.int32))

    @property
    def input_dim(self) -> int:
        return 2 * self.continuous_dim + self.channel_ids.numel()

    def _validate_mlp_output(self, mlp_output: torch.Tensor) -> None:
        if (mlp_output.ndim < 1 or mlp_output.shape[-1] != self.input_dim
                or mlp_output.dtype not in (torch.float32, torch.float64)
                or mlp_output.device != self.channel_ids.device or not torch.isfinite(mlp_output).all()):
            raise ValueError('Invalid hybrid MLP shape/dtype/device/finite values')

    def update(self, mlp_output: torch.Tensor) -> None:
        self._validate_mlp_output(mlp_output)
        c = self.continuous_dim
        self._continuous = Beta(softplus(mlp_output[..., :c]) + 1.,
                                softplus(mlp_output[..., c:2*c]) + 1., validate_args=True)
        self._categorical = Categorical(logits=mlp_output[..., 2*c:], validate_args=True)

    def sample(self) -> torch.Tensor:
        continuous = self._continuous.sample()
        channel = self.channel_ids[self._categorical.sample()].to(continuous.dtype)
        return torch.cat((continuous, channel.unsqueeze(-1)), dim=-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        self._validate_mlp_output(mlp_output)
        return self.as_deterministic_output_module()(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return _HybridDeterministicOutput(self.continuous_dim, self.channel_ids)

    @property
    def mean(self) -> torch.Tensor:
        channel = (self._categorical.probs * self.channel_ids).sum(dim=-1, keepdim=True)
        return torch.cat((self._continuous.mean, channel), dim=-1)

    @property
    def std(self) -> torch.Tensor:
        probs = self._categorical.probs
        mean = (probs * self.channel_ids).sum(dim=-1, keepdim=True)
        variance = (probs * (self.channel_ids - mean).square()).sum(dim=-1, keepdim=True)
        return torch.cat((self._continuous.stddev, variance.sqrt()), dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        return self._continuous.entropy().sum(dim=-1) + self._categorical.entropy()

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._continuous.concentration1, self._continuous.concentration0, self._categorical.logits)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        expected = (*self._categorical.batch_shape, self.output_dim)
        if (outputs.shape != expected or outputs.dtype != self._continuous.mean.dtype
                or outputs.device != self.channel_ids.device or not torch.isfinite(outputs).all()):
            raise ValueError('Invalid hybrid action shape/dtype/device/finite values')
        matches = outputs[..., -1, None] == self.channel_ids
        if not matches.any(dim=-1).all():
            raise ValueError('Invalid channel ID: expected an exact integer from channel_ids')
        index = matches.to(torch.int64).argmax(dim=-1)
        # Beta 自身验证 [0,1]；不采用 RSL BetaDistribution.log_prob 中的 epsilon clipping。
        return (self._continuous.log_prob(outputs[..., :-1]).sum(dim=-1)
                + self._categorical.log_prob(index))

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...],
                      new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        old_alpha, old_beta, old_logits = old_params
        new_alpha, new_beta, new_logits = new_params
        return (kl_divergence(Beta(old_alpha, old_beta), Beta(new_alpha, new_beta)).sum(dim=-1)
                + kl_divergence(Categorical(logits=old_logits), Categorical(logits=new_logits)))


class _HybridDeterministicOutput(nn.Module):
    """RSL JIT/ONNX 导出的纯 tensor head：连续均值与 channel 众数。"""

    def __init__(self, continuous_dim: int, channel_ids: torch.Tensor) -> None:
        super().__init__()
        self.continuous_dim = continuous_dim
        self.register_buffer('channel_ids', channel_ids.clone())

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        c = self.continuous_dim
        alpha = softplus(mlp_output[..., :c]) + 1.
        beta = softplus(mlp_output[..., c:2*c]) + 1.
        channel = self.channel_ids[mlp_output[..., 2*c:].argmax(dim=-1)].to(mlp_output.dtype)
        return torch.cat((alpha / (alpha + beta), channel.unsqueeze(-1)), dim=-1)
