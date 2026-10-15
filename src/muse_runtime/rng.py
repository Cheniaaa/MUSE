"""按world/实例/用途隔离的环境随机流。"""
import hashlib
import json

import torch


class WorldRng:
    def __init__(self, root_seed: int, world_count: int, device: str):
        self.root_seed = root_seed
        self.world_count = world_count
        self.device = device
        self._generators = {}

    def generator(self, world: int, instance: str, purpose: str) -> torch.Generator:
        key = (int(world), instance, purpose)
        if key not in self._generators:
            # 只在首次创建流时派生稳定种子，不使用每进程变化的Python hash。
            identity = json.dumps([self.root_seed, *key], ensure_ascii=True).encode()
            seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little') % (2**63)
            self._generators[key] = torch.Generator(device=self.device).manual_seed(seed)
        return self._generators[key]

    def uniform(self, world_ids, instance: str, purpose: str, shape: tuple[int, ...]):
        return torch.stack([
            torch.rand(shape, generator=self.generator(int(world), instance, purpose), device=self.device)
            for world in world_ids
        ])
