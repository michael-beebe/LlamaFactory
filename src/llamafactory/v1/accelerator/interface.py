# Copyright 2025 Bytedance Ltd. and the LlamaFactory team.
#
# This code is inspired by the Bytedance's VeOmni library.
# https://github.com/ByteDance-Seed/VeOmni/blob/v0.1.4/veomni/distributed/parallel_state.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A unified interface for model parallelism and data parallelism.

Supports model parallelism types:
- mp_replicate: Replicate model across multiple devices.
- mp_shard: Shard model across multiple devices.

And data parallelism types:
- dp: Data parallelism.
- cp: Context parallelism.
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Any, Optional

from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..utils import logging
from ..utils.types import DistributedConfig, ProcessGroup, TensorLike
from . import helper


logger = logging.get_logger(__name__)


def _build_mscclpp_meshes(current_device, strategy, timeout_sec):
    """Build (model_device_mesh, data_device_mesh) backed by torchcomms+MSCCL++.

    Both meshes are 2-D to match the standard LlamaFactory layout
    (mp_replicate × mp_shard, dp × cp). For the common case where one
    dimension equals the world size and the other is 1, the world-sized
    dim is backed by a TorchComm("mscclpp", ...) and the size-1 dim is
    backed by a degenerate per-rank TorchComm. This gives FSDP2 a valid
    DeviceMesh whose underlying ProcessGroup routes collectives through
    MSCCL++ (with transparent NCCL fallback for unsupported ops).
    """
    import os as _os
    import torch as _torch
    import torchcomms as _tc
    import torchcomms.device_mesh as _tcdm
    import mscclpp_torchcomms  # noqa: F401 — auto-registers backend .so path

    local_rank = int(_os.environ["LOCAL_RANK"])
    device = _torch.device(current_device.type, local_rank)

    def _make_1d_mesh(size: int, name: str):
        if size != helper.get_world_size():
            raise NotImplementedError(
                f"MSCCL++ hook only supports world-sized device meshes; "
                f"got size={size} (world={helper.get_world_size()}, name={name})."
            )
        comm = _tc.new_comm("mscclpp", device, name=f"lf_{name}")
        return _tcdm.init_device_mesh(mesh_dim_comms=(comm,), mesh_dim_names=(name,))

    world = helper.get_world_size()

    # Model mesh: collapse to 1-D when one dim is degenerate.
    mp_rep, mp_shard = strategy.model_mesh_shape
    if mp_rep == 1 and mp_shard == world:
        model_mesh = _make_1d_mesh(mp_shard, "mp_shard")
    elif mp_shard == 1 and mp_rep == world:
        model_mesh = _make_1d_mesh(mp_rep, "mp_replicate")
    else:
        raise NotImplementedError(
            f"MSCCL++ hook does not yet support 2-D HSDP meshes "
            f"(mp_replicate={mp_rep}, mp_shard={mp_shard})."
        )

    # Data mesh: same collapse rule. We reuse the same world-sized comm name
    # collision-free by appending a suffix.
    dp, cp = strategy.data_mesh_shape
    if dp == world and cp == 1:
        data_mesh = _make_1d_mesh(dp, "dp")
    elif cp == world and dp == 1:
        data_mesh = _make_1d_mesh(cp, "cp")
    else:
        raise NotImplementedError(
            f"MSCCL++ hook does not yet support 2-D data meshes (dp={dp}, cp={cp})."
        )

    logger.info_rank0(
        f"[MSCCL++] LlamaFactory device meshes initialized via torchcomms: "
        f"model_mesh={model_mesh}, data_mesh={data_mesh}"
    )
    return model_mesh, data_mesh


class Dim(StrEnum):
    """Dimension names."""

    MP_REPLICATE = "mp_replicate"
    MP_SHARD = "mp_shard"
    DP = "dp"
    CP = "cp"


@dataclass
class DistributedStrategy:
    """Distributed strategy."""

    mp_replicate_size: int = 1
    """Model parallel replicate size, default to 1."""
    mp_shard_size: int | None = None
    """Model parallel shard size, default to world_size // mp_replicate_size."""
    dp_size: int | None = None
    """Data parallel size, default to world_size // cp_size."""
    cp_size: int = 1
    """Context parallel size, default to 1."""

    def __post_init__(self) -> None:
        if not helper.is_distributed():
            self.mp_shard_size = 1
        elif self.mp_shard_size is None:
            self.mp_shard_size = helper.get_world_size() // self.mp_replicate_size
        elif self.mp_replicate_size * self.mp_shard_size != helper.get_world_size():
            raise ValueError(
                f"mp_replicate_size * mp_shard_size must equal to world_size, "
                f"got {self.mp_replicate_size} * {self.mp_shard_size} != {helper.get_world_size()}."
            )

        if not helper.is_distributed():
            self.dp_size = 1
        elif self.dp_size is None:
            self.dp_size = helper.get_world_size() // self.cp_size
        elif self.dp_size * self.cp_size != helper.get_world_size():
            raise ValueError(
                f"dp_size * cp_size must equal to world_size, "
                f"got {self.dp_size} * {self.cp_size} != {helper.get_world_size()}."
            )

    @property
    def model_mesh_shape(self) -> tuple[int, int]:
        """Model parallel mesh shape."""
        return (self.mp_replicate_size, self.mp_shard_size)

    @property
    def model_mesh_dim_names(self) -> tuple[str, str]:
        """Model parallel mesh dimension names."""
        return (Dim.MP_REPLICATE.value, Dim.MP_SHARD.value)

    @property
    def data_mesh_shape(self) -> tuple[int, int]:
        """Data parallel mesh shape."""
        return (self.dp_size, self.cp_size)

    @property
    def data_mesh_dim_names(self) -> tuple[str, str]:
        """Data parallel mesh dimension names."""
        return (Dim.DP.value, Dim.CP.value)


class DistributedInterface:
    """Distributed interface."""

    _instance: Optional["DistributedInterface"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "DistributedInterface":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, config: DistributedConfig | None = None) -> None:
        if self._initialized:
            return

        self.dist_config = config

        helper.set_device_index()
        self._is_distributed = helper.is_distributed()
        self._rank = helper.get_rank()
        self._world_size = helper.get_world_size()
        self._local_rank = helper.get_local_rank()
        self._local_world_size = helper.get_local_world_size()
        self.current_device = helper.get_current_device()
        self.device_count = helper.get_device_count()

        if config is None:
            self.strategy = DistributedStrategy()
            timeout = 18000
        else:
            self.strategy = DistributedStrategy(
                mp_replicate_size=config.get("mp_replicate_size", 1),
                mp_shard_size=config.get("mp_shard_size", None),
                dp_size=config.get("dp_size", None),
                cp_size=config.get("cp_size", 1),
            )
            timeout = config.get("timeout", 18000)

        if self._is_distributed:
            # MSCCL++ TorchComms hook: when LLAMAFACTORY_USE_MSCCLPP=1, route the
            # model/data device meshes through a torchcomms-backed ProcessGroup so
            # FSDP2 collectives (all_gather, reduce_scatter) flow through MSCCL++
            # instead of stock NCCL. Falls back to the standard path otherwise.
            import os as _os
            if _os.environ.get("LLAMAFACTORY_USE_MSCCLPP", "0") == "1":
                self.model_device_mesh, self.data_device_mesh = _build_mscclpp_meshes(
                    self.current_device,
                    self.strategy,
                    timeout_sec=timeout,
                )
            else:
                init_process_group(
                    timeout=timedelta(seconds=timeout), backend=helper.get_process_group_backend()
                )
                self.model_device_mesh = init_device_mesh(
                    device_type=self.current_device.type,
                    mesh_shape=self.strategy.model_mesh_shape,
                    mesh_dim_names=self.strategy.model_mesh_dim_names,
                )
                self.data_device_mesh = init_device_mesh(
                    device_type=self.current_device.type,
                    mesh_shape=self.strategy.data_mesh_shape,
                    mesh_dim_names=self.strategy.data_mesh_dim_names,
                )
        else:
            self.model_device_mesh = None
            self.data_device_mesh = None

        self._initialized = True
        logger.info_rank0(f"DistributedInterface initialized: {self}.")

    def __str__(self) -> str:
        return (
            f"DistributedInterface(strategy={self.strategy}), is_distributed={self._is_distributed}, "
            f"current_device={self.current_device}, rank={self._rank}, world_size={self._world_size}, "
            f"model_device_mesh={self.model_device_mesh}, data_device_mesh={self.data_device_mesh}"
        )

    def get_device_mesh(self, dim: Dim | None = None) -> DeviceMesh | None:
        """Get device mesh for specified dimension."""
        if dim is None:
            raise ValueError("dim must be specified.")
        elif not self._is_distributed:
            return None
        elif dim in self.strategy.data_mesh_dim_names:
            mesh = self.data_device_mesh
        else:
            mesh = self.model_device_mesh
        # The MSCCL++ hook collapses degenerate (size-1) dims so the mesh
        # may be 1-D with a single-named dim. If the requested dim is the
        # collapsed one (size 1), return the whole mesh — every collective
        # on it is a no-op anyway.
        if mesh.mesh_dim_names is not None and dim.value not in mesh.mesh_dim_names:
            return mesh
        return mesh[dim.value]

    def get_group(self, dim: Dim | None = None) -> Optional[ProcessGroup]:
        """Get process group for specified dimension."""
        if not self._is_distributed or dim is None:
            return None
        else:
            return self.get_device_mesh(dim).get_group()

    def get_rank(self, dim: Dim | None = None) -> int:
        """Get parallel rank for specified dimension."""
        if not self._is_distributed:
            return 0
        elif dim is None:
            return self._rank
        else:
            return self.get_device_mesh(dim).get_local_rank()

    def get_world_size(self, dim: Dim | None = None) -> int:
        """Get parallel size for specified dimension."""
        if not self._is_distributed:
            return 1
        elif dim is None:
            return self._world_size
        else:
            return self.get_device_mesh(dim).size()

    def get_local_rank(self) -> int:
        """Get parallel local rank."""
        return self._local_rank

    def get_local_world_size(self) -> int:
        """Get parallel local world size."""
        return self._local_world_size

    def all_gather(self, data: TensorLike, dim: Dim | None = Dim.DP) -> TensorLike:
        """Gather tensor across specified parallel group."""
        if self._is_distributed:
            return helper.operate_tensorlike(helper.all_gather, data, group=self.get_group(dim))
        else:
            return data

    def all_reduce(
        self, data: TensorLike, op: helper.ReduceOp = helper.ReduceOp.MEAN, dim: Dim | None = Dim.DP
    ) -> TensorLike:
        """Reduce tensor across specified parallel group."""
        if self._is_distributed:
            return helper.operate_tensorlike(helper.all_reduce, data, op=op, group=self.get_group(dim))
        else:
            return data

    def broadcast(self, data: TensorLike, src: int = 0, dim: Dim | None = Dim.DP) -> TensorLike:
        """Broadcast tensor across specified parallel group."""
        if self._is_distributed:
            return helper.operate_tensorlike(helper.broadcast, data, src=src, group=self.get_group(dim))
        else:
            return data

    def sync(self) -> None:
        """Synchronize all processes."""
        if self._is_distributed:
            helper.synchronize()

    def barrier(self) -> None:
        """Barrier all processes."""
        if self._is_distributed:
            barrier()

    def destroy(self) -> None:
        """Destroy all processes."""
        if self._is_distributed:
            destroy_process_group()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.accelerator.interface
    """
    print(DistributedInterface())
