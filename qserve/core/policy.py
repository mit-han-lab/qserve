# original file: https://github.com/vllm-project/vllm/blob/main/vllm/core/policy.py
# modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
from collections import deque
from typing import Deque

from qserve.sequence import SequenceGroup


class Policy:
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            )
        )


class FCFS(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time


class PolicyFactory:
    _POLICY_REGISTRY = {
        "fcfs": FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
