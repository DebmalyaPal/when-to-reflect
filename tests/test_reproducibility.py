import torch

from when_to_reflect.reproducibility import set_random_seed


def test_same_seed_reproduces_sampled_values() -> None:
    """Backend check behind the matched-seed rollout protocol."""

    def sample(seed: int) -> list[float]:
        set_random_seed(seed)
        return torch.rand(8).tolist()

    assert sample(42) == sample(42)
    assert sample(42) != sample(43)
