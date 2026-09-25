import torch

from when_to_reflect import device


def test_select_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: True)

    assert device.select_device() == torch.device("cuda")


def test_select_device_uses_mps_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device, "mps_available", lambda: True)

    assert device.select_device() == torch.device("mps")


def test_select_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    assert device.select_device() == torch.device("cpu")
