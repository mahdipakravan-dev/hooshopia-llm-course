from llm.env import get_device


def test_device_is_known():
    assert get_device() in {"cuda", "mps", "cpu"}
