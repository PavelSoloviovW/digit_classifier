import pytest

from digit_classifier.__main__ import parse_arguments


def test_parse_arguments_defaults():
    """Test default values when optional flags are omitted."""
    args = parse_arguments(["--dataset", "images/mnist_sample.png"])
    assert args.algorithm == "cnn"
    assert args.mode == "predict"
    assert args.dataset == "images/mnist_sample.png"


def test_parse_arguments_custom_long_flags():
    """Test custom values using long flags."""
    args = parse_arguments(
        ["--algorithm", "rf", "--mode", "train", "--dataset", "images/mnist_sample.png"]
    )
    assert args.algorithm == "rf"
    assert args.mode == "train"
    assert args.dataset == "images/mnist_sample.png"


def test_parse_arguments_custom_short_flags():
    """Test custom values using short flags (-a, -m, -d)."""
    args = parse_arguments(
        ["-a", "rand", "-m", "predict", "-d", "images/mnist_sample.png"]
    )
    assert args.algorithm == "rand"
    assert args.mode == "predict"
    assert args.dataset == "images/mnist_sample.png"


def test_parse_arguments_missing_required():
    """Test that omitting the required dataset argument raises SystemExit."""
    with pytest.raises(SystemExit):
        parse_arguments([])


def test_parse_arguments_invalid_choice():
    """Test that passing an unsupported algorithm choice raises SystemExit."""
    with pytest.raises(SystemExit):
        parse_arguments(
            ["--algorithm", "unknown_algo", "--dataset", "images/mnist_sample.png"]
        )
