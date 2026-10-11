# MNIST Digit Classifier

An object-oriented architecture for MNIST digit classification built with clean code principles, professional naming conventions, pluggable model implementations, and a robust unit-testing suite. The project uses a **Facade pattern** for unified prediction and is managed via the modern **uv** package manager.

## Project Structure

```text
├── .gitignore
├── .python-version
├── README.md
├── images/
│   ├── mnist_sample.png
│   └── non_mnist_sample.png
├── pyproject.toml
├── src/
│   └── digit_classifier/
│       ├── __init__.py
│       ├── __main__.py
│       ├── classifier.py
│       ├── commons.py
│       ├── model.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cnn_model.py
│       │   ├── rand_model.py
│       │   └── random_forest_model.py
│       └── utils/
│           └── generate_images.py
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   ├── test_cli.py
│   └── test_validation.py
└── uv.lock
```

## Features

* **Facade Design Pattern**: The `DigitClassifier` class provides a single, unified interface for prediction and mode handling, hiding underlying model complexity.
* **Dependency Injection (DI)**: Full support for injecting mock or custom model instances directly into the facade for isolated testing.
* **Pluggable Models**: Easily switch between a Convolutional Neural Network (`CNNModel`), a Random Forest classifier (`RandomForestModel`), and a baseline random model (`RandomModel`).
* **Strict Data Validation**: Core validation invariants ensure data shape integrity (`28x28`) before inference.
* **Image Generation Utility**: Built-in utility to generate sample images (valid MNIST shapes vs. invalid shapes) for quick integration testing.
* **CLI Support**: Fully interactive, testable command-line interface via `argparse` supporting execution modes (`predict`/`train`) and dataset paths.
* **Specification Compliance**: Training methods explicitly raise `NotImplementedError` per project requirements.

---

## Installation & Setup

Ensure you have **uv** installed, then sync the environment and dependencies:

```bash
uv sync
```

---

## Running Tests

The project includes a comprehensive, isolated unit-testing suite powered by `pytest` (covering the facade, dependency injection, data validation, and CLI argument parsing):

```bash
uv run pytest
```

---

## Usage

### 1. Generating Sample Images

You can generate sample images (both valid and invalid shapes) using the built-in utility script:

```bash
uv run python -m digit_classifier.utils.generate_images
```

### 2. Making Predictions via CLI

Run predictions using any of the available algorithms (`cnn`, `rf`, or `rand`):

```bash
uv run python -m digit_classifier --algorithm cnn --mode predict --dataset ./images/mnist_sample.png
```

### Available Arguments:

* `-a`, `--algorithm`: Choose the model implementation (`cnn`, `rf`, or `rand`). Default: `cnn`.
* `-m`, `--mode`: Choose execution mode (`predict` or `train`). Default: `predict`. *(Note: training triggers `NotImplementedError` per project requirements).*
* `-d`, `--dataset`: Path to the dataset image file.
