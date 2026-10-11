import argparse

from digit_classifier.classifier import DigitClassifier
from digit_classifier.commons import load_and_preprocess_image


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MNIST Digit Classifier CLI")
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="cnn",
        choices=["cnn", "rf", "rand"],
        help="Classification algorithm to use (cnn, rf, rand). Default: cnn",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "train"],
        help="Execution mode: predict or train. Default: predict",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset image file.",
    )
    return parser.parse_args(args)


def handle_train(classifier: DigitClassifier) -> None:
    """Handles the training execution flow."""
    try:
        classifier.train()
    except NotImplementedError as exception:
        print(f"Info: {exception} (Training is intentionally disabled)")


def handle_predict(classifier: DigitClassifier, dataset_path: str) -> None:
    """Handles the prediction execution flow with input validation."""
    try:
        image_array = load_and_preprocess_image(dataset_path)
        prediction = classifier.predict(image_array)
        print(f"Prediction result: {prediction}")
    except (FileNotFoundError, ValueError, TypeError) as exception:
        print(f"Validation Error: {exception}")


def main() -> None:
    """Main entry point orchestrating CLI execution."""
    args = parse_arguments()

    print(
        f"Initializing classifier [{args.algorithm.upper()}] in [{args.mode.upper()}] mode..."
    )

    classifier = DigitClassifier(algorithm=args.algorithm)
    if args.mode == "train":
        handle_train(classifier)
    else:
        handle_predict(classifier, args.dataset)


if __name__ == "__main__":
    main()
