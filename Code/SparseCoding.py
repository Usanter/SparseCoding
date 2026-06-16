from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn import datasets


DEFAULT_ALPHA = 0.05
DEFAULT_LAMBDA = 0.15
DEFAULT_MAX_ITER = 15
DEFAULT_ISTA_MAX_ITER = 75
DEFAULT_TOLERANCE = 1e-4
DEFAULT_RANDOM_STATE = 0


def _as_2d_float_array(matrix: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    return array


def convergence(matrix_before: np.ndarray, matrix_after: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return True while the update is still above the requested tolerance."""
    before = np.asarray(matrix_before, dtype=float)
    after = np.asarray(matrix_after, dtype=float)
    return np.linalg.norm(before - after) > tolerance


# Backward-compatible alias.
Convergence = convergence


def shrink(values: np.ndarray | Iterable[float], coef: float) -> np.ndarray:
    """Apply soft-thresholding without mutating the input array."""
    if coef < 0:
        raise ValueError("coef must be non-negative")
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.maximum(np.abs(array) - coef, 0.0)



def ista(
    initial_codes: np.ndarray | Iterable[Iterable[float]],
    dictionary: np.ndarray,
    samples: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    lambda_coef: float | None = DEFAULT_LAMBDA,
    max_iter: int = DEFAULT_ISTA_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Infer sparse codes with iterative soft thresholding."""
    codes = _as_2d_float_array(initial_codes, "initial_codes").copy()
    dictionary = _as_2d_float_array(dictionary, "dictionary")
    samples = _as_2d_float_array(samples, "samples")

    if dictionary.shape[0] != samples.shape[0]:
        raise ValueError("dictionary and samples must share the same feature dimension")
    if dictionary.shape[1] != codes.shape[0] or samples.shape[1] != codes.shape[1]:
        raise ValueError("codes shape must be (n_atoms, n_samples)")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    lambda_value = DEFAULT_LAMBDA if lambda_coef is None else float(lambda_coef)
    if lambda_value < 0:
        raise ValueError("lambda_coef must be non-negative")

    gram = dictionary.T @ dictionary
    cross = dictionary.T @ samples

    for _ in range(max_iter):
        previous = codes.copy()
        gradient = gram @ codes - cross
        codes = shrink(codes - alpha * gradient, alpha * lambda_value)
        if not convergence(previous, codes, tolerance=tolerance):
            break
    return codes


# Backward-compatible alias.
ISTA = ista



def compute_cost(samples: np.ndarray, dictionary: np.ndarray, codes: np.ndarray, lambda_coef: float = DEFAULT_LAMBDA) -> float:
    """Return the average sparse coding objective value."""
    samples = _as_2d_float_array(samples, "samples")
    dictionary = _as_2d_float_array(dictionary, "dictionary")
    codes = _as_2d_float_array(codes, "codes")

    if dictionary.shape[0] != samples.shape[0]:
        raise ValueError("dictionary and samples must share the same feature dimension")
    if dictionary.shape[1] != codes.shape[0] or samples.shape[1] != codes.shape[1]:
        raise ValueError("codes shape must be (n_atoms, n_samples)")
    if lambda_coef < 0:
        raise ValueError("lambda_coef must be non-negative")

    residual = samples - dictionary @ codes
    n_samples = samples.shape[1]
    return float((0.5 * np.linalg.norm(residual) ** 2 + lambda_coef * np.linalg.norm(codes, ord=1)) / n_samples)



def _initial_dictionary(samples: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_features, n_samples = samples.shape
    if k <= n_samples:
        indices = rng.choice(n_samples, size=k, replace=False)
        dictionary = samples[:, indices].copy()
    else:
        dictionary = rng.normal(size=(n_features, k))
    norms = np.linalg.norm(dictionary, axis=0)
    norms[norms == 0] = 1.0
    return dictionary / norms



def sparse_coding(
    x: np.ndarray | Iterable[Iterable[float]],
    k: int = 16,
    alpha: float = DEFAULT_ALPHA,
    lambda_coef: float = DEFAULT_LAMBDA,
    max_iter: int = DEFAULT_MAX_ITER,
    ista_max_iter: int = DEFAULT_ISTA_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    return_costs: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn a dictionary and sparse codes for samples arranged as (features, samples)."""
    samples = _as_2d_float_array(x, "x")
    n_features, n_samples = samples.shape

    if k <= 0:
        raise ValueError("k must be positive")
    if k > n_features * 4:
        raise ValueError("k is unexpectedly large for the provided feature space")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if lambda_coef < 0:
        raise ValueError("lambda_coef must be non-negative")
    if max_iter <= 0 or ista_max_iter <= 0:
        raise ValueError("iteration counts must be positive")

    rng = np.random.default_rng(random_state)
    dictionary = _initial_dictionary(samples, k, rng)
    codes = np.zeros((k, n_samples), dtype=float)
    costs = []

    for _ in range(max_iter):
        previous_dictionary = dictionary.copy()
        codes = ista(
            codes,
            dictionary,
            samples,
            alpha=alpha,
            lambda_coef=lambda_coef,
            max_iter=ista_max_iter,
            tolerance=tolerance,
        )

        gram = codes @ codes.T
        regularized_gram = gram + 1e-6 * np.eye(k)
        dictionary = (samples @ codes.T) @ np.linalg.pinv(regularized_gram)

        zero_columns = np.linalg.norm(dictionary, axis=0) == 0
        if np.any(zero_columns):
            dictionary[:, zero_columns] = _initial_dictionary(samples, int(np.sum(zero_columns)), rng)

        norms = np.linalg.norm(dictionary, axis=0)
        norms[norms == 0] = 1.0
        dictionary = dictionary / norms

        costs.append(compute_cost(samples, dictionary, codes, lambda_coef=lambda_coef))
        if not convergence(previous_dictionary, dictionary, tolerance=tolerance):
            break

    if return_costs:
        return dictionary, codes, np.asarray(costs, dtype=float)
    return dictionary, codes



def load_demo_digits(num_samples: int = 32) -> np.ndarray:
    """Load a small digits subset for examples and tests."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    digits = datasets.load_digits()
    num = min(num_samples, digits.data.shape[0])
    return digits.data[:num].T.astype(float)



def save_dictionary_plot(dictionary: np.ndarray, output_path: str | Path, image_shape: tuple[int, int] = (8, 8)) -> Path:
    """Save the learned atoms as a grid image."""
    dictionary = _as_2d_float_array(dictionary, "dictionary")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = max(1, math.ceil(math.sqrt(dictionary.shape[1])))
    cols = max(1, math.ceil(dictionary.shape[1] / rows))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, atom in zip(axes_array, dictionary.T):
        axis.imshow(atom.reshape(image_shape), cmap="gray")
        axis.axis("off")
    for axis in axes_array[dictionary.shape[1] :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output



def run_demo(
    output_dir: str | Path,
    num_samples: int = 32,
    k: int = 16,
    max_iter: int = 8,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Run the main sparse-coding workflow and persist demo artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    samples = load_demo_digits(num_samples=num_samples)
    dictionary, codes, costs = sparse_coding(
        samples,
        k=k,
        max_iter=max_iter,
        random_state=random_state,
        return_costs=True,
    )

    dictionary_path = output / "dictionary.npy"
    codes_path = output / "codes.npy"
    costs_path = output / "costs.npy"
    np.save(dictionary_path, dictionary)
    np.save(codes_path, codes)
    np.save(costs_path, costs)
    plot_path = save_dictionary_plot(dictionary, output / "dictionary.png")

    return {
        "output_dir": str(output.resolve()),
        "dictionary_path": str(dictionary_path.resolve()),
        "codes_path": str(codes_path.resolve()),
        "costs_path": str(costs_path.resolve()),
        "plot_path": str(plot_path.resolve()),
        "costs": costs,
        "dictionary_shape": dictionary.shape,
        "codes_shape": codes.shape,
    }



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse coding demo on the scikit-learn digits dataset.")
    parser.add_argument("--demo", action="store_true", help="Run the demo workflow and save artifacts.")
    parser.add_argument("--output-dir", default="demo_output", help="Directory where demo artifacts will be written.")
    parser.add_argument("--samples", type=int, default=32, help="Number of digit samples to use.")
    parser.add_argument("--atoms", type=int, default=16, help="Number of dictionary atoms to learn.")
    parser.add_argument("--max-iter", type=int, default=8, help="Maximum alternating-optimization iterations.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for reproducibility.")
    return parser



def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return 0

    summary = run_demo(
        output_dir=args.output_dir,
        num_samples=args.samples,
        k=args.atoms,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    final_cost = float(summary["costs"][-1]) if len(summary["costs"]) else float("nan")
    print(f"Demo artifacts written to: {summary['output_dir']}")
    print(f"Dictionary shape: {summary['dictionary_shape']}")
    print(f"Code shape: {summary['codes_shape']}")
    print(f"Final cost: {final_cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
