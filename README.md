# SparseCoding

SparseCoding is a small research-oriented Python project for learning sparse representations of data. It originates from master's internship work completed at IRIT in the Samova team, and the maintained code path in this repository trains a dictionary on the scikit-learn handwritten-digits dataset, infers sparse activation codes, and saves artifacts that can be inspected as a simple end-to-end demo.

The repository also contains several historical notebooks exploring traditional sparse coding, convolutional sparse coding, LC-KSVD, and speech experiments. Those notebooks are kept for reference, but they rely on older research dependencies and are not part of the supported demo workflow. A new supported Jupyter notebook demo is provided for a lightweight MNIST walkthrough.

## What the project does

- learns a dictionary of atoms from input samples
- infers sparse coefficients for reconstructing those samples
- visualizes the learned atoms as image tiles
- saves dictionary, code, and cost artifacts for inspection

## Installation and setup

### Requirements

- Python 3.10+
- pip

### Recommended setup

```bash
cd <repository-root>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r Code/requirements.txt
```

### Optional package metadata check

```bash
cd Code
python setup.py --name
```

## Usage

### Run the demo

```bash
cd <repository-root>
python Code/demo.py --demo --output-dir demo_output
```

The command writes these files into `demo_output/`:

- `dictionary.npy`: learned dictionary matrix
- `codes.npy`: sparse coefficient matrix
- `costs.npy`: optimization cost history
- `dictionary.png`: image grid of learned atoms

### Run the Jupyter notebook demo

1. Install Jupyter if needed: `pip install notebook`
2. From `<repository-root>`, run `jupyter notebook`
3. Open `Code/MNIST Demo.ipynb`

The notebook downloads MNIST from OpenML on first use, trains a compact dictionary on a small subset, and visualizes learned atoms plus reconstructions.

### Use the library from Python

```python
from SparseCoding import load_demo_digits, sparse_coding

samples = load_demo_digits(num_samples=16)
dictionary, codes, costs = sparse_coding(samples, k=8, return_costs=True)
```

## Configuration options

The demo CLI supports these options:

- `--output-dir`: where demo files are written
- `--samples`: number of digit samples to use
- `--atoms`: number of dictionary atoms to learn
- `--max-iter`: number of alternating-optimization iterations
- `--random-state`: reproducible random seed

The `sparse_coding()` function also accepts:

- `alpha`: ISTA learning rate
- `lambda_coef`: sparsity penalty
- `ista_max_iter`: inner sparse-code iterations
- `tolerance`: convergence threshold
- `return_costs`: whether to return the cost history

## Tests

Run the test suite with:

```bash
cd <repository-root>
python -m unittest discover -s tests -v
```

## Citation

This code is made available for research and educational use. If you use this repository or build on this work, please cite the following paper:

```bibtex
@inproceedings{rolland2019label,
  title={Label-consistent sparse auto-encoders},
  author={Rolland, Thomas and Basarab, Adrian and Pellegrini, Thomas},
  booktitle={Workshop on Signal Processing with Adaptative Sparse Structured Representations (SPARS 2019)},
  pages={1--2},
  year={2019}
}
```
