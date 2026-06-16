import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

import SparseCoding as sc


class SparseCodingTests(unittest.TestCase):
    def test_shrink_does_not_mutate_input(self):
        values = np.array([1.0, -0.5, 0.1])
        original = values.copy()

        result = sc.shrink(values, 0.2)

        np.testing.assert_allclose(result, np.array([0.8, -0.3, 0.0]))
        np.testing.assert_allclose(values, original)

    def test_sparse_coding_returns_expected_shapes_and_costs(self):
        samples = sc.load_demo_digits(num_samples=12)

        dictionary, codes, costs = sc.sparse_coding(
            samples,
            k=6,
            max_iter=4,
            ista_max_iter=20,
            random_state=0,
            return_costs=True,
        )

        self.assertEqual(dictionary.shape, (64, 6))
        self.assertEqual(codes.shape, (6, 12))
        self.assertGreaterEqual(len(costs), 1)
        self.assertTrue(np.all(np.isfinite(costs)))
        norms = np.linalg.norm(dictionary, axis=0)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_sparse_coding_rejects_invalid_k(self):
        samples = sc.load_demo_digits(num_samples=8)
        with self.assertRaises(ValueError):
            sc.sparse_coding(samples, k=0)

    def test_run_demo_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = sc.run_demo(tmpdir, num_samples=10, k=4, max_iter=3, random_state=0)

            self.assertTrue(Path(summary["dictionary_path"]).exists())
            self.assertTrue(Path(summary["codes_path"]).exists())
            self.assertTrue(Path(summary["costs_path"]).exists())
            self.assertTrue(Path(summary["plot_path"]).exists())

    def test_cli_demo_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CODE_DIR / "demo.py"),
                    "--demo",
                    "--output-dir",
                    tmpdir,
                    "--samples",
                    "10",
                    "--atoms",
                    "4",
                    "--max-iter",
                    "3",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Demo artifacts written to:", completed.stdout)
            for artifact in ("dictionary.npy", "codes.npy", "costs.npy", "dictionary.png"):
                self.assertTrue((Path(tmpdir) / artifact).exists())


    def test_mnist_notebook_demo_is_present_and_documented(self):
        notebook_path = CODE_DIR / "MNIST Demo.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

        self.assertIn("IRIT (Samova team)", joined_sources)
        self.assertIn("fetch_openml(\"mnist_784\"", joined_sources)
        self.assertIn("from SparseCoding import compute_cost, sparse_coding", joined_sources)

    def test_readme_mentions_internship_context_and_notebook_demo(self):
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("master's internship work completed at IRIT in the Samova team", readme_text)
        self.assertIn("Code/MNIST Demo.ipynb", readme_text)

if __name__ == "__main__":
    unittest.main()
