import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_utils import prepare_training_split


class PrepareTrainingSplitTests(unittest.TestCase):
    def test_small_dataset_can_be_split_without_stratification_error(self):
        X = np.arange(15).reshape(-1, 1)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        X_train, X_test, y_train, y_test = prepare_training_split(X, y, test_size=0.2)

        self.assertEqual(len(X_train) + len(X_test), len(y))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)


if __name__ == "__main__":
    unittest.main()
