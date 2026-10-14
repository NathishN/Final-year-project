"""
tests/test_data_integrity.py
============================
Automated tests for dataset splits, path integrity, and Grade 2 coverage.
"""

import os
import unittest
import pandas as pd

ROOT = r"C:\fruit_dataset"
MANIFEST_PATH = os.path.join(ROOT, "outputs", "final_audit", "split_manifest.csv")


class TestDataIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MANIFEST_PATH):
            raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
        cls.df = pd.read_csv(MANIFEST_PATH)

    def test_total_valid_samples(self):
        """Manifest must contain exactly 56,036 active valid samples."""
        self.assertEqual(len(self.df), 56036)

    def test_all_five_grades_present(self):
        """All five severity grades (0, 1, 2, 3, 4) must exist in manifest."""
        grades = set(self.df['severity_grade'].unique())
        self.assertEqual(grades, {0, 1, 2, 3, 4})

    def test_grade2_test_coverage(self):
        """Test set must contain exactly 119 Grade 2 samples from session 20200730."""
        g2_test = ((self.df['split'] == 'test') & (self.df['severity_grade'] == 2)).sum()
        self.assertEqual(g2_test, 119)

    def test_zero_cross_split_path_leakage(self):
        """Zero image paths may cross train, val, or test splits."""
        train_p = set(self.df[self.df['split'] == 'train']['image_path'])
        val_p = set(self.df[self.df['split'] == 'val']['image_path'])
        test_p = set(self.df[self.df['split'] == 'test']['image_path'])

        self.assertEqual(len(train_p.intersection(val_p)), 0, "Train-Val path leakage!")
        self.assertEqual(len(train_p.intersection(test_p)), 0, "Train-Test path leakage!")
        self.assertEqual(len(val_p.intersection(test_p)), 0, "Val-Test path leakage!")

    def test_zero_grade2_session_leakage(self):
        """Session 20200730 in Banana Grade 2 must exist exclusively in test split."""
        g2_bananas = self.df[(self.df['severity_grade'] == 2) & (self.df['fruit'] == 'Banana')]
        ses_20200730 = g2_bananas[g2_bananas['session_id'].astype(str) == '20200730']
        splits = set(ses_20200730['split'].unique())
        self.assertEqual(splits, {'test'})


if __name__ == '__main__':
    unittest.main()
