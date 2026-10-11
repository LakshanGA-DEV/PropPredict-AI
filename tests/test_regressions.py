"""Fast checks for the original leakage, inference, and data-validation bugs.

Run from the project directory:
    python -m unittest discover -s tests -v
These tests don't retrain any models.
"""
import unittest

import numpy as np
import pandas as pd

from feature_engineering import (
    INDIA_CATS, US_CATS, _generate_mock_india_data, _generate_mock_us_data,
    apply_encoding_map, apply_target_encoding, engineer_india_features,
    engineer_us_features, calibrated_interval,
)


class RegressionTests(unittest.TestCase):
    def test_price_never_becomes_an_india_feature(self):
        raw = _generate_mock_india_data(35)
        original = engineer_india_features(raw)
        modified = raw.copy()
        modified['price'] = modified['price'] * 100
        changed = engineer_india_features(modified)
        pd.testing.assert_frame_equal(original, changed)
        self.assertFalse({'price', 'emi_band', 'log_price'} & set(changed.columns))

    def test_price_never_becomes_a_us_feature(self):
        raw = _generate_mock_us_data(35)
        original = engineer_us_features(raw)
        modified = raw.copy()
        modified['price'] = modified['price'] * 100
        changed = engineer_us_features(modified)
        pd.testing.assert_frame_equal(original, changed)
        self.assertFalse({'price', 'price_rent_ratio', 'hoa_value_penalty', 'log_price'} & set(changed.columns))

    def test_out_of_fold_does_not_see_its_own_label(self):
        raw = pd.DataFrame({'city': ['C'] * 30, 'price': np.arange(30) * 100 + 1000})
        encoded, _ = apply_target_encoding(raw, ['city'])
        changed = raw.copy()
        changed.loc[0, 'price'] += 100000
        recoded, _ = apply_target_encoding(changed, ['city'])
        self.assertAlmostEqual(float(encoded.loc[0, 'city_encoded']),
                               float(recoded.loc[0, 'city_encoded']))

    def test_inference_city_encoding_uses_training_map(self):
        raw = _generate_mock_india_data(90)
        engineered = engineer_india_features(raw)
        engineered['price'] = raw['price'].values
        _, maps = apply_target_encoding(engineered, INDIA_CATS)
        new = engineer_india_features(raw.drop(columns='price').iloc[:2])
        output = apply_encoding_map(new, maps)
        self.assertEqual(output.loc[0, 'city_encoded'], maps['city'][raw.iloc[0]['city']])
        self.assertNotIn('city', output.columns)
        self.assertIn('builder_encoded', output.columns)

    def test_zipcode_encoding_stable_between_single_and_batch(self):
        raw = _generate_mock_us_data(90)
        train = engineer_us_features(raw)
        train['price'] = raw['price'].to_numpy()
        _, maps = apply_target_encoding(train, US_CATS)
        batch = apply_encoding_map(engineer_us_features(raw.drop(columns='price').iloc[:2]), maps)
        single = apply_encoding_map(engineer_us_features(raw.drop(columns='price').iloc[:1]), maps)
        for col in single.select_dtypes(include='number').columns:
            self.assertAlmostEqual(float(batch.iloc[0][col]), float(single.iloc[0][col]), places=8)

    def test_usa_full_training_uses_existing_tuner_functions(self):
        import ast
        import inspect
        import model_usa

        tree = ast.parse(inspect.getsource(model_usa.train_us_model))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn('tune_xgb_usa', called)
        self.assertIn('tune_lgb_usa', called)
        self.assertTrue(callable(model_usa.tune_xgb_usa))
        self.assertTrue(callable(model_usa.tune_lgb_usa))

    def test_calibrated_intervals_always_include_point_estimate(self):
        price = np.array([250000, 450000, 900000, 100000])
        q10 = np.array([300000, 200000, 700000, 120000])
        q90 = np.array([400000, 350000, 800000, 80000])
        low, high = calibrated_interval(price, q10, q90, adjustment=20000)
        self.assertTrue((low <= price).all())
        self.assertTrue((high >= price).all())
        self.assertTrue((low >= 0).all())
        self.assertTrue((high >= low).all())

    def test_calibrated_intervals_reject_non_finite_input(self):
        with self.assertRaises(ValueError):
            calibrated_interval([np.nan], [20], [30], 0)
        with self.assertRaises(ValueError):
            calibrated_interval([10], [8], [12], float('inf'))
        with self.assertRaises(ValueError):
            calibrated_interval([10], [8], [12], -1)

    def test_normal_training_saves_calibration(self):
        import inspect, model_usa, model_india
        for trainer in [model_usa.train_us_model, model_india.train_india_model]:
            source = inspect.getsource(trainer)
            self.assertIn('calibration_raw', source)
            self.assertIn('interval_adjustment', source)
            self.assertIn("'schema_version': 3", source)

    def test_synthetic_us_records_are_physically_consistent(self):
        raw = _generate_mock_us_data(400)
        self.assertTrue((raw['bsmt_finished_sqft'] <= raw['bsmt_total_sqft']).all())
        self.assertTrue((raw['year_remodeled'] >= raw['year_built']).all())


if __name__ == '__main__':
    unittest.main()
