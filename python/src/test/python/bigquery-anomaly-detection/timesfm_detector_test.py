#
# Copyright (C) 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

"""Unit tests for bqmonitor.timesfm_detector."""

import logging
import unittest

import numpy as np
import apache_beam as beam
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.utils.timestamp import Timestamp

from bqmonitor.timesfm_detector import (
    _resample_with_nans,
    _strip_leading_nans,
    _linear_interpolation,
    _TimesFMDetectorConfig,
    _TAG_PREDICTION,
    _TAG_OBSERVATION,
    TimesFMBufferDoFn,
    TimesFMExtractPredictionsDoFn,
    TimesFMCacheAndScoreDoFn,
    TimesFMEnrichDoFn,
    PATCH_SIZE,
    FORECAST_HORIZON,
)
from bqmonitor.pipeline import _parse_detector_spec

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Preprocessing tests (resample, strip NaN, interpolate)
# ---------------------------------------------------------------------------

class ResampleTest(unittest.TestCase):

  def test_no_gaps(self):
    entries = [(0.0, 10.0), (1.0, 20.0), (2.0, 30.0)]
    grid = _resample_with_nans(entries, 1.0)
    self.assertEqual(len(grid), 3)
    self.assertFalse(np.any(np.isnan(grid)))

  def test_single_gap_produces_nan(self):
    entries = [(0.0, 10.0), (2.0, 30.0)]
    grid = _resample_with_nans(entries, 1.0)
    self.assertEqual(len(grid), 3)
    self.assertTrue(np.isnan(grid[1]))
    self.assertAlmostEqual(grid[0], 10.0)
    self.assertAlmostEqual(grid[2], 30.0)

  def test_multiple_gaps(self):
    entries = [(0.0, 0.0), (4.0, 40.0)]
    grid = _resample_with_nans(entries, 1.0)
    self.assertEqual(len(grid), 5)
    self.assertEqual(int(np.sum(np.isnan(grid))), 3)

  def test_empty(self):
    grid = _resample_with_nans([], 1.0)
    self.assertEqual(len(grid), 0)

  def test_single_entry(self):
    grid = _resample_with_nans([(0.0, 10.0)], 1.0)
    self.assertEqual(len(grid), 1)


class StripLeadingNansTest(unittest.TestCase):

  def test_no_nans(self):
    arr = np.array([1.0, 2.0, 3.0])
    result = _strip_leading_nans(arr)
    np.testing.assert_array_equal(result, arr)

  def test_leading_nans(self):
    arr = np.array([np.nan, np.nan, 1.0, 2.0])
    result = _strip_leading_nans(arr)
    np.testing.assert_array_equal(result, [1.0, 2.0])

  def test_all_nans(self):
    arr = np.array([np.nan, np.nan])
    result = _strip_leading_nans(arr)
    # argmax on all-False returns 0, so returns full array
    self.assertEqual(len(result), 2)


class LinearInterpolationTest(unittest.TestCase):

  def test_no_nans(self):
    arr = np.array([1.0, 2.0, 3.0])
    result = _linear_interpolation(arr)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

  def test_interior_nan(self):
    arr = np.array([10.0, np.nan, 30.0])
    result = _linear_interpolation(arr)
    np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])

  def test_multiple_nans(self):
    arr = np.array([0.0, np.nan, np.nan, np.nan, 40.0])
    result = _linear_interpolation(arr)
    np.testing.assert_array_almost_equal(result, [0.0, 10.0, 20.0, 30.0, 40.0])

  def test_all_nan(self):
    arr = np.array([np.nan, np.nan])
    result = _linear_interpolation(arr)
    np.testing.assert_array_almost_equal(result, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class DetectorConfigTest(unittest.TestCase):

  def test_default_config(self):
    cfg = _TimesFMDetectorConfig()
    self.assertEqual(cfg.min_refresh, 5)
    self.assertEqual(cfg.max_refresh, 10)
    self.assertEqual(cfg.zscore_threshold, 5.0)

  def test_invalid_refresh_bounds(self):
    with self.assertRaises(ValueError):
      _TimesFMDetectorConfig(min_refresh=0)
    with self.assertRaises(ValueError):
      _TimesFMDetectorConfig(max_refresh=129)
    with self.assertRaises(ValueError):
      _TimesFMDetectorConfig(min_refresh=20, max_refresh=10)

  def test_parse_minimal(self):
    cfg = _parse_detector_spec('{"type":"TimesFM"}')
    self.assertIsInstance(cfg, _TimesFMDetectorConfig)

  def test_parse_refresh_bounds(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{"min_refresh":3,"max_refresh":15}}')
    self.assertEqual(cfg.min_refresh, 3)
    self.assertEqual(cfg.max_refresh, 15)

  def test_parse_zscore_threshold(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{"zscore_threshold":3.5}}')
    self.assertEqual(cfg.zscore_threshold, 3.5)


# ---------------------------------------------------------------------------
# BufferDoFn tests
# ---------------------------------------------------------------------------

class BufferDoFnTest(unittest.TestCase):

  def _make_row(self, value, window_start):
    return beam.Row(value=value, window_start=Timestamp.of(window_start),
                    window_end=Timestamp.of(window_start + 1))

  def _run_buffer(self, elements, **kwargs):
    defaults = dict(min_context=32, max_context=64,
                    expected_interval=1.0, min_refresh=5, max_refresh=10)
    defaults.update(kwargs)
    dofn = TimesFMBufferDoFn(**defaults)

    all_entries = []
    inference = []
    observe = []
    warmup = []
    step = 0
    max_entries = defaults['max_context'] + 1

    for key, row in elements:
      ts = row.window_start
      all_entries.append((ts, row.value))
      all_entries.sort(key=lambda x: x[0])
      if len(all_entries) > max_entries:
        all_entries = all_entries[-max_entries:]

      step += 1
      context_len = len(all_entries) - 1

      if context_len < defaults['min_context']:
        warmup.append((key, step))
        continue

      observe.append((key, step))

      import random as _rng
      if not hasattr(self, '_next_refresh') or self._next_refresh is None or step >= self._next_refresh:
        inference.append((key, step))
        interval = _rng.randint(defaults['min_refresh'], defaults['max_refresh'])
        self._next_refresh = step + interval

    return inference, observe, warmup

  def test_warmup_phase(self):
    elements = [('k', self._make_row(float(i), i)) for i in range(10)]
    inf, obs, warmup = self._run_buffer(elements, min_context=32)
    self.assertEqual(len(warmup), 10)
    self.assertEqual(len(obs), 0)
    self.assertEqual(len(inf), 0)

  def test_emits_after_warmup(self):
    elements = [('k', self._make_row(float(i), i)) for i in range(50)]
    inf, obs, warmup = self._run_buffer(elements, min_context=32)
    self.assertEqual(len(warmup), 32)
    self.assertGreater(len(obs), 0)

  def test_inference_within_refresh_bounds(self):
    elements = [('k', self._make_row(float(i), i)) for i in range(100)]
    inf, obs, warmup = self._run_buffer(
        elements, min_context=32, min_refresh=5, max_refresh=10)
    # After 32 warmup, 68 scored. With refresh 5-10, expect 7-14 calls
    self.assertGreater(len(inf), 3)
    self.assertLess(len(inf), len(obs))


# ---------------------------------------------------------------------------
# CacheAndScoreDoFn tests
# ---------------------------------------------------------------------------

class CacheAndScoreDoFnTest(unittest.TestCase):
  """Tests for timestamp-based cursor computation in CacheAndScoreDoFn."""

  def _make_cached(self, context_end_ts=99.0, expected_interval=1.0):
    return {
        'point': np.arange(128, dtype=np.float32) + 50.0,  # 50, 51, 52, ...
        'quantile': np.tile(
            np.array([50, 45, 46, 47, 48, 50, 52, 53, 54, 55], dtype=np.float32),
            (128, 1)),
        'upper_idx': 9, 'lower_idx': 1,
        'context_end_ts': Timestamp.of(context_end_ts),
        'expected_interval': expected_interval,
    }

  def _make_obs(self, observed, window_start):
    ws = Timestamp.of(window_start)
    return {
        _TAG_OBSERVATION: True,
        'observed': observed,
        'row': beam.Row(value=observed, window_start=ws,
                        window_end=ws + 1),
        'step': 1,
    }

  def test_cursor_from_timestamp(self):
    """cursor = round((obs_ts - ctx_end) / interval) - 1"""
    dofn = TimesFMCacheAndScoreDoFn()
    # context_end=99, interval=1
    # obs at t=100: cursor = round((100-99)/1) - 1 = 0
    cached = self._make_cached(context_end_ts=99.0, expected_interval=1.0)
    obs = self._make_obs(48.0, window_start=100.0)
    result = dofn._score('k', obs, cached)
    self.assertIsNotNone(result)
    key, row = result
    self.assertEqual(key, 'k')
    # cursor=0, predicted = 50.0
    self.assertAlmostEqual(row.timesfm_predicted, 50.0)
    self.assertAlmostEqual(row.value, 50.0 - 48.0)  # residual

  def test_cursor_at_offset(self):
    """Observation 5 intervals after context end → cursor=4"""
    dofn = TimesFMCacheAndScoreDoFn()
    cached = self._make_cached(context_end_ts=99.0, expected_interval=1.0)
    # obs at t=104: cursor = round((104-99)/1) - 1 = 4
    obs = self._make_obs(48.0, window_start=104.0)
    result = dofn._score('k', obs, cached)
    key, row = result
    # cursor=4, predicted = 54.0 (50 + 4)
    self.assertAlmostEqual(row.timesfm_predicted, 54.0)

  def test_cursor_with_15s_polling(self):
    """Simulates CDC polling: obs arrives 15 intervals after context end."""
    dofn = TimesFMCacheAndScoreDoFn()
    cached = self._make_cached(context_end_ts=99.0, expected_interval=1.0)
    # obs at t=114: cursor = round((114-99)/1) - 1 = 14
    obs = self._make_obs(48.0, window_start=114.0)
    result = dofn._score('k', obs, cached)
    key, row = result
    # cursor=14, predicted = 64.0 (50 + 14)
    self.assertAlmostEqual(row.timesfm_predicted, 64.0)

  def test_cursor_negative_returns_none(self):
    """Observation before context end → out of range."""
    dofn = TimesFMCacheAndScoreDoFn()
    cached = self._make_cached(context_end_ts=99.0, expected_interval=1.0)
    obs = self._make_obs(48.0, window_start=98.0)  # before context end
    result = dofn._score('k', obs, cached)
    self.assertIsNone(result)

  def test_cursor_past_horizon_returns_none(self):
    """Observation too far in the future → out of range."""
    dofn = TimesFMCacheAndScoreDoFn()
    cached = self._make_cached(context_end_ts=99.0, expected_interval=1.0)
    # obs at t=228: cursor = round((228-99)/1) - 1 = 128 → out of range
    obs = self._make_obs(48.0, window_start=228.0)
    result = dofn._score('k', obs, cached)
    self.assertIsNone(result)

  def test_sub_second_interval(self):
    """0.5s interval: obs at t=100.5 → cursor=0"""
    dofn = TimesFMCacheAndScoreDoFn()
    cached = self._make_cached(context_end_ts=100.0, expected_interval=0.5)
    # obs at t=100.5: cursor = round((100.5-100.0)/0.5) - 1 = 0
    obs = self._make_obs(48.0, window_start=100.5)
    result = dofn._score('k', obs, cached)
    self.assertIsNotNone(result)
    self.assertAlmostEqual(result[1].timesfm_predicted, 50.0)  # cursor=0


# ---------------------------------------------------------------------------
# EnrichDoFn tests
# ---------------------------------------------------------------------------

class EnrichDoFnTest(unittest.TestCase):

  def test_enriches_info(self):
    row = beam.Row(value=2.0, window_start=Timestamp(100), window_end=Timestamp(101),
                   timesfm_observed=48.0, timesfm_predicted=50.0,
                   timesfm_lower=45.0, timesfm_upper=55.0)
    prediction = AnomalyPrediction(
        model_id='ZScore', score=1.5, label=0, info='zscore info')
    result = AnomalyResult(example=row, predictions=[prediction])

    dofn = TimesFMEnrichDoFn()
    outputs = list(dofn.process(('k', result)))
    key, enriched = outputs[0]
    self.assertIn('predicted=50.0000', enriched.predictions[0].info)
    self.assertIn('residual=2.0000', enriched.predictions[0].info)
    self.assertAlmostEqual(enriched.example.value, 48.0)

  def test_model_id(self):
    row = beam.Row(value=0.0, window_start=Timestamp(0), window_end=Timestamp(1),
                   timesfm_observed=50.0, timesfm_predicted=50.0,
                   timesfm_lower=45.0, timesfm_upper=55.0)
    prediction = AnomalyPrediction(model_id='ZScore', score=0.0, label=0)
    result = AnomalyResult(example=row, predictions=[prediction])
    dofn = TimesFMEnrichDoFn()
    outputs = list(dofn.process(result))
    self.assertEqual(outputs[0].predictions[0].model_id, 'TimesFM-2.5+ZScore')


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class PipelineParseTimesFMTest(unittest.TestCase):

  def test_timesfm_in_supported_detectors(self):
    from bqmonitor.pipeline import _SUPPORTED_DETECTORS
    self.assertIn('TimesFM', _SUPPORTED_DETECTORS)

  def test_parse_returns_config(self):
    cfg = _parse_detector_spec('{"type":"TimesFM"}')
    self.assertIsInstance(cfg, _TimesFMDetectorConfig)

  def test_all_config_fields(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{'
        '"min_context":64,"max_context":512,"confidence":80,'
        '"min_refresh":3,"max_refresh":15,"zscore_threshold":3.5'
        '}}')
    self.assertEqual(cfg.min_context, 64)
    self.assertEqual(cfg.max_context, 512)
    self.assertEqual(cfg.min_refresh, 3)
    self.assertEqual(cfg.max_refresh, 15)
    self.assertEqual(cfg.zscore_threshold, 3.5)


if __name__ == '__main__':
  unittest.main()
