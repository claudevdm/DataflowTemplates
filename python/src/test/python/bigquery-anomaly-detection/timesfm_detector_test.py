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

import math
import logging
import unittest

import numpy as np
import apache_beam as beam
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.utils.timestamp import Timestamp

from bqmonitor.timesfm_detector import (
    _fill_gaps,
    _TimesFMDetectorConfig,
    TimesFMBufferDoFn,
    TimesFMBufferDoFnOLS,
    TimesFMResidualDoFn,
    TimesFMEnrichDoFn,
    PATCH_SIZE,
)
from bqmonitor.pipeline import _parse_detector_spec

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# _fill_gaps tests
# ---------------------------------------------------------------------------

class FillGapsTest(unittest.TestCase):
  """Tests for _fill_gaps()."""

  def test_no_gaps(self):
    entries = [(0.0, 10.0), (1.0, 20.0), (2.0, 30.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    self.assertEqual(len(filled), 3)

  def test_single_gap(self):
    entries = [(0.0, 10.0), (2.0, 30.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    self.assertEqual(len(filled), 3)
    # Interpolated value at t=1 should be 20.0
    self.assertAlmostEqual(filled[1][0], 1.0)
    self.assertAlmostEqual(filled[1][1], 20.0)

  def test_multiple_consecutive_gaps(self):
    entries = [(0.0, 0.0), (4.0, 40.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    self.assertEqual(len(filled), 5)
    for i, (ts, val) in enumerate(filled):
      self.assertAlmostEqual(ts, float(i), places=5)
      self.assertAlmostEqual(val, float(i * 10), places=5)

  def test_gap_threshold(self):
    # Gap of 1.4x interval should NOT be filled (threshold is 1.5x)
    entries = [(0.0, 10.0), (1.4, 24.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    self.assertEqual(len(filled), 2)

  def test_gap_above_threshold(self):
    # Gap of 1.6x interval should be filled
    entries = [(0.0, 10.0), (2.0, 30.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    self.assertEqual(len(filled), 3)

  def test_empty_entries(self):
    self.assertEqual(_fill_gaps([], expected_interval=1.0), [])

  def test_single_entry(self):
    entries = [(0.0, 10.0)]
    self.assertEqual(_fill_gaps(entries, expected_interval=1.0), entries)

  def test_auto_detect_interval(self):
    # When expected_interval=None, detect from median of diffs
    entries = [(0.0, 0.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0)]
    filled = _fill_gaps(entries, expected_interval=None)
    # Median diff is 5.0. Gap between 10.0 and 20.0 is 10.0 (2x > 1.5x)
    # Should fill with one value at t=15.0
    self.assertEqual(len(filled), 5)
    self.assertAlmostEqual(filled[3][0], 15.0)
    self.assertAlmostEqual(filled[3][1], 30.0)

  def test_preserves_original_entries(self):
    entries = [(0.0, 10.0), (1.0, 20.0), (3.0, 40.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    # Original entries should be at positions 0, 1, 3 (shifted by fill)
    self.assertAlmostEqual(filled[0][1], 10.0)
    self.assertAlmostEqual(filled[1][1], 20.0)
    self.assertAlmostEqual(filled[3][1], 40.0)

  def test_with_timestamp_objects(self):
    entries = [(Timestamp.of(0), 10.0), (Timestamp.of(2), 30.0)]
    filled = _fill_gaps(entries, expected_interval=1.0)
    self.assertEqual(len(filled), 3)


# ---------------------------------------------------------------------------
# _TimesFMDetectorConfig tests
# ---------------------------------------------------------------------------

class DetectorConfigTest(unittest.TestCase):
  """Tests for _TimesFMDetectorConfig and _parse_detector_spec."""

  def test_default_config(self):
    cfg = _TimesFMDetectorConfig()
    self.assertEqual(cfg.min_context, 128)
    self.assertEqual(cfg.max_context, 1024)
    self.assertEqual(cfg.confidence, 90)
    self.assertTrue(cfg.force_flip_invariance)
    self.assertTrue(cfg.truncate_negative)
    self.assertFalse(cfg.use_ordered_list_state)
    self.assertIsNone(cfg.expected_interval)

  def test_parse_minimal(self):
    cfg = _parse_detector_spec('{"type":"TimesFM"}')
    self.assertIsInstance(cfg, _TimesFMDetectorConfig)
    self.assertEqual(cfg.max_context, 1024)

  def test_parse_with_config(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{"max_context":2048,"confidence":80}}')
    self.assertEqual(cfg.max_context, 2048)
    self.assertEqual(cfg.confidence, 80)

  def test_parse_expected_interval(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{"expected_interval":60}}')
    self.assertEqual(cfg.expected_interval, 60)

  def test_parse_use_ordered_list_state(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{"use_ordered_list_state":true}}')
    self.assertTrue(cfg.use_ordered_list_state)


# ---------------------------------------------------------------------------
# TimesFMBufferDoFn tests
# ---------------------------------------------------------------------------

class BufferDoFnTest(unittest.TestCase):
  """Tests for TimesFMBufferDoFn (BagState version).

  Tests the DoFn logic directly by simulating stateful calls.
  """

  def _make_row(self, value, window_start):
    return beam.Row(value=value, window_start=float(window_start),
                    window_end=float(window_start + 60))

  def _run_dofn(self, elements, min_context=32, max_context=64,
                expected_interval=60.0):
    """Simulate the buffer DoFn by calling _emit with accumulated entries."""
    dofn = TimesFMBufferDoFn(
        min_context=min_context, max_context=max_context,
        expected_interval=expected_interval)

    all_entries = []
    main_results = []
    warmup_results = []
    max_entries = max_context + 1

    for key, row in elements:
      ts = row.window_start
      all_entries.append((ts, row.value))
      all_entries.sort(key=lambda x: x[0])
      if len(all_entries) > max_entries:
        all_entries = all_entries[-max_entries:]

      output = dofn._emit(key, row, all_entries)
      if output.tag == 'main':
        main_results.append(output.value)
      elif output.tag == 'warmup':
        warmup_results.append(output.value)

    return main_results, warmup_results

  def test_warmup_phase(self):
    elements = [('k', self._make_row(float(i), i * 60))
                for i in range(10)]
    main, warmup = self._run_dofn(elements, min_context=32)
    self.assertEqual(len(warmup), 10)
    self.assertEqual(len(main), 0)
    for key, result in warmup:
      self.assertEqual(result.predictions[0].label, -2)

  def test_emits_after_min_context(self):
    elements = [('k', self._make_row(float(i), i * 60))
                for i in range(40)]
    main, warmup = self._run_dofn(elements, min_context=32)
    self.assertEqual(len(warmup), 32)
    self.assertGreater(len(main), 0)

  def test_context_is_patch_aligned(self):
    elements = [('k', self._make_row(float(i), i * 60))
                for i in range(50)]
    main, _ = self._run_dofn(elements, min_context=32)
    for key, data in main:
      ctx = data['context']
      self.assertEqual(len(ctx) % PATCH_SIZE, 0,
                       f'Context length {len(ctx)} not patch-aligned')

  def test_context_excludes_observed(self):
    elements = [('k', self._make_row(float(i), i * 60))
                for i in range(40)]
    main, _ = self._run_dofn(elements, min_context=32)
    last_key, last_data = main[-1]
    self.assertAlmostEqual(last_data['observed'], 39.0)
    # Context[-1] should be the value before observed
    self.assertAlmostEqual(float(last_data['context'][-1]), 38.0)

  def test_max_context_trim(self):
    elements = [('k', self._make_row(float(i), i * 60))
                for i in range(100)]
    main, _ = self._run_dofn(elements, min_context=32, max_context=64)
    for _, data in main:
      self.assertLessEqual(len(data['context']), 64)

  def test_keyed_output(self):
    elements = [('mykey', self._make_row(float(i), i * 60))
                for i in range(40)]
    main, warmup = self._run_dofn(elements, min_context=32)
    for key, data in main:
      self.assertEqual(key, 'mykey')
    for key, result in warmup:
      self.assertEqual(key, 'mykey')

  def test_row_preserved_in_output(self):
    elements = [('k', self._make_row(float(i), i * 60))
                for i in range(40)]
    main, _ = self._run_dofn(elements, min_context=32)
    _, data = main[-1]
    self.assertIn('row', data)
    self.assertAlmostEqual(data['row'].window_start, 39.0 * 60)

  def test_warmup_info_message(self):
    elements = [('k', self._make_row(0.0, 0))]
    _, warmup = self._run_dofn(elements, min_context=32)
    _, result = warmup[0]
    self.assertIn('warmup:', result.predictions[0].info)
    self.assertIn('0/32', result.predictions[0].info)


# ---------------------------------------------------------------------------
# TimesFMScoreDoFn tests
# ---------------------------------------------------------------------------

class ResidualDoFnTest(unittest.TestCase):
  """Tests for TimesFMResidualDoFn."""

  def _make_prediction_result(self, observed, point_forecast, quantiles,
                              key='k'):
    row = beam.Row(value=observed, window_start=100.0, window_end=160.0)
    data = {'context': np.zeros(32), 'observed': observed, 'row': row}
    quant_array = np.zeros((128, 10))
    quant_array[0] = quantiles
    inference = {
        'point_forecast': np.array([point_forecast] + [0.0] * 127),
        'quantile_forecast': quant_array,
    }
    pr = PredictionResult(example=data, inference=inference,
                          model_id='TimesFM-2.5')
    return (key, pr)

  def _run(self, element, confidence=90):
    dofn = TimesFMResidualDoFn(confidence=confidence)
    return list(dofn.process(element))

  def test_residual_value(self):
    quantiles = [50.0, 45.0, 46.0, 47.0, 48.0, 50.0, 52.0, 53.0, 54.0, 55.0]
    element = self._make_prediction_result(48.0, 50.0, quantiles)
    results = self._run(element)
    self.assertEqual(len(results), 1)
    key, row = results[0]
    self.assertEqual(key, 'k')
    # residual = predicted - observed = 50 - 48 = 2
    self.assertAlmostEqual(row.value, 2.0)

  def test_negative_residual(self):
    quantiles = [50.0, 45.0, 46.0, 47.0, 48.0, 50.0, 52.0, 53.0, 54.0, 55.0]
    element = self._make_prediction_result(55.0, 50.0, quantiles)
    results = self._run(element)
    _, row = results[0]
    # residual = 50 - 55 = -5
    self.assertAlmostEqual(row.value, -5.0)

  def test_preserves_window_timestamps(self):
    quantiles = [50.0, 45.0, 46.0, 47.0, 48.0, 50.0, 52.0, 53.0, 54.0, 55.0]
    element = self._make_prediction_result(50.0, 50.0, quantiles)
    results = self._run(element)
    _, row = results[0]
    self.assertAlmostEqual(row.window_start, 100.0)
    self.assertAlmostEqual(row.window_end, 160.0)

  def test_carries_timesfm_metadata(self):
    quantiles = [50.0, 45.0, 46.0, 47.0, 48.0, 50.0, 52.0, 53.0, 54.0, 55.0]
    element = self._make_prediction_result(48.0, 50.0, quantiles)
    results = self._run(element)
    _, row = results[0]
    self.assertAlmostEqual(row.timesfm_observed, 48.0)
    self.assertAlmostEqual(row.timesfm_predicted, 50.0)
    self.assertAlmostEqual(row.timesfm_lower, 45.0)  # P10
    self.assertAlmostEqual(row.timesfm_upper, 55.0)  # P90

  def test_keyed_output(self):
    quantiles = [50.0] * 10
    element = self._make_prediction_result(50.0, 50.0, quantiles, key='mykey')
    results = self._run(element)
    key, _ = results[0]
    self.assertEqual(key, 'mykey')


class EnrichDoFnTest(unittest.TestCase):
  """Tests for TimesFMEnrichDoFn."""

  def _make_anomaly_result(self, residual, observed, predicted, lower, upper,
                           score=1.5, label=0, key='k'):
    row = beam.Row(
        value=residual,
        window_start=100.0, window_end=160.0,
        timesfm_observed=observed, timesfm_predicted=predicted,
        timesfm_lower=lower, timesfm_upper=upper)
    prediction = AnomalyPrediction(
        model_id='RobustZScore', score=score, label=label, info='zscore info')
    result = AnomalyResult(example=row, predictions=[prediction])
    if key is not None:
      return (key, result)
    return result

  def test_enriches_info(self):
    element = self._make_anomaly_result(
        residual=2.0, observed=48.0, predicted=50.0, lower=45.0, upper=55.0)
    dofn = TimesFMEnrichDoFn()
    results = list(dofn.process(element))
    key, result = results[0]
    info = result.predictions[0].info
    self.assertIn('predicted=50.0000', info)
    self.assertIn('bounds=[45.0000, 55.0000]', info)
    self.assertIn('residual=2.0000', info)
    self.assertIn('zscore info', info)

  def test_restores_observed_value(self):
    element = self._make_anomaly_result(
        residual=2.0, observed=48.0, predicted=50.0, lower=45.0, upper=55.0)
    dofn = TimesFMEnrichDoFn()
    results = list(dofn.process(element))
    _, result = results[0]
    # Example value should be the original observed, not the residual.
    self.assertAlmostEqual(result.example.value, 48.0)

  def test_model_id(self):
    element = self._make_anomaly_result(
        residual=0.0, observed=50.0, predicted=50.0, lower=45.0, upper=55.0)
    dofn = TimesFMEnrichDoFn()
    results = list(dofn.process(element))
    _, result = results[0]
    self.assertEqual(result.predictions[0].model_id, 'TimesFM-2.5+ZScore')

  def test_preserves_label_and_score(self):
    element = self._make_anomaly_result(
        residual=10.0, observed=40.0, predicted=50.0,
        lower=45.0, upper=55.0, score=4.5, label=1)
    dofn = TimesFMEnrichDoFn()
    results = list(dofn.process(element))
    _, result = results[0]
    self.assertAlmostEqual(result.predictions[0].score, 4.5)
    self.assertEqual(result.predictions[0].label, 1)

  def test_keyed_output(self):
    element = self._make_anomaly_result(
        residual=0.0, observed=50.0, predicted=50.0,
        lower=45.0, upper=55.0, key='mykey')
    dofn = TimesFMEnrichDoFn()
    results = list(dofn.process(element))
    key, _ = results[0]
    self.assertEqual(key, 'mykey')


# ---------------------------------------------------------------------------
# Integration: parse_detector_spec for TimesFM in pipeline
# ---------------------------------------------------------------------------

class PipelineParseTimesFMTest(unittest.TestCase):
  """Tests that TimesFM integrates with _parse_detector_spec."""

  def test_timesfm_in_supported_detectors(self):
    from bqmonitor.pipeline import _SUPPORTED_DETECTORS
    self.assertIn('TimesFM', _SUPPORTED_DETECTORS)

  def test_parse_returns_config(self):
    cfg = _parse_detector_spec('{"type":"TimesFM"}')
    self.assertIsInstance(cfg, _TimesFMDetectorConfig)

  def test_all_config_fields_passed(self):
    cfg = _parse_detector_spec(
        '{"type":"TimesFM","config":{'
        '"model_name":"custom/model",'
        '"min_context":64,'
        '"max_context":512,'
        '"confidence":80,'
        '"force_flip_invariance":false,'
        '"truncate_negative":false,'
        '"use_ordered_list_state":true,'
        '"expected_interval":30'
        '}}')
    self.assertEqual(cfg.model_name, 'custom/model')
    self.assertEqual(cfg.min_context, 64)
    self.assertEqual(cfg.max_context, 512)
    self.assertEqual(cfg.confidence, 80)
    self.assertFalse(cfg.force_flip_invariance)
    self.assertFalse(cfg.truncate_negative)
    self.assertTrue(cfg.use_ordered_list_state)
    self.assertEqual(cfg.expected_interval, 30)


if __name__ == '__main__':
  unittest.main()
