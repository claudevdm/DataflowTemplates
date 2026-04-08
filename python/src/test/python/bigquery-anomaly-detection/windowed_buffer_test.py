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

"""Tests for the windowed timeseries buffer and relative change detector."""

import shutil
import sys
import unittest

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_stream import TestStream
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.window import TimestampedValue
from apache_beam.utils.timestamp import Timestamp

from bqmonitor.windowed_buffer import WindowedTimeseries
from bqmonitor.windowed_relative_change import RelativeChangeDetector


def _make_row(value, window_start):
  return beam.Row(
      value=float(value),
      window_start=Timestamp.of(window_start),
      window_end=Timestamp.of(window_start + 1))


# ---------------------------------------------------------------------------
# WindowedTimeseries tests (batch)
# ---------------------------------------------------------------------------

class WindowedTimeseriesTest(unittest.TestCase):
  """Test WindowedTimeseries PTransform in batch mode."""

  def test_basic_collect(self):
    """Elements 1s apart, context_size=2, window=1s."""
    elements = [
        TimestampedValue(("k", _make_row(10, 1)), 1),
        TimestampedValue(("k", _make_row(20, 2)), 2),
        TimestampedValue(("k", _make_row(30, 3)), 3),
        TimestampedValue(("k", _make_row(40, 4)), 4),
        TimestampedValue(("k", _make_row(50, 5)), 5),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(elements)
          | WindowedTimeseries(context_size=2, window_duration_sec=1))
      # Extract (key, [values]) for easy assertion.
      values = result | beam.MapTuple(
          lambda k, rows: (k, [int(r.value) for r in rows]))
      assert_that(values, equal_to([
          ("k", [10]),
          ("k", [10, 20]),
          ("k", [10, 20, 30]),
          ("k", [20, 30, 40]),
          ("k", [30, 40, 50]),
      ]))

  def test_context_size_zero(self):
    """With context_size=0, each window has exactly one element."""
    elements = [
        TimestampedValue(("k", _make_row(10, 1)), 1),
        TimestampedValue(("k", _make_row(20, 2)), 2),
        TimestampedValue(("k", _make_row(30, 3)), 3),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(elements)
          | WindowedTimeseries(context_size=0, window_duration_sec=1))
      values = result | beam.MapTuple(
          lambda k, rows: (k, [int(r.value) for r in rows]))
      assert_that(values, equal_to([
          ("k", [10]),
          ("k", [20]),
          ("k", [30]),
      ]))

  def test_multiple_keys(self):
    """Each key gets independent timeseries."""
    elements = [
        TimestampedValue(("a", _make_row(10, 1)), 1),
        TimestampedValue(("a", _make_row(20, 2)), 2),
        TimestampedValue(("b", _make_row(100, 1)), 1),
        TimestampedValue(("b", _make_row(200, 2)), 2),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(elements)
          | WindowedTimeseries(context_size=1, window_duration_sec=1))
      values = result | beam.MapTuple(
          lambda k, rows: (k, [int(r.value) for r in rows]))
      assert_that(values, equal_to([
          ("a", [10]),
          ("a", [10, 20]),
          ("b", [100]),
          ("b", [100, 200]),
      ]))


# ---------------------------------------------------------------------------
# RelativeChangeDetector tests (batch)
# ---------------------------------------------------------------------------

class RelativeChangeDetectorTest(unittest.TestCase):
  """End-to-end tests for RelativeChangeDetector in batch mode."""

  def _run(self, elements, checker, context_size=1,
           window_duration_sec=1, **detector_kwargs):
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(elements)
          | WindowedTimeseries(
              context_size=context_size,
              window_duration_sec=window_duration_sec)
          | beam.ParDo(RelativeChangeDetector(**detector_kwargs)))
      assert_that(result, checker)

  def _make_elements(self, values, start=1):
    """Create timestamped (key, row) elements from a list of values."""
    return [
        TimestampedValue(("k", _make_row(v, start + i)), start + i)
        for i, v in enumerate(values)]

  def test_warmup(self):
    """First element has no context — warmup."""
    elements = self._make_elements([100.0])

    def _check(actual):
      assert len(actual) == 1
      assert actual[0][1].predictions[0].label == -2

    self._run(elements, _check,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=1)

  def test_warmup_with_lookback_3(self):
    """First 3 elements are warmup with lookback_windows=3."""
    elements = self._make_elements([10.0, 20.0, 30.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, -2, -2], f'{labels}'

    self._run(elements, _check, context_size=3,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=3)

  def test_decrease_detected(self):
    """25% decrease triggers alert."""
    elements = self._make_elements([100.0, 75.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, 1], f'{labels}'
      assert actual[1][1].predictions[0].score == -25.0

    self._run(elements, _check,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=1)

  def test_decrease_not_triggered(self):
    """10% decrease does not trigger 20% threshold."""
    elements = self._make_elements([100.0, 90.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, 0], f'{labels}'

    self._run(elements, _check,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=1)

  def test_increase_detected(self):
    """50% increase triggers alert."""
    elements = self._make_elements([100.0, 150.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, 1], f'{labels}'
      assert actual[1][1].predictions[0].score == 50.0

    self._run(elements, _check,
              direction='increase', threshold_pct=20.0,
              lookback_windows=1)

  def test_both_direction(self):
    """Both decrease and increase trigger with direction='both'."""
    elements = self._make_elements([100.0, 75.0, 120.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, 1, 1], f'{labels}'

    self._run(elements, _check, context_size=1,
              direction='both', threshold_pct=20.0,
              lookback_windows=1)

  def test_lookback_3_uses_mean(self):
    """lookback_windows=3 uses mean of last 3 context values."""
    elements = self._make_elements([100.0, 100.0, 100.0, 70.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      # First 3 are warmup, 4th has baseline=mean(100,100,100)=100, -30%
      assert labels == [-2, -2, -2, 1], f'{labels}'
      score = actual[3][1].predictions[0].score
      assert abs(score - (-30.0)) < 0.01, f'{score}'

    self._run(elements, _check, context_size=3,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=3)

  def test_zero_baseline(self):
    """Zero baseline with nonzero current triggers alert."""
    elements = self._make_elements([0.0, 50.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, 1], f'{labels}'
      # inf score stored as None
      assert actual[1][1].predictions[0].score is None

    self._run(elements, _check,
              direction='increase', threshold_pct=20.0,
              lookback_windows=1)

  def test_zero_to_zero(self):
    """Zero to zero is not an alert."""
    elements = self._make_elements([0.0, 0.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      assert labels == [-2, 0], f'{labels}'

    self._run(elements, _check,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=1)

  def test_many_elements(self):
    """Multiple elements processed in order."""
    elements = self._make_elements([100.0, 100.0, 80.0, 80.0, 60.0])

    def _check(actual):
      labels = [r.predictions[0].label for _, r in actual]
      # warmup, 0%, -20%, 0%, -25%
      assert labels == [-2, 0, 1, 0, 1], f'{labels}'

    self._run(elements, _check,
              direction='decrease', threshold_pct=20.0,
              lookback_windows=1)


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

_go_installed = shutil.which('go') is not None
_in_windows = sys.platform == "win32"


@unittest.skipUnless(_go_installed, 'Go is not installed.')
@unittest.skipIf(_in_windows, reason="Not supported on Windows")
class StreamingWindowedBufferTest(unittest.TestCase):
  """Test the windowed approach with TestStream + Prism."""

  def setUp(self):
    self.options = PipelineOptions([
        "--streaming",
        "--environment_type=LOOPBACK",
        "--runner=PrismRunner",
    ])

  def test_streaming_basic(self):
    """Elements arrive in a stream, get collected into timeseries."""
    test_stream = TestStream()
    for ts in range(1, 6):
      row = _make_row(ts * 10, ts)
      test_stream.add_elements([("k", row)], event_timestamp=ts)
      test_stream.advance_watermark_to(ts)
      test_stream.advance_processing_time(10)
    test_stream.advance_watermark_to_infinity()

    with TestPipeline(options=self.options) as p:
      result = (
          p
          | test_stream
          | WindowedTimeseries(context_size=2, window_duration_sec=1))
      values = result | beam.MapTuple(
          lambda k, rows: (k, [int(r.value) for r in rows]))
      assert_that(values, equal_to([
          ("k", [10]),
          ("k", [10, 20]),
          ("k", [10, 20, 30]),
          ("k", [20, 30, 40]),
          ("k", [30, 40, 50]),
      ]))


if __name__ == '__main__':
  unittest.main()
