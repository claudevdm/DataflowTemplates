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

"""Relative change anomaly detector using windowed timeseries.

Compares the current metric window value against the mean of the last N
windows and alerts if the percentage change exceeds a threshold.

Uses ``WindowedTimeseries`` to collect context — no stateful DoFn needed.

Example detector specs::

    {"type": "RelativeChange", "direction": "decrease", "threshold_pct": 20}
    {"type": "RelativeChange", "direction": "increase", "threshold_pct": 50,
     "lookback_windows": 3}

Pipeline integration::

    keyed_metrics
    | WindowedTimeseries(context_size=lookback_windows, ...)
    | beam.ParDo(RelativeChangeDetector(...))
    -> (key, AnomalyResult)
"""

import dataclasses
import math

import numpy as np

import apache_beam as beam
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult

_VALID_DIRECTIONS = ('decrease', 'increase', 'both')


@dataclasses.dataclass(frozen=True)
class RelativeChangeConfig:
  """Configuration for the RelativeChange detector."""
  direction: str
  threshold_pct: float
  lookback_windows: int

  def __post_init__(self):
    if self.direction not in _VALID_DIRECTIONS:
      raise ValueError(
          f"direction must be one of {_VALID_DIRECTIONS}, "
          f"got '{self.direction}'")
    if self.threshold_pct < 0:
      raise ValueError(
          f'threshold_pct must be >= 0, got {self.threshold_pct}')
    if self.lookback_windows < 1:
      raise ValueError(
          f'lookback_windows must be >= 1, got {self.lookback_windows}')


def _compute_pct_change(current, baseline):
  """Compute percentage change from baseline to current.

  Returns (pct_change, is_valid) where is_valid is False when
  baseline is zero (division undefined).
  """
  if baseline == 0:
    return (float('inf') if current != 0 else 0.0, current != 0)
  return ((current - baseline) / abs(baseline) * 100.0, True)


def _check_alert(pct_change, direction, threshold_pct):
  """Check if the percentage change triggers an alert."""
  if direction == 'decrease':
    return pct_change <= -threshold_pct
  elif direction == 'increase':
    return pct_change >= threshold_pct
  else:  # both
    return abs(pct_change) >= threshold_pct


class RelativeChangeDetector(beam.DoFn):
  """Stateless DoFn that detects relative changes in a timeseries.

  Receives ``(key, [row1, ..., row_current])`` from
  ``WindowedTimeseries``. The last row is the current value;
  preceding rows are context. Computes the mean of the last
  ``lookback_windows`` context values as the baseline.

  Outputs ``(key, AnomalyResult)``.
  """

  def __init__(self, direction='decrease', threshold_pct=20.0,
               lookback_windows=1):
    self._direction = direction
    self._threshold_pct = threshold_pct
    self._lookback_windows = lookback_windows

  def process(self, element):
    key, timeseries = element
    current_row = timeseries[-1]
    context_rows = timeseries[:-1]

    if len(context_rows) < self._lookback_windows:
      prediction = AnomalyPrediction(
          model_id='RelativeChange',
          score=None,
          label=-2,
          info=(f'warmup: {len(context_rows)}'
                f'/{self._lookback_windows}'))
      yield (key, AnomalyResult(
          example=current_row, predictions=[prediction]))
      return

    context_values = [row.value for row in context_rows]
    baseline = float(np.mean(context_values[-self._lookback_windows:]))
    current_value = current_row.value

    pct_change, is_valid = _compute_pct_change(current_value, baseline)

    if is_valid:
      is_alert = _check_alert(
          pct_change, self._direction, self._threshold_pct)
    else:
      is_alert = False

    info = (f'baseline={baseline:.4f} '
            f'current={current_value:.4f} '
            f'pct_change={pct_change:.2f}%')

    prediction = AnomalyPrediction(
        model_id='RelativeChange',
        score=pct_change if not math.isinf(pct_change) else None,
        label=1 if is_alert else 0,
        info=info)
    yield (key, AnomalyResult(
        example=current_row, predictions=[prediction]))
