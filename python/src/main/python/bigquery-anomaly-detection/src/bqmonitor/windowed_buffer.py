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

"""Windowed timeseries buffer using Beam windowing primitives.

Replaces the stateful ``TimestampBufferDoFn`` with a simpler approach:
expand each element into future windows so it appears as context,
then use ``FixedWindows + CombinePerKey`` to collect sorted timeseries.

No stateful DoFn, no timers, no buffer state machine.

Usage::

    keyed_metrics = ...  # (key, beam.Row(value, window_start, window_end))
    timeseries = keyed_metrics | WindowedTimeseries(
        context_size=3,
        window_duration_sec=60)
    # Output: (key, [row_t1, row_t2, row_t3, row_current])
"""

import apache_beam as beam
from apache_beam.transforms.window import FixedWindows
from apache_beam.transforms.window import GlobalWindows
from apache_beam.transforms.window import TimestampedValue


def _expand_to_windows(keyed_element, context_size, window_duration_sec):
  """Expand a keyed element into context_size+1 windows.

  Each element is emitted at its original timestamp (home window)
  and at timestamps shifted forward by 1..context_size windows,
  so it appears as context in future windows.

  Elements are tagged as 'home' or 'context' so trailing windows
  (containing only context) can be filtered out.
  """
  key, row = keyed_element
  event_time = row.window_start
  for i in range(context_size + 1):
    shifted = event_time + (i * window_duration_sec)
    is_home = (i == 0)
    yield TimestampedValue((key, (is_home, row)), shifted)


class _CollectTimeseriesFn(beam.CombineFn):
  """Collects (is_home, row) pairs into a sorted timeseries.

  Tracks whether any element in the window is a "home" element
  (vs only context from past windows). Output is
  (has_home, sorted_rows).
  """

  def create_accumulator(self):
    return (False, [])

  def add_input(self, accumulator, element):
    has_home, rows = accumulator
    is_home, row = element
    rows.append(row)
    return (has_home or is_home, rows)

  def merge_accumulators(self, accumulators):
    merged_has_home = False
    merged_rows = []
    for has_home, rows in accumulators:
      merged_has_home = merged_has_home or has_home
      merged_rows.extend(rows)
    return (merged_has_home, merged_rows)

  def extract_output(self, accumulator):
    has_home, rows = accumulator
    sorted_rows = sorted(rows, key=lambda r: r.window_start)
    return (has_home, sorted_rows)


class WindowedTimeseries(beam.PTransform):
  """Collect timeseries context using Beam windowing.

  For each element, expands it into ``context_size + 1`` windows so
  that each window receives the current element plus up to
  ``context_size`` preceding elements as context.

  Args:
      context_size: Number of preceding elements to include as context.
      window_duration_sec: Duration of each fixed window in seconds.
          Must match the spacing of input elements.

  Input:  ``(key, beam.Row(value, window_start, window_end))``
  Output: ``(key, [row1, row2, ..., row_current])``
      Sorted list of rows. The last element is the "current" one;
      preceding elements are context.
  """

  def __init__(self, context_size, window_duration_sec):
    self._context_size = context_size
    self._window_duration_sec = window_duration_sec

  def expand(self, pcoll):
    return (
        pcoll
        | 'ExpandToWindows' >> beam.FlatMap(
            _expand_to_windows,
            context_size=self._context_size,
            window_duration_sec=self._window_duration_sec)
        | 'FixedWindow' >> beam.WindowInto(
            FixedWindows(self._window_duration_sec))
        | 'CollectTimeseries' >> beam.CombinePerKey(
            _CollectTimeseriesFn())
        | 'GlobalWindow' >> beam.WindowInto(GlobalWindows())
        | 'FilterTrailing' >> beam.Filter(
            lambda kv: kv[1][0])  # keep only windows with home element
        | 'ExtractRows' >> beam.MapTuple(
            lambda key, home_rows: (key, home_rows[1])))
