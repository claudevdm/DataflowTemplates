"""TimesFM anomaly detector for the BigQuery anomaly detection pipeline.

Uses Google's TimesFM 2.5 foundation model (via HuggingFace transformers)
for zero-shot time-series forecasting and anomaly detection based on
residual z-score analysis.

Pipeline integration::

    BufferDoFn ─┬─ 'inference' (every N steps) ─→ RunInference ─→ predictions
                ├─ 'observe' (every step) ─────────────────────→ observations
                └─ 'warmup' ───────────────────────────────────→ warmup results
                                                                    │
    CacheAndScoreDoFn ←── Flatten(predictions, observations) ──────┘
        │  (stateful: caches predictions, scores observations)
        ▼
    (key, beam.Row(value=residual, ...))
        │
    AnomalyDetection(ZScore) → EnrichDoFn → AnomalyResult

Detector spec::

    {"type": "TimesFM"}
    {"type": "TimesFM", "config": {"max_context": 1024, "min_refresh": 5, "max_refresh": 10}}
"""

import dataclasses
import logging
import time

import numpy as np
import apache_beam as beam
from apache_beam.coders import FastPrimitivesCoder
from apache_beam.coders import PickleCoder
from apache_beam.transforms.userstate import OrderedListStateSpec
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.transforms.userstate import BagStateSpec
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.utils.timestamp import Duration
from apache_beam.utils.timestamp import Timestamp

from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import Optional

# Sentinel: any timestamp before this is "uninitialized"
_MIN_TIMESTAMP = Timestamp(0)

_LOGGER = logging.getLogger(__name__)

PATCH_SIZE = 32
FORECAST_HORIZON = 128

# Tags for distinguishing predictions vs observations in CacheAndScoreDoFn
_TAG_PREDICTION = '__prediction__'
_TAG_OBSERVATION = '__observation__'


def _resample_with_nans(entries, expected_interval):
    """Resample (timestamp, value) entries onto an evenly-spaced grid.

    Places known values at their nearest grid position and fills gaps
    with NaN. The resulting array has no timestamp gaps — every grid
    position has either a real value or NaN.

    Args:
        entries: Sorted list of (Timestamp|float, value) tuples.
        expected_interval: Expected spacing between consecutive entries
            (seconds, as float).

    Returns:
        np.ndarray of values on the evenly-spaced grid.
    """
    if len(entries) < 2 or expected_interval <= 0:
        return np.array([e[1] for e in entries], dtype=np.float64)

    t_start = Timestamp.of(entries[0][0])
    t_end = Timestamp.of(entries[-1][0])
    span_secs = float(t_end - t_start)
    n_points = round(span_secs / expected_interval) + 1
    grid = np.full(n_points, np.nan, dtype=np.float64)

    for ts, val in entries:
        offset_secs = float(Timestamp.of(ts) - t_start)
        idx = round(offset_secs / expected_interval)
        if 0 <= idx < n_points:
            grid[idx] = val

    n_gaps = int(np.sum(np.isnan(grid)))
    if n_gaps > 0:
        _LOGGER.warning(
            '[TimesFM] _resample: %d/%d grid positions are NaN '
            '(interval=%.2fs, %d raw entries)',
            n_gaps, n_points, expected_interval, len(entries))

    return grid


def _strip_leading_nans(arr):
    """Remove contiguous NaN values from the beginning of an array.

    Adapted from the native TimesFM package.
    """
    isnan = np.isnan(arr)
    if not np.any(isnan):
        return arr
    first_valid = np.argmax(~isnan)
    return arr[first_valid:]


def _linear_interpolation(arr):
    """Fill NaN values via linear interpolation.

    Adapted from the native TimesFM package. Interior NaNs are
    interpolated between neighbors; edge NaNs are filled with the
    nearest valid value (via np.interp extrapolation).
    """
    nans = np.isnan(arr)
    if not np.any(nans):
        return arr

    arr = arr.copy()
    nans_idx = np.where(nans)[0]
    valid_idx = np.where(~nans)[0]
    valid_vals = arr[~nans]

    if len(valid_vals) == 0:
        return np.zeros_like(arr)

    arr[nans] = np.interp(nans_idx, valid_idx, valid_vals)
    return arr


@dataclasses.dataclass(frozen=True)
class _TimesFMDetectorConfig:
    """Lightweight config container for TimesFM detector."""
    model_name: str = 'google/timesfm-2.5-200m-transformers'
    min_context: int = 128
    max_context: int = 1024
    confidence: int = 90
    force_flip_invariance: bool = True
    truncate_negative: bool = True
    use_ordered_list_state: bool = False
    expected_interval: Optional[float] = None
    zscore_threshold: float = 5.0
    min_refresh: int = 5   # min steps between forecast refreshes per key
    max_refresh: int = 10  # max steps between forecast refreshes per key

    def __post_init__(self):
        if self.min_refresh < 1:
            raise ValueError(
                f'min_refresh must be >= 1, got {self.min_refresh}')
        if self.max_refresh > FORECAST_HORIZON:
            raise ValueError(
                f'max_refresh must be <= {FORECAST_HORIZON} (forecast horizon), '
                f'got {self.max_refresh}')
        if self.min_refresh > self.max_refresh:
            raise ValueError(
                f'min_refresh ({self.min_refresh}) must be <= '
                f'max_refresh ({self.max_refresh})')


# ---------------------------------------------------------------------------
# ModelHandler
# ---------------------------------------------------------------------------

class TimesFMModelHandler(ModelHandler[dict, PredictionResult, Any]):
    """TimesFM model handler. Returns full 128-step forecast."""

    def __init__(self, model_name='google/timesfm-2.5-200m-transformers',
                 expected_interval=1.0,
                 force_flip_invariance=True, truncate_negative=True, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._expected_interval = expected_interval
        self._force_flip_invariance = force_flip_invariance
        self._truncate_negative = truncate_negative

    def load_model(self):
        import torch
        from transformers import TimesFm2_5ModelForPrediction
        _LOGGER.warning('[TimesFM] Loading model from %s', self._model_name)
        t0 = time.time()
        model = TimesFm2_5ModelForPrediction.from_pretrained(self._model_name)
        model.eval()
        model.requires_grad_(False)
        _LOGGER.warning('[TimesFM] Model loaded in %.1fs', time.time() - t0)
        return model

    def _prepare_context(self, context):
        """Prepare a context array for TimesFM inference.

        TimesFM requires evenly-spaced input divisible by patch_size=32.

        If context is a list of (timestamp, value) tuples:
        1. Resample onto an evenly-spaced grid (gaps become NaN)
        2. Strip leading NaNs
        3. Linearly interpolate interior NaNs
        4. Truncate to nearest multiple of 32

        If context is a flat np.ndarray (already evenly spaced):
        1. Just patch-align by truncation

        Adapted from the native TimesFM package's preprocessing.
        """
        if isinstance(context, np.ndarray):
            n = len(context)
            aligned = (n // PATCH_SIZE) * PATCH_SIZE
            return context[-aligned:] if aligned > 0 else context

        # Context is list of (ts, value) — resample, interpolate, align
        grid = _resample_with_nans(context, self._expected_interval)
        grid = _strip_leading_nans(grid)
        grid = _linear_interpolation(grid)
        values = grid.astype(np.float32)
        aligned = (len(values) // PATCH_SIZE) * PATCH_SIZE
        return values[-aligned:] if aligned > 0 else values

    def run_inference(self, batch, model, inference_args=None):
        import torch

        # Prepare contexts: fill gaps and patch-align
        prepared = []
        for item in batch:
            ctx = item['context']
            prepared.append(self._prepare_context(ctx))

        _LOGGER.info('[TimesFM] run_inference: batch_size=%d context_lens=%s',
                     len(batch), [len(c) for c in prepared])

        for ctx in prepared:
            if len(ctx) % PATCH_SIZE != 0:
                raise ValueError(
                    f'Context length {len(ctx)} not divisible by {PATCH_SIZE}.')

        past_values = [torch.tensor(c, dtype=torch.float32) for c in prepared]
        fcl = max(len(c) for c in prepared)
        t0 = time.time()
        with torch.no_grad():
            outputs = model(
                past_values=past_values, forecast_context_len=fcl,
                force_flip_invariance=self._force_flip_invariance,
                truncate_negative=self._truncate_negative)
        _LOGGER.info('[TimesFM] inference done in %.0fms',
                     (time.time() - t0) * 1000)

        point_fc = outputs.mean_predictions.numpy()
        quantile_fc = outputs.full_predictions.numpy()
        return [
            PredictionResult(
                example=item,
                inference={'point_forecast': point_fc[i],
                           'quantile_forecast': quantile_fc[i]},
                model_id='TimesFM-2.5')
            for i, item in enumerate(batch)
        ]

    def get_num_bytes(self, batch):
        return sum(
            item['context'].nbytes if isinstance(item['context'], np.ndarray)
            else len(item['context']) * 8
            for item in batch)

    def get_metrics_namespace(self):
        return 'BeamML_TimesFM'


# ---------------------------------------------------------------------------
# BufferDoFn
# ---------------------------------------------------------------------------

class TimesFMBufferDoFn(beam.DoFn):
    """Buffers per-key metric values using BagState with event-time timers.

    On ``process()``: buffers the element and sets a timer for the
    **earliest unprocessed** element timestamp. On ``on_timer()``:
    emits ALL observations with timestamps up to the fire time.

    All buffer entries, timers, and last_processed state use
    ``element_timestamp`` (Beam's element timestamp, typically
    window_end - 1ms). This is the same timestamp space the watermark
    tracks, so timer scheduling and unprocessed filtering are simple
    comparisons with no coordinate-system bridging.

    The ``row.window_start`` is preserved inside the row for downstream
    use (context building, cursor computation) but is not used for
    buffer ordering or timer logic.

    Outputs:
        'inference': (key, {"context": list[(ts,val)]}) — for RunInference
        'observe':   (key, {"observed": float, "row": beam.Row, "step": int})
        'warmup':    (key, AnomalyResult) or AnomalyResult
    """

    BUFFER_BAG = BagStateSpec('buffer', PickleCoder())
    ROWS_BAG = BagStateSpec('rows', PickleCoder())
    STEP_COUNTER = ReadModifyWriteStateSpec('step', beam.coders.VarIntCoder())
    NEXT_REFRESH = ReadModifyWriteStateSpec(
        'next_refresh', beam.coders.VarIntCoder())
    LAST_PROCESSED_TS = ReadModifyWriteStateSpec(
        'last_processed', PickleCoder())
    TIMER_TARGET = ReadModifyWriteStateSpec(
        'timer_target', PickleCoder())
    EMIT_TIMER = beam.transforms.userstate.TimerSpec(
        'emit', beam.transforms.timeutil.TimeDomain.WATERMARK)

    def __init__(self, min_context=128, max_context=1024, confidence=90,
                 expected_interval=None, min_refresh=5, max_refresh=10,
                 verbose=False):
        self._min_context = max(PATCH_SIZE, min_context)
        self._max_context = (max_context // PATCH_SIZE) * PATCH_SIZE
        self._confidence = confidence
        self._expected_interval = expected_interval
        self._min_refresh = min_refresh
        self._max_refresh = max_refresh
        self._verbose = verbose

    def process(self, element,
                buffer_bag=beam.DoFn.StateParam(BUFFER_BAG),
                rows_bag=beam.DoFn.StateParam(ROWS_BAG),
                last_processed=beam.DoFn.StateParam(LAST_PROCESSED_TS),
                timer_target=beam.DoFn.StateParam(TIMER_TARGET),
                emit_timer=beam.DoFn.TimerParam(EMIT_TIMER),
                key=beam.DoFn.KeyParam,
                element_timestamp=beam.DoFn.TimestampParam):
        """Buffer the element and set timer for earliest unprocessed."""
        if isinstance(element, tuple) and len(element) == 2:
            key, row = element
        else:
            key, row = None, element

        # Buffer keyed by element_timestamp (what the watermark tracks).
        # The row carries window_start for downstream context building.
        buffer_bag.add((element_timestamp, row.value))
        rows_bag.add((element_timestamp, row))

        if self._verbose:
            bag_size = len(list(buffer_bag.read()))
            _LOGGER.warning(
                '[TimesFM Buffer] process: key=%s ts=%s '
                'value=%.2f bag_size=%d',
                key, element_timestamp, row.value, bag_size)

        # Set timer to earliest unprocessed element timestamp.
        lp = last_processed.read()
        is_new = lp is None or element_timestamp > lp

        if is_new:
            current_target = timer_target.read()
            if current_target is None or element_timestamp < current_target:
                timer_target.write(element_timestamp)
                emit_timer.set(element_timestamp)
                if self._verbose:
                    _LOGGER.warning(
                        '[TimesFM Buffer] TIMER SET to %s',
                        element_timestamp)

    @beam.transforms.userstate.on_timer(EMIT_TIMER)
    def on_emit(self,
                buffer_bag=beam.DoFn.StateParam(BUFFER_BAG),
                rows_bag=beam.DoFn.StateParam(ROWS_BAG),
                step_counter=beam.DoFn.StateParam(STEP_COUNTER),
                next_refresh=beam.DoFn.StateParam(NEXT_REFRESH),
                last_processed=beam.DoFn.StateParam(LAST_PROCESSED_TS),
                timer_target=beam.DoFn.StateParam(TIMER_TARGET),
                emit_timer=beam.DoFn.TimerParam(EMIT_TIMER),
                key=beam.DoFn.KeyParam,
                fire_timestamp=beam.DoFn.TimestampParam):
        """Fires when watermark passes. Emit ALL unprocessed observations."""
        import random

        lp = last_processed.read() or _MIN_TIMESTAMP

        all_entries = sorted(buffer_bag.read(), key=lambda x: x[0])
        all_rows = sorted(rows_bag.read(), key=lambda x: x[0])
        row_by_ts = {ts: r for ts, r in all_rows}

        # Everything with ts > last_processed and ts <= fire_timestamp
        # is safe to process (watermark guarantees completeness).
        unprocessed = [(ts, v) for ts, v in all_entries
                       if ts > lp and ts <= fire_timestamp]

        # Trim: keep max_context entries before the oldest unprocessed
        # element (so it has full context for inference), plus all
        # unprocessed and future entries. Only already-processed entries
        # beyond max_context distance are discarded.
        # This prevents the trim from starving the timer chain of context
        # when it's catching up through a large backlog.
        if unprocessed:
            oldest_unprocessed_idx = next(
                i for i, (ts, _) in enumerate(all_entries)
                if ts == unprocessed[0][0])
            trim_start = max(0, oldest_unprocessed_idx - self._max_context)
        else:
            trim_start = max(0, len(all_entries) - self._max_context)

        if trim_start > 0:
            cutoff_ts = all_entries[trim_start][0]
            all_entries = all_entries[trim_start:]
            buffer_bag.clear()
            for entry in all_entries:
                buffer_bag.add(entry)
            all_rows = [(ts, r) for ts, r in all_rows if ts >= cutoff_ts]
            rows_bag.clear()
            for entry in all_rows:
                rows_bag.add(entry)

        if not unprocessed:
            return

        if self._verbose:
            _LOGGER.warning(
                '[TimesFM Buffer] key=%s on_timer: fire_ts=%s '
                'unprocessed=%d buffer=%d lp=%s',
                key, fire_timestamp, len(unprocessed), len(all_entries), lp)

        step = step_counter.read() or 0

        for entry_ts, val in unprocessed:
            row = row_by_ts.get(entry_ts)
            if row is None:
                continue

            step += 1
            context_len = sum(1 for ts_e, _ in all_entries
                              if ts_e < entry_ts)

            # Warmup
            if context_len < self._min_context:
                prediction = AnomalyPrediction(
                    model_id='TimesFM-2.5+ZScore', score=None, label=-2,
                    info=f'warmup: {context_len}/{self._min_context}')
                result = AnomalyResult(example=row, predictions=[prediction])
                if key is not None:
                    yield beam.pvalue.TaggedOutput('warmup', (key, result))
                else:
                    yield beam.pvalue.TaggedOutput('warmup', result)
                continue

            # Emit observation
            yield beam.pvalue.TaggedOutput('observe', (key, {
                _TAG_OBSERVATION: True,
                'observed': float(val),
                'row': row,
                'step': step,
            }))

            # Emit inference if refresh is due
            refresh_at = next_refresh.read()
            if refresh_at is None or step >= refresh_at:
                # Context uses window_start from the row for time-series
                # alignment (what TimesFM needs for interpolation).
                ctx_entries = [
                    (row_by_ts[ts_e].window_start, v_e)
                    for ts_e, v_e in all_entries
                    if ts_e < entry_ts and ts_e in row_by_ts]
                if len(ctx_entries) >= 2:
                    context_end_ts = ctx_entries[-1][0]
                    interval = random.randint(
                        self._min_refresh, self._max_refresh)
                    next_refresh.write(step + interval)

                    if self._verbose:
                        _LOGGER.warning(
                            '[TimesFM Buffer] key=%s INFERENCE: step=%d '
                            'raw_ctx=%d ctx_end=%s obs_ts=%s',
                            key, step, len(ctx_entries),
                            context_end_ts, row.window_start)

                    yield beam.pvalue.TaggedOutput('inference', (key, {
                        'context': ctx_entries,
                        'context_end_ts': context_end_ts,
                        'expected_interval': self._expected_interval,
                    }))

        step_counter.write(step)
        last_processed.write(fire_timestamp)

        # Reschedule timer for next unprocessed element.
        remaining = [(ts, v) for ts, v in all_entries
                     if ts > fire_timestamp]
        if remaining:
            next_ts = remaining[0][0]
            timer_target.write(next_ts)
            emit_timer.set(next_ts)
            if self._verbose:
                _LOGGER.warning(
                    '[TimesFM Buffer] TIMER RESCHEDULED to %s '
                    '(%d remaining)', next_ts, len(remaining))
        else:
            timer_target.write(None)


class TimesFMBufferDoFnOLS(beam.DoFn):
    """Same as TimesFMBufferDoFn but uses OrderedListState."""

    BUFFER = OrderedListStateSpec('buffer', FastPrimitivesCoder())
    ROWS = OrderedListStateSpec('rows', PickleCoder())
    STEP_COUNTER = ReadModifyWriteStateSpec('step', beam.coders.VarIntCoder())
    NEXT_REFRESH = ReadModifyWriteStateSpec(
        'next_refresh', beam.coders.VarIntCoder())
    LAST_PROCESSED_TS = ReadModifyWriteStateSpec(
        'last_processed', PickleCoder())
    TIMER_TARGET = ReadModifyWriteStateSpec(
        'timer_target', PickleCoder())
    EMIT_TIMER = beam.transforms.userstate.TimerSpec(
        'emit', beam.transforms.timeutil.TimeDomain.WATERMARK)

    def __init__(self, min_context=128, max_context=1024, confidence=90,
                 expected_interval=None, min_refresh=5, max_refresh=10,
                 verbose=False):
        self._min_context = max(PATCH_SIZE, min_context)
        self._max_context = (max_context // PATCH_SIZE) * PATCH_SIZE
        self._confidence = confidence
        self._expected_interval = expected_interval
        self._min_refresh = min_refresh
        self._max_refresh = max_refresh
        self._verbose = verbose

    def process(self, element,
                buffer=beam.DoFn.StateParam(BUFFER),
                rows=beam.DoFn.StateParam(ROWS),
                last_processed=beam.DoFn.StateParam(LAST_PROCESSED_TS),
                timer_target=beam.DoFn.StateParam(TIMER_TARGET),
                emit_timer=beam.DoFn.TimerParam(EMIT_TIMER),
                key=beam.DoFn.KeyParam,
                element_timestamp=beam.DoFn.TimestampParam):
        if isinstance(element, tuple) and len(element) == 2:
            key, row = element
        else:
            key, row = None, element

        buffer.add((element_timestamp, row.value))
        rows.add((element_timestamp, row))

        lp = last_processed.read() or _MIN_TIMESTAMP

        if element_timestamp > lp:
            current_target = timer_target.read()
            if current_target is None or element_timestamp < current_target:
                timer_target.write(element_timestamp)
                emit_timer.set(element_timestamp)

    @beam.transforms.userstate.on_timer(EMIT_TIMER)
    def on_emit(self,
                buffer=beam.DoFn.StateParam(BUFFER),
                rows=beam.DoFn.StateParam(ROWS),
                step_counter=beam.DoFn.StateParam(STEP_COUNTER),
                next_refresh=beam.DoFn.StateParam(NEXT_REFRESH),
                last_processed=beam.DoFn.StateParam(LAST_PROCESSED_TS),
                timer_target=beam.DoFn.StateParam(TIMER_TARGET),
                emit_timer=beam.DoFn.TimerParam(EMIT_TIMER),
                key=beam.DoFn.KeyParam,
                fire_timestamp=beam.DoFn.TimestampParam):
        import random

        lp = last_processed.read() or _MIN_TIMESTAMP

        all_entries = list(buffer.read())
        all_rows = list(rows.read())
        row_by_ts = {ts: r for ts, r in all_rows}

        unprocessed = [(ts, v) for ts, v in all_entries
                       if ts > lp and ts <= fire_timestamp]

        # Trim: keep max_context entries before the oldest unprocessed.
        if unprocessed:
            oldest_unprocessed_idx = next(
                i for i, (ts, _) in enumerate(all_entries)
                if ts == unprocessed[0][0])
            trim_start = max(0, oldest_unprocessed_idx - self._max_context)
        else:
            trim_start = max(0, len(all_entries) - self._max_context)

        if trim_start > 0:
            cutoff_ts = all_entries[trim_start][0]
            buffer.clear_range(Timestamp(0), cutoff_ts)
            all_entries = all_entries[trim_start:]

        if not unprocessed:
            return

        if self._verbose:
            _LOGGER.warning(
                '[TimesFM Buffer OLS] key=%s on_timer: fire_ts=%s '
                'unprocessed=%d buffer=%d',
                key, fire_timestamp, len(unprocessed), len(all_entries))

        step = step_counter.read() or 0

        for entry_ts, val in unprocessed:
            row = row_by_ts.get(entry_ts)
            if row is None:
                continue

            step += 1
            context_len = sum(1 for ts_e, _ in all_entries
                              if ts_e < entry_ts)

            if context_len < self._min_context:
                prediction = AnomalyPrediction(
                    model_id='TimesFM-2.5+ZScore', score=None, label=-2,
                    info=f'warmup: {context_len}/{self._min_context}')
                result = AnomalyResult(example=row, predictions=[prediction])
                if key is not None:
                    yield beam.pvalue.TaggedOutput('warmup', (key, result))
                else:
                    yield beam.pvalue.TaggedOutput('warmup', result)
                continue

            yield beam.pvalue.TaggedOutput('observe', (key, {
                _TAG_OBSERVATION: True,
                'observed': float(val),
                'row': row,
                'step': step,
            }))

            refresh_at = next_refresh.read()
            if refresh_at is None or step >= refresh_at:
                ctx_entries = [
                    (row_by_ts[ts_e].window_start, v_e)
                    for ts_e, v_e in all_entries
                    if ts_e < entry_ts and ts_e in row_by_ts]
                if len(ctx_entries) >= 2:
                    context_end_ts = ctx_entries[-1][0]
                    interval = random.randint(
                        self._min_refresh, self._max_refresh)
                    next_refresh.write(step + interval)

                    yield beam.pvalue.TaggedOutput('inference', (key, {
                        'context': ctx_entries,
                        'context_end_ts': context_end_ts,
                        'expected_interval': self._expected_interval,
                    }))

        step_counter.write(step)
        last_processed.write(fire_timestamp)

        remaining = [(ts, v) for ts, v in all_entries
                     if ts > fire_timestamp]
        if remaining:
            next_ts = remaining[0][0]
            timer_target.write(next_ts)
            emit_timer.set(next_ts)
            if self._verbose:
                _LOGGER.warning(
                    '[TimesFM Buffer OLS] TIMER RESCHEDULED to %s '
                    '(%d remaining)', next_ts, len(remaining))
        else:
            timer_target.write(None)


# ---------------------------------------------------------------------------
# ExtractPredictions DoFn
# ---------------------------------------------------------------------------

class TimesFMExtractPredictionsDoFn(beam.DoFn):
    """Extracts full forecast from PredictionResult and tags for merging.

    Input:  (key, PredictionResult) from KeyedModelHandler
    Output: (key, {__prediction__: True, point_forecast: [...], quantile_forecast: [...]})
    """

    def __init__(self, confidence=90):
        self._upper_idx = confidence // 10
        self._lower_idx = 10 - self._upper_idx

    def process(self, element):
        key, prediction_result = element
        inference = prediction_result.inference
        example = prediction_result.example

        yield (key, {
            _TAG_PREDICTION: True,
            'point_forecast': inference['point_forecast'],       # [128]
            'quantile_forecast': inference['quantile_forecast'],  # [128, 10]
            'upper_idx': self._upper_idx,
            'lower_idx': self._lower_idx,
            'context_end_ts': example.get('context_end_ts'),
            'expected_interval': example.get('expected_interval'),
        })


# ---------------------------------------------------------------------------
# CacheAndScoreDoFn (stateful)
# ---------------------------------------------------------------------------

class TimesFMCacheAndScoreDoFn(beam.DoFn):
    """Stateful DoFn that caches predictions and scores observations.

    Receives two types of elements (distinguished by tag):
    - Predictions: full 128-step forecast from RunInference, with
      ``context_end_ts`` and ``expected_interval`` metadata
    - Observations: individual observed values from BufferDoFn

    Cursor computation is timestamp-based: for an observation at time T,
    the cursor into the 128-step forecast is::

        cursor = round((T - context_end_ts) / expected_interval) - 1

    This ensures correct alignment regardless of element arrival order
    or gaps between CDC polling intervals.

    Output: (key, beam.Row(value=residual, ...)) for ZScore detection
    """

    CACHE = ReadModifyWriteStateSpec('cache', PickleCoder())
    PENDING = BagStateSpec('pending', PickleCoder())

    def __init__(self, verbose=False):
        self._verbose = verbose

    def process(self, element,
                cache=beam.DoFn.StateParam(CACHE),
                pending=beam.DoFn.StateParam(PENDING),
                key=beam.DoFn.KeyParam):
        if isinstance(element, tuple) and len(element) == 2:
            key, data = element
        else:
            key, data = None, element

        if _TAG_PREDICTION in data:
            # New predictions arrived — cache them with timing metadata
            cached = {
                'point': data['point_forecast'],
                'quantile': data['quantile_forecast'],
                'upper_idx': data['upper_idx'],
                'lower_idx': data['lower_idx'],
                'context_end_ts': Timestamp.of(
                    data.get('context_end_ts', 0)),
                'expected_interval': data.get('expected_interval', 1.0),
            }
            cache.write(cached)

            if self._verbose:
                _LOGGER.warning(
                    '[TimesFM Cache] key=%s received predictions: '
                    'context_end_ts=%s interval=%.1f point[0:5]=[%s]',
                    key, cached['context_end_ts'],
                    cached['expected_interval'],
                    ', '.join(f'{float(v):.2f}'
                              for v in data['point_forecast'][:5]))

            # Score any pending observations
            pending_obs = sorted(pending.read(),
                                 key=lambda x: x['row'].window_start)
            pending.clear()

            for obs in pending_obs:
                result = self._score(key, obs, cached)
                if result is not None:
                    yield result

        elif _TAG_OBSERVATION in data:
            # Observation — try to score from cache
            cached = cache.read()

            if cached is not None:
                result = self._score(key, data, cached)
                if result is not None:
                    yield result
                else:
                    # Cursor out of range — queue for next prediction
                    pending.add(data)
            else:
                # No cache yet — queue for later
                pending.add(data)
                if self._verbose:
                    _LOGGER.warning(
                        '[TimesFM Cache] key=%s queued observation '
                        '(no cache)', key)

    def _score(self, key, obs_data, cached):
        """Score an observation against cached predictions.

        Computes the cursor from the observation's timestamp relative to
        the context end timestamp. Returns None if the cursor is out of
        the forecast horizon [0, 128).
        """
        observed = obs_data['observed']
        row = obs_data['row']
        obs_ts = row.window_start
        context_end_ts = cached['context_end_ts']
        expected_interval = cached['expected_interval'] or 1.0

        # Compute cursor: how many intervals after context_end is this
        # observation?
        # prediction[0] = forecast for context_end + 1*interval
        # prediction[n] = forecast for context_end + (n+1)*interval
        offset_secs = float(obs_ts - context_end_ts)
        cursor = round(offset_secs / expected_interval) - 1

        if cursor < 0 or cursor >= FORECAST_HORIZON:
            _LOGGER.warning(
                '[TimesFM Score] key=%s obs_ts=%s context_end=%s '
                'interval=%.1f cursor=%d OUT OF RANGE',
                key, obs_ts, context_end_ts, expected_interval, cursor)
            return None

        point = float(cached['point'][cursor])
        quantiles = cached['quantile'][cursor]
        upper = float(quantiles[cached['upper_idx']])
        lower = float(quantiles[cached['lower_idx']])
        residual = point - observed

        if self._verbose:
            _LOGGER.warning(
                '[TimesFM Score] key=%s observed=%.4f predicted=%.4f '
                'residual=%.4f cursor=%d (obs_ts=%s ctx_end=%s) '
                'P10=%.4f P90=%.4f step=%s',
                key, observed, point, residual, cursor,
                obs_ts, context_end_ts,
                lower, upper, obs_data.get('step', '?'))

        residual_row = beam.Row(
            value=residual,
            window_start=row.window_start,
            window_end=row.window_end,
            timesfm_observed=observed,
            timesfm_predicted=point,
            timesfm_lower=lower,
            timesfm_upper=upper,
        )

        if key is not None:
            return (key, residual_row)
        return residual_row


# ---------------------------------------------------------------------------
# EnrichDoFn
# ---------------------------------------------------------------------------

class TimesFMEnrichDoFn(beam.DoFn):
    """Enriches AnomalyResult from ZScore with TimesFM prediction metadata."""

    def __init__(self, verbose=False):
        self._verbose = verbose

    def process(self, element):
        key, result = _unpack_result(element)
        prediction = result.predictions[0]
        example = result.example

        observed = getattr(example, 'timesfm_observed', None)
        predicted = getattr(example, 'timesfm_predicted', None)
        lower = getattr(example, 'timesfm_lower', None)
        upper = getattr(example, 'timesfm_upper', None)
        residual = example.value

        zscore_info = prediction.info or ''
        timesfm_info = ''
        if predicted is not None:
            timesfm_info = (f'predicted={predicted:.4f} '
                            f'bounds=[{lower:.4f}, {upper:.4f}] '
                            f'residual={residual:.4f}')
        combined_info = f'{timesfm_info} | {zscore_info}' if zscore_info else timesfm_info

        original_row = beam.Row(
            value=observed if observed is not None else residual,
            window_start=example.window_start,
            window_end=example.window_end,
        )

        enriched_prediction = AnomalyPrediction(
            model_id='TimesFM-2.5+ZScore',
            score=prediction.score,
            label=prediction.label,
            threshold=prediction.threshold,
            info=combined_info,
        )

        enriched_result = AnomalyResult(
            example=original_row,
            predictions=[enriched_prediction],
        )

        if self._verbose:
            _LOGGER.warning(
                '[TimesFM+ZScore] key=%s observed=%.4f predicted=%.4f '
                'residual=%.4f zscore=%s label=%d',
                key,
                observed if observed is not None else 0,
            predicted if predicted is not None else 0,
            residual,
            prediction.score,
            prediction.label if prediction.label is not None else -2)

        if key is not None:
            yield (key, enriched_result)
        else:
            yield enriched_result


def _unpack_result(element):
    if isinstance(element, tuple) and len(element) == 2:
        return element[0], element[1]
    return None, element


# ---------------------------------------------------------------------------
# Debug BufferV1 — structurally identical to TimesFMBufferDoFn but minimal
# ---------------------------------------------------------------------------

class DebugBufferV1(beam.DoFn):
    """Debug copy of BufferDoFn in same module as TimesFMBufferDoFn."""

    BUFFER_BAG = BagStateSpec('buffer', PickleCoder())
    ROWS_BAG = BagStateSpec('rows', PickleCoder())
    STEP_COUNTER = ReadModifyWriteStateSpec('step', beam.coders.VarIntCoder())
    NEXT_REFRESH = ReadModifyWriteStateSpec(
        'next_refresh', beam.coders.VarIntCoder())
    LAST_PROCESSED_TS = ReadModifyWriteStateSpec(
        'last_processed', PickleCoder())
    TIMER_TARGET = ReadModifyWriteStateSpec('timer_target', PickleCoder())
    EMIT_TIMER = beam.transforms.userstate.TimerSpec(
        'emit', beam.transforms.timeutil.TimeDomain.WATERMARK)

    def __init__(self, min_context=128, max_context=1024, confidence=90,
                 expected_interval=None, min_refresh=5, max_refresh=10):
        self._min_context = max(PATCH_SIZE, min_context)
        self._max_context = (max_context // PATCH_SIZE) * PATCH_SIZE
        self._confidence = confidence
        self._expected_interval = expected_interval
        self._min_refresh = min_refresh
        self._max_refresh = max_refresh

    def process(self, element,
                buffer_bag=beam.DoFn.StateParam(BUFFER_BAG),
                rows_bag=beam.DoFn.StateParam(ROWS_BAG),
                last_processed=beam.DoFn.StateParam(LAST_PROCESSED_TS),
                timer_target=beam.DoFn.StateParam(TIMER_TARGET),
                emit_timer=beam.DoFn.TimerParam(EMIT_TIMER),
                key=beam.DoFn.KeyParam,
                element_timestamp=beam.DoFn.TimestampParam):
        if isinstance(element, tuple) and len(element) == 2:
            key, row = element
        else:
            key, row = None, element

        ws = row.window_start
        buffer_bag.add((ws, row.value))
        rows_bag.add((ws, row))

        bag_size = len(list(buffer_bag.read()))

        lp = last_processed.read()
        is_new = lp is None or element_timestamp > lp

        if is_new:
            current_target = timer_target.read()
            if current_target is None or element_timestamp < current_target:
                timer_target.write(element_timestamp)
                emit_timer.set(element_timestamp)
                _LOGGER.warning(
                    '[DebugV1] TIMER SET to %s bag=%d',
                    element_timestamp, bag_size)

    @beam.transforms.userstate.on_timer(EMIT_TIMER)
    def on_emit(self,
                buffer_bag=beam.DoFn.StateParam(BUFFER_BAG),
                rows_bag=beam.DoFn.StateParam(ROWS_BAG),
                step_counter=beam.DoFn.StateParam(STEP_COUNTER),
                next_refresh=beam.DoFn.StateParam(NEXT_REFRESH),
                last_processed=beam.DoFn.StateParam(LAST_PROCESSED_TS),
                timer_target=beam.DoFn.StateParam(TIMER_TARGET),
                emit_timer=beam.DoFn.TimerParam(EMIT_TIMER),
                key=beam.DoFn.KeyParam,
                fire_timestamp=beam.DoFn.TimestampParam):
        lp = last_processed.read() or _MIN_TIMESTAMP
        fire_cutoff = fire_timestamp + Duration(micros=1000)

        all_entries = sorted(buffer_bag.read(), key=lambda x: x[0])
        all_rows = sorted(rows_bag.read(), key=lambda x: x[0])
        row_by_ts = {rts: r for rts, r in all_rows}

        # Trim buffer to max_context
        max_entries = self._max_context + 1
        if len(all_entries) > max_entries:
            cutoff_ts = all_entries[-max_entries][0]
            all_entries = all_entries[-max_entries:]
            buffer_bag.clear()
            for entry in all_entries:
                buffer_bag.add(entry)
            all_rows = [(rts, r) for rts, r in all_rows if rts >= cutoff_ts]
            rows_bag.clear()
            for entry in all_rows:
                rows_bag.add(entry)

        unprocessed = [(ts, v) for ts, v in all_entries
                       if ts > lp and ts <= fire_cutoff]

        if not unprocessed:
            return

        _LOGGER.warning(
            '[DebugV1] ON_TIMER: fire_ts=%s lp=%s '
            'buffer=%d unprocessed=%d',
            fire_timestamp, lp, len(all_entries), len(unprocessed))

        step = step_counter.read() or 0
        for entry_ts, val in unprocessed:
            row = row_by_ts.get(entry_ts)
            if row is None:
                continue
            step += 1
            context_len = sum(1 for ts_e, _ in all_entries
                              if ts_e < entry_ts)
            if context_len < self._min_context:
                prediction = AnomalyPrediction(
                    model_id='Test', score=None, label=-2,
                    info=f'warmup: {context_len}/{self._min_context}')
                result = AnomalyResult(example=row, predictions=[prediction])
                yield beam.pvalue.TaggedOutput('warmup', (key, result))
                continue

            yield beam.pvalue.TaggedOutput('observe', (key, {
                'observed': float(val), 'row': row, 'step': step}))

            import random
            refresh_at = next_refresh.read()
            if refresh_at is None or step >= refresh_at:
                ctx_entries = [(ts_e, v_e) for ts_e, v_e in all_entries
                               if ts_e < entry_ts]
                if len(ctx_entries) >= 2:
                    interval = random.randint(
                        self._min_refresh, self._max_refresh)
                    next_refresh.write(step + interval)
                    yield beam.pvalue.TaggedOutput('inference', (key, {
                        'context': ctx_entries,
                        'context_end_ts': ctx_entries[-1][0],
                        'expected_interval': self._expected_interval}))

        step_counter.write(step)
        last_processed.write(fire_timestamp)

        remaining = [(ts, v) for ts, v in all_entries
                     if ts > fire_timestamp]
        if remaining:
            next_timer = remaining[0][0] + Duration(micros=999000)
            timer_target.write(next_timer)
            emit_timer.set(next_timer)
        else:
            timer_target.write(None)
