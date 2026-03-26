"""TimesFM anomaly detector for the BigQuery anomaly detection pipeline.

Uses Google's TimesFM 2.5 foundation model (via HuggingFace transformers)
for zero-shot time-series forecasting and anomaly detection based on
quantile prediction intervals.

Pipeline integration::

    ReadCDC -> ComputeMetric -> WindowInto(GlobalWindows)
           -> TimesFMBufferDoFn -> RunInference(KeyedModelHandler(...))
           -> TimesFMScoreDoFn -> AnomalyResult

Detector spec::

    {"type": "TimesFM"}
    {"type": "TimesFM", "config": {"max_context": 1024, "confidence": 90}}
"""

import dataclasses
import logging
import time

import numpy as np
import apache_beam as beam
from apache_beam.coders import FastPrimitivesCoder
from apache_beam.transforms.userstate import OrderedListStateSpec
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.utils.timestamp import Timestamp

from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import Optional

_LOGGER = logging.getLogger(__name__)

PATCH_SIZE = 32


def _fill_gaps(entries, expected_interval=None):
    """Fill gaps in a sorted (timestamp, value) sequence via linear interpolation.

    Detects the expected interval from the data if not provided (median of
    consecutive diffs). Any gap larger than 1.5x the expected interval is
    filled with linearly interpolated values at the expected interval spacing.

    Args:
        entries: List of (timestamp, value) tuples, sorted by timestamp.
            Timestamps can be floats (seconds) or Timestamp objects.
        expected_interval: Expected spacing between consecutive entries
            (in seconds). If None, inferred from the median of consecutive
            timestamp diffs.

    Returns:
        List of (timestamp, value) tuples with gaps filled. Original
        entries are preserved; only new interpolated entries are added.
    """
    if len(entries) < 2:
        return entries

    # Convert timestamps to float seconds for arithmetic
    def _to_secs(ts):
        return float(ts)

    timestamps = [_to_secs(e[0]) for e in entries]
    values = [e[1] for e in entries]

    # If no interval provided, detect from median of consecutive diffs.
    # This is a fallback — normally the interval is derived from the
    # metric_spec window size in build_pipeline.
    if expected_interval is None:
        if len(timestamps) < 2:
            return entries
        diffs = [timestamps[i + 1] - timestamps[i]
                 for i in range(len(timestamps) - 1)]
        diffs.sort()
        expected_interval = diffs[len(diffs) // 2]  # median

    if expected_interval <= 0:
        return entries

    gap_threshold = expected_interval * 1.5
    filled = []
    n_filled = 0

    for i in range(len(entries)):
        filled.append(entries[i])

        if i < len(entries) - 1:
            gap = timestamps[i + 1] - timestamps[i]
            if gap > gap_threshold:
                # Number of missing steps
                n_missing = round(gap / expected_interval) - 1
                if n_missing > 0:
                    v_start = values[i]
                    v_end = values[i + 1]
                    t_start = timestamps[i]

                    for step in range(1, n_missing + 1):
                        frac = step / (n_missing + 1)
                        interp_ts = t_start + step * expected_interval
                        interp_val = v_start + (v_end - v_start) * frac
                        filled.append((interp_ts, interp_val))
                        n_filled += 1

    if n_filled > 0:
        _LOGGER.warning(
            '[TimesFM] _fill_gaps: filled %d missing values '
            '(interval=%.1fs, %d -> %d entries)',
            n_filled, expected_interval, len(entries), len(filled))

    return filled


@dataclasses.dataclass(frozen=True)
class _TimesFMDetectorConfig:
    """Lightweight config container for TimesFM detector.

    Parsed by ``_parse_detector_spec`` in ``pipeline.py`` and used to
    construct the handler, buffer, and scorer in ``build_pipeline``.
    """
    model_name: str = 'google/timesfm-2.5-200m-transformers'
    min_context: int = 128
    max_context: int = 1024
    confidence: int = 90
    force_flip_invariance: bool = True
    truncate_negative: bool = True
    use_ordered_list_state: bool = False
    expected_interval: Optional[float] = None  # seconds; None = auto-detect
    zscore_threshold: float = 5.0  # z-score cutoff for residual anomaly detection


# ---------------------------------------------------------------------------
# ModelHandler
# ---------------------------------------------------------------------------

class TimesFMModelHandler(ModelHandler[dict, PredictionResult, Any]):
    """TimesFM model handler. Each call runs the model (no caching).

    Input element: dict with ``"context"`` (np.ndarray, patch-aligned)
        and ``"observed"`` (float).
    Output: PredictionResult with 128-step point and quantile forecasts.
    """

    def __init__(
            self,
            model_name='google/timesfm-2.5-200m-transformers',
            force_flip_invariance=True,
            truncate_negative=True,
            **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._force_flip_invariance = force_flip_invariance
        self._truncate_negative = truncate_negative

    def load_model(self):
        import torch
        from transformers import TimesFm2_5ModelForPrediction

        _LOGGER.warning(
            '[TimesFM] Loading model from %s (device=%s)',
            self._model_name,
            'cuda' if torch.cuda.is_available() else 'cpu')
        t0 = time.time()
        model = TimesFm2_5ModelForPrediction.from_pretrained(self._model_name)
        model.eval()
        model.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters())
        _LOGGER.warning(
            '[TimesFM] Model loaded in %.1fs (%d params, %.0f MB)',
            time.time() - t0, n_params, n_params * 4 / 1e6)
        return model

    def run_inference(self, batch, model, inference_args=None):
        import torch

        batch_size = len(batch)
        contexts = [item['context'] for item in batch]
        context_lens = [len(c) for c in contexts]

        _LOGGER.warning(
            '[TimesFM] run_inference: batch_size=%d context_lens=%s '
            'force_flip=%s truncate_neg=%s',
            batch_size, context_lens,
            self._force_flip_invariance, self._truncate_negative)

        for i, ctx in enumerate(contexts):
            if len(ctx) % PATCH_SIZE != 0:
                raise ValueError(
                    f'Context length {len(ctx)} is not divisible by '
                    f'{PATCH_SIZE}.')
            _LOGGER.warning(
                '[TimesFM]   batch[%d]: ctx_len=%d ctx_tail5=%s observed=%.4f',
                i, len(ctx),
                [round(float(v), 4) for v in ctx[-5:]],
                batch[i]['observed'])

        past_values = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        fcl = max(context_lens)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(
                past_values=past_values,
                forecast_context_len=fcl,
                force_flip_invariance=self._force_flip_invariance,
                truncate_negative=self._truncate_negative,
            )
        infer_ms = (time.time() - t0) * 1000

        point_fc = outputs.mean_predictions.numpy()
        quantile_fc = outputs.full_predictions.numpy()

        _LOGGER.warning(
            '[TimesFM] inference done in %.0fms. '
            'point_shape=%s quantile_shape=%s',
            infer_ms, point_fc.shape, quantile_fc.shape)

        results = []
        for i, item in enumerate(batch):
            point_0 = float(point_fc[i][0])
            quant_0 = quantile_fc[i][0]
            observed = item['observed']

            _LOGGER.warning(
                '[TimesFM]   result[%d]: observed=%.4f predicted=%.4f '
                'P10=%.4f P90=%.4f point_5=[%s] quantile_shape=%s',
                i, observed, point_0,
                float(quant_0[1]), float(quant_0[9]),
                ', '.join(f'{float(v):.4f}' for v in point_fc[i][:5]),
                quantile_fc[i].shape)

            results.append(PredictionResult(
                example=item,
                inference={
                    'point_forecast': point_fc[i],
                    'quantile_forecast': quantile_fc[i],
                },
                model_id='TimesFM-2.5',
            ))
        return results

    def get_num_bytes(self, batch):
        return sum(item['context'].nbytes for item in batch)

    def get_metrics_namespace(self):
        return 'BeamML_TimesFM'


# ---------------------------------------------------------------------------
# BufferDoFn (BagState — default)
# ---------------------------------------------------------------------------

class TimesFMBufferDoFn(beam.DoFn):
    """Buffers per-key metric values using BagState (portable).

    Receives ``(key, beam.Row)`` elements in GlobalWindows (after
    ``ComputeMetric`` + rewindow). Accumulates values and emits context
    arrays for downstream RunInference.

    Output ``'main'``: ``(key, dict)`` with ``context`` (np.ndarray) and
        ``observed`` (float) plus the original ``beam.Row`` in ``row``.
    Output ``'warmup'``: ``AnomalyResult`` with label=-2 (warmup sentinel).
    """

    BUFFER_BAG = beam.transforms.userstate.BagStateSpec(
        'buffer', beam.coders.PickleCoder())

    def __init__(self, min_context=128, max_context=1024, confidence=90,
                 expected_interval=None):
        self._min_context = max(PATCH_SIZE, min_context)
        self._max_context = (max_context // PATCH_SIZE) * PATCH_SIZE
        self._confidence = confidence
        self._expected_interval = expected_interval

    def _emit(self, key, row, all_entries):
        """Build output from sorted (timestamp, value) entries."""
        # Fill gaps via linear interpolation before building context
        all_entries = _fill_gaps(all_entries, self._expected_interval)

        values = [e[1] for e in all_entries]
        observed = values[-1]
        context_len = len(values) - 1

        _LOGGER.warning(
            '[TimesFM Buffer] key=%s buffer_len=%d context_len=%d '
            'min_context=%d max_context=%d observed=%.4f '
            'window_start=%s window_end=%s',
            key, len(all_entries), context_len,
            self._min_context, self._max_context, observed,
            row.window_start, row.window_end)

        if context_len < self._min_context:
            _LOGGER.warning(
                '[TimesFM Buffer] key=%s WARMUP: %d/%d context points',
                key, context_len, self._min_context)
            prediction = AnomalyPrediction(
                model_id='TimesFM-2.5',
                score=None,
                label=-2,
                threshold=None,
                info=f'warmup: {context_len}/{self._min_context} context points')
            result = AnomalyResult(example=row, predictions=[prediction])
            if key is not None:
                return beam.pvalue.TaggedOutput('warmup', (key, result))
            return beam.pvalue.TaggedOutput('warmup', result)

        usable = (context_len // PATCH_SIZE) * PATCH_SIZE
        context = np.array(values[-usable - 1:-1], dtype=np.float32)

        _LOGGER.warning(
            '[TimesFM Buffer] key=%s EMIT: usable_context=%d '
            'ctx_tail5=%s observed=%.4f',
            key, usable,
            [round(float(v), 4) for v in context[-5:]],
            float(observed))

        return beam.pvalue.TaggedOutput('main', (key, {
            'context': context,
            'observed': float(observed),
            'row': row,
        }))

    def process(self, element,
                buffer_bag=beam.DoFn.StateParam(BUFFER_BAG),
                key=beam.DoFn.KeyParam):
        if isinstance(element, tuple) and len(element) == 2:
            key, row = element
        else:
            key, row = None, element

        value = row.value
        ts = row.window_start

        _LOGGER.warning(
            '[TimesFM Buffer] key=%s process: value=%.4f ts=%s',
            key, value, ts)

        buffer_bag.add((ts, value))
        all_entries = sorted(buffer_bag.read(), key=lambda x: x[0])

        max_entries = self._max_context + 1
        if len(all_entries) > max_entries:
            trimmed = len(all_entries) - max_entries
            _LOGGER.warning(
                '[TimesFM Buffer] key=%s TRIM: %d -> %d entries (removed %d)',
                key, len(all_entries), max_entries, trimmed)
            all_entries = all_entries[-max_entries:]
            buffer_bag.clear()
            for entry in all_entries:
                buffer_bag.add(entry)

        yield self._emit(key, row, all_entries)


# ---------------------------------------------------------------------------
# BufferDoFn (OrderedListState — efficient alternative)
# ---------------------------------------------------------------------------

class TimesFMBufferDoFnOLS(beam.DoFn):
    """Same as TimesFMBufferDoFn but uses OrderedListState.

    More efficient (no read-all + sort) but requires runner support
    (Dataflow, FnApiRunner).
    """

    BUFFER = OrderedListStateSpec('buffer', FastPrimitivesCoder())

    def __init__(self, min_context=128, max_context=1024, confidence=90,
                 expected_interval=None):
        self._min_context = max(PATCH_SIZE, min_context)
        self._max_context = (max_context // PATCH_SIZE) * PATCH_SIZE
        self._confidence = confidence
        self._expected_interval = expected_interval

    def _emit(self, key, row, all_entries):
        all_entries = _fill_gaps(all_entries, self._expected_interval)

        values = [e[1] for e in all_entries]
        observed = values[-1]
        context_len = len(values) - 1

        _LOGGER.warning(
            '[TimesFM Buffer OLS] key=%s buffer_len=%d context_len=%d '
            'min_context=%d observed=%.4f window_start=%s',
            key, len(all_entries), context_len,
            self._min_context, observed, row.window_start)

        if context_len < self._min_context:
            _LOGGER.warning(
                '[TimesFM Buffer OLS] key=%s WARMUP: %d/%d',
                key, context_len, self._min_context)
            prediction = AnomalyPrediction(
                model_id='TimesFM-2.5',
                score=None,
                label=-2,
                threshold=None,
                info=f'warmup: {context_len}/{self._min_context} context points')
            result = AnomalyResult(example=row, predictions=[prediction])
            if key is not None:
                return beam.pvalue.TaggedOutput('warmup', (key, result))
            return beam.pvalue.TaggedOutput('warmup', result)

        usable = (context_len // PATCH_SIZE) * PATCH_SIZE
        context = np.array(values[-usable - 1:-1], dtype=np.float32)

        _LOGGER.warning(
            '[TimesFM Buffer OLS] key=%s EMIT: usable=%d ctx_tail5=%s',
            key, usable,
            [round(float(v), 4) for v in context[-5:]])

        return beam.pvalue.TaggedOutput('main', (key, {
            'context': context,
            'observed': float(observed),
            'row': row,
        }))

    def process(self, element,
                buffer=beam.DoFn.StateParam(BUFFER),
                key=beam.DoFn.KeyParam):
        if isinstance(element, tuple) and len(element) == 2:
            key, row = element
        else:
            key, row = None, element

        value = row.value
        ts = Timestamp.of(row.window_start)

        _LOGGER.warning(
            '[TimesFM Buffer OLS] key=%s process: value=%.4f ts=%s',
            key, value, ts)

        buffer.add((ts, value))
        all_entries = list(buffer.read())

        max_entries = self._max_context + 1
        if len(all_entries) > max_entries:
            cutoff_ts = all_entries[-max_entries][0]
            _LOGGER.warning(
                '[TimesFM Buffer OLS] key=%s TRIM: %d -> %d',
                key, len(all_entries), max_entries)
            buffer.clear_range(Timestamp(0), cutoff_ts)
            all_entries = all_entries[-max_entries:]

        yield self._emit(key, row, all_entries)


# ---------------------------------------------------------------------------
# ScoreDoFn
# ---------------------------------------------------------------------------

class TimesFMResidualDoFn(beam.DoFn):
    """Computes residuals from TimesFM forecasts for downstream ZScore detection.

    Receives ``(key, PredictionResult)`` from ``KeyedModelHandler`` and
    emits ``(key, beam.Row)`` where ``value = predicted - observed`` (the
    residual). The Row carries the original window timestamps plus TimesFM
    prediction metadata so the downstream anomaly result can include both
    the statistical score and the model's confidence intervals.

    The residual feeds into a standard anomaly detector (e.g. RobustZScore)
    which learns the normal distribution of prediction errors and flags
    when the error is abnormally large.
    """

    def __init__(self, confidence=90):
        self._upper_idx = confidence // 10       # 90 -> 9 (P90)
        self._lower_idx = 10 - self._upper_idx   # 90 -> 1 (P10)

    def process(self, element):
        key, prediction_result = element
        data = prediction_result.example
        inference = prediction_result.inference

        observed = data['observed']
        row = data['row']

        point = float(inference['point_forecast'][0])
        quantiles = inference['quantile_forecast'][0]
        upper = float(quantiles[self._upper_idx])
        lower = float(quantiles[self._lower_idx])
        residual = point - observed

        _LOGGER.warning(
            '[TimesFM Residual] key=%s observed=%.4f predicted=%.4f '
            'residual=%.4f P10=%.4f P90=%.4f window=[%s, %s]',
            key, observed, point, residual, lower, upper,
            row.window_start, row.window_end)

        # Emit a beam.Row with value=residual for the ZScore detector,
        # plus the original window info and TimesFM metadata in extra fields.
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
            yield (key, residual_row)
        else:
            yield residual_row


class TimesFMEnrichDoFn(beam.DoFn):
    """Enriches AnomalyResult from the ZScore detector with TimesFM metadata.

    The ZScore detector produces AnomalyResult with the residual as the
    example value. This DoFn replaces the example with the original observed
    value and appends TimesFM prediction info to the AnomalyPrediction.
    """

    def process(self, element):
        key, result = _unpack_result(element)
        prediction = result.predictions[0]
        example = result.example

        # Extract TimesFM metadata from the residual Row.
        observed = getattr(example, 'timesfm_observed', None)
        predicted = getattr(example, 'timesfm_predicted', None)
        lower = getattr(example, 'timesfm_lower', None)
        upper = getattr(example, 'timesfm_upper', None)
        residual = example.value

        # Build enriched info string combining ZScore and TimesFM details.
        zscore_info = prediction.info or ''
        timesfm_info = ''
        if predicted is not None:
            timesfm_info = (f'predicted={predicted:.4f} '
                            f'bounds=[{lower:.4f}, {upper:.4f}] '
                            f'residual={residual:.4f}')

        combined_info = f'{timesfm_info} | {zscore_info}' if zscore_info else timesfm_info

        # Reconstruct the example Row with the original observed value
        # (not the residual) for the sink table.
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

        _LOGGER.warning(
            '[TimesFM+ZScore] key=%s observed=%.4f predicted=%.4f '
            'residual=%.4f zscore=%.4f label=%d window=[%s, %s]',
            key,
            observed if observed is not None else 0,
            predicted if predicted is not None else 0,
            residual,
            prediction.score if prediction.score is not None else 0,
            prediction.label if prediction.label is not None else -2,
            example.window_start, example.window_end)

        if key is not None:
            yield (key, enriched_result)
        else:
            yield enriched_result


def _unpack_result(element):
    """Unpack a possibly-keyed AnomalyResult element."""
    if isinstance(element, tuple) and len(element) == 2:
        return element[0], element[1]
    return None, element
