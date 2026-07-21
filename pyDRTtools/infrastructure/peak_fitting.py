# -*- coding: utf-8 -*-
"""Process-backed Peak One/Peak All execution."""

import os

from .parallel_fitting import (
    FIT_ALL_NATIVE_THREADS_PER_PROCESS,
    FitProcessManager,
    _disable_cvxopt_progress,
    _native_thread_limit,
    _silence_process_output,
)


def peak_process_task(key, entry, _mode, params, signature):
    """Analyze one spectrum in a child process without importing Qt."""
    native_threads = max(1, int(params.get('native_threads', 1) or 1))
    for name in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
    ):
        os.environ[name] = str(native_threads)

    with _silence_process_output():
        _disable_cvxopt_progress()
        with _native_thread_limit(native_threads):
            from ..services.peak_results import analyze_peak_entry

            result = analyze_peak_entry(entry, params)
    return key, result, signature


class PeakProcessManager(FitProcessManager):
    """Reuse the proven adaptive scheduler with a peak-analysis task."""

    def start(self, tasks, params, signature):
        return super().start(tasks, 'peak', params, signature)

    def _submit_until_limit(self):
        if self.executor is None:
            return
        while self.pending_keys and len(self.futures) < self.worker_count:
            key = self.pending_keys.popleft()
            if key not in self.tasks:
                continue
            future = self.executor.submit(
                peak_process_task,
                key,
                self.tasks[key],
                self.mode,
                self.params,
                self.signature,
            )
            self.futures[future] = key
