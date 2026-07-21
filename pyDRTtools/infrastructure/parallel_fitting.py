# -*- coding: utf-8 -*-
# FIT_PARALLEL_SAFE_STAGED_ADAPTIVE_20260720
"""Process-based fitting backend for SuperDRTtools.

This module intentionally contains no Qt or Matplotlib imports.  Windows
``spawn`` workers can therefore import the fitting backend without importing
the GUI stack.  Each worker fits one EIS file and uses one native numerical
thread; parallelism comes from running several independent worker processes.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from typing import Dict, Iterable, List, Optional, Tuple


# Safety-first process policy.
#
# A SciPy/CVXOPT worker is memory-heavy during Windows ``spawn``.  Starting
# physical_cores * 1.5 workers at once can exhaust the commit/page-file before
# Python has a chance to report an exception, which may freeze the whole OS.
# Therefore workers are started in stages and never exceed the physical-core
# count by default.  On a 16-core/32-thread CPU this means a safe maximum of 16
# fitting processes; Windows may show about 50% because it reports usage over
# 32 logical processors, even though all 16 physical cores are occupied.
FIT_ALL_PROCESS_HARD_CAP = 32
FIT_ALL_NATIVE_THREADS_PER_PROCESS = 1
FIT_ALL_INITIAL_PHYSICAL_MULTIPLIER = 1
FIT_ALL_MAX_PHYSICAL_MULTIPLIER = 1.0
FIT_ALL_MIN_INITIAL_WORKERS = 2

# Staged ramp-up avoids the simultaneous SciPy DLL import/memory spike.
FIT_ALL_SCALE_STEP = 1
FIT_ALL_WARMUP_SECONDS = 8.0
FIT_ALL_SCALE_INTERVAL_SECONDS = 3.0
FIT_ALL_SCALE_CPU_BELOW_PERCENT = 92.0
FIT_ALL_LOW_CPU_SAMPLES_REQUIRED = 2

# Reduce concurrency gradually after a page-file/DLL resource failure.
FIT_ALL_FALLBACK_STEP = 1

# Commit-memory guard.  A fixed 6 GiB reserve forced otherwise healthy systems
# down to one worker whenever available commit dipped slightly below that
# value.  Use a proportional reserve instead: preserve enough headroom for the
# desktop and GUI while allowing more workers on machines with modest page
# files.  Scale-up remains staged and rechecks commit before every new worker.
FIT_ALL_COMMIT_RESERVE_FRACTION = 0.35
FIT_ALL_MIN_COMMIT_RESERVE_GIB = 2.0
FIT_ALL_MAX_COMMIT_RESERVE_GIB = 6.0
FIT_ALL_START_COMMIT_PER_PROCESS_MIB = 768
FIT_ALL_SCALE_COMMIT_PER_ADDED_PROCESS_MIB = 512

_RESOURCE_ERROR_MARKERS = (
    "页面文件太小",
    "paging file is too small",
    "not enough memory resources",
    "insufficient system resources",
    "dll load failed",
    "commitment limit",
    "out of memory",
    "cannot allocate memory",
    "memoryerror",
)

_CHILD_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _cpu_counts() -> Tuple[int, int]:
    """Return ``(physical, logical)`` CPU counts with safe fallbacks."""
    logical = max(1, int(os.cpu_count() or 1))
    try:
        import psutil

        physical = int(psutil.cpu_count(logical=False) or 0)
    except Exception:
        physical = 0

    if physical <= 0:
        # Most desktop CPUs expose two logical processors per physical core.
        # On a non-SMT CPU this fallback is conservative, while the logical
        # count still acts as the absolute upper bound.
        physical = max(1, logical // 2) if logical > 2 else logical
    return max(1, physical), logical


def recommended_worker_plan(task_count: int) -> Tuple[int, int, int, int]:
    """Return ``(initial, maximum, physical, logical)`` worker counts.

    Workers start at about half the physical cores and ramp up one at a time.
    The default maximum is the physical-core count, not the logical-thread
    count.  This avoids the misleading attempt to reach 100% in Task Manager
    by launching one memory-heavy SciPy process per SMT thread.
    """
    tasks = max(1, int(task_count))
    physical, logical = _cpu_counts()
    maximum_by_physical = int(math.ceil(
        physical * FIT_ALL_MAX_PHYSICAL_MULTIPLIER
    ))
    maximum = max(
        1,
        min(tasks, maximum_by_physical, logical, FIT_ALL_PROCESS_HARD_CAP),
    )
    preferred = int(math.ceil(
        physical * FIT_ALL_INITIAL_PHYSICAL_MULTIPLIER
    ))
    if maximum > 1:
        preferred = max(FIT_ALL_MIN_INITIAL_WORKERS, preferred)
    initial = max(1, min(tasks, preferred, maximum))
    return initial, maximum, physical, logical


def recommended_initial_workers(task_count: int) -> Tuple[int, int, int]:
    """Backward-compatible wrapper retained for external callers."""
    initial, _maximum, physical, _logical = recommended_worker_plan(task_count)
    return initial, physical, FIT_ALL_PROCESS_HARD_CAP


def _available_commit_bytes() -> Optional[int]:
    """Return currently available commit/page-file bytes when detectable."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPageFile)
        except Exception:
            pass

    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return int(vm.available + swap.free)
    except Exception:
        return None


def _memory_safe_worker_count(requested_workers: int) -> Tuple[int, Optional[float]]:
    """Limit the initial spawn to the currently available commit memory.

    This is a guard against an OS-level freeze, not a performance model.  The
    scheduler may still ramp up later after the first workers have loaded and
    actual commit usage is known.
    """
    requested = max(1, int(requested_workers))
    available = _available_commit_bytes()
    if available is None:
        return requested, None

    reserve = _commit_reserve_bytes(available)
    per_worker = int(FIT_ALL_START_COMMIT_PER_PROCESS_MIB * (1024 ** 2))
    usable = max(0, available - reserve)
    allowed = max(1, int(usable // max(1, per_worker)))
    return max(1, min(requested, allowed)), available / (1024 ** 3)


def _commit_reserve_bytes(available: int) -> int:
    """Return a proportional system reserve bounded by safe desktop limits."""
    gib = 1024 ** 3
    minimum = int(FIT_ALL_MIN_COMMIT_RESERVE_GIB * gib)
    maximum = int(FIT_ALL_MAX_COMMIT_RESERVE_GIB * gib)
    proportional = int(max(0, available) * FIT_ALL_COMMIT_RESERVE_FRACTION)
    return min(maximum, max(minimum, proportional))


def _set_worker_below_normal_priority() -> None:
    """Keep Windows responsive while all fitting workers are busy."""
    if os.name != "nt":
        return
    try:
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(
            handle, BELOW_NORMAL_PRIORITY_CLASS
        )
    except Exception:
        pass


def _worker_initializer(native_threads: int) -> None:
    value = str(max(1, int(native_threads)))
    for name in _CHILD_THREAD_ENV_VARS:
        os.environ[name] = value
    _set_worker_below_normal_priority()
    _disable_cvxopt_progress()


def _read_windows_cpu_times():
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            return None

        def as_int(value):
            return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

        return as_int(idle), as_int(kernel), as_int(user)
    except Exception:
        return None


@contextmanager
def _child_spawn_environment(n_threads: int = FIT_ALL_NATIVE_THREADS_PER_PROCESS):
    """Temporarily configure numerical libraries before Windows spawns workers."""
    value = str(max(1, int(n_threads)))
    previous = {name: os.environ.get(name) for name in _CHILD_THREAD_ENV_VARS}
    try:
        for name in _CHILD_THREAD_ENV_VARS:
            os.environ[name] = value
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


@contextmanager
def _native_thread_limit(n_threads: int):
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        yield
        return

    try:
        ctx = threadpool_limits(limits=max(1, int(n_threads)))
    except Exception:
        ctx = nullcontext()
    with ctx:
        yield


@contextmanager
def _silence_process_output():
    """Silence Python and native solver output inside a child process."""
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    saved_fds: Dict[int, int] = {}

    try:
        for stream in (old_stdout, old_stderr):
            try:
                stream.flush()
            except Exception:
                pass

        for fd in (1, 2):
            try:
                saved_fds[fd] = os.dup(fd)
                os.dup2(devnull.fileno(), fd)
            except Exception:
                pass

        sys.stdout = devnull
        sys.stderr = devnull
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        for fd, saved_fd in saved_fds.items():
            try:
                os.dup2(saved_fd, fd)
            except Exception:
                pass
            try:
                os.close(saved_fd)
            except Exception:
                pass
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            devnull.close()
        except Exception:
            pass


def _disable_cvxopt_progress() -> None:
    try:
        from cvxopt import solvers

        solvers.options["show_progress"] = False
    except Exception:
        pass


def _prepare_entry_for_fit(entry):
    if entry is None:
        raise ValueError("Empty entry")
    fit_entry = copy.deepcopy(entry)
    fit_entry.fit_data_source = "raw"
    return fit_entry


def _finalize_fitted_entry(fitted_entry, source_entry):
    if fitted_entry is None:
        return fitted_entry

    try:
        import numpy as np

        fitted_entry.fit_data_source = "raw"
        fitted_entry.freq = np.asarray(source_entry.freq, dtype=float).copy()
        fitted_entry.Z_prime = np.asarray(source_entry.Z_prime, dtype=float).copy()
        fitted_entry.Z_double_prime = np.asarray(source_entry.Z_double_prime, dtype=float).copy()
        fitted_entry.Z_exp = np.asarray(source_entry.Z_exp, dtype=complex).copy()
        fitted_entry.tau = np.asarray(source_entry.tau, dtype=float).copy()
        fitted_entry.tau_fine = np.asarray(source_entry.tau_fine, dtype=float).copy()
    except Exception:
        pass

    attrs = (
        "freq_0", "Z_prime_0", "Z_double_prime_0", "Z_exp_0",
        "mask_manual_raw", "mask_auto_raw", "mask_total_raw", "mask_settings",
        "visible_keep_raw", "active_raw_indices",
        "kk_valid", "kk_c", "kk_max_m", "kk_fit_type", "kk_selected_m", "kk_mu", "kk_tau",
        "kk_R0", "kk_R", "kk_L", "kk_Z_fit", "kk_res_re_pct", "kk_res_im_pct",
        "kk_res_re_raw_pct", "kk_res_im_raw_pct", "kk_Z_fit_raw", "kk_signature",
    )
    for attr in attrs:
        try:
            if hasattr(source_entry, attr):
                setattr(fitted_entry, attr, copy.deepcopy(getattr(source_entry, attr)))
        except Exception:
            pass
    return fitted_entry


def run_fit_entry(entry, mode: str, params: dict):
    """Run one fit without touching Qt objects.

    Imports are intentionally lazy so a spawned worker reaches the output
    redirection and thread-limit setup before NumPy/SciPy/CVXOPT are loaded.
    """
    if entry is None:
        raise ValueError("Empty entry")

    params = dict(params or {})
    from ..algorithms import basics
    from ..algorithms.runs import BHT_run, Bayesian_run, simple_run

    if mode == "simple":
        source_entry = entry
        fit_entry = _prepare_entry_for_fit(entry)
        fitted = simple_run(
            fit_entry,
            rbf_type=params["rbf_type"],
            data_used=params["data_used"],
            induct_used=params["induct_used"],
            der_used=params["der_used"],
            cv_type=params["cv_type"],
            reg_param=params["reg_param"],
            shape_control=params["shape_control"],
            coeff=params["coeff"],
        )
        if params["cv_type"] == "custom":
            fitted.lambda_value = params["reg_param"]
        else:
            fitted.lambda_value = basics.optimal_lambda(
                fitted.A_re,
                fitted.A_im,
                fitted.b_re,
                fitted.b_im,
                fitted.M,
                params["data_used"],
                params["induct_used"],
                -3,
                params["cv_type"],
            )
        return _finalize_fitted_entry(fitted, source_entry)

    if mode == "bayesian":
        source_entry = entry
        fit_entry = _prepare_entry_for_fit(entry)
        fitted = Bayesian_run(
            fit_entry,
            rbf_type=params["rbf_type"],
            data_used=params["data_used"],
            induct_used=params["induct_used"],
            der_used=params["der_used"],
            cv_type=params["cv_type"],
            reg_param=params["reg_param"],
            shape_control=params["shape_control"],
            coeff=params["coeff"],
            NMC_sample=params["sample_number"],
        )
        return _finalize_fitted_entry(fitted, source_entry)

    if mode == "BHT":
        return BHT_run(
            entry,
            params["rbf_type"],
            params["der_used"],
            params["shape_control"],
            params["coeff"],
        )

    raise ValueError(f"Unknown mode: {mode}")


def fit_process_task(key: str, entry, mode: str, params: dict, signature):
    """Process entry point: fit one file and return a serializable result."""
    params = dict(params or {})
    native_threads = max(
        1,
        int(params.get("native_threads", FIT_ALL_NATIVE_THREADS_PER_PROCESS) or 1),
    )

    for name in _CHILD_THREAD_ENV_VARS:
        os.environ[name] = str(native_threads)

    # The scientific stack is imported by run_fit_entry only after output has
    # been redirected.  This also prevents interleaved CVXOPT iteration tables.
    with _silence_process_output():
        _disable_cvxopt_progress()
        with _native_thread_limit(native_threads):
            fitted_entry = run_fit_entry(entry, mode, params)
    return key, fitted_entry, signature


def _exception_text(exc: BaseException) -> str:
    parts: List[str] = []
    current: Optional[BaseException] = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            parts.append(text)
        current = current.__cause__ or current.__context__
    return " | ".join(parts) or exc.__class__.__name__


def is_resource_failure(exc: BaseException) -> bool:
    """Return True for page-file/DLL failures and broken process pools."""
    if isinstance(exc, MemoryError):
        return True

    cls_name = exc.__class__.__name__.lower()
    if cls_name == "brokenprocesspool":
        return True

    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror in (8, 14, 1450, 1455):
            return True

    text = _exception_text(exc).lower()
    return any(marker.lower() in text for marker in _RESOURCE_ERROR_MARKERS)


def next_lower_worker_count(current_workers: int) -> int:
    """Return the next gradual fallback level (one process at a time)."""
    current = max(1, int(current_workers))
    if current <= 1:
        return 1
    return max(1, current - max(1, int(FIT_ALL_FALLBACK_STEP)))


class FitProcessManager:
    """Non-Qt Fit All scheduler with adaptive process concurrency.

    A process pool is created with a logical-CPU upper bound, but only the
    current concurrency limit is submitted.  The first workers are warmed up
    before more files are submitted, and the maximum remains bounded by the
    physical-core count and commit-memory safety margin.  If a
    page-file/DLL failure breaks the pool, completed results are kept and the
    unfinished files are retried with a small step-down in concurrency.
    """

    def __init__(self):
        self.executor: Optional[ProcessPoolExecutor] = None
        self.futures = {}
        self.tasks: Dict[str, object] = {}
        self.pending_keys = deque()
        self.mode: Optional[str] = None
        self.params: dict = {}
        self.signature = None

        # ``worker_count`` is the current active-concurrency target; the
        # executor itself may support up to ``maximum_worker_count`` processes.
        self.worker_count = 0
        self.initial_worker_count = 0
        self.maximum_worker_count = 0
        self.physical_cores = 1
        self.logical_cores = 1
        self.hard_cap = FIT_ALL_PROCESS_HARD_CAP

        self.retry_count = 0
        self.scale_count = 0
        self.active = False

        self._started_at = 0.0
        self._last_scale_check = 0.0
        self._low_cpu_samples = 0
        self._last_cpu_percent: Optional[float] = None
        self.initial_available_commit_gib: Optional[float] = None
        self._psutil_cpu_available = False
        self._windows_cpu_times = None

    def _prime_cpu_sampler(self) -> None:
        self._last_cpu_percent = None
        self._windows_cpu_times = _read_windows_cpu_times()
        try:
            import psutil

            psutil.cpu_percent(interval=None)
            self._psutil_cpu_available = True
        except Exception:
            self._psutil_cpu_available = False

    def _sample_cpu_percent(self) -> Optional[float]:
        if self._psutil_cpu_available:
            try:
                import psutil

                value = float(psutil.cpu_percent(interval=None))
                self._last_cpu_percent = value
                return value
            except Exception:
                self._psutil_cpu_available = False

        current = _read_windows_cpu_times()
        previous = self._windows_cpu_times
        self._windows_cpu_times = current
        if current is None or previous is None:
            return self._last_cpu_percent

        idle_delta = current[0] - previous[0]
        kernel_delta = current[1] - previous[1]
        user_delta = current[2] - previous[2]
        total_delta = kernel_delta + user_delta
        if total_delta <= 0:
            return self._last_cpu_percent

        busy_delta = max(0, total_delta - idle_delta)
        value = max(0.0, min(100.0, 100.0 * busy_delta / total_delta))
        self._last_cpu_percent = value
        return value

    def start(self, tasks: Iterable[Tuple[str, object]], mode: str, params: dict, signature) -> dict:
        self.shutdown(cancel=True, terminate=True)
        self.tasks = {str(key): entry for key, entry in tasks}
        if not self.tasks:
            raise ValueError("No Fit All tasks")

        self.mode = str(mode)
        self.params = dict(params or {})
        self.params["native_threads"] = FIT_ALL_NATIVE_THREADS_PER_PROCESS
        self.signature = signature

        initial, maximum, physical, logical = recommended_worker_plan(len(self.tasks))
        self.initial_worker_count = initial
        self.maximum_worker_count = maximum
        self.physical_cores = physical
        self.logical_cores = logical
        self.hard_cap = FIT_ALL_PROCESS_HARD_CAP
        self.retry_count = 0
        self.scale_count = 0
        self._started_at = time.monotonic()
        self._last_scale_check = self._started_at
        self._low_cpu_samples = 0
        self._prime_cpu_sampler()

        initial, available_commit_gib = _memory_safe_worker_count(initial)
        memory_limited_initial = initial < self.initial_worker_count
        self.initial_available_commit_gib = available_commit_gib
        workers = self._start_with_submission_fallback(initial)
        self.worker_count = workers
        self.active = True
        return {
            "worker_count": workers,
            "initial_worker_count": self.initial_worker_count,
            "maximum_worker_count": self.maximum_worker_count,
            "physical_cores": physical,
            "logical_cores": logical,
            "hard_cap": self.hard_cap,
            "task_count": len(self.tasks),
            "initial_available_commit_gib": self.initial_available_commit_gib,
            "memory_limited_initial": memory_limited_initial,
        }

    def _start_with_submission_fallback(self, workers: int) -> int:
        current = max(1, min(int(workers), self.maximum_worker_count or int(workers)))
        last_error: Optional[BaseException] = None
        while current >= 1:
            try:
                self._start_pool(current)
                return current
            except Exception as exc:
                last_error = exc
                self._shutdown_executor(cancel=True, terminate=True)
                if current <= 1 or not is_resource_failure(exc):
                    raise
                current = next_lower_worker_count(current)
                self.retry_count += 1
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to start Fit All process pool")

    def _rebuild_pending_queue(self) -> None:
        submitted = set(self.futures.values())
        self.pending_keys = deque(key for key in self.tasks if key not in submitted)

    def _start_pool(self, active_workers: int) -> None:
        if not self.tasks:
            self.active = False
            return

        mp.freeze_support()
        context = mp.get_context("spawn")
        executor_max = max(1, int(self.maximum_worker_count or active_workers))
        with _child_spawn_environment(self.params.get("native_threads", 1)):
            executor = ProcessPoolExecutor(
                max_workers=executor_max,
                mp_context=context,
                initializer=_worker_initializer,
                initargs=(self.params.get("native_threads", 1),),
            )

        self.executor = executor
        self.futures = {}
        self.worker_count = max(1, min(int(active_workers), executor_max))
        self._rebuild_pending_queue()
        try:
            self._submit_until_limit()
        except Exception:
            self._shutdown_executor(cancel=True, terminate=True)
            raise
        self.active = True

    def _submit_until_limit(self) -> None:
        if self.executor is None:
            return
        while self.pending_keys and len(self.futures) < self.worker_count:
            key = self.pending_keys.popleft()
            if key not in self.tasks:
                continue
            entry = self.tasks[key]
            future = self.executor.submit(
                fit_process_task,
                key,
                entry,
                self.mode,
                self.params,
                self.signature,
            )
            self.futures[future] = key

    def _commit_allows_scale(self, added_workers: int) -> Tuple[bool, Optional[float]]:
        available = _available_commit_bytes()
        if available is None:
            return True, None
        reserve = _commit_reserve_bytes(available)
        incremental = int(
            max(1, int(added_workers))
            * FIT_ALL_SCALE_COMMIT_PER_ADDED_PROCESS_MIB
            * (1024 ** 2)
        )
        return available >= reserve + incremental, available / (1024 ** 3)

    def _maybe_scale_up(self) -> Optional[dict]:
        if not self.pending_keys:
            return None
        if self.worker_count >= self.maximum_worker_count:
            return None

        now = time.monotonic()
        if now - self._started_at < FIT_ALL_WARMUP_SECONDS:
            return None
        if now - self._last_scale_check < FIT_ALL_SCALE_INTERVAL_SECONDS:
            return None
        self._last_scale_check = now

        cpu_percent = self._sample_cpu_percent()
        if cpu_percent is None:
            # Without a CPU sampler, scale cautiously after two intervals.
            self._low_cpu_samples += 1
        elif cpu_percent < FIT_ALL_SCALE_CPU_BELOW_PERCENT:
            self._low_cpu_samples += 1
        else:
            self._low_cpu_samples = 0
            return None

        if self._low_cpu_samples < FIT_ALL_LOW_CPU_SAMPLES_REQUIRED:
            return None

        old_workers = self.worker_count
        new_workers = min(
            self.maximum_worker_count,
            old_workers + FIT_ALL_SCALE_STEP,
            old_workers + len(self.pending_keys),
        )
        if new_workers <= old_workers:
            return None

        allowed, available_commit_gib = self._commit_allows_scale(new_workers - old_workers)
        if not allowed:
            # Do not reduce the current concurrency merely because the reserve
            # is low; just postpone the next increase.
            self._low_cpu_samples = 0
            return None

        self.worker_count = new_workers
        self.scale_count += 1
        self._low_cpu_samples = 0
        self._submit_until_limit()
        return {
            "from_workers": old_workers,
            "to_workers": new_workers,
            "cpu_percent": cpu_percent,
            "available_commit_gib": available_commit_gib,
            "scale_count": self.scale_count,
        }

    def poll(self) -> dict:
        """Collect results, increase concurrency, and recover from pool failure."""
        results = []
        errors = []
        retry = None
        scale = None

        if not self.active:
            return {
                "results": results,
                "errors": errors,
                "retry": retry,
                "scale": scale,
                "done": True,
            }

        done_futures = [future for future in list(self.futures) if future.done()]
        resource_failure = None

        for future in done_futures:
            key = self.futures.pop(future, None)
            if key is None:
                continue
            try:
                result = future.result()
                results.append(result)
                self.tasks.pop(key, None)
            except Exception as exc:
                if is_resource_failure(exc):
                    resource_failure = exc
                    # Keep the task payload so it can be submitted again.
                else:
                    errors.append((key, _exception_text(exc)))
                    self.tasks.pop(key, None)

        if resource_failure is not None:
            old_workers = self.worker_count
            self._shutdown_executor(cancel=True, terminate=True)

            if self.tasks and old_workers > 1:
                requested_workers = next_lower_worker_count(old_workers)
                self.worker_count = requested_workers
                try:
                    new_workers = self._start_with_submission_fallback(requested_workers)
                    self.worker_count = new_workers
                    # After a real page-file/DLL failure, do not auto-scale above
                    # the worker count that actually restarted successfully.
                    self.maximum_worker_count = min(self.maximum_worker_count, new_workers)
                    self.retry_count += 1
                    retry = {
                        "from_workers": old_workers,
                        "to_workers": new_workers,
                        "remaining": len(self.tasks),
                        "reason": _exception_text(resource_failure),
                        "retry_count": self.retry_count,
                    }
                except Exception as exc:
                    message = _exception_text(exc)
                    for key in list(self.tasks):
                        errors.append((key, message))
                    self.tasks.clear()
                    self.active = False
            elif self.tasks:
                message = _exception_text(resource_failure)
                for key in list(self.tasks):
                    errors.append((key, message))
                self.tasks.clear()
                self.active = False
        else:
            # Refill the current concurrency before deciding whether more
            # parallelism is useful.
            self._submit_until_limit()
            scale = self._maybe_scale_up()

        if not self.tasks:
            self._shutdown_executor(cancel=False, terminate=False)
            self.active = False
        elif not self.futures and not self.pending_keys:
            # Defensive recovery for an unexpected empty queue.
            self._rebuild_pending_queue()
            self._submit_until_limit()

        return {
            "results": results,
            "errors": errors,
            "retry": retry,
            "scale": scale,
            "done": not self.active,
            "worker_count": self.worker_count,
            "maximum_worker_count": self.maximum_worker_count,
            "remaining": len(self.tasks),
            "queued": len(self.pending_keys),
            "running": len(self.futures),
            "cpu_percent": self._last_cpu_percent,
        }

    def _shutdown_executor(self, cancel: bool, terminate: bool) -> None:
        futures = self.futures or {}
        if cancel:
            for future in list(futures):
                try:
                    future.cancel()
                except Exception:
                    pass

        executor = self.executor
        if terminate and executor is not None:
            try:
                for process in list(getattr(executor, "_processes", {}).values()):
                    try:
                        process.terminate()
                    except Exception:
                        pass
            except Exception:
                pass

        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=bool(cancel))
            except TypeError:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
            except Exception:
                pass

        self.executor = None
        self.futures = {}
        self.pending_keys = deque()

    def shutdown(self, cancel: bool = False, terminate: bool = False) -> None:
        self._shutdown_executor(cancel=cancel, terminate=terminate)
        if cancel:
            self.tasks.clear()
        self.active = False
        self.worker_count = 0

