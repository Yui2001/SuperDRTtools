# -*- coding: utf-8 -*-
"""Matplotlib canvas and rendering primitives used by GUI controllers."""

import matplotlib as mpl

mpl.use("Qt5Agg")
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

import matplotlib.pyplot as plt
import numpy as np
from numpy import absolute, angle
from PyQt5 import QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..services.eis_state import _safe_kk_residual_max


class Figure_Canvas(FigureCanvas):

    def __init__(self, parent=None, width=7.5, height=6.5, dpi=100):  # create Figure under matplotlib.pyplot

        # deactivate the popping of another figure panel
        plt.ioff()

        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        # Never call ``plt.close('all')`` here. Other SuperDRTtools windows may
        # still own live Qt-embedded canvases; closing them globally can cause
        # a native Qt crash when their graphics scenes repaint.
        fig = Figure(figsize=(width, height), dpi=dpi)

        # initalizing parent
        FigureCanvas.__init__(self, fig)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        self.setParent(parent)
        self.axes = fig.add_subplot(111)  # using add_subplot method
        fig.tight_layout()
        fig.subplots_adjust(left=0.14, bottom=0.13, right=0.9, top=0.9)
        try:
            self.mpl_connect('scroll_event', self._on_scroll_zoom)
        except Exception:
            pass
        try:
            self.mpl_connect('motion_notify_event', self._on_general_motion)
            self.mpl_connect('figure_leave_event', self._on_general_leave)
        except Exception:
            pass
        self._kk_hover_payload = None
        self._kk_hover_annotation = None
        self._eis_hover_payload = None
        self._eis_plot_payload = None

    @staticmethod
    def _zoom_axis_limits(cur_min, cur_max, center, scale_factor, log_scale=False):
        if not np.isfinite(cur_min) or not np.isfinite(cur_max) or cur_min == cur_max:
            return cur_min, cur_max
        if center is None or not np.isfinite(center):
            center = (cur_min + cur_max) / 2.0
        if log_scale:
            if cur_min <= 0 or cur_max <= 0 or center <= 0:
                return cur_min, cur_max
            lmin, lmax, lc = np.log10(cur_min), np.log10(cur_max), np.log10(center)
            new_min = 10.0 ** (lc - (lc - lmin) * scale_factor)
            new_max = 10.0 ** (lc + (lmax - lc) * scale_factor)
            return new_min, new_max
        new_min = center - (center - cur_min) * scale_factor
        new_max = center + (cur_max - center) * scale_factor
        return new_min, new_max

    def _on_scroll_zoom(self, event):
        ax = getattr(event, 'inaxes', None)
        if ax is None:
            return
        base_scale = 1.2
        scale_factor = 1.0 / base_scale if getattr(event, 'button', None) == 'up' else base_scale

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_center = event.xdata if getattr(event, 'xdata', None) is not None else (x0 + x1) / 2.0
        y_center = event.ydata if getattr(event, 'ydata', None) is not None else (y0 + y1) / 2.0

        x_reversed = x0 > x1
        y_reversed = y0 > y1
        x_low, x_high = (x1, x0) if x_reversed else (x0, x1)
        y_low, y_high = (y1, y0) if y_reversed else (y0, y1)

        new_x_low, new_x_high = self._zoom_axis_limits(x_low, x_high, x_center, scale_factor,
                                                       log_scale=(ax.get_xscale() == 'log'))
        new_y_low, new_y_high = self._zoom_axis_limits(y_low, y_high, y_center, scale_factor,
                                                       log_scale=(ax.get_yscale() == 'log'))
        ax.set_xlim((new_x_high, new_x_low) if x_reversed else (new_x_low, new_x_high))
        ax.set_ylim((new_y_high, new_y_low) if y_reversed else (new_y_low, new_y_high))
        try:
            self.draw_idle()
        except Exception:
            self.draw()

    @staticmethod
    def _format_hover_frequency(value: float) -> str:
        try:
            v = float(value)
        except Exception:
            return str(value)
        if v == 0:
            return '0'
        if 1e-2 <= abs(v) < 1e4:
            out = f"{v:.3f}"
            out = out.rstrip('0').rstrip('.')
            return out
        return f"{v:.3e}"

    def _hide_kk_hover(self):
        try:
            QtWidgets.QToolTip.hideText()
        except Exception:
            pass

    def _on_general_motion(self, event):
        eis_payload = getattr(self, '_eis_hover_payload', None)
        if eis_payload and getattr(event, 'inaxes', None) is self.axes:
            xs = np.asarray(eis_payload.get('x', []), dtype=float)
            ys = np.asarray(eis_payload.get('y', []), dtype=float)
            if xs.size:
                pts = np.column_stack([xs, ys])
                try:
                    disp = self.axes.transData.transform(pts)
                except Exception:
                    disp = None
                if disp is not None and disp.size:
                    dist2 = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
                    idx = int(np.argmin(dist2))
                    d = float(np.sqrt(dist2[idx]))
                    if d <= 12.0:
                        freq_val = float(np.asarray(eis_payload.get('freq', []), dtype=float)[idx])
                        x_val = float(xs[idx])
                        y_val = float(ys[idx])
                        re_res = np.asarray(eis_payload.get('res_re', []), dtype=float)
                        im_res = np.asarray(eis_payload.get('res_im', []), dtype=float)
                        masked = np.asarray(eis_payload.get('masked', []), dtype=bool)
                        re_txt = 'N/A' if idx >= re_res.size or not np.isfinite(re_res[idx]) else f"{re_res[idx]:.2f}%"
                        im_txt = 'N/A' if idx >= im_res.size or not np.isfinite(im_res[idx]) else f"{im_res[idx]:.2f}%"
                        masked_txt = 'Yes' if idx < masked.size and bool(masked[idx]) else 'No'
                        text = (
                            f"Frequency: {self._format_hover_frequency(freq_val)} Hz\n"
                            f"Z': {x_val:.4g} Ω\n"
                            f"-Z'': {y_val:.4g} Ω\n"
                            f"KK Re residual: {re_txt}\n"
                            f"KK Im residual: {im_txt}\n"
                            f"Masked: {masked_txt}"
                        )
                        try:
                            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
                        except Exception:
                            pass
                        return

        payload = getattr(self, '_kk_hover_payload', None)
        if not payload or getattr(event, 'inaxes', None) is not self.axes:
            self._hide_kk_hover()
            return

        best = None
        threshold_px = 15.0
        for label, xs, ys in payload:
            pts = np.column_stack([xs, ys])
            try:
                disp = self.axes.transData.transform(pts)
            except Exception:
                continue
            if disp.size == 0:
                continue
            dist2 = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
            idx = int(np.argmin(dist2))
            d = float(np.sqrt(dist2[idx]))
            if best is None or d < best[0]:
                best = (d, label, float(xs[idx]), float(ys[idx]))

        if best is None or best[0] > threshold_px:
            self._hide_kk_hover()
            return

        _, label, x_val, y_val = best
        text = (
            f"Frequency: {self._format_hover_frequency(x_val)} Hz\n"
            f"{label}: {y_val:.2f}%"
        )
        try:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)
        except Exception:
            pass

    def _on_general_leave(self, event):
        self._hide_kk_hover()

    def EIS_data(self, entry):  # plot the data
        legend_needed = False
        self._kk_hover_payload = None
        self._eis_hover_payload = None
        self._eis_plot_payload = None

        if entry.method == 'BHT':  # for BHT run
            self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im,
                           'k', label='$Z_\mu$(Regressed)', linewidth=3)
            self.axes.plot(entry.mu_Z_H_re_agm, -entry.mu_Z_H_im_agm,
                           'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            legend_needed = True

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im, 'k', linewidth=2, label='DRT fit')
            legend_needed = True

        if getattr(entry, 'kk_valid', False) and hasattr(entry, 'kk_Z_fit'):
            self.axes.plot(entry.kk_Z_fit.real, -entry.kk_Z_fit.imag,
                           color='orange', linewidth=2.5, label='K-K fit')
            legend_needed = True

        freq0 = np.asarray(getattr(entry, 'freq_0', getattr(entry, 'freq', [])), dtype=float).reshape(-1)
        zre0 = np.asarray(getattr(entry, 'Z_prime_0', getattr(entry, 'Z_prime', [])), dtype=float).reshape(-1)
        zim0 = np.asarray(getattr(entry, 'Z_double_prime_0', getattr(entry, 'Z_double_prime', [])), dtype=float).reshape(-1)
        visible_keep = np.asarray(getattr(entry, 'visible_keep_raw', np.ones(freq0.size, dtype=bool)), dtype=bool).reshape(-1)
        if visible_keep.size != freq0.size:
            visible_keep = np.ones(freq0.size, dtype=bool)
        masked = np.asarray(getattr(entry, 'mask_total_raw', np.zeros(freq0.size, dtype=bool)), dtype=bool).reshape(-1)
        if masked.size != freq0.size:
            masked = np.zeros(freq0.size, dtype=bool)
        kk_re = np.asarray(getattr(entry, 'kk_res_re_raw_pct', np.full(freq0.size, np.nan)), dtype=float).reshape(-1)
        kk_im = np.asarray(getattr(entry, 'kk_res_im_raw_pct', np.full(freq0.size, np.nan)), dtype=float).reshape(-1)
        if kk_re.size != freq0.size:
            kk_re = np.full(freq0.size, np.nan, dtype=float)
        if kk_im.size != freq0.size:
            kk_im = np.full(freq0.size, np.nan, dtype=float)
        kk_max = _safe_kk_residual_max(kk_re, kk_im, freq0.size)

        disp_idx = np.where(visible_keep)[0].astype(int)
        if disp_idx.size:
            severe = np.zeros(disp_idx.size, dtype=int)
            kk_disp = kk_max[disp_idx]
            severe[(np.isfinite(kk_disp)) & (kk_disp > 1.0) & (kk_disp <= 5.0)] = 1
            severe[(np.isfinite(kk_disp)) & (kk_disp > 5.0)] = 2
            masked_disp = masked[disp_idx]

            labels = {0: 'Experimental data', 1: 'KK residual > 1%', 2: 'KK residual > 5%'}
            colors = {0: '#D62728', 1: '#FF8C00', 2: '#7A3DB8'}
            for sev in (0, 1, 2):
                idx_vis = disp_idx[(severe == sev) & (~masked_disp)]
                if idx_vis.size:
                    self.axes.scatter(zre0[idx_vis], -zim0[idx_vis], s=34, c=colors[sev], alpha=0.95,
                                      edgecolors='none', label=labels[sev])
                    legend_needed = True
            for sev in (0, 1, 2):
                idx_mask = disp_idx[(severe == sev) & masked_disp]
                if idx_mask.size:
                    self.axes.scatter(zre0[idx_mask], -zim0[idx_mask], s=34, c=colors[sev], alpha=0.25,
                                      edgecolors='none', label='Masked point' if sev == 0 else None)
                    legend_needed = True

            self._eis_hover_payload = {
                'raw_indices': disp_idx,
                'x': zre0[disp_idx],
                'y': -zim0[disp_idx],
                'freq': freq0[disp_idx],
                'res_re': kk_re[disp_idx],
                'res_im': kk_im[disp_idx],
                'masked': masked[disp_idx],
            }
            self._eis_plot_payload = self._eis_hover_payload

        if legend_needed:
            handles, labels = self.axes.get_legend_handles_labels()
            unique = {}
            for h, lab in zip(handles, labels):
                if lab and lab not in unique:
                    unique[lab] = h
            self.axes.legend(list(unique.values()), list(unique.keys()), frameon=False, fontsize=11, loc='best')

        self.axes.set_xlabel('$Z^{\prime}/\Omega$')
        self.axes.set_ylabel('-$Z^{\prime \prime}/\Omega$')
        self.axes.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        self.axes.axis('equal')

    def Magnitude(self, entry):  # plot the magnitude

        if entry.method == 'BHT':  # for BHT run
            self.axes.semilogx(entry.freq, absolute(entry.mu_Z_re + 1j * entry.mu_Z_im),
                               'k', label='$Z_\mu$(Regressed)', linewidth=2)
            self.axes.semilogx(entry.freq, absolute(entry.mu_Z_H_re_agm + 1j * entry.mu_Z_H_im_agm),
                               'b', label='$Z_H$(Hilbert transform)', linewidth=2)
            self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.semilogx(entry.freq, absolute(entry.mu_Z_re + 1j * entry.mu_Z_im), 'k', linewidth=3)
            self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, absolute(entry.Z_exp), 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('$|Z|/\Omega$')

    def Phase(self, entry):  # plot the phase

        if entry.method == 'BHT':  # for BHT run
            self.axes.semilogx(entry.freq, angle(entry.mu_Z_re + 1j * entry.mu_Z_im, deg=True),
                               'k', label='$Z_\mu$(Regressed)', linewidth=2)
            self.axes.semilogx(entry.freq, angle(entry.mu_Z_H_re_agm + 1j * entry.mu_Z_H_im_agm, deg=True),
                               'b', label='$Z_H$(Hilbert transform)', linewidth=2)
            self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg=True), 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.semilogx(entry.freq, angle(entry.mu_Z_re + 1j * entry.mu_Z_im, deg=True), 'k', linewidth=3)
            self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg=True), 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, angle(entry.Z_exp, deg=True), 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('$angle/^\circ$')

    def Re_data(self, entry):  # plot the real part

        if entry.method == 'BHT':  # for BHT run
            self.axes.fill_between(entry.freq, entry.mu_Z_H_re_agm - 3 * entry.band_re_agm,
                                   entry.mu_Z_H_re_agm + 3 * entry.band_re_agm, facecolor='lightgrey')
            self.axes.semilogx(entry.freq, entry.mu_Z_re, 'k', label='$Z_\mu$(Regressed)', linewidth=3)
            self.axes.semilogx(entry.freq, entry.mu_Z_H_re_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            self.axes.semilogx(entry.freq, entry.Z_prime, 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or Bayesian run
            self.axes.semilogx(entry.freq, entry.mu_Z_re, 'k', linewidth=3)
            self.axes.semilogx(entry.freq, entry.Z_prime, 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, entry.Z_prime, 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('$Z^{\prime}/\Omega$')

    def Im_data(self, entry):  # plot of the imaginary part

        if entry.method == 'BHT':  # for BHT run
            self.axes.fill_between(entry.freq, -entry.mu_Z_H_im_agm - 3 * entry.band_im_agm,
                                   -entry.mu_Z_H_im_agm + 3 * entry.band_im_agm, facecolor='lightgrey')
            self.axes.semilogx(entry.freq, -entry.mu_Z_im, 'k', label='$Z_\mu$(Regressed)', linewidth=3)
            self.axes.semilogx(entry.freq, -entry.mu_Z_H_im_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')

        elif entry.method != 'none':  # for simple or bayesian run
            self.axes.semilogx(entry.freq, -entry.mu_Z_im, 'k')
            self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')

        else:  # no computation of the imported data yet
            self.axes.semilogx(entry.freq, -entry.Z_double_prime, 'or')

        self.axes.set_xlim([min(entry.freq), max(entry.freq)])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel('-$Z^{\prime \prime}/\Omega$')

    def DRT_residual(self, entry):  # plot DRT residual analysis

        self._kk_hover_payload = None

        if entry.method == 'none':
            self.axes.text(0.5, 0.5, 'Run DRT fitting to view residuals.',
                           ha='center', va='center', transform=self.axes.transAxes)
            self.axes.set_axis_off()
            return

        freq = np.asarray(entry.freq, dtype=float)
        z_abs = np.asarray(np.abs(entry.Z_exp), dtype=float)
        if z_abs.size != freq.size:
            z_abs = np.ones_like(freq, dtype=float)
        z_abs_safe = np.where(np.isfinite(z_abs) & (z_abs > 0), z_abs, np.nan)

        if entry.method == 'BHT':
            res_re = np.asarray(entry.res_H_re, dtype=float)
            res_im = np.asarray(entry.res_H_im, dtype=float)
        else:
            res_re = np.asarray(entry.res_re, dtype=float)
            res_im = np.asarray(entry.res_im, dtype=float)

        res_re_pct = 100.0 * res_re / z_abs_safe
        res_im_pct = 100.0 * res_im / z_abs_safe

        self.axes.semilogx(freq, res_re_pct, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Re}}$')
        self.axes.semilogx(freq, res_im_pct, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Im}}$')

        y_max = float(np.nanmax(np.abs(np.concatenate([res_re_pct, res_im_pct]))))
        if (not np.isfinite(y_max)) or y_max <= 0:
            y_max = 1.0

        self.axes.set_xlim([float(np.max(freq)), float(np.min(freq))])
        self.axes.set_ylim([-1.15 * y_max, 1.15 * y_max])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel(r'$\Delta\,(\%)$')
        self.axes.axhline(0.0, color='0.5', linewidth=0.8)
        self.axes.grid(True, which='both', linestyle='--', alpha=0.35)
        self.axes.legend(frameon=False, fontsize=12, loc='best')

        self._kk_hover_payload = [
            ('Real Residual', freq, res_re_pct),
            ('Imag Residual', freq, res_im_pct),
        ]

    def Re_residual(self, entry):
        self.DRT_residual(entry)

    def Im_residual(self, entry):
        self.DRT_residual(entry)

    def KK_residual(self, entry):  # plot K-K residual analysis

        self._kk_hover_payload = None

        if not getattr(entry, 'kk_valid', False) or not hasattr(entry, 'kk_res_re_pct'):
            self.axes.text(0.5, 0.5, 'Run K-K analysis to view residuals.',
                           ha='center', va='center', transform=self.axes.transAxes)
            self.axes.set_axis_off()
            return

        freq = np.asarray(entry.freq, dtype=float)
        res_re = np.asarray(entry.kk_res_re_pct, dtype=float)
        res_im = np.asarray(entry.kk_res_im_pct, dtype=float)

        self.axes.semilogx(freq, res_re, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Re}}$')
        self.axes.semilogx(freq, res_im, '-o', linewidth=1.6, markersize=4,
                           label=r'$\Delta_{\mathrm{Im}}$')

        y_max = float(np.nanmax(np.abs(np.concatenate([res_re, res_im]))))
        if (not np.isfinite(y_max)) or y_max <= 0:
            y_max = 1.0

        self.axes.set_xlim([float(np.max(freq)), float(np.min(freq))])
        self.axes.set_ylim([-1.15 * y_max, 1.15 * y_max])
        self.axes.set_xlabel('$f/Hz$')
        self.axes.set_ylabel(r'$\Delta\,(\%)$')
        self.axes.axhline(0.0, color='0.5', linewidth=0.8)
        self.axes.grid(True, which='both', linestyle='--', alpha=0.35)
        self.axes.legend(frameon=False, fontsize=12, loc='best')

        self._kk_hover_payload = [
            ('Real Residual', freq, res_re),
            ('Imag Residual', freq, res_im),
        ]

    def DRT_data(self, entry):  # plot the DRT

        if entry.method == 'none':
            return

        elif entry.method == 'simple':
            self.axes.semilogx(entry.out_tau_vec, entry.gamma, 'k', linewidth=3)
            y_min = 0
            y_max = max(entry.gamma)

        elif entry.method == 'credit':
            self.axes.fill_between(entry.out_tau_vec, entry.lower_bound, entry.upper_bound, facecolor='lightgrey')
            self.axes.semilogx(entry.out_tau_vec, entry.gamma, color='black', label='MAP', linewidth=3)
            self.axes.semilogx(entry.out_tau_vec, entry.mean, color='blue', label='mean', linewidth=3)
            self.axes.semilogx(entry.out_tau_vec, entry.lower_bound, color='black')
            self.axes.semilogx(entry.out_tau_vec, entry.upper_bound, color='black')
            self.axes.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(entry.upper_bound)

        elif entry.method == 'BHT':
            self.axes.semilogx(entry.out_tau_vec, entry.mu_gamma_fine_re, 'b', linewidth=3, label='$Mean Re$')
            self.axes.semilogx(entry.out_tau_vec, entry.mu_gamma_fine_im, 'k', linewidth=3, label='$Mean Im$')
            y_min = min(np.concatenate((entry.mu_gamma_fine_re, entry.mu_gamma_fine_im)))
            y_max = max(np.concatenate((entry.mu_gamma_fine_re, entry.mu_gamma_fine_im)))

        elif entry.method == 'peak':

            self.axes.semilogx(
                entry.out_tau_vec, entry.gamma, color='black', linewidth=3, label='DRT'
            )
            components = list(getattr(entry, 'peak_components', []) or [])
            results = list(getattr(entry, 'peak_results', []) or [])
            if components and str(getattr(entry, 'peak_method', 'separate')).lower() != 'combine':
                colors = plt.get_cmap('tab20')(np.linspace(0.0, 1.0, max(2, len(components))))
                for index, component in enumerate(components):
                    name = (
                        str(results[index].get('name', f'Peak {index + 1}'))
                        if index < len(results) else f'Peak {index + 1}'
                    )
                    self.axes.semilogx(
                        entry.out_tau_vec,
                        component,
                        color=colors[index],
                        linewidth=2,
                        label=name,
                    )
                self.axes.legend(frameon=False, fontsize=10, loc='best')
            elif components:
                self.axes.semilogx(
                    entry.out_tau_vec,
                    np.asarray(entry.gamma_fit_tot, dtype=float),
                    color='green',
                    linewidth=2,
                    label='Combined peak fit',
                )
                self.axes.legend(frameon=False, fontsize=10, loc='best')
            else:
                # Backward-compatible rendering for projects created before
                # structured peak results were introduced.
                fitted = getattr(entry, 'out_gamma_fit', [])
                if isinstance(fitted, (list, tuple)) and len(fitted) == entry.N_peaks:
                    colors = plt.get_cmap('tab20')(np.linspace(0.0, 1.0, max(2, entry.N_peaks)))
                    for index in range(entry.N_peaks):
                        self.axes.semilogx(
                            entry.out_tau_vec, fitted[index], color=colors[index], linewidth=2
                        )
                else:
                    self.axes.semilogx(entry.out_tau_vec, fitted, color='green', linewidth=2)
            y_min = 0
            y_max = max(entry.gamma)

        self.axes.set_xlabel(r'$\tau/s$')
        self.axes.set_ylabel(r'$\gamma( \tau)/\Omega$')
        self.axes.set_ylim([y_min, 1.1 * y_max])
        self.axes.set_xlim([min(entry.out_tau_vec), max(entry.out_tau_vec)])
        if entry.method == 'peak':
            self._add_peak_annotations(entry, float(y_max))

    def _add_peak_annotations(self, entry, y_max):
        """Place compact, pickable result labels beside fitted peak maxima."""
        results = list(getattr(entry, 'peak_results', []) or [])
        if not results:
            self._peak_annotations = []
            return

        colors = plt.get_cmap('tab20')(np.linspace(0.0, 1.0, max(2, len(results))))
        annotations = []
        for index, result in enumerate(results):
            tau = float(result.get('tau_s', np.nan))
            height = float(result.get('height_ohm', np.nan))
            if not np.isfinite(tau) or tau <= 0 or not np.isfinite(height):
                continue
            name = str(result.get('name', f'Peak {index + 1}'))
            resistance = float(result.get('resistance_ohm', np.nan))
            fraction = float(result.get('fraction_percent', np.nan))
            text = (
                f'{name}\n'
                f'tau={tau:.3e} s\n'
                f'R={resistance:.3g} ohm ({fraction:.1f}%)'
            )
            # Put labels below exceptionally tall peaks and above the others.
            offset_y = -54 if y_max > 0 and height > 0.78 * y_max else 14
            annotation = self.axes.annotate(
                text,
                xy=(tau, height),
                xytext=(0, offset_y),
                textcoords='offset points',
                ha='center',
                va='top' if offset_y < 0 else 'bottom',
                fontsize=8.5,
                color='#1D1D1F',
                bbox={
                    'boxstyle': 'round,pad=0.3',
                    'facecolor': 'white',
                    'edgecolor': colors[index],
                    'alpha': 0.88,
                },
                arrowprops={'arrowstyle': '-', 'color': colors[index], 'linewidth': 0.9},
                annotation_clip=True,
                picker=True,
            )
            annotation.set_gid(f'peak-result:{index}')
            annotations.append(annotation)
        self._peak_annotations = annotations

    def Peak_comparison(self, labels, series, y_lim=None):
        """Plot resistance contributions for named peaks across files."""
        labels = list(labels or [])
        if not labels or not series:
            return

        x_values = np.arange(len(labels), dtype=float)
        colors = plt.get_cmap('tab10')(np.linspace(0.0, 1.0, max(2, len(series))))
        lines = []
        for index, (name, values) in enumerate(series.items()):
            values = np.asarray(values, dtype=float)
            line, = self.axes.plot(
                x_values,
                values,
                marker='o',
                markersize=6,
                linewidth=2,
                color=colors[index],
                label=str(name),
            )
            line.set_gid(str(name))
            lines.append(line)

        self.axes.set_xticks(x_values)
        self.axes.set_xticklabels(labels, rotation=35, ha='right')
        self.axes.set_xlim(-0.35, max(0.35, len(labels) - 0.65))
        self.axes.set_xlabel('File')
        self.axes.set_ylabel(r'Peak resistance / $\Omega$')
        if y_lim is not None:
            ymin, ymax = y_lim
            auto_ymin, auto_ymax = self.axes.get_ylim()
            self.axes.set_ylim(
                auto_ymin if ymin is None else float(ymin),
                auto_ymax if ymax is None else float(ymax),
            )
        self.axes.grid(True, axis='y', linestyle='--', alpha=0.3)
        self.axes.legend(frameon=False, fontsize=10, loc='best')
        self._peak_comparison_lines = lines
        self._peak_comparison_labels = labels
        self.figure.subplots_adjust(bottom=0.22)

    def DRT_comparison(self, entries, colors, y_lim=None, keys=None):
        """Overlay multiple DRT curves in the same axes (log-x)."""
        if not entries:
            return

        # expose plotted lines for GUI interactions (pick/highlight)
        self._drt_comp_lines = []

        if keys is None:
            keys = [None] * len(entries)

        xs_min = None
        xs_max = None
        ymins = []
        ymaxs = []

        for entry, color, key in zip(entries, colors, keys):
            if getattr(entry, 'method', 'none') == 'none':
                continue
            if not hasattr(entry, 'out_tau_vec'):
                continue

            x = entry.out_tau_vec

            # Choose a single representative curve per file (matching DRT plot intent)
            y = None
            if entry.method in ('simple', 'peak'):
                y = getattr(entry, 'gamma', None)
            elif entry.method == 'credit':
                y = getattr(entry, 'gamma', None)  # MAP curve
            elif entry.method == 'BHT':
                y = getattr(entry, 'mu_gamma_fine_re', None)
            else:
                y = getattr(entry, 'gamma', None)

            if y is None:
                continue

            line, = self.axes.semilogx(x, y, color=color, linewidth=2, picker=5)
            line._drt_original_color = color
            line._drt_original_linewidth = 2.0
            self._drt_comp_lines.append(line)
            if key is not None:
                try:
                    line.set_gid(key)
                except Exception:
                    pass

            xs_min = float(np.min(x)) if xs_min is None else min(xs_min, float(np.min(x)))
            xs_max = float(np.max(x)) if xs_max is None else max(xs_max, float(np.max(x)))
            ymins.append(float(np.min(y)))
            ymaxs.append(float(np.max(y)))

        if xs_min is None or xs_max is None or not ymins or not ymaxs:
            return

        self.axes.set_xlabel(r'$\tau/s$')
        self.axes.set_ylabel(r'$\gamma( \tau)/\Omega$')
        self.axes.set_xlim([xs_min, xs_max])

        auto_ymin = min(ymins)
        auto_ymax = max(ymaxs)
        # mimic single DRT plot: start at 0 if all positive
        if auto_ymin > 0:
            auto_ymin = 0.0

        if y_lim is not None:
            ymin, ymax = y_lim
            if ymin is None:
                ymin = auto_ymin
            if ymax is None:
                ymax = auto_ymax * 1.1
            self.axes.set_ylim([ymin, ymax])
        else:
            self.axes.set_ylim([auto_ymin, auto_ymax * 1.1])

    def Score(self, entry):  # plot the EIS score

        if entry.method != 'BHT':
            return

        else:  # for BHT method
            Re_part = np.array(
                [entry.out_scores['s_res_re'][0], entry.out_scores['s_res_re'][1], entry.out_scores['s_res_re'][2],
                 entry.out_scores['s_mu_re'], entry.out_scores['s_HD_re'], entry.out_scores['s_JSD_re']])
            Im_part = np.array(
                [entry.out_scores['s_res_im'][0], entry.out_scores['s_res_im'][1], entry.out_scores['s_res_im'][2],
                 entry.out_scores['s_mu_im'], entry.out_scores['s_HD_im'], entry.out_scores['s_JSD_im']])
            x = np.arange(6)  # the label locations
            width = 0.35  # the width of the bars

            self.axes.bar(x - width / 2, Re_part * 100, width, label='Re part', color='blue')
            self.axes.bar(x + width / 2, Im_part * 100, width, label='Im part', color='black')
            self.axes.plot([-0.5, 5.5], [100, 100], '--k')

            self.axes.legend(frameon=False, fontsize=15)
            self.axes.set_ylim([0, 125])
            self.axes.set_xlim([-0.5, 5.5])
            self.axes.set_xticks(x)
            self.axes.set_xticklabels(
                (r'$s_{1\sigma}$', r'$s_{2\sigma}$', r'$s_{3\sigma}$', r'$s_{\mu}$', r'$s_{\rm HD}$', r'$s_{\rm JSD}$'))
            self.axes.set_yticks([0, 50, 100])
            self.axes.set_ylabel(r'$\rm Scores (\%)$')
