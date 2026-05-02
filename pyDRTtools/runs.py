# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Adeleke Maradesa, Baptiste Py'
__date__ = '28th June 2024'

"""
This file stores all the functions that are shared by all three DRT methods, i.e., simple, Bayesian, and Bayesian Hilbert Transform.
References: 
    [1] T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta. 184 (2015) 483-499.
    [2] M. Saccoccio, T. H. Wan, C. Chen,F. Ciucci, Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: Ridge and lasso regression methods - A theoretical and experimental study, Electrochimica Acta. 147 (2014) 470-482.
    [3] J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
    [4] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the regularization parameter in the distribution of relaxation times, Journal of the Electrochemical Society. 170 (2023) 030502.
"""

# Maths and data related packages
import numpy as np
import sys
import re
from numpy import log, log10, sqrt
import pandas as pd
from math import pi
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import importlib
from cvxopt import matrix, solvers

# pyDRTtools related package
#
from . import peak_analysis as peaks
# from . import parameter_selection as param
from . import basics
from . import nearest_PD as nPD
import importlib

importlib.reload(basics)
importlib.reload(peaks)
# importlib.reload(param)
# from . import deep_learning as deep
from . import BHT
from . import HMC
import time


class EIS_object(object):

    # The EIS_object class stores the input data and the DRT result.

    def __init__(self, freq, Z_prime, Z_double_prime):

        """
        This is EIS_object class
        Inputs:
            freq: frequency of the EIS measurement
            Z_prime: real part of the impedance
            Z_double_prime: imaginery part of the impedance
        """
        # define an EIS_object
        self.freq = freq
        self.Z_prime = Z_prime
        self.Z_double_prime = Z_double_prime
        self.Z_exp = Z_prime + 1j * Z_double_prime

        # keep a copy of the original data
        self.freq_0 = freq
        self.Z_prime_0 = Z_prime
        self.Z_double_prime_0 = Z_double_prime
        self.Z_exp_0 = Z_prime + 1j * Z_double_prime

        self.tau = 1 / freq  # we assume that the collocation points equal to 1/freq as default
        self.tau_fine = np.logspace(log10(self.tau.min()) - 0.5, log10(self.tau.max()) + 0.5, 10 * freq.shape[0])
        ## select custom collocation

        # tau_fine = np.logspace(tau_min, tau_max, num = N_taus, endpoint=True)

        self.method = 'none'

    @classmethod
    def from_file(cls, filename):
        """Load EIS data from file.

        Supported formats:
          - Plain 3-column numeric CSV/TXT: freq, Z', Z''
          - CHI exported EIS files (CSV/TXT): header metadata + column names such as
            Freq/Hz, Z'/ohm, Z"/ohm ...
        """

        def _is_chi_like(head_lines):
            joined = "\n".join(head_lines)
            # CHI typically has metadata lines with " = " and a header with Freq/Hz
            return "Freq" in joined and "Hz" in joined and ("Z'" in joined or "Z\"" in joined or "Z" in joined)

        def _detect_header_and_delim(lines):
            header_idx = None
            header_line = None
            for i, line in enumerate(lines):
                if ("Freq" in line and "Hz" in line) and (
                        ("Z'" in line) or ('Z"' in line) or ("Z\"" in line) or ("Z''" in line)):
                    header_idx = i
                    header_line = line
                    break
            if header_idx is None:
                raise ValueError("CHI header line not found")
            # delimiter heuristic
            comma = header_line.count(",")
            tab = header_line.count("\t")
            semi = header_line.count(";")
            if tab >= comma and tab >= semi and tab > 0:
                delim = "\t"
            elif semi > comma and semi > 0:
                delim = ";"
            else:
                delim = ","
            return header_idx, delim

        def _normalize(s: str) -> str:
            s = str(s)
            s = s.strip().lower()
            s = re.sub(r"\s+", "", s)
            return s

        def _pick_col(df_cols, candidates):
            norm_cols = [_normalize(c) for c in df_cols]
            for cand in candidates:
                cand_n = _normalize(cand)
                for i, c in enumerate(norm_cols):
                    if cand_n == c:
                        return df_cols[i]
            # fallback contains-match
            for cand in candidates:
                cand_n = _normalize(cand)
                for i, c in enumerate(norm_cols):
                    if cand_n in c:
                        return df_cols[i]
            return None

        def _read_plain_3col_csv(path):
            # auto-sep, no header, robust to tabs/spaces
            df = pd.read_csv(path, header=None, engine="python", sep=None, comment="#")
            if df.shape[1] < 3:
                raise ValueError("Plain CSV needs >=3 columns")
            f = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            zr = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            zi = pd.to_numeric(df.iloc[:, 2], errors="coerce")
            out = pd.DataFrame({"f": f, "zr": zr, "zi": zi}).dropna()
            out = out[out["f"] > 0]
            if out.shape[0] < 3:
                raise ValueError("Plain CSV parsed too few rows")
            return out["f"].to_numpy(float), out["zr"].to_numpy(float), out["zi"].to_numpy(float)

        def _read_plain_3col_txt(path):
            data = np.loadtxt(path)
            if data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Plain TXT needs >=3 columns")
            f = data[:, 0].astype(float)
            zr = data[:, 1].astype(float)
            zi = data[:, 2].astype(float)
            mask = np.isfinite(f) & np.isfinite(zr) & np.isfinite(zi) & (f > 0)
            f, zr, zi = f[mask], zr[mask], zi[mask]
            if f.shape[0] < 3:
                raise ValueError("Plain TXT parsed too few rows")
            return f, zr, zi

        def _read_chi(path):
            # read only the header region to locate the column header line
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()

            header_idx, delim = _detect_header_and_delim(lines)

            # parse with pandas using the discovered header line
            df = pd.read_csv(path, skiprows=header_idx, header=0, engine="python", sep=delim)
            if df.shape[0] == 0:
                raise ValueError("CHI file contains no data rows")

            freq_col = _pick_col(df.columns, ["Freq/Hz", "Frequency/Hz", "Freq(Hz)", "Freq"])
            zre_col = _pick_col(df.columns, ["Z'/ohm", "Z'(ohm)", "Zre/ohm", "Zre"])
            zim_col = _pick_col(df.columns, ['Z"/ohm', 'Z"(ohm)', "Z''/ohm", "Zim/ohm", "Zim"])

            if freq_col is None:
                # last-resort: any column containing 'freq'
                freq_col = _pick_col(df.columns, ["freq"])
            if zre_col is None:
                zre_col = _pick_col(df.columns, ["z'"])
            if zim_col is None:
                zim_col = _pick_col(df.columns, ['z"', "z''", "zim"])

            if freq_col is None or zre_col is None or zim_col is None:
                raise ValueError("CHI required columns not found")

            f = pd.to_numeric(df[freq_col], errors="coerce")
            zr = pd.to_numeric(df[zre_col], errors="coerce")
            zi = pd.to_numeric(df[zim_col], errors="coerce")

            out = pd.DataFrame({"f": f, "zr": zr, "zi": zi}).dropna()
            out = out[out["f"] > 0]
            if out.shape[0] < 3:
                raise ValueError("CHI parsed too few rows")

            return out["f"].to_numpy(float), out["zr"].to_numpy(float), out["zi"].to_numpy(float)

        # Fast sniff: read a few lines to decide format
        head_lines = []
        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(80):
                    line = fh.readline()
                    if not line:
                        break
                    head_lines.append(line)
        except Exception:
            head_lines = []

        is_chi = _is_chi_like(head_lines)

        if filename.lower().endswith(".csv"):
            if is_chi:
                freq, Z_prime, Z_double_prime = _read_chi(filename)
            else:
                try:
                    freq, Z_prime, Z_double_prime = _read_plain_3col_csv(filename)
                except Exception:
                    # fallback to CHI parser if plain read fails
                    freq, Z_prime, Z_double_prime = _read_chi(filename)

        elif filename.lower().endswith(".txt"):
            if is_chi:
                freq, Z_prime, Z_double_prime = _read_chi(filename)
            else:
                try:
                    freq, Z_prime, Z_double_prime = _read_plain_3col_txt(filename)
                except Exception:
                    freq, Z_prime, Z_double_prime = _read_chi(filename)

        else:
            raise ValueError(f"Unsupported file extension: {filename}")

        return cls(freq, Z_prime, Z_double_prime)

    def plot_DRT(self):  # plot the DRT result

        basics.pretty_plot(4, 4)
        plt.rc('font', family='serif', size=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('text', usetex=True)

        if self.method == 'simple':
            plt.plot(self.out_tau_vec, self.gamma, 'k')
            y_min = 0
            y_max = max(self.gamma)

        elif self.method == 'credit':
            plt.fill_between(self.out_tau_vec, self.lower_bound, self.upper_bound, facecolor='lightgrey')
            plt.plot(self.out_tau_vec, self.gamma, color='black', label='MAP')
            plt.plot(self.out_tau_vec, self.mean, color='blue', label='mean')
            plt.plot(self.out_tau_vec, self.lower_bound, color='black', linewidth=1)
            plt.plot(self.out_tau_vec, self.upper_bound, color='black', linewidth=1)
            plt.legend(frameon=False, fontsize=15)
            y_min = 0
            y_max = max(self.upper_bound)

        elif self.method == 'BHT':
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_re, 'b', linewidth=1)
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_im, 'k', linewidth=1)
            y_min = min(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
            y_max = max(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))

        else:
            return

        plt.xscale('log')
        plt.xlim(self.out_tau_vec.min(), self.out_tau_vec.max())
        plt.ylim(y_min, y_max * 1.1)
        plt.xlabel(r'$f/{\rm Hz}$', fontsize=20)
        plt.ylabel(r'$\gamma(\tau)/\Omega$', fontsize=20)

        plt.show()


#

def simple_run(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1, der_used='1st order',
               cv_type='GCV', reg_param=1E-3, shape_control='FWHM Coefficient', coeff=0.5):
    """
    This function enables to compute the DRT using ridge regression (also known as Tikhonov regression)
    References:
        T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta 184 (2015) 483-499.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type
        shape_control: option for controlling the shape of the radial basis function (RBF)
        coeff: magnitude of the shape control
    """

    # Step 1.1: define the optimization bounds
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    ###
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    # Step 1.2: compute epsilon
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)

    # Step 1.3: compute A matrix
    ## assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type)
    entry.A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type)
    entry.A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type)

    # Step 1.4: compute M matrix  assemble_M_1(tau_vec, epsilon, rbf_type)
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type)
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type)

    # Step 2: conduct ridge regularization
    if data_used == 'Combined Re-Im Data':  # select both parts of the impedance for the simple run

        if induct_used == 0 or induct_used == 2:  # without considering the inductance
            N_RL = 1  # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:, 0] = 1

            entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp

            entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
            entry.M[N_RL:, N_RL:] = entry.M_temp

            # optimally select the regularization level
            # initial guess for the hyperparameter
            log_lambda_0 = log(reg_param)  # initial guess for lambda
            #
            if cv_type == 'custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                           data_used, induct_used, log_lambda_0, cv_type)

            print('The value of the regularization parameter is', entry.lambda_value)  # to check the value of lambda

            # recover the DRT using cvxopt
            H_combined, c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im,
                                                                 entry.M, entry.lambda_value)
            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0] + N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0] + N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined), G, h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler, it will be used if needed
            entry.mu_Z_re = entry.A_re @ x
            entry.mu_Z_im = entry.A_im @ x
            entry.res_re = entry.mu_Z_re - entry.b_re
            entry.res_im = entry.mu_Z_im - entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re, entry.res_im]))
            inv_V = 1 / sigma_re_im ** 2 * np.eye(N_freqs)

            Sigma_inv = (entry.A_re.T @ inv_V @ entry.A_re) + (entry.A_im.T @ inv_V @ entry.A_im) + (
                        entry.lambda_value / sigma_re_im ** 2) * entry.M
            mu_numerator = entry.A_re.T @ inv_V @ entry.b_re + entry.A_im.T @ inv_V @ entry.b_im

        elif induct_used == 1:  # considering the inductance
            N_RL = 2
            entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:, 1] = 1

            entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:, 0] = 2 * pi * entry.freq

            entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
            entry.M[N_RL:, N_RL:] = entry.M_temp

            # optimally select the regularization level
            log_lambda_0 = log(reg_param)  # initial guess for lambda
            if cv_type == 'custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                           data_used, induct_used, log_lambda_0, cv_type)

            print('The value of the regularization parameter is', entry.lambda_value)  # to check the value of lambda

            # recover the DRT using cvxopt
            H_combined, c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im,
                                                                 entry.M, entry.lambda_value)
            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0] + N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0] + N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined), G, h)
            x = np.array(sol['x']).flatten()

            entry.mu_Z_re = entry.A_re @ x
            entry.mu_Z_im = entry.A_im @ x
            entry.res_re = entry.mu_Z_re - entry.b_re
            entry.res_im = entry.mu_Z_im - entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re, entry.res_im]))
            inv_V = 1 / sigma_re_im ** 2 * np.eye(N_freqs)

            Sigma_inv = (entry.A_re.T @ inv_V @ entry.A_re) + (entry.A_im.T @ inv_V @ entry.A_im) + (
                        entry.lambda_value / sigma_re_im ** 2) * entry.M
            mu_numerator = entry.A_re.T @ inv_V @ entry.b_re + entry.A_im.T @ inv_V @ entry.b_im

    elif data_used == 'Im Data':  # select the imaginary part of the impedance for the simple run

        if induct_used == 0 or induct_used == 2:  # without considering the inductance
            N_RL = 0  # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp

            entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp

            entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
            entry.M[N_RL:, N_RL:] = entry.M_temp

            # optimally select the regularization level
            log_lambda_0 = log(reg_param)  # initial guess for lambda
            if cv_type == 'custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                           data_used, induct_used, log_lambda_0, cv_type)

            print('The value of the regularization parameter is', entry.lambda_value)  # to check the value of lambda

            # recover the DRT using cvxopt
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraints
            ## bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0] + N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0] + N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im), G, h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re @ x
            entry.mu_Z_im = entry.A_im @ x
            entry.res_re = entry.mu_Z_re - entry.b_re
            entry.res_im = entry.mu_Z_im - entry.b_im

            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1 / sigma_re_im ** 2 * np.eye(N_freqs)

            Sigma_inv = (entry.A_im.T @ inv_V @ entry.A_im) + (entry.lambda_value / sigma_re_im ** 2) * entry.M
            mu_numerator = entry.A_im.T @ inv_V @ entry.b_im


        elif induct_used == 1:  # considering the inductance
            N_RL = 1
            entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp

            entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:, 0] = 2 * pi * entry.freq

            entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
            entry.M[N_RL:, N_RL:] = entry.M_temp

            # optimally select the regularization level
            log_lambda_0 = log(reg_param)  # initial guess for lambda
            if cv_type == 'custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                           data_used, induct_used, log_lambda_0, cv_type)

            print('The value of the regularization parameter is', entry.lambda_value)  # to check the value of lambda

            # recover the DRT using cvxopt

            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            #
            # enforce positivity constraints
            # bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0] + N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0] + N_RL))
            # Formulate the quadratic programming problem
            ##
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im), G, h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re @ x
            entry.mu_Z_im = entry.A_im @ x
            entry.res_re = entry.mu_Z_re - entry.b_re
            entry.res_im = entry.mu_Z_im - entry.b_im

            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1 / sigma_re_im ** 2 * np.eye(N_freqs)

            Sigma_inv = (entry.A_im.T @ inv_V @ entry.A_im) + (entry.lambda_value / sigma_re_im ** 2) * entry.M
            mu_numerator = entry.A_im.T @ inv_V @ entry.b_im

    elif data_used == 'Re Data':  # select the real part of the impedance for the simple run
        N_RL = 1
        entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
        entry.A_re[:, N_RL:] = entry.A_re_temp
        entry.A_re[:, 0] = 1

        entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
        entry.A_im[:, N_RL:] = entry.A_im_temp

        entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
        entry.M[N_RL:, N_RL:] = entry.M_temp

        # optimally select the regularization level
        log_lambda_0 = log(reg_param)  # initial guess for lambda
        if cv_type == 'custom':
            entry.lambda_value = reg_param
        else:
            entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                       data_used, induct_used, log_lambda_0, cv_type)

        print('The value of the regularization parameter is', entry.lambda_lambda)  # to check the value of lambda

        # recover the DRT using cvxopt
        H_re, c_re = basics.quad_format_separate(entry.A_re, entry.b_re, entry.M, entry.lambda_value)

        # enforce positivity constraints
        # ## bound matrix
        G = matrix(-np.identity(entry.b_re.shape[0] + N_RL))
        h = matrix(np.zeros(entry.b_re.shape[0] + N_RL))
        # Formulate the quadratic programming problem
        ###
        # Solve the quadratic programming problem
        sol = solvers.qp(matrix(H_re), matrix(c_re), G, h)
        x = np.array(sol['x']).flatten()

        # prepare for HMC sampler
        entry.mu_Z_re = entry.A_re @ x
        entry.mu_Z_im = entry.A_im @ x
        entry.res_re = entry.mu_Z_re - entry.b_re
        entry.res_im = entry.mu_Z_im - entry.b_im

        # only consider std of residuals in the real part
        sigma_re_im = np.std(entry.res_re)
        inv_V = 1 / sigma_re_im ** 2 * np.eye(N_freqs)

        Sigma_inv = (entry.A_re.T @ inv_V @ entry.A_re) + (entry.lambda_value / sigma_re_im ** 2) * entry.M
        mu_numerator = entry.A_re.T @ inv_V @ entry.b_re

    entry.Sigma_inv = (Sigma_inv + Sigma_inv.T) / 2

    # test if the covariance matrix is positive definite
    if (nPD.is_PD(entry.Sigma_inv) == False):
        entry.Sigma_inv = nPD.nearest_PD(entry.Sigma_inv)  # if not, use the nearest positive definite matrix

    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    entry.mu = np.linalg.solve(L_Sigma_inv, mu_numerator)
    entry.mu = np.linalg.solve(L_Sigma_inv.T, entry.mu)
    # entry.mu = np.linalg.solve(entry.Sigma_inv, mu_numerator)

    # Step 3: obtaining the result of inductance, resistance, and gamma
    if N_RL == 0:
        entry.L, entry.R = 0, 0
    elif N_RL == 1 and data_used == 'Im Data':
        entry.L, entry.R = x[0], 0
    elif N_RL == 1 and data_used != 'Im Data':
        entry.L, entry.R = 0, x[0]
    elif N_RL == 2:
        entry.L, entry.R = x[0:2]

    entry.x = x[N_RL:]
    entry.out_tau_vec, entry.gamma = basics.x_to_gamma(x[N_RL:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.N_RL = N_RL
    entry.method = 'simple'

    return entry


def Bayesian_run(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1, der_used='1st order',
                 cv_type='GCV', reg_param=1E-3, shape_control='FWHM Coefficient', coeff=0.5, NMC_sample=2000):
    """
    This function enables to recover the DRT with its uncertainty in a Bayesian framework.
    References:
        F. Ciucci, C. Chen, Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach, Electrochimica Acta 167 (2015) 439-454.
        M. B. Effat, F. Ciucci, Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data, Electrochimica Acta 247 (2017) 1117-1129.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type
        shape_control: option for controlling the shape of the radial basis function (RBF)
        coeff: magnitude of the shape control
        NMC_sample: number of samples for the HMC sampler
    """

    simple_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
               der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control, coeff=coeff)

    # using HMC sampler to sample the truncated Gaussian distribution

    # object_A.plot_DRT()
    entry.mu = entry.mu[entry.N_RL:]  # reshape to avoid error as
    entry.Sigma_inv = entry.Sigma_inv[entry.N_RL:, entry.N_RL:]

    # Step 1: Cholesky Transform instead of direct inverse
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    L_Sigma_agm = np.linalg.inv(L_Sigma_inv)
    entry.Sigma = L_Sigma_agm.T @ L_Sigma_agm

    # Step 2: set up the boundary constraints
    F = np.eye(entry.x.shape[0])
    g = np.finfo(float).eps * np.ones(entry.mu.shape[0])
    initial_X = entry.x

    # Step 3: use generate_tmg from HMC_exact.py to sample the truncated Gaussian distribution
    entry.Xs = HMC.generate_tmg(F, g, entry.Sigma, entry.mu, initial_X, cov=True, L=NMC_sample)
    entry.lower_bound = np.quantile(entry.Xs[:, 501:], .005, axis=1)
    entry.upper_bound = np.quantile(entry.Xs[:, 501:], .995, axis=1)
    entry.mean = np.mean(entry.Xs[:, 501:], axis=1)

    # Step 4: map array to gamma
    entry.out_tau_vec, entry.lower_bound = basics.x_to_gamma(entry.lower_bound, entry.tau_fine, entry.tau,
                                                             entry.epsilon, rbf_type)
    entry.out_tau_vec, entry.upper_bound = basics.x_to_gamma(entry.upper_bound, entry.tau_fine, entry.tau,
                                                             entry.epsilon, rbf_type)
    entry.out_tau_vec, entry.mean = basics.x_to_gamma(entry.mean, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)

    entry.method = 'credit'

    return entry


# Hilbert run

def BHT_run(entry, rbf_type='Gaussian', der_used='1st order', shape_control='FWHM Coefficient', coeff=0.5):
    """
    This function enables to assess the compliance of an EIS spectrum to the Kramers-Kronig relations.
    References:
       J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
       F. Ciucci, The Gaussian process hilbert transform (GP-HT): Testing the consistency of electrochemical impedance spectroscopy data, Journal of the Electrochemical Society. 167-12 (2020) 126503.
    Inputs:
       entry: an EIS spectrum
       rbf_type: discretization function
       der_used: order of the derivative considered for the M matrix
       shape_control: option for controlling the shape of the radial basis function (RBF)
       coeff: magnitude of the shape control
    """

    omega_vec = 2 * pi * entry.freq
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]

    # Step 1: construct the A matrix
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)

    A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type)
    A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type)

    # add resistance column and inductance column to A_re and A_im
    entry.A_re = np.append(np.ones([N_freqs, 1]), A_re_temp, axis=1)
    entry.A_im = np.append(omega_vec.reshape(N_freqs, 1), A_im_temp, axis=1)
    entry.A_H_re = A_re_temp
    entry.A_H_im = A_im_temp
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag

    # Step 2: construct the M matrix
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type)

    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type)
    entry.M = np.zeros((N_taus + 1, N_taus + 1))
    entry.M[1:, 1:] = entry.M_temp

    # Step 3: perform Hilbert transform estimation (try for "max_attempts" until no error occur for the HT_single_est)
    max_attempts = 20  # number of attempt
    for attempt in range(max_attempts):
        try:
            # generate a random initial guess for theta_0 in the range
            theta_0 = 10 ** (6 * np.random.rand(3, 1) - 3)
            # perform the Hilbert transform estimation for the real part of the impedance data
            out_dict_real = BHT.HT_single_est(theta_0, entry.Z_exp.real, entry.A_re, entry.A_H_im, entry.M, N_freqs,
                                              N_taus)
            # update theta_0 based on the result of the real part estimation
            theta_0 = out_dict_real['theta']
            # perform the Hilbert transform estimation for the imaginary part of the impedance data
            out_dict_imag = BHT.HT_single_est(theta_0, entry.Z_exp.imag, entry.A_im, entry.A_H_re, entry.M, N_freqs,
                                              N_taus)

            # exit the loop if both estimations are successful
            break
        except Exception as e:
            # print the error message and the attempt number if an error occurs
            print(f'Error Occurred: {e}. Attempt {attempt + 1}/{max_attempts}. Trying another initial condition.')
    else:
        raise RuntimeError(f"Failed to execute successfully after {max_attempts} attempts.")

    # Step 4: score the EIS
    entry.out_scores = BHT.EIS_score(theta_0, entry.freq, entry.Z_exp, out_dict_real, out_dict_imag, N_MC_samples=10000)

    # Step 5: display the bands and the Hilbert fitting of the real and the imaginary parts

    # Step 5.1: Real part

    # Step 5.1.1: Bayesian regression
    entry.mu_Z_re = out_dict_real.get('mu_Z')
    entry.cov_Z_re = np.diag(out_dict_real.get('Sigma_Z'))

    entry.mu_R_inf = out_dict_real.get('mu_gamma')[0]
    entry.cov_R_inf = np.diag(out_dict_real.get('Sigma_gamma'))[0]

    # Step 5.1.2: DRT part
    entry.mu_Z_DRT_re = out_dict_real.get('mu_Z_DRT')
    entry.cov_Z_DRT_re = np.diag(out_dict_real.get('Sigma_Z_DRT'))

    # Step 5.1.3: HT prediction
    entry.mu_Z_H_im = out_dict_real.get('mu_Z_H')
    entry.cov_Z_H_im = np.diag(out_dict_real.get('Sigma_Z_H'))

    # Step 5.1.4: sigma_n estimation
    entry.sigma_n_re = out_dict_real.get('theta')[0]

    # Step 5.1.5: mu_gamma estimation
    entry.mu_gamma_re = out_dict_real.get('mu_gamma')
    entry.out_tau_vec, entry.mu_gamma_fine_re = basics.x_to_gamma(entry.mu_gamma_re[1:], entry.tau_fine, entry.tau,
                                                                  entry.epsilon, rbf_type)

    # Step 5.2: Imaginary part

    # Step 5.2.1: Bayesian regression
    entry.mu_Z_im = out_dict_imag.get('mu_Z')
    entry.cov_Z_im = np.diag(out_dict_imag.get('Sigma_Z'))

    entry.mu_L_0 = out_dict_imag.get('mu_gamma')[0]
    entry.cov_L_0 = np.diag(out_dict_imag.get('Sigma_gamma'))[0]

    # Step 5.2.2: DRT part
    entry.mu_Z_DRT_im = out_dict_imag.get('mu_Z_DRT')
    entry.cov_Z_DRT_im = np.diag(out_dict_imag.get('Sigma_Z_DRT'))

    # Step 5.2.3: HT prediction
    entry.mu_Z_H_re = out_dict_imag.get('mu_Z_H')
    entry.cov_Z_H_re = np.diag(out_dict_imag.get('Sigma_Z_H'))

    # Step 5.2.4: sigma_n estimation
    entry.sigma_n_im = out_dict_imag.get('theta')[0]

    # Step 5.2.5: mu_gamma estimation
    entry.mu_gamma_im = out_dict_imag.get('mu_gamma')
    entry.out_tau_vec, entry.mu_gamma_fine_im = basics.x_to_gamma(entry.mu_gamma_im[1:], entry.tau_fine, entry.tau,
                                                                  entry.epsilon, rbf_type)

    # Step 6: plot the fits
    entry.mu_Z_H_re_agm = entry.mu_R_inf + entry.mu_Z_H_re
    entry.band_re_agm = sqrt(entry.cov_R_inf + entry.cov_Z_H_re + entry.sigma_n_im ** 2)

    entry.mu_Z_H_im_agm = omega_vec * entry.mu_L_0 + entry.mu_Z_H_im
    entry.band_im_agm = sqrt((omega_vec ** 2) * entry.cov_L_0 + entry.cov_Z_H_im + entry.sigma_n_re ** 2)

    # Step 7: residuals of the Hilbert DRT
    entry.res_H_re = entry.mu_Z_H_re_agm - entry.b_re
    entry.res_H_im = entry.mu_Z_H_im_agm - entry.b_im

    entry.method = 'BHT'

    return entry


# For Peak Analysis

def peak_analysis(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1, der_used='1st order',
                  cv_type='GCV', reg_param=1E-3, shape_control='FWHM Coefficient', coeff=0.5, peak_method='separate',
                  N_peaks=1):
    """
    This function identifies the DRT peaks.

    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type
        shape_control: option for controlling the shape of the radial basis function (RBF)
        coeff: magnitude of the shape control
        N_peaks: desired number of peaks
        peak_method: option for the fit of the recovered DRT, either 'separate' to separately fit each peak, or 'combine' to optimize all the peaks at once
    """

    # Step 1: define the necessary quantities before the subsequent optimizations
    entry.N_peaks = int(N_peaks)

    # Run initial simple analysis
    simple_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
               der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control=shape_control, coeff=coeff)

    # upper and lower log tau values
    log_tau_min = np.min(np.log(entry.out_tau_vec))
    log_tau_max = np.max(np.log(entry.out_tau_vec))

    # Find peaks in the gamma spectrum with a lower threshold for height
    dynamic_threshold = 0.05 * np.mean(entry.gamma)  # Lower threshold for small peak detection
    peak_indices, _ = find_peaks(entry.gamma, height=dynamic_threshold, distance=5)

    # Adjust N_peaks if fewer peaks are found than specified
    N_peaks = min(len(peak_indices), entry.N_peaks)
    num_peaks_found = len(peak_indices)

    if entry.N_peaks > num_peaks_found:
        print(f"Warning: Adjusting N_peaks to {num_peaks_found}.")
        entry.N_peaks = num_peaks_found

    # Set bounds for optimization
    bounds = []
    for _ in range(N_peaks):
        bounds.extend([
            (0, 1.3 * np.sqrt(np.max(entry.gamma))),  # Bounds for sigma_f (peak height)
            (log_tau_min, log_tau_max),  # Bounds for mu_log_tau (peak location)
            (1.0 / (log_tau_max - log_tau_min), 10)  # Bounds for inv_sigma (peak width)
        ])

    # Define the objective function for optimization
    def objective(params):
        return np.sum((peaks.peak_fct(params, entry.out_tau_vec, entry.N_peaks, 'Gaussian') - entry.gamma) ** 2)

    # Parameter optimization using differential evolution
    out_fit_tot = differential_evolution(objective, bounds, popsize=30, mutation=(0.5, 1.5), recombination=0.9,
                                         strategy='best1bin', seed=42, maxiter=200, workers=1)

    # Refine parameters using L-BFGS-B
    theta_fit_tot = out_fit_tot.x
    out_fit_tot = minimize(objective, theta_fit_tot, bounds=bounds, method='L-BFGS-B')
    theta_fit_tot = out_fit_tot.x
    entry.gamma_fit_tot = peaks.peak_fct(theta_fit_tot, entry.out_tau_vec, entry.N_peaks, 'Gaussian')

    # Generate individual Gaussian fits for each peak
    gamma_fit_list = [0] * N_peaks
    for k in range(entry.N_peaks):
        params = theta_fit_tot[3 * k:3 * k + 3]
        gamma_fit = peaks.peak_fct(params, entry.out_tau_vec, 1, 'Gaussian')
        gamma_fit_list[k] = gamma_fit

    if peak_method == 'separate':  # separate fit of the DRT
        entry.out_gamma_fit = gamma_fit_list
        entry.Gaussian = np.array(gamma_fit_list)
        entry.num_vectors = entry.Gaussian.shape[0]
        entry.column_headings = [f'Gaussian_{i + 1}' for i in range(entry.num_vectors)]
        entry.df = pd.DataFrame(entry.Gaussian.T, columns=entry.column_headings)
    else:  # combine fit of the DRT
        entry.out_gamma_fit = entry.gamma_fit_tot

    entry.method = 'peak'

    return entry
