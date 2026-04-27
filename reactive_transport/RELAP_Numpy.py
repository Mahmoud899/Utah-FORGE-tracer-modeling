"""
NumPy-backed versions of the RELAP model classes and simulation functions.

This module mirrors reactive_transport/RELAP_v5.py but replaces all mpmath
arithmetic (sqrt, exp, tanh) with numpy equivalents. Every __call__ method
accepts either a scalar complex128 value OR a 1-D complex128 numpy array,
which enables the batched evaluation required by dehoog_invert.

Simulation functions (Simulate_RELAP_Relative, Simulate_RELAP_Dimensionless)
use dehoog_invert from laplace_inversion.py instead of mpmath.invertlaplace,
giving the same results at ~50-100x higher speed.
"""

import numpy as np

from .laplace_inversion import dehoog_invert


# ---------------------------------------------------------------------------
# Ground-water transfer function classes
# ---------------------------------------------------------------------------

class GroundWaterFiniteMatrixSolution:
    """Laplace-domain solution for finite matrix dual-porosity transport."""

    def __init__(
        self,
        mean_residence_time,
        peclet_number,
        dualPorosity_param,
        abDm,
        thermal_degradation_coeff,
        fracture_retardation,
        matrix_retardation,
    ):
        self.mean_residence_time = mean_residence_time
        self.peclet_number = peclet_number
        self.dualPorosity_param = dualPorosity_param
        self.abDm = abDm
        self.thermal_degradation_coeff = thermal_degradation_coeff
        self.fracture_retardation = fracture_retardation
        self.matrix_retardation = matrix_retardation

    def __call__(self, s):
        Rf  = self.fracture_retardation
        Rm  = self.matrix_retardation
        Pe  = self.peclet_number
        t   = self.mean_residence_time
        dP  = self.dualPorosity_param
        abDm = self.abDm
        k   = self.thermal_degradation_coeff

        term1 = np.tanh(np.sqrt(Rm * (s + k)) * abDm)
        term2 = Rf * (s + k) + dP * np.sqrt(Rm * (s + k)) * term1
        term3 = 4 * t / Pe * term2
        term4 = np.sqrt(1 + term3)
        return np.exp(Pe / 2 * (1 - term4))


class GroundWaterFiniteMatrixSolution_2:
    """Alternative finite-matrix formulation with explicit porosity/water ratios."""

    def __init__(
        self,
        mean_residence_time,
        peclet_number,
        matrix_diffusion_parameter,
        porosity_ratio,
        water_ratio,
        thermal_degradation_coeff,
        fracture_retardation,
        matrix_retardation,
    ):
        self.mean_residence_time = mean_residence_time
        self.peclet_number = peclet_number
        self.matrix_diffusion_parameter = matrix_diffusion_parameter
        self.porosity_ratio = porosity_ratio
        self.water_ratio = water_ratio
        self.thermal_degradation_coeff = thermal_degradation_coeff
        self.fracture_retardation = fracture_retardation
        self.matrix_retardation = matrix_retardation

    def __call__(self, s):
        Rf  = self.fracture_retardation
        Rm  = self.matrix_retardation
        Pe  = self.peclet_number
        t   = self.mean_residence_time
        x1  = self.matrix_diffusion_parameter
        x2  = self.porosity_ratio
        x3  = self.water_ratio
        k   = self.thermal_degradation_coeff

        term1 = np.tanh(np.sqrt(Rm * (s + k)) * (1 / x1) * (1 / 2 * x3 - 1))
        term2 = Rf * (s + k) + x2 * x1 * np.sqrt(Rm * (s + k)) * term1
        term3 = 4 * t / Pe * term2
        term4 = np.sqrt(1 + term3)
        return np.exp(Pe / 2 * (1 - term4))


class GroundWaterInfiniteMatrixSolution:
    """Laplace-domain solution for infinite matrix (single/dual porosity) transport."""

    def __init__(
        self,
        mean_residence_time,
        peclet_number,
        dualPorosity_param,
        thermal_degradation_coeff,
        fracture_retardation,
        matrix_retardation,
    ):
        self.mean_residence_time = mean_residence_time
        self.peclet_number = peclet_number
        self.dualPorosity_param = dualPorosity_param
        self.thermal_degradation_coeff = thermal_degradation_coeff
        self.fracture_retardation = fracture_retardation
        self.matrix_retardation = matrix_retardation

    def __call__(self, s):
        Rf  = self.fracture_retardation
        Rm  = self.matrix_retardation
        Pe  = self.peclet_number
        t   = self.mean_residence_time
        k   = self.thermal_degradation_coeff
        dP  = self.dualPorosity_param

        term2 = Rf * (s + k) + dP * np.sqrt(Rm * (s + k))
        term3 = 4 * t / Pe * term2
        term4 = np.sqrt(1 + term3)
        return np.exp(Pe / 2 * (1 - term4))


# ---------------------------------------------------------------------------
# Input / boundary condition classes
# ---------------------------------------------------------------------------

class Input_Pulses_of_Tracer:
    """Laplace-domain representation of one or more tracer injection pulses."""

    def __init__(
        self,
        background_concentration,
        injection_concentrations,
        injection_durations,
    ):
        self.background_concentration = background_concentration
        self.injection_concentrations = injection_concentrations
        self.injection_durations = injection_durations
        self.relative_concentrations = injection_concentrations - background_concentration
        self.num_slugs = len(self.injection_durations)

    def __call__(self, s):
        C_R = self.relative_concentrations
        T_p = self.injection_durations

        first_slug      = C_R[0] * (1 - np.exp(-s * T_p[0])) / s
        displacing_water = C_R[-1] * np.exp(-s * T_p[-1]) / s

        if len(self.injection_concentrations) > 2:
            subsequent_slugs = 0
            for i in range(1, self.num_slugs):
                subsequent_slugs = (
                    subsequent_slugs
                    + C_R[i] * (np.exp(-s * T_p[i - 1]) - np.exp(-s * T_p[i])) / s
                )
            return first_slug + subsequent_slugs + displacing_water
        else:
            return first_slug + displacing_water


class Tracer_Injection_with_Background_Concentration:
    """Laplace-domain input for a single slug with a non-zero background."""

    def __init__(
        self,
        background_concentration,
        injection_concentration,
        injection_duration,
    ):
        dimensionless_background_concentration = background_concentration / (
            background_concentration - injection_concentration
        )
        self.dimensionless_background_concentration = dimensionless_background_concentration
        self.injection_duration = injection_duration

    def __call__(self, s):
        C_0D = self.dimensionless_background_concentration
        T_p  = self.injection_duration
        return (1 - (1 - C_0D) * np.exp(-s * T_p)) / s


# ---------------------------------------------------------------------------
# Modifier / node classes
# ---------------------------------------------------------------------------

class WellboreStorage:
    def __init__(self, wellbore_storage_coeff):
        self.wellbore_storage_coeff = wellbore_storage_coeff

    def __call__(self, s):
        return self.wellbore_storage_coeff / (self.wellbore_storage_coeff + s)


class PipelineDelay:
    def __init__(self, delay_time):
        self.delay_time = delay_time

    def __call__(self, s):
        return np.exp(-1 * self.delay_time * s)


class Recirculation:
    def __init__(self, recirculation_ratio):
        self.recirculation_ratio = recirculation_ratio

    def __call__(self, F):
        return F / (1 - self.recirculation_ratio * F)


class RELAP_Modifed:
    """
    Composite RELAP Laplace-domain model: groundwater × input × modifiers.

    All sub-callables accept numpy arrays so the composed __call__ also
    accepts numpy arrays, enabling batch evaluation in dehoog_invert.
    """

    def __init__(
        self,
        Ground_Water,
        Input_Instance,
        wellbore_storage_node=lambda s: 1,
        pipeline_delay_node=lambda s: 1,
        recirculation=False,
    ):
        self.Ground_Water = Ground_Water
        self.Input_Instance = Input_Instance
        self.wellbore_storage_node = wellbore_storage_node
        self.pipeline_delay_node = pipeline_delay_node
        self.recirculation = recirculation

    def __call__(self, s):
        loop_solution = (
            self.Ground_Water(s)
            * self.pipeline_delay_node(s)
            * self.wellbore_storage_node(s)
        )
        if self.recirculation:
            loop_solution = self.recirculation(loop_solution)
        return loop_solution * self.Input_Instance(s)


# ---------------------------------------------------------------------------
# Simulation functions (use dehoog_invert instead of mpmath.invertlaplace)
# ---------------------------------------------------------------------------

def Simulate_RELAP_Relative(
    RELAP_instance,
    time_points,
    background_concentration,
    M=10,
    dps=8,
):
    """
    Compute absolute tracer concentration at each time point.

    Uses dehoog_invert to invert RELAP_instance(s) over all time points
    in a single batch per time point (replacing the per-point invertlaplace loop).

    Parameters
    ----------
    RELAP_instance : callable
        Laplace-domain model; must accept a complex128 numpy array.
    time_points : array-like
        Times at which to evaluate the concentration.
    background_concentration : float
        Added to the inverted relative concentration.
    M, dps : int
        Passed to dehoog_invert (default M=10 matches mpmath at dps=8).

    Returns
    -------
    absolute_concentration : ndarray, float64
    """
    time_points = np.asarray(time_points, dtype=float)
    relative_concentration = dehoog_invert(RELAP_instance, time_points, M=M, dps=dps)
    return background_concentration + relative_concentration


def Simulate_RELAP_Dimensionless(
    RELAP_instance,
    time_points,
    injection_concentration,
    background_concentration,
    normalize=False,
    M=10,
    dps=8,
):
    """
    Compute dimensionless (or dimensional) concentration at each time point.

    Parameters
    ----------
    normalize : bool
        If True, return the raw dimensionless concentration without scaling.
    """
    time_points = np.asarray(time_points, dtype=float)
    dimensionless_concentration = dehoog_invert(RELAP_instance, time_points, M=M, dps=dps)
    if normalize:
        return dimensionless_concentration
    return background_concentration + (
        injection_concentration - background_concentration
    ) * dimensionless_concentration
