import numpy as np
from mpmath import *

mp.dps = 8
mp.pretty = True


class GroundWaterFiniteMatrixSolution:
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
        Rf = self.fracture_retardation
        Rm = self.matrix_retardation
        Pe = self.peclet_number
        t = self.mean_residence_time
        dP = self.dualPorosity_param
        abDm = self.abDm
        k = self.thermal_degradation_coeff

        term1 = tanh(sqrt(Rm * (s + k)) * abDm)
        term2 = Rf * (s + k) + dP * sqrt(Rm * (s + k)) * term1
        term3 = 4 * t / Pe * term2
        term4 = sqrt(1 + term3)
        term5 = exp(Pe / 2 * (1 - term4))

        return term5


class GroundWaterFiniteMatrixSolution_2:
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
        Rf = self.fracture_retardation
        Rm = self.matrix_retardation
        Pe = self.peclet_number
        t = self.mean_residence_time
        x1 = self.matrix_diffusion_parameter
        x2 = self.porosity_ratio
        x3 = self.water_ratio
        k = self.thermal_degradation_coeff

        term1 = tanh(sqrt(Rm * (s + k)) * (1 / x1) * (1 / 2 * x3 - 1))
        term2 = Rf * (s + k) + x2 * x1 * sqrt(Rm * (s + k)) * term1
        term3 = 4 * t / Pe * term2
        term4 = sqrt(1 + term3)
        term5 = exp(Pe / 2 * (1 - term4))

        return term5


class GroundWaterInfiniteMatrixSolution:
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
        Rf = self.fracture_retardation
        Rm = self.matrix_retardation
        Pe = self.peclet_number
        t = self.mean_residence_time
        k = self.thermal_degradation_coeff
        dP = self.dualPorosity_param

        term2 = Rf * (s + k) + dP * sqrt(Rm * (s + k))
        term3 = 4 * t / Pe * term2
        term4 = sqrt(1 + term3)
        term5 = exp(Pe / 2 * (1 - term4))

        return term5


class Input_Pulses_of_Tracer:
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

        first_slug = C_R[0] * (1 - exp(-s * T_p[0])) / s
        displacing_water = C_R[-1] * exp(-s * T_p[-1]) / s

        if len(self.injection_concentrations) > 2:
            subsequent_slugs = 0
            for i in range(1, self.num_slugs):
                subsequent_slugs = (
                    subsequent_slugs
                    + C_R[i] * (exp(-s * T_p[i - 1]) - exp(-s * T_p[i])) / s
                )
            slugs_superposition = first_slug + subsequent_slugs + displacing_water
        else:
            slugs_superposition = first_slug + displacing_water

        return slugs_superposition


class Tracer_Injection_with_Background_Concentration:
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
        T_p = self.injection_duration
        input_function = (1 - (1 - C_0D) * exp(-s * T_p)) / s
        return input_function


class WellboreStorage:
    def __init__(self, wellbore_storage_coeff):
        self.wellbore_storage_coeff = wellbore_storage_coeff

    def __call__(self, s):
        return self.wellbore_storage_coeff / (self.wellbore_storage_coeff + s)


class PipelineDelay:
    def __init__(self, delay_time):
        self.delay_time = delay_time

    def __call__(self, s):
        return exp(-1 * self.delay_time * s)


class Recirculation:
    def __init__(self, recirculation_ratio):
        self.recirculation_ratio = recirculation_ratio

    def __call__(self, F):
        return F / (1 - self.recirculation_ratio * F)


class RELAP_Modifed:
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
            self.Ground_Water(s) * self.pipeline_delay_node(s) * self.wellbore_storage_node(s)
        )
        if self.recirculation:
            loop_solution = self.recirculation(loop_solution)
        full_solution = loop_solution * self.Input_Instance(s)
        return full_solution


def Simulate_RELAP_Dimensionless(
    RELAP_instance,
    time_points,
    injection_concentration,
    background_concentration,
    normalize=False,
):
    dimensionless_concentration = [
        invertlaplace(RELAP_instance, time_point, method="dehoog") for time_point in time_points
    ]
    dimensionless_concentration = np.array(dimensionless_concentration)
    if normalize:
        return dimensionless_concentration
    dimensional_concentration = background_concentration + (
        injection_concentration - background_concentration
    ) * dimensionless_concentration
    return dimensional_concentration


def Simulate_RELAP_Relative(
    RELAP_instance,
    time_points,
    background_concentration,
):
    relative_concentration = [
        invertlaplace(RELAP_instance, time_point, method="dehoog") for time_point in time_points
    ]
    relative_concentration = np.array(relative_concentration, dtype=np.float64)
    absolute_concentration = background_concentration + relative_concentration

    return absolute_concentration

