import numpy as np
import mpmath as mpm

from .RELAP_v5 import *


def simulateSinglePorosity(
    mean_residence_time,
    peclet_number,
    frac_retard,
    time_points,
    bckgrnd_conc,
    inj_concs,
    inj_durs,
    recRatio=0,
    wsCoef=0,
    dps=8,
):
    mpm.mp.dps = dps
    gw_inf = GroundWaterInfiniteMatrixSolution(
        mean_residence_time,
        peclet_number,
        0,
        0,
        fracture_retardation=frac_retard,
        matrix_retardation=1,
    )

    tracer_injection = Input_Pulses_of_Tracer(
        background_concentration=bckgrnd_conc,
        injection_concentrations=inj_concs,
        injection_durations=inj_durs,
    )

    reciculation = Recirculation(recRatio)
    if wsCoef > 0:
        wellbore_storage = WellboreStorage(wsCoef)
    else:
        wellbore_storage = lambda x: 1
    relap_instance = RELAP_Modifed(
        Ground_Water=gw_inf,
        Input_Instance=tracer_injection,
        recirculation=reciculation,
        wellbore_storage_node=wellbore_storage,
    )

    conc_values = Simulate_RELAP_Relative(
        relap_instance,
        time_points=time_points,
        background_concentration=bckgrnd_conc,
    )

    return np.array(conc_values, dtype=np.float64)


def simulateDualPorosity(
    mean_residence_time,
    peclet_number,
    frac_retard,
    time_points,
    bckgrnd_conc,
    inj_concs,
    inj_durs,
    dualPorosity_param,
    matrix_retardation,
    recRatio=0,
    wsCoef=0,
    delay_time=0,
    dps=8,
):
    mpm.mp.dps = dps
    gw_inf = GroundWaterInfiniteMatrixSolution(
        mean_residence_time,
        peclet_number,
        dualPorosity_param,
        0,
        fracture_retardation=frac_retard,
        matrix_retardation=matrix_retardation,
    )

    tracer_injection = Input_Pulses_of_Tracer(
        background_concentration=bckgrnd_conc,
        injection_concentrations=inj_concs,
        injection_durations=inj_durs,
    )

    reciculation = Recirculation(recRatio)
    pipelinedelay = PipelineDelay(delay_time=delay_time)
    if wsCoef > 0:
        wellbore_storage = WellboreStorage(wsCoef)
    else:
        wellbore_storage = lambda x: 1

    relap_instance = RELAP_Modifed(
        Ground_Water=gw_inf,
        Input_Instance=tracer_injection,
        recirculation=reciculation,
        pipeline_delay_node=pipelinedelay,
        wellbore_storage_node=wellbore_storage,
    )

    conc_values = Simulate_RELAP_Relative(
        relap_instance,
        time_points=time_points,
        background_concentration=bckgrnd_conc,
    )

    return np.array(conc_values, dtype=np.float64)


def simulateDualPorosityFinite(
    mean_residence_time,
    peclet_number,
    frac_retard,
    time_points,
    bckgrnd_conc,
    inj_concs,
    inj_durs,
    dualPorosity_param,
    abDm,
    matrix_retardation,
    recRatio=0,
    delay_time=0,
    wsCoef=0,
    dps=8,
):
    mpm.mp.dps = dps
    gw_inf = GroundWaterFiniteMatrixSolution(
        mean_residence_time,
        peclet_number,
        dualPorosity_param,
        abDm,
        0,
        fracture_retardation=frac_retard,
        matrix_retardation=matrix_retardation,
    )

    tracer_injection = Input_Pulses_of_Tracer(
        background_concentration=bckgrnd_conc,
        injection_concentrations=inj_concs,
        injection_durations=inj_durs,
    )

    reciculation = Recirculation(recRatio)
    pipelinedelay = PipelineDelay(delay_time=delay_time)
    if wsCoef > 0:
        wellbore_storage = WellboreStorage(wsCoef)
    else:
        wellbore_storage = lambda x: 1

    relap_instance = RELAP_Modifed(
        Ground_Water=gw_inf,
        Input_Isntance=tracer_injection,
        recirculation=reciculation,
        pipeline_delay_node=pipelinedelay,
        wellbore_storage_node=wellbore_storage,
    )

    conc_values = Simulate_RELAP_Relative(
        relap_instance,
        time_points=time_points,
        background_concentration=bckgrnd_conc,
    )

    return np.array(conc_values, dtype=np.float64)


def simulateDualPorosityFinite2(
    time_points,
    bckgrnd_conc,
    inj_concs,
    inj_durs,
    mrt,
    pe,
    frac_retard,
    phiRb,
    diffCoef,
    fracSpacing,
    matr_retard,
    recRatio=0,
    delay_time=0,
    wsCoef=0,
    dps=8,
):
    mpm.mp.dps = dps
    dpp = phiRb * np.sqrt(diffCoef * 3600)
    abDm = np.sqrt(matr_retard / (diffCoef * 3600)) * fracSpacing / 2
    gw_inf = GroundWaterFiniteMatrixSolution(
        mrt,
        pe,
        dpp,
        abDm,
        0,
        fracture_retardation=frac_retard,
        matrix_retardation=matr_retard,
    )

    tracer_injection = Input_Pulses_of_Tracer(
        background_concentration=bckgrnd_conc,
        injection_concentrations=inj_concs,
        injection_durations=inj_durs,
    )

    reciculation = Recirculation(recRatio)
    pipelinedelay = PipelineDelay(delay_time=delay_time)
    if wsCoef > 0:
        wellbore_storage = WellboreStorage(wsCoef)
    else:
        wellbore_storage = lambda x: 1

    relap_instance = RELAP_Modifed(
        Ground_Water=gw_inf,
        Input_Instance=tracer_injection,
        recirculation=reciculation,
        pipeline_delay_node=pipelinedelay,
        wellbore_storage_node=wellbore_storage,
    )

    conc_values = Simulate_RELAP_Relative(
        relap_instance,
        time_points=time_points,
        background_concentration=bckgrnd_conc,
    )

    return np.array(conc_values, dtype=np.float64)

