#!/usr/bin/env python3

import math
from bisect import bisect_left
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from werkzeug.exceptions import BadRequest
from flask import current_app

from api.helpers.carbon_intensity_c3lab import get_carbon_intensity_list as get_carbon_intensity_list_c3lab
from api.helpers.carbon_intensity_azure import get_carbon_intensity_list as get_carbon_intensity_list_azure
from api.helpers.carbon_intensity_emap import get_carbon_intensity_list as get_carbon_intensity_list_emap
from api.models.common import CarbonDataSource


FLOAT_PRECISION = 8


def get_carbon_intensity_list(iso: str, start: datetime, end: datetime,
        carbon_data_source: CarbonDataSource, use_prediction: bool,
        desired_renewable_ratio: float = None) -> list[dict]:
    """Retrieve the carbon intensity time series data in the given time window.

        Args:
            iso: the ISO region name.
            start: the start time.
            end: the end time.
            carbon_data_source: the source of the carbon data.
            use_prediction: whether to use prediction or actual data.

        Returns:
            A list of time series data.
    """
    current_app.logger.info(f'Getting carbon intensity for {iso} in range ({start}, {end})')
    match carbon_data_source:
        case CarbonDataSource.C3Lab:
            return get_carbon_intensity_list_c3lab(iso, start, end, use_prediction, desired_renewable_ratio)
        case CarbonDataSource.Azure:
            if desired_renewable_ratio is not None:
                raise ValueError('Azure carbon data source does not support custom renewable ratio.')
            return get_carbon_intensity_list_azure(iso, start, end, use_prediction)
        case CarbonDataSource.EMap:
            if desired_renewable_ratio is not None:
                raise ValueError('Electricity map carbon data source does not support custom renewable ratio.')
            return get_carbon_intensity_list_emap(iso, start, end, use_prediction)
        case _:
            raise NotImplementedError()

def convert_carbon_intensity_list_to_dict(l_carbon_intensity: list[dict]) -> dict[datetime, float]:
    d_carbon_intensity_by_timestamp: dict[datetime, float] = {}
    for d in l_carbon_intensity:
        timestamp = d['timestamp']
        carbon_intensity = d['carbon_intensity']
        d_carbon_intensity_by_timestamp[timestamp] = carbon_intensity
    return d_carbon_intensity_by_timestamp


def get_carbon_intensity_interval(timestamps: list[datetime]) -> timedelta:
    """Deduce the interval from a series of timestamps returned from the database."""
    if len(timestamps) == 0:
        raise ValueError("Invalid argument: empty carbon intensity list.")
    if len(timestamps) == 1:
        return timedelta(hours=1)
    timestamp_deltas = np.diff(timestamps)
    values, counts = np.unique(timestamp_deltas, return_counts=True)
    return values[np.argmax(counts)]


def calculate_total_carbon_emissions_naive(start: datetime, runtime: timedelta,
                                     max_delay: timedelta,
                                     input_transfer_time: timedelta,
                                     output_transfer_time: timedelta,
                                     compute_carbon_emission_rates: pd.Series,
                                     transfer_carbon_emission_rates: pd.Series,
                                     ) -> tuple[float, timedelta]:
    """Calculate the total carbon emission, including both compute and data transfer emissions.

        Args:
            start: start time of a workload.
            runtime: runtime of a workload.
            max_delay: the amount of delay that a workload can tolerate.
            transfer_input_time: time to transfer input data.
            transfer_output_time: time to transfer output data.
            compute_carbon_emission_rates: the compute carbon emission rate in gCO2/s.
            transfer_carbon_emission_rates: the aggregated data transfer carbon emission rate in gCO2/s.

        Returns:
            Total carbon emissions in kgCO2.
            Optimal delay of start time, if applicable.
    """
    current_app.logger.info('Calculating total carbon emissions ...')
    perf_start_time = time.time()
    if runtime <= timedelta():
        raise BadRequest("Runtime must be positive.")

    if input_transfer_time + output_transfer_time > max_delay:
        raise ValueError("Not enough time to finish before deadline.")

    perf_counter_step_size = 0

    def _integrate_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the integral of a series, and return the x and y values for a later np.interp call.

            Args:
                series: A step function represented by a pandas series with a datetime index and numeric values.

            Returns:
                A tuple of two numpy arrays, the first one is the UNIX timestamp values and the second one is the y values, or cumulative sum of the original series.
        """
        # Calculate the size of each step, in seconds.
        timestamp_seconds = series.index.astype(np.int64) // (10**9)
        steps_seconds = (-timestamp_seconds.to_series().diff(-1)).to_numpy(na_value=0)
        # Intergral of the original step function
        cumsum = np.cumsum(steps_seconds * series.to_numpy())
        return np.array(timestamp_seconds), np.insert(cumsum[:-1], 0, 0)

    def _calculate_carbon_emission_in_interval(start: datetime, end: datetime,
                                               carbon_emission_cumsum: tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate the carbon emission in the given interval, using the cumulative sum on the timeseries."""
        if carbon_emission_cumsum[0].size == 0 or start >= end:
            return 0.
        # Convert start and end to UNIX timestamp.
        (start_timestamp, end_timestamp) = (start.timestamp(), end.timestamp())

        # look up in existing carbon emission rates' cumulative carbon emission.
        (x_values, y_values) = carbon_emission_cumsum
        (start_cumsum, end_cumsum) = np.interp([start_timestamp, end_timestamp], x_values, y_values)
        return round(end_cumsum - start_cumsum, FLOAT_PRECISION)

    def _calculate_total_emission(curr_wait_times: list[datetime], breakdown=False):
        """Calculate the total carbon emissions based on the current wait times."""
        (input_wait, compute_wait, output_wait) = curr_wait_times
        input_transfer_start = start + input_wait
        input_transfer_end = input_transfer_start + input_transfer_time
        compute_start = input_transfer_end + compute_wait
        compute_end = compute_start + runtime
        output_transfer_start = compute_end + output_wait
        output_transfer_end = output_transfer_start + output_transfer_time

        input_transfer_emission = _calculate_carbon_emission_in_interval(
                input_transfer_start,
                input_transfer_end,
                transfer_carbon_cumsum)
        compute_emission = _calculate_carbon_emission_in_interval(
                compute_start,
                compute_end,
                compute_carbon_cumsum)
        output_transfer_emission = _calculate_carbon_emission_in_interval(
                output_transfer_start,
                output_transfer_end,
                transfer_carbon_cumsum)
        if breakdown:
            return (compute_emission , input_transfer_emission + output_transfer_emission)
        else:
            return input_transfer_emission + compute_emission + output_transfer_emission

    def _get_marginal_emission_rate_delta_and_step_size(curr_wait_times: list[datetime],
                                                        moving_index: int) -> tuple[float, timedelta]:
        """Calculate the total marginal emission rate delta by moving the n-th wait time and the minimum step size across all three steps, while the emission rates stay the same."""
        def _impl_single_interval(start: datetime, end: datetime, carbon_emission_rates: pd.Series):
            """Calculates the marginal emission rate delta and step size while the emission rate stays the same."""
            def _get_value_at_timestamp(target: datetime) -> float:
                index = carbon_emission_rates.index.searchsorted(target, side='right') - 1
                return carbon_emission_rates[index]
            def _get_next_timestamp(target: datetime) -> datetime:
                index = carbon_emission_rates.index.searchsorted(target, side='right')
                return carbon_emission_rates.index[index]
            marginal_rate_start = _get_value_at_timestamp(start)
            marginal_rate_end = _get_value_at_timestamp(end)
            step_size_start = _get_next_timestamp(start) - start
            step_size_end = _get_next_timestamp(end) - end
            return marginal_rate_end - marginal_rate_start, min(step_size_start, step_size_end)

        assert moving_index >= 0 and moving_index < NUM_TIME_VARIABLES, "Invalid moving index."

        (input_wait, compute_wait, output_wait) = curr_wait_times
        input_transfer_start = start + input_wait
        input_transfer_end = input_transfer_start + input_transfer_time
        compute_start = input_transfer_end + compute_wait
        compute_end = compute_start + runtime
        output_transfer_start = compute_end + output_wait
        output_transfer_end = output_transfer_start + output_transfer_time

        sum_rate_delta = 0
        min_step_size = timedelta(days=365)
        # Moving the first time afects the latter two steps, and moving the second time affects the last step.
        if moving_index <= 0 and not transfer_carbon_emission_rates.empty:
            input_rate_delta, input_step_size = _impl_single_interval(input_transfer_start, input_transfer_end,
                                                                      transfer_carbon_emission_rates)
            sum_rate_delta += input_rate_delta
            min_step_size = min(min_step_size, input_step_size)
        if moving_index <= 1:
            compute_rate_delta, compute_step_size = _impl_single_interval(compute_start, compute_end,
                                                                          compute_carbon_emission_rates)
            sum_rate_delta += compute_rate_delta
            min_step_size = min(min_step_size, compute_step_size)
        if moving_index <= 2 and not transfer_carbon_emission_rates.empty:
            output_rate_delta, output_step_size = _impl_single_interval(output_transfer_start, output_transfer_end,
                                                                        transfer_carbon_emission_rates)
            sum_rate_delta += output_rate_delta
            min_step_size = min(min_step_size, output_step_size)

        nonlocal perf_counter_step_size
        perf_counter_step_size += 1

        return sum_rate_delta, min_step_size

    def _advance_wait_times_and_get_emission_delta(curr_wait_times: list[timedelta],
                      total_wait_limit: timedelta) -> tuple[list[timedelta], float]:
        """Advance the wait time to the next step and return the accumulated emission delta."""
        moving_index = NUM_TIME_VARIABLES - 1
        marginal_emission_rate_delta, step_size = _get_marginal_emission_rate_delta_and_step_size(curr_wait_times, moving_index)
        if sum(curr_wait_times, timedelta()) + step_size <= total_wait_limit:
            curr_wait_times[moving_index] += step_size
            emission_delta = marginal_emission_rate_delta * step_size.total_seconds()
            return emission_delta
        else:
            while moving_index > 0:
                curr_wait_times[moving_index] = timedelta()
                moving_index -= 1
                _, step_size = _get_marginal_emission_rate_delta_and_step_size(curr_wait_times, moving_index)
                if sum(curr_wait_times, timedelta()) + step_size <= total_wait_limit:
                    curr_wait_times[moving_index] += step_size
                    return math.nan
            curr_wait_times = [timedelta()] * len(curr_wait_times)
            raise StopIteration()

    # Transform the carbon intensity rates into cumulative sum for faster lookup.
    compute_carbon_cumsum = _integrate_series(compute_carbon_emission_rates)
    transfer_carbon_cumsum = _integrate_series(transfer_carbon_emission_rates)

    NUM_TIME_VARIABLES = 3  # input wait, compute wait and output wait
    t_total_wait_limit = max_delay - input_transfer_time - output_transfer_time

    curr_wait_times = [timedelta()] * NUM_TIME_VARIABLES
    current_emission = _calculate_total_emission(curr_wait_times)
    min_time_values = curr_wait_times.copy()
    min_total_emission = current_emission

    perf_counter_calculate_total = 0

    try:
        while True:
            emission_delta = _advance_wait_times_and_get_emission_delta(curr_wait_times, t_total_wait_limit)
            if math.isnan(emission_delta):
                # Re-calculate total emission is delta is unknown
                current_emission = _calculate_total_emission(curr_wait_times)
                perf_counter_calculate_total += 1
            else:
                current_emission += emission_delta
            if current_emission < min_total_emission and not math.isclose(current_emission, min_total_emission):
                min_total_emission = current_emission
                min_time_values = curr_wait_times.copy()
    except StopIteration:
        pass

    perf_elapsed = time.time() - perf_start_time
    current_app.logger.info('calculate_total_carbon_emissions() took %.3f seconds' % perf_elapsed)
    current_app.logger.info('perf_counter_calculate_total = %d' % perf_counter_calculate_total)
    current_app.logger.info('perf_counter_step_size = %d' % perf_counter_step_size)

    curr_wait_times = min_time_values.copy()

    (input_wait, compute_wait, output_wait) = curr_wait_times
    input_transfer_start = start + input_wait
    input_transfer_end = input_transfer_start + input_transfer_time
    compute_start = input_transfer_end + compute_wait
    compute_end = compute_start + runtime
    output_transfer_start = compute_end + output_wait
    output_transfer_end = output_transfer_start + output_transfer_time

    return (_calculate_total_emission(curr_wait_times, True), {
        'input_transfer_start': input_transfer_start,
        'input_transfer_duration': input_transfer_time,
        'input_transfer_end': input_transfer_end,
        'compute_start': compute_start,
        'compute_duration': runtime,
        'compute_end': compute_end,
        'output_transfer_start': output_transfer_start,
        'output_transfer_duration': output_transfer_time,
        'output_transfer_end': output_transfer_end,
        'min_start': start,
        'max_end': start + runtime + max_delay,
        'total_transfer_time': input_transfer_time + output_transfer_time,
    })

def calculate_total_carbon_emissions_linear(start: datetime, runtime: timedelta,
                                     max_delay: timedelta,
                                     input_transfer_time: timedelta,
                                     output_transfer_time: timedelta,
                                     compute_carbon_emission_rates: pd.Series,
                                     transfer_carbon_emission_rates: pd.Series,
                                     ) -> tuple[float, timedelta]:
    """Calculate the total carbon emission, including both compute and data transfer emissions.

        Args:
            start: start time of a workload.
            runtime: runtime of a workload.
            max_delay: the amount of delay that a workload can tolerate.
            transfer_input_time: time to transfer input data.
            transfer_output_time: time to transfer output data.
            compute_carbon_emission_rates: the compute carbon emission rate in gCO2/s.
            transfer_carbon_emission_rates: the aggregated data transfer carbon emission rate in gCO2/s.

        Returns:
            Total carbon emissions in kgCO2.
            Optimal delay of start time, if applicable.
    """
    current_app.logger.info('Calculating total carbon emissions ...')
    if runtime <= timedelta():
        raise BadRequest("Runtime must be positive.")

    if input_transfer_time + output_transfer_time > max_delay:
        raise ValueError("Not enough time to finish before deadline.")

    perf_start_time = time.time()

    # Linear algorithm described in appendix A of the paper.

    def calculate_integral_optimized(D, T_min, T_max, steps):
        """Calculate the integral of a step function with a given window [T_min, T_max] and step size D."""
        if len(steps) == 0:
            return {}

        integral = {}
        last_step_time = T_min
        last_step_value = 0

        # Add a sentinel value at the end of steps for easy calculation
        steps_with_sentinel = steps + [(float('inf'), 0)]

        # Pre-calculate the integral up to each step
        precalc_integral = [0]
        for step_time, step_value in steps_with_sentinel:
            precalc_integral.append(precalc_integral[-1] + last_step_value * (step_time - last_step_time))
            last_step_time = step_time
            last_step_value = step_value

        # Calculate the integral at each time t using the pre-calculated values
        for t in range(T_min, T_max + 1):
            # Find the steps that bound the current window [t, t+D)
            start_index = bisect_left(steps_with_sentinel, (t, float('inf'))) - 1
            end_index = bisect_left(steps_with_sentinel, (t + D, float('inf'))) - 1

            # Calculate the total using precalculated integrals and adjusting for the partial areas
            total = precalc_integral[end_index + 1] - precalc_integral[start_index + 1]
            total -= steps_with_sentinel[start_index][1] * (t - steps_with_sentinel[start_index][0])
            total += steps_with_sentinel[end_index][1] * ((t + D) - steps_with_sentinel[end_index][0])

            # Need to round to avoid floating point inequality for later comparison.
            integral[t] = round(total, FLOAT_PRECISION)

        return integral

    # Procedure to get optimal points
    def get_optimal_points(f_I, T_min, T_max, reverse) -> list[int]:
        """Get a sorted list of optimal points for a given integral function f_I.

            Optimal points are defined as the series of turning points (where the derivative of f_I changes) and the values are non-increasaing. We do not need to consider other points because they are less optimal (has higher total carbon emissions on the curve).
        """
        if len(f_I) == 0:
            return []

        OP = []
        last_value = float('inf')
        interval = range(T_min, T_max + 1) if not reverse else range(T_max, T_min - 1, -1)

        def is_tuning_point(t):
            """Check if the derivative of f_I changes at time t."""
            # Approximate the derivative by the difference
            # assuming the delta is small, here it's taken as 1 for simplicity
            delta = 1
            t_prev = max(t - delta, T_min)
            t_next = min(t + delta, T_max)

            # Approximate the derivative by the difference
            f_prime_previous = f_I[t] - f_I[t_prev]
            f_prime_next = f_I[t_next] - f_I[t]

            # Check if this is a turning point
            return not math.isclose(f_prime_previous, f_prime_next, rel_tol=1e-6)

        for t in interval:
            if t != T_min and t != T_max and not is_tuning_point(t):
                continue
            current_value = f_I[t]
            # In the reverse case, we consider all the points of equal value, because a later point with equal value can still be picked.
            # In the non-reverse case, we only consider the first point of equal value (hence strict comparison), because an earlier time with equal value is always available and thus preferred.
            if (reverse and current_value <= last_value) or (not reverse and current_value < last_value):
                last_value = current_value
                OP.append(t)

        if reverse:
            OP.reverse()
        return OP

    # Main function to optimize total carbon
    def optimize_total_carbon(f1_steps, f2_steps, f3_steps, T0, T4, D1, D2, D3):
        integrals = {}
        OPs = {}
        assert T0 == 0, "T0 should be set to 0."

        perf_start_time = time.time()

        # Calculate integrals and optimal points for each function
        for i, (f_steps, D) in enumerate(zip([f1_steps, f2_steps, f3_steps], [D1, D2, D3]), start=1):
            Tmin = T0 + sum([D1, D2, D3][:i-1])
            Tmax = T4 - sum([D1, D2, D3][i-1:])
            integral = calculate_integral_optimized(D, Tmin, Tmax, f_steps)
            integrals[i] = integral

            if i == 1:
                OPs[i] = get_optimal_points(integral, Tmin, Tmax, False)
            elif i == 3:
                # TODO: debug difference from existing example (see graph)
                OPs[i] = get_optimal_points(integral, Tmin, Tmax, True)

        perf_elapsed = time.time() - perf_start_time
        current_app.logger.debug('Pre-calculation of integral and optimal points took %.3f seconds' % perf_elapsed)

        min_integral_total = float('inf')
        T_optimal: list[int] = [np.nan, np.nan, np.nan]
        min_integrals: list[float] = [np.nan, np.nan, np.nan]

        # Find the optimal time intervals
        for t2 in range(T0 + D1, T4 - D2 - D3 + 1):
            # Calculate minimum integral for f1
            if integrals[1]:
                t1_max = t2 - D1
                op1 = max([op for op in OPs[1] if op <= t1_max], default=t1_max)
                # Order matters in argmin call below, as we want to pick the earlier time in case of equal values.
                # In this case, op1 <= t1_max
                optimal_t1 = min((integrals[1][op1], op1), (integrals[1][t1_max], t1_max), key=lambda x: x[0])[1]
                min_integral_1 = integrals[1][optimal_t1]
            else:
                optimal_t1 = 0
                min_integral_1 = 0

            # Calculate minimum integral for f3
            if integrals[3]:
                t3_min = t2 + D2
                op3 = min([op for op in OPs[3] if op >= t3_min], default=t3_min)
                # Order matters in argmin call below, as we want to pick the earlier time in case of equal values.
                # In this case, t3_min <= op3
                optimal_t3 = min((integrals[3][t3_min], t3_min), (integrals[3][op3], op3), key=lambda x: x[0])[1]
                min_integral_3 = integrals[3][optimal_t3]
            else:
                optimal_t3 = t2 + D2
                min_integral_3 = 0

            # Compare total integral
            integral_total = min_integral_1 + integrals[2][t2] + min_integral_3
            if integral_total < min_integral_total:
                min_integral_total = integral_total
                T_optimal = [optimal_t1, t2, optimal_t3]
                min_integrals = [min_integral_1, integrals[2][t2], min_integral_3]

        perf_elapsed = time.time() - perf_start_time
        current_app.logger.debug('optimize_total_carbon() took %.3f seconds' % perf_elapsed)

        return T_optimal, min_integrals

    def wrapper_optimize_total_carbon(carbon_rates_1: pd.Series, carbon_rates_2: pd.Series, carbon_rates_3: pd.Series,
                                      T0: datetime, T4: datetime,
                                      D1: timedelta, D2: timedelta, D3: timedelta) -> \
                                        tuple[list[datetime], list[float]]:
        """Wrapper function to convert the parameters to seconds and call the integer-input optimize carbon function.

        Args:
            carbon_rates_1, carbon_rates_2, carbon_rates_3: carbon emission rates for the three step functions: input transfer, compute, and output transfer.
            T0_datetime: minimum start time of the entire job.
            T4_datetime: maximum end time of the entire job.
            D1, D2, D3: duration of each step: input transfer, compute, and output transfer.

        Returns:
            A tuple of: 1) a list of optimal start times for each step, and 2) a list of the corresponding carbon emissions.
        """
        def datetime_to_seconds(start_time, end_time):
            """Converts datetime objects to a total number of seconds since the start_time."""
            return int((end_time - start_time).total_seconds())

        def timedelta_to_seconds(td):
            """Converts timedelta objects to a total number of seconds."""
            return int(td.total_seconds())

        # Convert series to steps for our algorithm
        def convert_series_to_steps(f_series, T0_datetime):
            """Converts a pandas Series with a datetime index to a list of steps with seconds as time."""
            steps = [(int((t - T0_datetime).total_seconds()), value) for t, value in f_series.items()]
            return steps

        # Convert datetime and timedelta to seconds
        T0s = datetime_to_seconds(T0, T0)
        T4s = datetime_to_seconds(T0, T4)
        D1s = timedelta_to_seconds(D1)
        D2s = timedelta_to_seconds(D2)
        D3s = timedelta_to_seconds(D3)

        # Convert carbon rates to step functions
        f1_steps = convert_series_to_steps(carbon_rates_1, T0)
        f2_steps = convert_series_to_steps(carbon_rates_2, T0)
        f3_steps = convert_series_to_steps(carbon_rates_3, T0)

        ts, emissions = optimize_total_carbon(f1_steps, f2_steps, f3_steps, T0s, T4s, D1s, D2s, D3s)
        return [T0 + timedelta(seconds=t) for t in ts], emissions

    start_times, emissions = wrapper_optimize_total_carbon(
        transfer_carbon_emission_rates,
        compute_carbon_emission_rates,
        transfer_carbon_emission_rates,
        start,
        start + runtime + max_delay,
        input_transfer_time,
        runtime,
        output_transfer_time
    )

    perf_elapsed = time.time() - perf_start_time
    current_app.logger.info('calculate_total_carbon_emissions() took %.3f seconds' % perf_elapsed)

    (input_transfer_start, compute_start, output_transfer_start) = start_times
    input_transfer_end = input_transfer_start + input_transfer_time
    compute_end = compute_start + runtime
    output_transfer_end = output_transfer_start + output_transfer_time

    emission_output = (emissions[1], emissions[0] + emissions[2])

    return (emission_output, {
        'input_transfer_start': input_transfer_start,
        'input_transfer_duration': input_transfer_time,
        'input_transfer_end': input_transfer_end,
        'compute_start': compute_start,
        'compute_duration': runtime,
        'compute_end': compute_end,
        'output_transfer_start': output_transfer_start,
        'output_transfer_duration': output_transfer_time,
        'output_transfer_end': output_transfer_end,
        'min_start': start,
        'max_end': start + runtime + max_delay,
        'total_transfer_time': input_transfer_time + output_transfer_time,
    })
