"""
Formant Feature Extraction Library

Extracts formant (F1, F2) features from audio intervals.
Functions operate on parselmouth.Sound objects with specified time intervals.

Functions:
    extract_f1_mean: Mean first formant frequency
    extract_f1_sd: Standard deviation of F1
    extract_f1_range: Peak-to-peak F1 range
    extract_f1_max: Maximum F1 value
    extract_f1_max_time: Time of maximum F1 relative to interval start
    extract_f1_min: Minimum F1 value
    extract_f1_min_time: Time of minimum F1 relative to interval start
    extract_f1_midpoint: F1 at temporal midpoint of interval
    extract_f1_contour: F1 at specified percentage intervals
    extract_f2_mean: Mean second formant frequency
    extract_f2_sd: Standard deviation of F2
    extract_f2_range: Peak-to-peak F2 range
    extract_f2_max: Maximum F2 value
    extract_f2_max_time: Time of maximum F2 relative to interval start
    extract_f2_min: Minimum F2 value
    extract_f2_min_time: Time of minimum F2 relative to interval start
    extract_f2_midpoint: F2 at temporal midpoint of interval
    extract_f2_contour: F2 at specified percentage intervals

All functions return None if formant extraction fails.
"""

import numpy as np
import parselmouth

# Configuration constants
MAX_FORMANT = 5500
TIME_STEP = 0.01
MAX_N_FORMANTS = 5
WINDOW_LENGTH = 0.025

def _extract_formant_values(sound, start_time, end_time, formant_number):
    """Internal helper: Extract formant values and times from interval."""
    try:
        formant_obj = sound.to_formant_burg(
            time_step=TIME_STEP,
            max_number_of_formants=MAX_N_FORMANTS,
            maximum_formant=MAX_FORMANT,
            window_length=WINDOW_LENGTH
        )
        
        formant_times = formant_obj.xs()
        time_mask = (formant_times >= start_time) & (formant_times <= end_time)
        
        if not np.any(time_mask):
            return None, None, None
        
        formant_values = []
        valid_times = []
        
        for t in formant_times[time_mask]:
            try:
                value = formant_obj.get_value_at_time(formant_number, t)
                if not np.isnan(value):
                    formant_values.append(value)
                    valid_times.append(t)
            except:
                continue
        
        if len(formant_values) == 0:
            return None, None, None
        
        return np.array(formant_values), np.array(valid_times), formant_obj
    
    except Exception:
        return None, None, None

def _extract_formant_stat(sound, start_time, end_time, formant_number, stat_func):
    """Internal helper for formant statistics."""
    values, _, _ = _extract_formant_values(sound, start_time, end_time, formant_number)
    if values is None or len(values) == 0:
        return None
    
    if stat_func == 'mean':
        return float(np.mean(values))
    elif stat_func == 'std':
        if len(values) < 2:
            return None
        return float(np.std(values))
    elif stat_func == 'range':
        return float(np.ptp(values))
    elif stat_func == 'max':
        return float(np.max(values))
    elif stat_func == 'min':
        return float(np.min(values))

def extract_f1_mean(sound, start_time, end_time):
    """Extract mean first formant frequency from interval."""
    return _extract_formant_stat(sound, start_time, end_time, 1, 'mean')

def extract_f1_sd(sound, start_time, end_time):
    """Extract standard deviation of first formant."""
    return _extract_formant_stat(sound, start_time, end_time, 1, 'std')

def extract_f1_range(sound, start_time, end_time):
    """Extract peak-to-peak range of first formant."""
    return _extract_formant_stat(sound, start_time, end_time, 1, 'range')

def extract_f1_max(sound, start_time, end_time):
    """Extract maximum first formant value."""
    return _extract_formant_stat(sound, start_time, end_time, 1, 'max')

def extract_f1_max_time(sound, start_time, end_time):
    """Extract time of maximum F1 relative to interval start."""
    values, times, _ = _extract_formant_values(sound, start_time, end_time, 1)
    if values is None or len(values) == 0:
        return None
    
    max_idx = np.argmax(values)
    max_time = times[max_idx]
    return float(max_time - start_time)

def extract_f1_min(sound, start_time, end_time):
    """Extract minimum first formant value."""
    return _extract_formant_stat(sound, start_time, end_time, 1, 'min')

def extract_f1_min_time(sound, start_time, end_time):
    """Extract time of minimum F1 relative to interval start."""
    values, times, _ = _extract_formant_values(sound, start_time, end_time, 1)
    if values is None or len(values) == 0:
        return None
    
    min_idx = np.argmin(values)
    min_time = times[min_idx]
    return float(min_time - start_time)

def extract_f1_midpoint(sound, start_time, end_time):
    """Extract F1 at temporal midpoint (50%) of interval."""
    return _extract_formant_at_percentage(sound, start_time, end_time, 1, 50)

def extract_f2_mean(sound, start_time, end_time):
    """Extract mean second formant frequency from interval."""
    return _extract_formant_stat(sound, start_time, end_time, 2, 'mean')

def extract_f2_sd(sound, start_time, end_time):
    """Extract standard deviation of second formant."""
    return _extract_formant_stat(sound, start_time, end_time, 2, 'std')

def extract_f2_range(sound, start_time, end_time):
    """Extract peak-to-peak range of second formant."""
    return _extract_formant_stat(sound, start_time, end_time, 2, 'range')

def extract_f2_max(sound, start_time, end_time):
    """Extract maximum second formant value."""
    return _extract_formant_stat(sound, start_time, end_time, 2, 'max')

def extract_f2_max_time(sound, start_time, end_time):
    """Extract time of maximum F2 relative to interval start."""
    values, times, _ = _extract_formant_values(sound, start_time, end_time, 2)
    if values is None or len(values) == 0:
        return None
    
    max_idx = np.argmax(values)
    max_time = times[max_idx]
    return float(max_time - start_time)

def extract_f2_min(sound, start_time, end_time):
    """Extract minimum second formant value."""
    return _extract_formant_stat(sound, start_time, end_time, 2, 'min')

def extract_f2_min_time(sound, start_time, end_time):
    """Extract time of minimum F2 relative to interval start."""
    values, times, _ = _extract_formant_values(sound, start_time, end_time, 2)
    if values is None or len(values) == 0:
        return None
    
    min_idx = np.argmin(values)
    min_time = times[min_idx]
    return float(min_time - start_time)

def extract_f2_midpoint(sound, start_time, end_time):
    """Extract F2 at temporal midpoint (50%) of interval."""
    return _extract_formant_at_percentage(sound, start_time, end_time, 2, 50)

def _extract_formant_at_percentage(sound, start_time, end_time, formant_number, percentage):
    """Internal helper: Extract formant at specific percentage of interval."""
    values, times, formant_obj = _extract_formant_values(sound, start_time, end_time, formant_number)
    if values is None or len(values) == 0:
        return None
    
    target_time = start_time + (percentage / 100.0) * (end_time - start_time)
    
    try:
        formant_value = formant_obj.get_value_at_time(formant_number, target_time)
        if not np.isnan(formant_value):
            return float(formant_value)
    except:
        pass
    
    nearest_idx = np.argmin(np.abs(times - target_time))
    return float(values[nearest_idx])

def extract_f1_contour(sound, start_time, end_time, percentage_increment):
    """
    Extract F1 contour at specified percentage increments.
    
    Parameters:
        sound: parselmouth.Sound object
        start_time, end_time: Interval boundaries in seconds
        percentage_increment: Integer percentage increment (e.g., 10, 25, 32)
    
    Returns:
        Dictionary with keys like "F1At0", "F1At25", etc.
        Returns None if formant extraction fails.
    """
    return _extract_formant_contour(sound, start_time, end_time, 1, percentage_increment)

def extract_f2_contour(sound, start_time, end_time, percentage_increment):
    """
    Extract F2 contour at specified percentage increments.
    
    Parameters:
        sound: parselmouth.Sound object
        start_time, end_time: Interval boundaries in seconds
        percentage_increment: Integer percentage increment (e.g., 10, 25, 32)
    
    Returns:
        Dictionary with keys like "F2At0", "F2At25", etc.
        Returns None if formant extraction fails.
    """
    return _extract_formant_contour(sound, start_time, end_time, 2, percentage_increment)

def _extract_formant_contour(sound, start_time, end_time, formant_number, percentage_increment):
    """Internal helper for formant contour extraction."""
    values, times, formant_obj = _extract_formant_values(sound, start_time, end_time, formant_number)
    if values is None or len(values) == 0:
        return None
    
    duration = end_time - start_time
    contour_points = {}
    
    percentages = [0]
    
    current = percentage_increment
    while current < 100:
        percentages.append(current)
        current += percentage_increment
    
    if 100 not in percentages:
        percentages.append(100)
    
    prefix = "F1At" if formant_number == 1 else "F2At"
    
    for pct in percentages:
        target_time = start_time + (pct / 100.0) * duration
        formant_value = None
        
        try:
            formant_value = formant_obj.get_value_at_time(formant_number, target_time)
            if np.isnan(formant_value):
                raise ValueError("NaN value at exact time")
        except:
            nearest_idx = np.argmin(np.abs(times - target_time))
            formant_value = values[nearest_idx]
        
        contour_points[f"{prefix}{pct}"] = float(formant_value)
    
    return contour_points
