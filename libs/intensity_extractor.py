"""
Intensity Feature Extraction Library

Extracts intensity (loudness) features from audio intervals.
Functions operate on parselmouth.Sound objects with specified time intervals.

Functions:
    extract_intensity_mean: Mean intensity
    extract_intensity_sd: Standard deviation of intensity
    extract_intensity_range: Peak-to-peak intensity range
    extract_intensity_max: Maximum intensity value
    extract_intensity_max_time: Time of maximum intensity relative to interval start
    extract_intensity_min: Minimum intensity value
    extract_intensity_min_time: Time of minimum intensity relative to interval start
    extract_intensity_midpoint: Intensity at temporal midpoint of interval
    extract_intensity_contour: Intensity at specified percentage intervals
    extract_intensity_ratio: C/V ratio (consonant/vowel intensity ratio)

All functions return None if intensity extraction fails.
"""

import numpy as np
import parselmouth

def _extract_intensity_values(sound, start_time, end_time):
    """Internal helper: Extract intensity values and times from interval."""
    try:
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        intensity_times = intensity.xs()
        
        time_mask = (intensity_times >= start_time) & (intensity_times <= end_time)
        
        if not np.any(time_mask):
            return None, None, None
        
        valid_values = intensity_values[time_mask]
        valid_times = intensity_times[time_mask]
        
        if len(valid_values) == 0:
            return None, None, None
        
        return valid_values, valid_times, intensity
    
    except Exception:
        return None, None, None

def extract_intensity_mean(sound, start_time, end_time):
    """Extract mean intensity from interval."""
    values, _, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    return float(np.mean(values))

def extract_intensity_sd(sound, start_time, end_time):
    """Extract standard deviation of intensity."""
    values, _, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) < 2:
        return None
    return float(np.std(values))

def extract_intensity_range(sound, start_time, end_time):
    """Extract peak-to-peak intensity range."""
    values, _, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    return float(np.ptp(values))

def extract_intensity_max(sound, start_time, end_time):
    """Extract maximum intensity value."""
    values, _, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    return float(np.max(values))

def extract_intensity_max_time(sound, start_time, end_time):
    """Extract time of maximum intensity relative to interval start."""
    values, times, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    
    max_idx = np.argmax(values)
    max_time = times[max_idx]
    return float(max_time - start_time)

def extract_intensity_min(sound, start_time, end_time):
    """Extract minimum intensity value."""
    values, _, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    return float(np.min(values))

def extract_intensity_min_time(sound, start_time, end_time):
    """Extract time of minimum intensity relative to interval start."""
    values, times, _ = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    
    min_idx = np.argmin(values)
    min_time = times[min_idx]
    return float(min_time - start_time)

def extract_intensity_midpoint(sound, start_time, end_time):
    """Extract intensity at temporal midpoint (50%) of interval."""
    return _extract_intensity_at_percentage(sound, start_time, end_time, 50)

def _extract_intensity_at_percentage(sound, start_time, end_time, percentage):
    """Internal helper: Extract intensity at specific percentage of interval."""
    values, times, intensity = _extract_intensity_values(sound, start_time, end_time)
    if values is None or len(values) == 0:
        return None
    
    target_time = start_time + (percentage / 100.0) * (end_time - start_time)
    
    try:
        intensity_value = intensity.get_value(target_time)
        if not np.isnan(intensity_value):
            return float(intensity_value)
    except:
        pass
    
    nearest_idx = np.argmin(np.abs(times - target_time))
    return float(values[nearest_idx])

def extract_intensity_contour(sound, start_time, end_time, percentage_increment):
    """
    Extract intensity contour at specified percentage increments.
    
    Parameters:
        sound: parselmouth.Sound object
        start_time, end_time: Interval boundaries in seconds
        percentage_increment: Integer percentage increment (e.g., 10, 25, 32)
    
    Returns:
        Dictionary with keys like "IntensityAt0", "IntensityAt25", etc.
        Returns None if intensity extraction fails.
    """
    values, times, intensity = _extract_intensity_values(sound, start_time, end_time)
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
    
    for pct in percentages:
        target_time = start_time + (pct / 100.0) * duration
        intensity_value = None
        
        try:
            intensity_value = intensity.get_value(target_time)
            if np.isnan(intensity_value):
                raise ValueError("NaN value at exact time")
        except:
            nearest_idx = np.argmin(np.abs(times - target_time))
            intensity_value = values[nearest_idx]
        
        contour_points[f"IntensityAt{pct}"] = float(intensity_value)
    
    return contour_points

def extract_intensity_ratio(sound, consonant_start, consonant_end, 
                           prev_vowel_start=None, prev_vowel_end=None,
                           next_vowel_start=None, next_vowel_end=None):
    """
    Extract C/V intensity ratio (consonant/vowel).
    
    Parameters:
        sound: parselmouth.Sound object
        consonant_start, consonant_end: Consonant interval boundaries
        prev_vowel_start, prev_vowel_end: Previous vowel interval (optional)
        next_vowel_start, next_vowel_end: Next vowel interval (optional)
    
    Returns:
        Float intensity ratio (consonant intensity / average vowel intensity)
        Returns None if vowel intensity data is insufficient.
    """
    # Extract consonant intensity
    consonant_intensity = extract_intensity_mean(sound, consonant_start, consonant_end)
    if consonant_intensity is None:
        return None
    
    vowel_intensities = []
    
    # Extract previous vowel intensity if provided
    if prev_vowel_start is not None and prev_vowel_end is not None:
        prev_vowel_intensity = extract_intensity_mean(sound, prev_vowel_start, prev_vowel_end)
        if prev_vowel_intensity is not None:
            vowel_intensities.append(prev_vowel_intensity)
    
    # Extract next vowel intensity if provided
    if next_vowel_start is not None and next_vowel_end is not None:
        next_vowel_intensity = extract_intensity_mean(sound, next_vowel_start, next_vowel_end)
        if next_vowel_intensity is not None:
            vowel_intensities.append(next_vowel_intensity)
    
    # Need at least one vowel intensity
    if len(vowel_intensities) == 0:
        return None
    
    avg_vowel_intensity = np.mean(vowel_intensities)
    
    if avg_vowel_intensity > 0:
        return float(consonant_intensity / avg_vowel_intensity)
    
    return None
