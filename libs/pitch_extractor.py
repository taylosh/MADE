"""
Pitch Feature Extraction Library

Extracts pitch (F0) features from audio intervals.
Functions operate on parselmouth.Sound objects with specified time intervals.

Functions:
    check_interval_has_voicing: Check if interval contains voiced frames
    extract_pitch_mean: Mean fundamental frequency
    extract_pitch_sd: Standard deviation of F0
    extract_pitch_range: Peak-to-peak F0 range
    extract_pitch_max: Maximum F0 value
    extract_pitch_max_time: Time of maximum F0 relative to interval start
    extract_pitch_min: Minimum F0 value
    extract_pitch_min_time: Time of minimum F0 relative to interval start
    extract_pitch_midpoint: F0 at temporal midpoint of interval
    extract_pitch_contour: F0 at specified percentage intervals

All functions return None if no voiced frames are found.
"""

import numpy as np
import parselmouth

# Configuration constants
PITCH_FLOOR = 25
PITCH_CEILING = 1000
TIME_STEP = 0.005

def _extract_voiced_pitch_values(sound, start_time, end_time):
    """Internal helper: Extract voiced pitch values and times from interval."""
    pitch = sound.to_pitch(
        time_step=TIME_STEP,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING
    )
    
    pitch_values = pitch.selected_array['frequency']
    pitch_times = pitch.xs()
    
    time_mask = (pitch_times >= start_time) & (pitch_times <= end_time)
    voiced_mask = (pitch_values > 0) & time_mask
    
    voiced_frames = pitch_values[voiced_mask]
    voiced_times = pitch_times[voiced_mask]
    
    if len(voiced_frames) == 0:
        return None, None, None
    
    return voiced_frames, voiced_times, pitch

def check_interval_has_voicing(sound, start_time, end_time):
    """Check if interval contains any voiced frames."""
    voiced_frames, _, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    return voiced_frames is not None and len(voiced_frames) > 0

def extract_pitch_mean(sound, start_time, end_time):
    """Extract mean fundamental frequency from interval."""
    voiced_frames, _, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    return float(np.mean(voiced_frames))

def extract_pitch_sd(sound, start_time, end_time):
    """Extract standard deviation of fundamental frequency."""
    voiced_frames, _, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) < 2:
        return None
    return float(np.std(voiced_frames))

def extract_pitch_range(sound, start_time, end_time):
    """Extract peak-to-peak range of fundamental frequency."""
    voiced_frames, _, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    return float(np.ptp(voiced_frames))

def extract_pitch_max(sound, start_time, end_time):
    """Extract maximum fundamental frequency value."""
    voiced_frames, _, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    return float(np.max(voiced_frames))

def extract_pitch_max_time(sound, start_time, end_time):
    """Extract time of maximum F0 relative to interval start."""
    voiced_frames, voiced_times, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    
    max_idx = np.argmax(voiced_frames)
    max_time = voiced_times[max_idx]
    return float(max_time - start_time)

def extract_pitch_min(sound, start_time, end_time):
    """Extract minimum fundamental frequency value."""
    voiced_frames, _, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    return float(np.min(voiced_frames))

def extract_pitch_min_time(sound, start_time, end_time):
    """Extract time of minimum F0 relative to interval start."""
    voiced_frames, voiced_times, _ = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    
    min_idx = np.argmin(voiced_frames)
    min_time = voiced_times[min_idx]
    return float(min_time - start_time)

def extract_pitch_midpoint(sound, start_time, end_time):
    """Extract F0 at temporal midpoint (50%) of interval."""
    return _extract_pitch_at_percentage(sound, start_time, end_time, 50)

def _extract_pitch_at_percentage(sound, start_time, end_time, percentage):
    """Internal helper: Extract pitch at specific percentage of interval."""
    voiced_frames, voiced_times, pitch = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
        return None
    
    target_time = start_time + (percentage / 100.0) * (end_time - start_time)
    
    try:
        pitch_value = pitch.get_value_at_time(target_time)
        if pitch_value > 0:
            return float(pitch_value)
    except:
        pass
    
    nearest_idx = np.argmin(np.abs(voiced_times - target_time))
    return float(voiced_frames[nearest_idx])

def extract_pitch_contour(sound, start_time, end_time, percentage_increment):
    """
    Extract pitch contour at specified percentage increments.
    
    Parameters:
        sound: parselmouth.Sound object
        start_time, end_time: Interval boundaries in seconds
        percentage_increment: Integer percentage increment (e.g., 10, 25, 32)
    
    Returns:
        Dictionary with keys like "PitchAt0", "PitchAt25", etc.
        Returns None if no voiced frames found.
    """
    voiced_frames, voiced_times, pitch = _extract_voiced_pitch_values(sound, start_time, end_time)
    if voiced_frames is None or len(voiced_frames) == 0:
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
        pitch_value = None
        
        try:
            pitch_value = pitch.get_value_at_time(target_time)
            if pitch_value <= 0:
                raise ValueError("Unvoiced at exact time")
        except:
            nearest_idx = np.argmin(np.abs(voiced_times - target_time))
            pitch_value = voiced_frames[nearest_idx]
        
        contour_points[f"PitchAt{pct}"] = float(pitch_value)
    
    return contour_points
