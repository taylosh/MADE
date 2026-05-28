"""
mfcc_extractor.py - MFCC (Mel-Frequency Cepstral Coefficient) Extraction Module
@author: taylosh
Created on 28 May 2026

Provides functions for extracting MFCC coefficients and their derivatives from acoustic intervals.
Based on Praat implementations from Praat Mass Analyzer and One Script to Rule Them All.

Key Functions:
    - extract_mfcc: Extract 12 MFCC coefficients from an interval
    - extract_mfcc_contour: Extract MFCC coefficients across multiple time windows
    - extract_mfcc_delta: Extract first derivative (delta) of MFCC coefficients
    - extract_mfcc_deltadelta: Extract second derivative (delta-delta) of MFCC coefficients
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import warnings

warnings.filterwarnings("ignore", category=Warning)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MFCC_CONFIG = {
    "number_of_coefficients": 12,      # Number of MFCC coefficients to extract (1-12)
    "window_length": 0.025,            # Analysis window length in seconds
    "time_step": 0.010,                # Time step between frames in seconds
    "first_filter_frequency": 100.0,   # First filter frequency in Hz
    "distance_between_filters": 100.0, # Distance between filters in Hz
    "maximum_frequency": 0.0,          # Maximum frequency (0 = Nyquist/2)
    "number_of_windows": 10,           # Number of time windows for contour (default 10 = 10% increments)
}


# =============================================================================
# CORE MFCC EXTRACTION FUNCTIONS
# =============================================================================

def extract_mfcc(sound, start_time, end_time, 
                 num_coefficients=DEFAULT_MFCC_CONFIG["number_of_coefficients"],
                 window_length=DEFAULT_MFCC_CONFIG["window_length"],
                 time_step=DEFAULT_MFCC_CONFIG["time_step"],
                 first_filter=DEFAULT_MFCC_CONFIG["first_filter_frequency"],
                 filter_distance=DEFAULT_MFCC_CONFIG["distance_between_filters"],
                 max_freq=DEFAULT_MFCC_CONFIG["maximum_frequency"]):
    """
    Extract average MFCC coefficients from an interval.
    
    Args:
        sound: Parselmouth Sound object
        start_time: Start time in seconds
        end_time: End time in seconds
        num_coefficients: Number of MFCC coefficients (default 12)
        window_length: Analysis window length in seconds
        time_step: Time step between frames
        first_filter: First filter frequency in Hz
        filter_distance: Distance between filters in Hz
        max_freq: Maximum frequency (0 = Nyquist/2)
    
    Returns:
        Dictionary with keys 'MFCC_01' through 'MFCC_12' (or up to num_coefficients)
        Returns None if interval is too short or extraction fails
    """
    try:
        duration = end_time - start_time
        
        # Need at least 2x window length for reliable extraction
        if duration < window_length * 2:
            return None
        
        # Extract the interval
        interval_sound = call(sound, "Extract part", start_time, end_time, "rectangular", 1.0, "yes")
        
        # Calculate appropriate time step if not specified
        if time_step <= 0:
            time_step = window_length / 4
        
        # To MFCC
        mfcc_object = call(interval_sound, "To MFCC", 
                          num_coefficients, window_length, time_step,
                          first_filter, filter_distance, max_freq)
        
        # Convert to TableOfReal for easier access
        mfcc_table = call(mfcc_object, "To TableOfReal", "no")
        
        # Get number of frames
        num_frames = call(mfcc_table, "Get number of rows")
        
        if num_frames == 0:
            call(interval_sound, "Remove")
            call(mfcc_object, "Remove")
            call(mfcc_table, "Remove")
            return None
        
        # Calculate average for each coefficient across all frames
        result = {}
        for coeff in range(1, num_coefficients + 1):
            sum_values = 0.0
            valid_frames = 0
            
            for frame in range(1, num_frames + 1):
                value = call(mfcc_table, "Get value", frame, coeff)
                if value != undefined and not np.isnan(value):
                    sum_values += value
                    valid_frames += 1
            
            if valid_frames > 0:
                avg_value = sum_values / valid_frames
                result[f"MFCC_{coeff:02d}"] = avg_value
            else:
                result[f"MFCC_{coeff:02d}"] = np.nan
        
        # Clean up
        call(interval_sound, "Remove")
        call(mfcc_object, "Remove")
        call(mfcc_table, "Remove")
        
        return result
        
    except Exception as e:
        warnings.warn(f"MFCC extraction failed: {e}")
        return None


def extract_mfcc_contour(sound, start_time, end_time, 
                         num_windows=DEFAULT_MFCC_CONFIG["number_of_windows"],
                         num_coefficients=DEFAULT_MFCC_CONFIG["number_of_coefficients"],
                         window_length=DEFAULT_MFCC_CONFIG["window_length"],
                         first_filter=DEFAULT_MFCC_CONFIG["first_filter_frequency"],
                         filter_distance=DEFAULT_MFCC_CONFIG["distance_between_filters"],
                         max_freq=DEFAULT_MFCC_CONFIG["maximum_frequency"]):
    """
    Extract MFCC coefficients across multiple time windows (contour).
    
    Args:
        sound: Parselmouth Sound object
        start_time: Start time in seconds
        end_time: End time in seconds
        num_windows: Number of time windows (default 10 = 10% increments)
        num_coefficients: Number of MFCC coefficients per window
        window_length: Analysis window length in seconds
        first_filter: First filter frequency in Hz
        filter_distance: Distance between filters in Hz
        max_freq: Maximum frequency (0 = Nyquist/2)
    
    Returns:
        Dictionary with keys like 'MFCC_01_0pct', 'MFCC_01_10pct', ..., 'MFCC_12_100pct'
        Returns None if extraction fails
    """
    try:
        duration = end_time - start_time
        
        if duration < window_length:
            return None
        
        result = {}
        
        # Calculate time points for each window
        for i in range(num_windows + 1):
            percentage = int((i / num_windows) * 100)
            
            # Window centers at specific percentages
            if i == 0:
                # First window at onset
                window_center = start_time + (window_length / 2)
            elif i == num_windows:
                # Last window at offset
                window_center = end_time - (window_length / 2)
            else:
                window_center = start_time + (i / num_windows) * duration
            
            # Ensure window is within bounds
            window_start = max(start_time, window_center - window_length / 2)
            window_end = min(end_time, window_center + window_length / 2)
            
            if window_end - window_start < window_length / 2:
                continue
            
            # Extract sub-interval
            window_sound = call(sound, "Extract part", window_start, window_end, "rectangular", 1.0, "yes")
            
            # To MFCC
            mfcc_object = call(window_sound, "To MFCC", 
                              num_coefficients, window_length, 0.0,
                              first_filter, filter_distance, max_freq)
            
            # Get average MFCC for this window
            mfcc_table = call(mfcc_object, "To TableOfReal", "no")
            num_frames = call(mfcc_table, "Get number of rows")
            
            if num_frames > 0:
                for coeff in range(1, num_coefficients + 1):
                    sum_values = 0.0
                    for frame in range(1, num_frames + 1):
                        value = call(mfcc_table, "Get value", frame, coeff)
                        if value != undefined and not np.isnan(value):
                            sum_values += value
                    
                    if num_frames > 0:
                        avg_value = sum_values / num_frames
                        key = f"MFCC_{coeff:02d}_{percentage}pct"
                        result[key] = avg_value
            
            # Clean up
            call(window_sound, "Remove")
            call(mfcc_object, "Remove")
            call(mfcc_table, "Remove")
        
        return result if result else None
        
    except Exception as e:
        warnings.warn(f"MFCC contour extraction failed: {e}")
        return None


def extract_mfcc_delta(sound, start_time, end_time,
                       num_coefficients=DEFAULT_MFCC_CONFIG["number_of_coefficients"],
                       window_length=DEFAULT_MFCC_CONFIG["window_length"],
                       time_step=DEFAULT_MFCC_CONFIG["time_step"],
                       first_filter=DEFAULT_MFCC_CONFIG["first_filter_frequency"],
                       filter_distance=DEFAULT_MFCC_CONFIG["distance_between_filters"],
                       max_freq=DEFAULT_MFCC_CONFIG["maximum_frequency"]):
    """
    Extract delta (first derivative) of MFCC coefficients.
    Delta represents the rate of change of MFCCs over time.
    
    Args:
        sound: Parselmouth Sound object
        start_time: Start time in seconds
        end_time: End time in seconds
        num_coefficients: Number of MFCC coefficients
        window_length: Analysis window length
        time_step: Time step for delta calculation
        first_filter: First filter frequency
        filter_distance: Distance between filters
        max_freq: Maximum frequency
    
    Returns:
        Dictionary with keys 'MFCC_Delta_01' through 'MFCC_Delta_12'
    """
    try:
        duration = end_time - start_time
        
        if duration < window_length * 2:
            return None
        
        # Extract interval
        interval_sound = call(sound, "Extract part", start_time, end_time, "rectangular", 1.0, "yes")
        
        # To MFCC
        if time_step <= 0:
            time_step = window_length / 4
        
        mfcc_object = call(interval_sound, "To MFCC", 
                          num_coefficients, window_length, time_step,
                          first_filter, filter_distance, max_freq)
        
        # Compute delta (using Praat's built-in function)
        delta_object = call(mfcc_object, "To MFCC", "yes")  # 'yes' for delta
        
        # Convert to TableOfReal
        delta_table = call(delta_object, "To TableOfReal", "no")
        num_frames = call(delta_table, "Get number of rows")
        
        if num_frames == 0:
            call(interval_sound, "Remove")
            call(mfcc_object, "Remove")
            call(delta_object, "Remove")
            call(delta_table, "Remove")
            return None
        
        # Average delta values across frames
        result = {}
        for coeff in range(1, num_coefficients + 1):
            sum_values = 0.0
            valid_frames = 0
            
            for frame in range(1, num_frames + 1):
                value = call(delta_table, "Get value", frame, coeff)
                if value != undefined and not np.isnan(value):
                    sum_values += value
                    valid_frames += 1
            
            if valid_frames > 0:
                result[f"MFCC_Delta_{coeff:02d}"] = sum_values / valid_frames
            else:
                result[f"MFCC_Delta_{coeff:02d}"] = np.nan
        
        # Clean up
        call(interval_sound, "Remove")
        call(mfcc_object, "Remove")
        call(delta_object, "Remove")
        call(delta_table, "Remove")
        
        return result
        
    except Exception as e:
        warnings.warn(f"MFCC delta extraction failed: {e}")
        return None


def extract_mfcc_deltadelta(sound, start_time, end_time,
                            num_coefficients=DEFAULT_MFCC_CONFIG["number_of_coefficients"],
                            window_length=DEFAULT_MFCC_CONFIG["window_length"],
                            time_step=DEFAULT_MFCC_CONFIG["time_step"],
                            first_filter=DEFAULT_MFCC_CONFIG["first_filter_frequency"],
                            filter_distance=DEFAULT_MFCC_CONFIG["distance_between_filters"],
                            max_freq=DEFAULT_MFCC_CONFIG["maximum_frequency"]):
    """
    Extract delta-delta (second derivative) of MFCC coefficients.
    Delta-delta represents the acceleration of MFCC change over time.
    
    Args:
        sound: Parselmouth Sound object
        start_time: Start time in seconds
        end_time: End time in seconds
        num_coefficients: Number of MFCC coefficients
        window_length: Analysis window length
        time_step: Time step for delta calculation
        first_filter: First filter frequency
        filter_distance: Distance between filters
        max_freq: Maximum frequency
    
    Returns:
        Dictionary with keys 'MFCC_DeltaDelta_01' through 'MFCC_DeltaDelta_12'
    """
    try:
        duration = end_time - start_time
        
        if duration < window_length * 2:
            return None
        
        # Extract interval
        interval_sound = call(sound, "Extract part", start_time, end_time, "rectangular", 1.0, "yes")
        
        # To MFCC
        if time_step <= 0:
            time_step = window_length / 4
        
        mfcc_object = call(interval_sound, "To MFCC", 
                          num_coefficients, window_length, time_step,
                          first_filter, filter_distance, max_freq)
        
        # Compute delta
        delta_object = call(mfcc_object, "To MFCC", "yes")
        
        # Compute delta-delta (apply delta again)
        deltadelta_object = call(delta_object, "To MFCC", "yes")
        
        # Convert to TableOfReal
        deltadelta_table = call(deltadelta_object, "To TableOfReal", "no")
        num_frames = call(deltadelta_table, "Get number of rows")
        
        if num_frames == 0:
            call(interval_sound, "Remove")
            call(mfcc_object, "Remove")
            call(delta_object, "Remove")
            call(deltadelta_object, "Remove")
            call(deltadelta_table, "Remove")
            return None
        
        # Average delta-delta values across frames
        result = {}
        for coeff in range(1, num_coefficients + 1):
            sum_values = 0.0
            valid_frames = 0
            
            for frame in range(1, num_frames + 1):
                value = call(deltadelta_table, "Get value", frame, coeff)
                if value != undefined and not np.isnan(value):
                    sum_values += value
                    valid_frames += 1
            
            if valid_frames > 0:
                result[f"MFCC_DeltaDelta_{coeff:02d}"] = sum_values / valid_frames
            else:
                result[f"MFCC_DeltaDelta_{coeff:02d}"] = np.nan
        
        # Clean up
        call(interval_sound, "Remove")
        call(mfcc_object, "Remove")
        call(delta_object, "Remove")
        call(deltadelta_object, "Remove")
        call(deltadelta_table, "Remove")
        
        return result
        
    except Exception as e:
        warnings.warn(f"MFCC delta-delta extraction failed: {e}")
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_mfcc_config():
    """Return the default MFCC configuration dictionary."""
    return DEFAULT_MFCC_CONFIG.copy()


def update_mfcc_config(**kwargs):
    """Update MFCC configuration with custom values."""
    DEFAULT_MFCC_CONFIG.update(kwargs)
    return DEFAULT_MFCC_CONFIG.copy()


def validate_mfcc_config(config):
    """Validate MFCC configuration parameters."""
    valid = True
    errors = []
    
    if config.get("number_of_coefficients", 12) < 1 or config.get("number_of_coefficients", 12) > 20:
        errors.append("number_of_coefficients must be between 1 and 20")
        valid = False
    
    if config.get("window_length", 0.025) <= 0:
        errors.append("window_length must be positive")
        valid = False
    
    if config.get("first_filter_frequency", 100.0) < 0:
        errors.append("first_filter_frequency must be non-negative")
        valid = False
    
    if config.get("distance_between_filters", 100.0) <= 0:
        errors.append("distance_between_filters must be positive")
        valid = False
    
    return valid, errors
