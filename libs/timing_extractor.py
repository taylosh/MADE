"""
Timing Feature Extraction Library

Extracts timing features from audio intervals.
Functions operate on interval time boundaries.

Functions:
    extract_interval_start: Start time of interval
    extract_interval_end: End time of interval  
    extract_interval_duration: Duration of interval

These are simple calculations but provided for consistency with other extractors.
"""

def extract_interval_start(start_time, end_time):
    """
    Extract start time of interval.
    
    Parameters:
        start_time: Start time in seconds
        end_time: End time in seconds (unused but kept for consistent signature)
    
    Returns:
        Float start time
    """
    return float(start_time)

def extract_interval_end(start_time, end_time):
    """
    Extract end time of interval.
    
    Parameters:
        start_time: Start time in seconds (unused but kept for consistent signature)
        end_time: End time in seconds
    
    Returns:
        Float end time
    """
    return float(end_time)

def extract_interval_duration(start_time, end_time):
    """
    Extract duration of interval.
    
    Parameters:
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        Float duration in seconds
    """
    return float(end_time - start_time)
