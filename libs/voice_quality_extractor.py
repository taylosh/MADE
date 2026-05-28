"""
Voice Quality Feature Extraction Library

Extracts voice quality features from audio intervals.
Functions operate on parselmouth.Sound objects with specified time intervals.

Functions:
    extract_hnr: Harmonics-to-Noise Ratio
    extract_h1h2: H1-H2 difference (spectral tilt)
    extract_jitter: Local jitter percentage
    extract_shimmer: Local shimmer percentage

All functions return None if extraction fails.
"""

import numpy as np
import parselmouth
from scipy.signal import find_peaks, butter, filtfilt

# Configuration constants
PITCH_FLOOR = 50
PITCH_CEILING = 600

def _check_voiced_segment(sound, start_time, end_time):
    """Check if interval contains voiced frames for voice quality analysis."""
    try:
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        pitch = segment.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        pitch_values = pitch.selected_array['frequency']
        voiced_frames = pitch_values[pitch_values > 0]
        return len(voiced_frames) > 0
    except Exception:
        return False

def extract_hnr(sound, start_time, end_time):
    """Extract Harmonics-to-Noise Ratio."""
    try:
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        harmonicity = segment.to_harmonicity()
        values = harmonicity.values[harmonicity.values != -200]
        
        if len(values) == 0:
            return None
        
        return float(np.mean(values))
    except Exception:
        return None

def extract_h1h2(sound, start_time, end_time):
    """Extract H1-H2 difference (spectral tilt measure)."""
    try:
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        
        # Apply pre-emphasis
        nyquist = segment.sampling_frequency / 2
        b, a = butter(4, 100/nyquist, btype='high')
        sound_pre = segment.copy()
        sound_pre.values = filtfilt(b, a, sound_pre.values)
        
        # Get spectrum
        spectrum = sound_pre.to_spectrum()
        freqs = spectrum.xs()
        amps_linear = spectrum.values[0]
        amps_db = 20 * np.log10(np.maximum(amps_linear, 1e-10))
        
        # Subtract linear regression slope
        slope, intercept = np.polyfit(np.log10(freqs[1:]), amps_db[1:], 1)
        amps_db_corrected = amps_db - (intercept + slope * np.log10(freqs))
        
        # Find H1 (strongest peak in 80-400Hz)
        h1_range = (80, 400)
        h1_mask = (freqs >= h1_range[0]) & (freqs <= h1_range[1])
        peaks, props = find_peaks(amps_db_corrected[h1_mask], prominence=5, width=2)
        
        if len(peaks) == 0:
            return None
        
        h1_idx = peaks[np.argmax(props["prominences"])]
        h1_amp = amps_db_corrected[h1_mask][h1_idx]
        
        # Find H2 (near 2*H1 frequency)
        h1_freq = freqs[h1_mask][h1_idx]
        h2_range = (1.8 * h1_freq, 2.2 * h1_freq)
        h2_mask = (freqs >= h2_range[0]) & (freqs <= h2_range[1])
        peaks, props = find_peaks(amps_db_corrected[h2_mask], prominence=5, width=2)
        
        if len(peaks) == 0:
            return None
        
        h2_idx = peaks[np.argmax(props["prominences"])]
        h2_amp = amps_db_corrected[h2_mask][h2_idx]
        
        return float(h1_amp - h2_amp)
        
    except Exception:
        return None

def extract_jitter(sound, start_time, end_time):
    """Extract local jitter percentage."""
    if not _check_voiced_segment(sound, start_time, end_time):
        return None
    
    try:
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        point_process = parselmouth.praat.call(
            segment, "To PointProcess (periodic, cc)", 
            PITCH_FLOOR, PITCH_CEILING
        )
        
        jitter = parselmouth.praat.call(
            point_process, "Get jitter (local)", 
            0, 0, 0.0001, 0.02, 1.3
        )
        
        return float(jitter * 100)
    except Exception:
        return None

def extract_shimmer(sound, start_time, end_time):
    """Extract local shimmer percentage."""
    if not _check_voiced_segment(sound, start_time, end_time):
        return None
    
    try:
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        point_process = parselmouth.praat.call(
            segment, "To PointProcess (periodic, cc)", 
            PITCH_FLOOR, PITCH_CEILING
        )
        
        shimmer = parselmouth.praat.call(
            [segment, point_process], "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        
        return float(shimmer * 100)
    except Exception:
        return None
