"""
Spectral Feature Extraction Library

Extracts spectral features from audio intervals, primarily for fricative analysis.
Functions operate on parselmouth.Sound objects with specified time intervals.

Functions:
    extract_cog: Center of Gravity (dB-weighted)
    extract_spectral_centroid: Spectral centroid (linear power-weighted)
    extract_spectral_spread: Spectral variance
    extract_spectral_skewness: Spectral asymmetry

All functions return None if spectral extraction fails.
"""

import numpy as np
import parselmouth

# Configuration constants for Spanish /s/ analysis
SPECTRAL_MIN_FREQ = 2000
SPECTRAL_MAX_FREQ = 8000

def _extract_spectrum_segment(sound, start_time, end_time):
    """Internal helper: Extract spectrum from interval."""
    try:
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        spectrum = segment.to_spectrum()
        return spectrum
    except Exception:
        return None

def extract_cog(sound, start_time, end_time):
    """Extract Center of Gravity (dB-weighted spectral mean)."""
    spectrum = _extract_spectrum_segment(sound, start_time, end_time)
    if spectrum is None:
        return None
    
    try:
        frequencies = spectrum.xs()
        magnitudes = spectrum.values[0]
        
        # Filter to Spanish /s/ range, fallback to wider range
        mask = (frequencies >= SPECTRAL_MIN_FREQ) & (frequencies <= SPECTRAL_MAX_FREQ)
        if not np.any(mask):
            mask = (frequencies >= 1000) & (frequencies <= 10000)
        
        freqs = frequencies[mask]
        mags = magnitudes[mask]
        
        if len(freqs) == 0:
            return None
        
        # Convert to dB scale
        mags_db = 20 * np.log10(np.maximum(mags, 1e-10))
        db_sum = np.sum(mags_db)
        
        if db_sum == 0:
            return None
        
        cog = np.sum(freqs * mags_db) / db_sum
        return float(cog)
        
    except Exception:
        return None

def extract_spectral_centroid(sound, start_time, end_time):
    """Extract spectral centroid (linear power-weighted mean)."""
    spectrum = _extract_spectrum_segment(sound, start_time, end_time)
    if spectrum is None:
        return None
    
    try:
        frequencies = spectrum.xs()
        magnitudes = spectrum.values[0]
        
        # Filter to Spanish /s/ range, fallback to wider range
        mask = (frequencies >= SPECTRAL_MIN_FREQ) & (frequencies <= SPECTRAL_MAX_FREQ)
        if not np.any(mask):
            mask = (frequencies >= 1000) & (frequencies <= 10000)
        
        freqs = frequencies[mask]
        mags = magnitudes[mask]
        
        if len(freqs) == 0:
            return None
        
        # Convert to power
        power = mags ** 2
        total_power = np.sum(power)
        
        if total_power == 0:
            return None
        
        centroid = np.sum(freqs * power) / total_power
        return float(centroid)
        
    except Exception:
        return None

def extract_spectral_spread(sound, start_time, end_time):
    """Extract spectral spread (variance)."""
    centroid = extract_spectral_centroid(sound, start_time, end_time)
    if centroid is None:
        return None
    
    try:
        spectrum = _extract_spectrum_segment(sound, start_time, end_time)
        frequencies = spectrum.xs()
        magnitudes = spectrum.values[0]
        
        # Filter to Spanish /s/ range, fallback to wider range
        mask = (frequencies >= SPECTRAL_MIN_FREQ) & (frequencies <= SPECTRAL_MAX_FREQ)
        if not np.any(mask):
            mask = (frequencies >= 1000) & (frequencies <= 10000)
        
        freqs = frequencies[mask]
        mags = magnitudes[mask]
        
        if len(freqs) == 0:
            return None
        
        # Convert to power
        power = mags ** 2
        total_power = np.sum(power)
        
        if total_power == 0:
            return None
        
        spread = np.sum(((freqs - centroid) ** 2) * power) / total_power
        return float(spread)
        
    except Exception:
        return None

def extract_spectral_skewness(sound, start_time, end_time):
    """Extract spectral skewness (asymmetry)."""
    centroid = extract_spectral_centroid(sound, start_time, end_time)
    spread = extract_spectral_spread(sound, start_time, end_time)
    
    if centroid is None or spread is None or spread == 0:
        return None
    
    try:
        spectrum = _extract_spectrum_segment(sound, start_time, end_time)
        frequencies = spectrum.xs()
        magnitudes = spectrum.values[0]
        
        # Filter to Spanish /s/ range, fallback to wider range
        mask = (frequencies >= SPECTRAL_MIN_FREQ) & (frequencies <= SPECTRAL_MAX_FREQ)
        if not np.any(mask):
            mask = (frequencies >= 1000) & (frequencies <= 10000)
        
        freqs = frequencies[mask]
        mags = magnitudes[mask]
        
        if len(freqs) == 0:
            return None
        
        # Convert to power
        power = mags ** 2
        total_power = np.sum(power)
        
        if total_power == 0:
            return None
        
        skewness = np.sum(((freqs - centroid) ** 3) * power) / (total_power * (spread ** 1.5))
        return float(skewness)
        
    except Exception:
        return None
