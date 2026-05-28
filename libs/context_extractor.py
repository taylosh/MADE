"""
context_extractor.py - Contextual Information Extraction Module
@author: taylosh
Created on 28 May 2026

Provides functions for extracting contextual information from TextGrid tiers.
Based on Praat implementations from Praat Mass Analyzer and One Script to Rule Them All.

Key Functions:
    - Phonetic context: Preceding/following phones and their stress
    - Lexical context: Preceding/following words and counts
    - Syllable context: Preceding/following syllables and stress patterns
    - Word-level: Duration, PVI, syllable structure
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=Warning)


# =============================================================================
# PHONETIC CONTEXT FUNCTIONS
# =============================================================================

def extract_preceding_phone(textgrid, tier_name, interval_index):
    """
    Extract the label of the preceding phone/interval.
    
    Args:
        textgrid: Praatio TextGrid object
        tier_name: Name of the tier
        interval_index: Current interval index (1-based)
    
    Returns:
        String label of preceding interval, or None if at boundary
    """
    try:
        tier = textgrid.getTier(tier_name)
        if interval_index > 1:
            preceding = tier.entries[interval_index - 2]  # 0-based index
            label = preceding.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_following_phone(textgrid, tier_name, interval_index):
    """
    Extract the label of the following phone/interval.
    
    Args:
        textgrid: Praatio TextGrid object
        tier_name: Name of the tier
        interval_index: Current interval index (1-based)
    
    Returns:
        String label of following interval, or None if at boundary
    """
    try:
        tier = textgrid.getTier(tier_name)
        if interval_index < len(tier.entries):
            following = tier.entries[interval_index]  # 0-based index
            label = following.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_preceding_phone_2(textgrid, tier_name, interval_index):
    """
    Extract the label two phones/intervals before the current.
    
    Args:
        textgrid: Praatio TextGrid object
        tier_name: Name of the tier
        interval_index: Current interval index (1-based)
    
    Returns:
        String label two intervals before, or None if out of range
    """
    try:
        tier = textgrid.getTier(tier_name)
        if interval_index > 2:
            preceding = tier.entries[interval_index - 3]
            label = preceding.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_following_phone_2(textgrid, tier_name, interval_index):
    """
    Extract the label two phones/intervals after the current.
    
    Args:
        textgrid: Praatio TextGrid object
        tier_name: Name of the tier
        interval_index: Current interval index (1-based)
    
    Returns:
        String label two intervals after, or None if out of range
    """
    try:
        tier = textgrid.getTier(tier_name)
        if interval_index < len(tier.entries) - 1:
            following = tier.entries[interval_index + 1]
            label = following.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_preceding_stress(phone_label):
    """
    Extract stress level from a phone label (e.g., 'AA1' -> 1).
    
    Args:
        phone_label: Phone label string (e.g., 'AA0', 'AE1', 'IY2')
    
    Returns:
        Integer stress level (0, 1, 2) or None if no stress marker
    """
    if not phone_label:
        return None
    
    # Check last character for stress numbers
    if phone_label and phone_label[-1].isdigit():
        return int(phone_label[-1])
    
    # Check for patterns like 'AA1' where number might be embedded
    import re
    match = re.search(r'[0-9]$', phone_label)
    if match:
        return int(match.group())
    
    return 0  # Default unstressed if no number found


def extract_following_stress(phone_label):
    """
    Extract stress level from a phone label.
    Same as preceding stress - included for API consistency.
    """
    return extract_preceding_stress(phone_label)


# =============================================================================
# LEXICAL CONTEXT FUNCTIONS
# =============================================================================

def extract_preceding_word(textgrid, word_tier_name, current_word_index):
    """
    Extract the label of the preceding word.
    
    Args:
        textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        current_word_index: Current word interval index (1-based)
    
    Returns:
        String label of preceding word, or None if at boundary
    """
    try:
        tier = textgrid.getTier(word_tier_name)
        if current_word_index > 1:
            preceding = tier.entries[current_word_index - 2]
            label = preceding.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_following_word(textgrid, word_tier_name, current_word_index):
    """
    Extract the label of the following word.
    
    Args:
        textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        current_word_index: Current word interval index (1-based)
    
    Returns:
        String label of following word, or None if at boundary
    """
    try:
        tier = textgrid.getTier(word_tier_name)
        if current_word_index < len(tier.entries):
            following = tier.entries[current_word_index]
            label = following.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_preceding_word_count(textgrid, word_tier_name, current_word_index):
    """
    Count number of words preceding the current word.
    
    Args:
        textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        current_word_index: Current word interval index (1-based)
    
    Returns:
        Integer count of preceding words
    """
    try:
        return current_word_index - 1
    except Exception:
        return 0


def extract_following_word_count(textgrid, word_tier_name, current_word_index):
    """
    Count number of words following the current word.
    
    Args:
        textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        current_word_index: Current word interval index (1-based)
    
    Returns:
        Integer count of following words
    """
    try:
        tier = textgrid.getTier(word_tier_name)
        return len(tier.entries) - current_word_index
    except Exception:
        return 0


def extract_context_window(textgrid, word_tier_name, current_word_index, window_size=2):
    """
    Extract a window of words around the target word.
    
    Args:
        textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        current_word_index: Current word interval index (1-based)
        window_size: Number of words before and after (default 2)
    
    Returns:
        String of space-separated words in context window
    """
    try:
        tier = textgrid.getTier(word_tier_name)
        context_words = []
        
        # Words before
        start_idx = max(0, current_word_index - 1 - window_size)
        for i in range(start_idx, current_word_index - 1):
            word = tier.entries[i].label.strip()
            if word:
                context_words.append(word)
        
        # Target word
        target = tier.entries[current_word_index - 1].label.strip()
        context_words.append(f"[{target}]")
        
        # Words after
        end_idx = min(len(tier.entries), current_word_index + window_size)
        for i in range(current_word_index, end_idx):
            word = tier.entries[i].label.strip()
            if word:
                context_words.append(word)
        
        return " ".join(context_words) if context_words else None
    except Exception:
        return None


# =============================================================================
# SYLLABLE CONTEXT FUNCTIONS
# =============================================================================

def extract_preceding_syllable(textgrid, syllable_tier_name, current_syllable_index):
    """
    Extract the label of the preceding syllable.
    
    Args:
        textgrid: Praatio TextGrid object
        syllable_tier_name: Name of the syllable tier
        current_syllable_index: Current syllable interval index (1-based)
    
    Returns:
        String label of preceding syllable, or None if at boundary
    """
    try:
        tier = textgrid.getTier(syllable_tier_name)
        if current_syllable_index > 1:
            preceding = tier.entries[current_syllable_index - 2]
            label = preceding.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_following_syllable(textgrid, syllable_tier_name, current_syllable_index):
    """
    Extract the label of the following syllable.
    
    Args:
        textgrid: Praatio TextGrid object
        syllable_tier_name: Name of the syllable tier
        current_syllable_index: Current syllable interval index (1-based)
    
    Returns:
        String label of following syllable, or None if at boundary
    """
    try:
        tier = textgrid.getTier(syllable_tier_name)
        if current_syllable_index < len(tier.entries):
            following = tier.entries[current_syllable_index]
            label = following.label.strip()
            return label if label else "boundary"
        return "boundary"
    except Exception:
        return None


def extract_current_syllable_stress(syllable_label):
    """
    Extract stress level from a syllable label.
    
    Args:
        syllable_label: Syllable label (e.g., 'Syllable1', '1', 'primary')
    
    Returns:
        Integer stress level (0=unstressed, 1=primary, 2=secondary) or None
    """
    if not syllable_label:
        return 0
    
    syllable_lower = syllable_label.lower()
    
    # Check for explicit stress markers
    if 'primary' in syllable_lower or syllable_label == '1':
        return 1
    if 'secondary' in syllable_lower or syllable_label == '2':
        return 2
    if 'unstressed' in syllable_lower or syllable_label == '0':
        return 0
    
    # Check for numbers in the label
    import re
    match = re.search(r'[0-9]', syllable_label)
    if match:
        num = int(match.group())
        if num <= 2:
            return num
    
    return 0  # Default unstressed


# =============================================================================
# WORD-LEVEL FUNCTIONS
# =============================================================================

def extract_word_duration(word_tier, word_index):
    """
    Extract duration of a word interval.
    
    Args:
        word_tier: Praatio Tier object
        word_index: Word interval index (1-based)
    
    Returns:
        Duration in seconds, or None if not found
    """
    try:
        entry = word_tier.entries[word_index - 1]
        return entry.end - entry.start
    except Exception:
        return None


def extract_left_word_duration(word_tier, word_index):
    """
    Extract duration of the word to the left.
    
    Args:
        word_tier: Praatio Tier object
        word_index: Current word interval index (1-based)
    
    Returns:
        Duration in seconds, or None if at boundary
    """
    try:
        if word_index > 1:
            entry = word_tier.entries[word_index - 2]
            return entry.end - entry.start
        return None
    except Exception:
        return None


def extract_right_word_duration(word_tier, word_index):
    """
    Extract duration of the word to the right.
    
    Args:
        word_tier: Praatio Tier object
        word_index: Current word interval index (1-based)
    
    Returns:
        Duration in seconds, or None if at boundary
    """
    try:
        if word_index < len(word_tier.entries):
            entry = word_tier.entries[word_index]
            return entry.end - entry.start
        return None
    except Exception:
        return None


# =============================================================================
# SYLLABLE-LEVEL FUNCTIONS
# =============================================================================

def extract_syllable_count(word_textgrid, word_tier_name, word_index, syllable_tier_name):
    """
    Count number of syllables in a word.
    
    Args:
        word_textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        word_index: Word interval index (1-based)
        syllable_tier_name: Name of the syllable tier
    
    Returns:
        Integer count of syllables, or 0 if not found
    """
    try:
        word_tier = word_textgrid.getTier(word_tier_name)
        word_entry = word_tier.entries[word_index - 1]
        word_start = word_entry.start
        word_end = word_entry.end
        
        syllable_tier = word_textgrid.getTier(syllable_tier_name)
        count = 0
        
        for entry in syllable_tier.entries:
            if entry.start >= word_start and entry.end <= word_end:
                if entry.label.strip():
                    count += 1
        
        return count
    except Exception:
        return 0


def extract_foot_count(word_textgrid, word_tier_name, word_index, syllable_tier_name):
    """
    Count number of metrical feet in a word (simplified: cluster of syllables with one primary stress).
    
    Args:
        word_textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        word_index: Word interval index (1-based)
        syllable_tier_name: Name of the syllable tier
    
    Returns:
        Integer count of feet (rough estimate)
    """
    try:
        word_tier = word_textgrid.getTier(word_tier_name)
        word_entry = word_tier.entries[word_index - 1]
        word_start = word_entry.start
        word_end = word_entry.end
        
        syllable_tier = word_textgrid.getTier(syllable_tier_name)
        stress_pattern = []
        
        for entry in syllable_tier.entries:
            if entry.start >= word_start and entry.end <= word_end:
                stress = extract_current_syllable_stress(entry.label)
                stress_pattern.append(stress)
        
        # Count primary stresses (each primary stress starts a new foot)
        return sum(1 for s in stress_pattern if s == 1) if stress_pattern else 0
    except Exception:
        return 0


def extract_stress_pattern(word_textgrid, word_tier_name, word_index, syllable_tier_name):
    """
    Extract stress pattern as a string.
    
    Args:
        word_textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        word_index: Word interval index (1-based)
        syllable_tier_name: Name of the syllable tier
    
    Returns:
        String of stress numbers (e.g., '101', '20')
    """
    try:
        word_tier = word_textgrid.getTier(word_tier_name)
        word_entry = word_tier.entries[word_index - 1]
        word_start = word_entry.start
        word_end = word_entry.end
        
        syllable_tier = word_textgrid.getTier(syllable_tier_name)
        pattern = []
        
        for entry in syllable_tier.entries:
            if entry.start >= word_start and entry.end <= word_end:
                stress = extract_current_syllable_stress(entry.label)
                pattern.append(str(stress))
        
        return "".join(pattern) if pattern else ""
    except Exception:
        return ""


def extract_current_stress(syllable_label):
    """
    Extract stress level of current syllable.
    Wrapper for extract_current_syllable_stress for API consistency.
    """
    return extract_current_syllable_stress(syllable_label)


def extract_onset_type(syllable_label, phone_tier=None, syllable_start=None, syllable_end=None):
    """
    Determine the type of syllable onset.
    
    Args:
        syllable_label: Syllable label (optional if phone tier provided)
        phone_tier: Phone tier for detailed analysis
        syllable_start: Start time of syllable
        syllable_end: End time of syllable
    
    Returns:
        String: 'none', 'single', 'cluster', or None
    """
    # This is a simplified version - full implementation would analyze phone tier
    if not syllable_label:
        return 'none'
    
    # Simple heuristic: if syllable label starts with a vowel, no onset
    vowel_starts = {'a', 'e', 'i', 'o', 'u', 'æ', 'ɑ', 'ɔ', 'ə', 'ɝ', 'ɹ'}
    if syllable_label and syllable_label[0].lower() in vowel_starts:
        return 'none'
    
    # Check for consonant clusters (multiple consonants before vowel)
    # This is simplified; full implementation would need phone-level analysis
    if len(syllable_label) > 2:
        return 'cluster'
    
    return 'single'


def extract_coda_type(syllable_label, phone_tier=None, syllable_start=None, syllable_end=None):
    """
    Determine the type of syllable coda.
    
    Args:
        syllable_label: Syllable label (optional if phone tier provided)
        phone_tier: Phone tier for detailed analysis
        syllable_start: Start time of syllable
        syllable_end: End time of syllable
    
    Returns:
        String: 'none', 'single', 'cluster', or None
    """
    # Simplified version - full implementation would analyze phone tier
    if not syllable_label:
        return 'none'
    
    # Check if syllable ends with a vowel (no coda)
    vowel_ends = {'a', 'e', 'i', 'o', 'u', 'æ', 'ɑ', 'ɔ', 'ə', 'ɝ', 'ɹ'}
    if syllable_label and syllable_label[-1].lower() in vowel_ends:
        return 'none'
    
    # Check for consonant clusters
    if len(syllable_label) > 2:
        return 'cluster'
    
    return 'single'


def extract_word_final(word_textgrid, word_tier_name, word_index):
    """
    Determine if current word is at the end of the utterance/file.
    
    Args:
        word_textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        word_index: Word interval index (1-based)
    
    Returns:
        Boolean (1/0) indicating word-final position
    """
    try:
        tier = word_textgrid.getTier(word_tier_name)
        return 1 if word_index == len(tier.entries) else 0
    except Exception:
        return 0


def extract_prepausal(word_textgrid, word_tier_name, word_index, silence_threshold=0.3):
    """
    Determine if word is prepausal (followed by silence).
    
    Args:
        word_textgrid: Praatio TextGrid object
        word_tier_name: Name of the word tier
        word_index: Word interval index (1-based)
        silence_threshold: Gap duration to consider as pause (seconds)
    
    Returns:
        Boolean (1/0) indicating prepausal position
    """
    try:
        tier = word_textgrid.getTier(word_tier_name)
        
        if word_index >= len(tier.entries):
            return 1  # End of file counts as prepausal
        
        current = tier.entries[word_index - 1]
        next_word = tier.entries[word_index]
        
        gap = next_word.start - current.end
        
        return 1 if gap >= silence_threshold else 0
    except Exception:
        return 0


# =============================================================================
# PVI (PAIRWISE VARIABILITY INDEX) FUNCTIONS
# =============================================================================

def extract_pvi(durations):
    """
    Calculate Pairwise Variability Index from a list of durations.
    PVI = Σ |d_n - d_{n+1}| / (n-1)
    
    Args:
        durations: List of duration values
    
    Returns:
        Float PVI value, or None if insufficient data
    """
    if not durations or len(durations) < 2:
        return None
    
    total_diff = 0.0
    for i in range(len(durations) - 1):
        total_diff += abs(durations[i] - durations[i + 1])
    
    return total_diff / (len(durations) - 1)


def extract_nucleus_duration(phone_duration, is_vowel=True):
    """
    Extract duration of syllable nucleus (typically the vowel).
    
    Args:
        phone_duration: Duration of the phone/segment
        is_vowel: Whether the current phone is a vowel
    
    Returns:
        Duration if vowel, otherwise None
    """
    return phone_duration if is_vowel else None


def extract_last_nucleus_duration(previous_nucleus_duration):
    """
    Get the duration of the previous syllable nucleus.
    This is typically stored in state between intervals.
    
    Args:
        previous_nucleus_duration: Duration from previous nucleus
    
    Returns:
        Previous nucleus duration, or None if not available
    """
    return previous_nucleus_duration


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_vowel(phone_label):
    """
    Determine if a phone label represents a vowel.
    
    Args:
        phone_label: Phone label string
    
    Returns:
        Boolean indicating if phone is a vowel
    """
    if not phone_label:
        return False
    
    # Common vowel sets (expandable)
    vowels = {
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'eh', 'er', 'ey',
        'ih', 'iy', 'ow', 'oy', 'uh', 'uw', 'a', 'e', 'i', 'o', 'u',
        'ɑ', 'æ', 'ʌ', 'ɔ', 'ə', 'ɝ', 'ɪ', 'i', 'oʊ', 'ʊ', 'u', 'aɪ', 'aʊ', 'ɔɪ'
    }
    
    # Strip stress markers (e.g., 'AA1' -> 'aa')
    base_label = re.sub(r'[0-9]', '', phone_label.lower())
    
    return base_label in vowels


def get_phone_tier_index(textgrid, tier_name):
    """
    Get the index of a tier by name.
    
    Args:
        textgrid: Praatio TextGrid object
        tier_name: Name of the tier
    
    Returns:
        Tier index (0-based) or None if not found
    """
    try:
        return textgrid.tierNames.index(tier_name)
    except ValueError:
        return None
