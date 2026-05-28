"""
MADE.py - Master Acoustic Feature Extraction Controller
@author: taylosh
Created on 21 Jan 2026
Last edited on 28 May 2026

A comprehensive, interactive CLI tool for automated phonetic analysis. 
This controller manages the full pipeline from file discovery to structured data export.

Key Functionalities:
- Recursive Search: Automatically pairs WAV and TextGrid files across directories.
- Multi-Tier Processing: Supports iterative selection of multiple TextGrid tiers.
- Granular Filtering: Allows processing of all, non-empty, or string-specific intervals.
- Modular Extraction: Fully customizable feature selection via interactive sub-menus.

Libraries (contained in ./libs):
    pitch_extractor: Pitch/F0 statistics and contours
    formant_extractor: Vowel resonance (F1, F2) and contours
    intensity_extractor: Loudness statistics and contours
    voice_quality_extractor: Glottal source metrics (HNR, Jitter, Shimmer)
    timing_extractor: Temporal measurements and duration
    spectral_extractor: Fricative analysis (CoG, Skewness, Centroid)
    mfcc_extractor: MFCC coefficient extraction
    context_extractor: Contextual label extraction

User Configuration:
- Supports custom time-step increments for contour data (default 10%).
- Automatically generates mandatory metadata: filename, label, start/end times, and voicing status.
"""

import os
import numpy as np
import pandas as pd
import warnings
from praatio import textgrid
import parselmouth
from pathlib import Path
import re
from datetime import datetime

# Rich imports for beautiful progress bars
try:
    from rich.console import Console
    from rich.progress import (
        Progress, 
        SpinnerColumn, 
        BarColumn, 
        TextColumn, 
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskProgressColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for beautiful progress bars: pip install rich")

# Silence mathematical warnings for cleaner CLI output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize Rich console if available
if RICH_AVAILABLE:
    console = Console()

# --- Package Initialization ---
libs_path = Path("./libs")
init_file = libs_path / "__init__.py"
if not libs_path.exists():
    libs_path.mkdir(parents=True, exist_ok=True)
if not init_file.exists():
    init_file.touch()

# --- Modular Imports ---
from libs import (
    pitch_extractor,
    formant_extractor,
    intensity_extractor,
    voice_quality_extractor,
    timing_extractor,
    spectral_extractor,
    mfcc_extractor,
    context_extractor,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_user_input(prompt, default=None):
    """Get user input with optional default value."""
    if default is not None:
        res = input(f"{prompt} [{default}]: ").strip()
        return res if res else default
    return input(f"{prompt}: ").strip()


def get_boolean_choice(prompt, default=False):
    """Get yes/no choice from user."""
    default_str = "y" if default else "n"
    choice = get_user_input(f"{prompt} (y/n)", default_str).lower()
    return choice == 'y'


def get_numbered_choice(prompt, options, allow_multiple=True):
    """
    Display numbered options and get user selection(s).
    Returns list of selected items.
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    if allow_multiple:
        print("\nEnter numbers (comma-separated) or 'all':")
        choice = get_user_input("Selection", "all").lower()
        
        if choice == 'all':
            return options
        
        selected = []
        for num_str in choice.split(','):
            num_str = num_str.strip()
            if num_str.isdigit() and 1 <= int(num_str) <= len(options):
                selected.append(options[int(num_str) - 1])
        return selected if selected else options
    else:
        while True:
            choice = get_user_input("Selection")
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")


def detect_interval_type(tier_name):
    """Auto-detect interval type from tier name."""
    tier_lower = tier_name.lower()
    if 'phone' in tier_lower or 'segment' in tier_lower:
        return "phone"
    elif 'word' in tier_lower:
        return "word"
    elif 'syllable' in tier_lower:
        return "syllable"
    else:
        return "unknown"


def get_filter_description(interval_filter):
    """Get human-readable description of a filter."""
    filter_type_name = IntervalFilter.FILTER_TYPES[interval_filter.filter_type]
    if interval_filter.filter_values:
        return f"{filter_type_name}: {', '.join(interval_filter.filter_values)}"
    return filter_type_name


def format_time(seconds):
    """Format seconds into HH:MM:SS or MM:SS format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# =============================================================================
# TIER SELECTION CLASS
# =============================================================================

class TierSelector:
    """Handles selection of tiers from a TextGrid."""
    
    SELECTION_METHODS = {
        "1": "Exact name match",
        "2": "Name starts with",
        "3": "Name ends with",
        "4": "Name contains",
        "5": "Tier number"
    }
    
    def __init__(self, textgrid_path=None):
        """
        Initialize tier selector.
        If textgrid_path is provided, shows available tiers.
        """
        self.textgrid_path = textgrid_path
        self.available_tiers = []
        if textgrid_path:
            self._load_available_tiers()
    
    def _load_available_tiers(self):
        """Load tier names from a TextGrid file."""
        try:
            tg = textgrid.openTextgrid(str(self.textgrid_path), includeEmptyIntervals=True)
            self.available_tiers = tg.tierNames
            print(f"\nAvailable tiers in {Path(self.textgrid_path).name}:")
            for i, tier_name in enumerate(self.available_tiers, 1):
                print(f"  {i}. {tier_name}")
        except Exception as e:
            print(f"Warning: Could not read TextGrid: {e}")
            self.available_tiers = []
    
    def select_tiers(self):
        """
        Interactive tier selection.
        Returns list of selected tier names.
        """
        selected_tiers = []
        
        print("\n" + "="*60)
        print("TIER SELECTION")
        print("="*60)
        
        while True:
            print(f"\n--- Selecting tier #{len(selected_tiers) + 1} ---")
            
            # Ask how they want to identify the tier
            method_choice = get_numbered_choice(
                "How would you like to identify this tier?",
                list(self.SELECTION_METHODS.values()),
                allow_multiple=False
            )
            
            # Find the method key
            method_key = [k for k, v in self.SELECTION_METHODS.items() if v == method_choice][0]
            
            tier_name = None
            
            if method_key == "5":  # Tier number
                while True:
                    tier_num = get_user_input("Enter tier number")
                    if tier_num.isdigit() and 1 <= int(tier_num) <= len(self.available_tiers):
                        tier_name = self.available_tiers[int(tier_num) - 1]
                        break
                    print(f"Invalid number. Please enter 1-{len(self.available_tiers)}.")
            else:
                # Name-based selection
                search_term = get_user_input("Enter search term")
                
                if method_key == "1":  # Exact match
                    if search_term in self.available_tiers:
                        tier_name = search_term
                    else:
                        print(f"Tier '{search_term}' not found. Available tiers: {', '.join(self.available_tiers)}")
                        continue
                
                elif method_key == "2":  # Starts with
                    matches = [t for t in self.available_tiers if t.startswith(search_term)]
                    if len(matches) == 1:
                        tier_name = matches[0]
                    elif len(matches) > 1:
                        print(f"Multiple tiers start with '{search_term}':")
                        tier_name = get_numbered_choice("Select tier", matches, allow_multiple=False)
                    else:
                        print(f"No tiers start with '{search_term}'")
                        continue
                
                elif method_key == "3":  # Ends with
                    matches = [t for t in self.available_tiers if t.endswith(search_term)]
                    if len(matches) == 1:
                        tier_name = matches[0]
                    elif len(matches) > 1:
                        print(f"Multiple tiers end with '{search_term}':")
                        tier_name = get_numbered_choice("Select tier", matches, allow_multiple=False)
                    else:
                        print(f"No tiers end with '{search_term}'")
                        continue
                
                elif method_key == "4":  # Contains
                    matches = [t for t in self.available_tiers if search_term.lower() in t.lower()]
                    if len(matches) == 1:
                        tier_name = matches[0]
                    elif len(matches) > 1:
                        print(f"Multiple tiers contain '{search_term}':")
                        tier_name = get_numbered_choice("Select tier", matches, allow_multiple=False)
                    else:
                        print(f"No tiers contain '{search_term}'")
                        continue
            
            if tier_name:
                selected_tiers.append(tier_name)
                print(f"  ✓ Added tier: '{tier_name}'")
            
            # Ask if they want to select another tier
            if not get_boolean_choice("Select another tier?", default=False):
                break
        
        return selected_tiers


# =============================================================================
# INTERVAL FILTER CLASS
# =============================================================================

class IntervalFilter:
    """Handles selection of which intervals to extract from a tier."""
    
    FILTER_TYPES = {
        "1": "All intervals (including empty)",
        "2": "All non-empty intervals",
        "3": "Specific label(s) - exact match",
        "4": "Labels containing substring",
        "5": "Labels matching regex pattern",
        "6": "Full tier (single interval from start to end)"
    }
    
    def __init__(self, tier_name):
        self.tier_name = tier_name
        self.filter_type = None
        self.filter_values = []  # For specific labels, substrings, or regex patterns
    
    def configure(self):
        """Interactive configuration of interval filter."""
        print(f"\n--- Filtering intervals on tier: '{self.tier_name}' ---")
        
        filter_choice = get_numbered_choice(
            "Which intervals should be extracted?",
            list(self.FILTER_TYPES.values()),
            allow_multiple=False
        )
        
        self.filter_type = [k for k, v in self.FILTER_TYPES.items() if v == filter_choice][0]
        
        # Handle filter-specific configurations
        if self.filter_type in ["3", "4", "5"]:
            print(f"\n--- Defining {filter_choice.lower()} ---")
            while True:
                value = get_user_input("Enter value")
                self.filter_values.append(value)
                print(f"  ✓ Added: '{value}'")
                if not get_boolean_choice("Add another value?", default=False):
                    break
        
        return self
    
    def matches(self, label):
        """
        Check if an interval label matches the filter.
        Returns True if interval should be extracted.
        """
        label = label.strip() if label else ""
        
        if self.filter_type == "1":  # All intervals
            return True
        
        elif self.filter_type == "2":  # Non-empty only
            return len(label) > 0
        
        elif self.filter_type == "3":  # Exact match
            return label in self.filter_values
        
        elif self.filter_type == "4":  # Contains substring
            return any(substring.lower() in label.lower() for substring in self.filter_values)
        
        elif self.filter_type == "5":  # Regex pattern
            return any(re.search(pattern, label) for pattern in self.filter_values)
        
        elif self.filter_type == "6":  # Full tier
            return True  # Special handling in main loop
        
        return True
    
    def is_full_tier(self):
        """Returns True if this filter represents the entire tier as one interval."""
        return self.filter_type == "6"


# =============================================================================
# FEATURE SELECTION SYSTEM
# =============================================================================

class FeatureSelector:
    """
    Handles selection of acoustic features to extract.
    Supports category-level selection followed by submenu for each category.
    """
    
    # Complete Feature Map (26 sections as designed)
    # [FEATURE_MAP dictionary remains the same - omitted for brevity]
    # In the actual file, this would contain all 26 categories defined earlier
    FEATURE_MAP = {}  # Placeholder - full map would be here
    
    def __init__(self):
        """Initialize feature selector with empty selections dictionary."""
        self.selections = {}
        self._selection_cache = {}
    
    # [All FeatureSelector methods remain the same - omitted for brevity]
    # In the actual file, these would contain the full implementation


# =============================================================================
# CSV OUTPUT CONFIGURATION
# =============================================================================

class CSVOutputConfig:
    """Configuration for CSV output column ordering."""
    
    # Fixed columns (always present, in this exact order)
    FIXED_COLUMNS = [
        "file_name",
        "tier_name",
        "interval_label",
        "interval_type",
        "preceding_label",
        "following_label",
        "has_voicing",
        "start_time",
        "end_time",
        "duration",
    ]
    
    def __init__(self, feature_selector):
        self.feature_selector = feature_selector
        self.dynamic_columns = self._build_dynamic_columns()
    
    def _build_dynamic_columns(self):
        """Build dynamic column list based on user feature selections."""
        dynamic_columns = []
        
        all_selected_features = {}
        for context_key, categories in self.feature_selector.selections.items():
            for cat_key, cat_config in categories.items():
                if cat_key not in all_selected_features:
                    all_selected_features[cat_key] = cat_config
        
        for cat_key in sorted(self.feature_selector.FEATURE_MAP.keys(), key=int):
            if cat_key in all_selected_features:
                cat_config = all_selected_features[cat_key]
                selected_features = cat_config.get('features', [])
                
                for feature_name in selected_features:
                    if 'Contour' in feature_name:
                        step = cat_config.get('contour_step', 10)
                        num_points = (100 // step) + 1
                        for i in range(num_points):
                            percentage = i * step
                            if percentage == 0:
                                col_name = f"{feature_name}_onset"
                            elif percentage == 100:
                                col_name = f"{feature_name}_offset"
                            else:
                                col_name = f"{feature_name}_{percentage}pct"
                            dynamic_columns.append(col_name)
                    else:
                        dynamic_columns.append(feature_name)
        
        return dynamic_columns
    
    @property
    def all_columns(self):
        return self.FIXED_COLUMNS + self.dynamic_columns
    
    def create_empty_row(self):
        return {col: None for col in self.all_columns}


# =============================================================================
# EXTRACTION ENGINE WITH RICH PROGRESS BAR
# =============================================================================

class ExtractionEngine:
    """
    Main extraction engine that processes files and extracts features.
    Includes Rich progress bar for long-running extractions.
    """
    
    def __init__(self, feature_selector, csv_config):
        self.feature_selector = feature_selector
        self.csv_config = csv_config
        self.progress_counter = 0
        self.start_time = None
    
    def _get_feature_function(self, feature_name):
        """
        Map feature name to its extraction function.
        Returns (function, is_contour, requires_voicing)
        """
        feature_registry = {
            # Pitch features
            "PitchMean": (pitch_extractor.extract_pitch_mean, False, True),
            "PitchSD": (pitch_extractor.extract_pitch_sd, False, True),
            "PitchMedian": (pitch_extractor.extract_pitch_median, False, True),
            "PitchRange": (pitch_extractor.extract_pitch_range, False, True),
            "PitchMin": (pitch_extractor.extract_pitch_min, False, True),
            "PitchMinTime": (pitch_extractor.extract_pitch_min_time, False, True),
            "PitchMax": (pitch_extractor.extract_pitch_max, False, True),
            "PitchMaxTime": (pitch_extractor.extract_pitch_max_time, False, True),
            "PitchMidpoint": (pitch_extractor.extract_pitch_midpoint, False, True),
            "PitchContour": (pitch_extractor.extract_pitch_contour, True, True),
            
            # Spectral moments
            "SpectralCentroid": (spectral_extractor.extract_spectral_centroid, False, False),
            "SpectralSpread": (spectral_extractor.extract_spectral_spread, False, False),
            "SpectralSkewness": (spectral_extractor.extract_spectral_skewness, False, False),
            "SpectralKurtosis": (spectral_extractor.extract_spectral_kurtosis, False, False),
            "SpectralCentroidContour": (spectral_extractor.extract_centroid_contour, True, False),
            "SpectralSpreadContour": (spectral_extractor.extract_spread_contour, True, False),
            "SpectralSkewnessContour": (spectral_extractor.extract_skewness_contour, True, False),
            "SpectralKurtosisContour": (spectral_extractor.extract_kurtosis_contour, True, False),
            
            # Band energy
            "BandEnergy_0_500": (spectral_extractor.extract_band_energy_0_500, False, False),
            "BandEnergy_500_1k": (spectral_extractor.extract_band_energy_500_1k, False, False),
            "BandEnergy_1k_2k": (spectral_extractor.extract_band_energy_1k_2k, False, False),
            "BandEnergy_2k_4k": (spectral_extractor.extract_band_energy_2k_4k, False, False),
            "BandEnergy_4k_8k": (spectral_extractor.extract_band_energy_4k_8k, False, False),
            
            # Intensity
            "IntensityMean": (intensity_extractor.extract_intensity_mean, False, False),
            "IntensitySD": (intensity_extractor.extract_intensity_sd, False, False),
            "IntensityRange": (intensity_extractor.extract_intensity_range, False, False),
            "IntensityMin": (intensity_extractor.extract_intensity_min, False, False),
            "IntensityMax": (intensity_extractor.extract_intensity_max, False, False),
            "IntensityContour": (intensity_extractor.extract_intensity_contour, True, False),
            
            # Duration
            "PhoneDuration": (timing_extractor.extract_interval_duration, False, False),
            "WordDuration": (context_extractor.extract_word_duration, False, False),
            
            # MFCC
            "MFCC_01_12": (mfcc_extractor.extract_mfcc, False, True),
            "MFCCContour": (mfcc_extractor.extract_mfcc_contour, True, True),
            "MFCC_Delta_01_12": (mfcc_extractor.extract_mfcc_delta, False, True),
            "MFCC_DeltaDelta_01_12": (mfcc_extractor.extract_mfcc_deltadelta, False, True),
            
            # Zero-crossing
            "ZeroCrossingCount": (timing_extractor.extract_zero_crossing_count, False, False),
            "ZeroCrossingRate": (timing_extractor.extract_zero_crossing_rate, False, False),
            
            # VOT
            "VOT": (timing_extractor.extract_vot, False, False),
            "VOTStart": (timing_extractor.extract_vot_start, False, False),
            "VOTEnd": (timing_extractor.extract_vot_end, False, False),
        }
        
        return feature_registry.get(feature_name, (None, False, False))
    
    def extract_contour_values(self, func, sound, start, end, step, feature_name):
        """Extract contour values and format as dictionary with column names."""
        result = func(sound, start, end, step)
        
        if not result or not isinstance(result, dict):
            return {}
        
        formatted = {}
        num_points = (100 // step) + 1
        
        for i in range(num_points):
            percentage = i * step
            if percentage == 0:
                col_name = f"{feature_name}_onset"
            elif percentage == 100:
                col_name = f"{feature_name}_offset"
            else:
                col_name = f"{feature_name}_{percentage}pct"
            
            if percentage in result:
                formatted[col_name] = result[percentage]
            elif str(percentage) in result:
                formatted[col_name] = result[str(percentage)]
            elif i in result:
                formatted[col_name] = result[i]
        
        return formatted
    
    def _get_preceding_label(self, tier, entry):
        """Get label of preceding interval."""
        try:
            for i, e in enumerate(tier.entries):
                if e.start == entry.start and e.end == entry.end:
                    if i > 0:
                        return tier.entries[i-1].label
                    break
        except:
            pass
        return None
    
    def _get_following_label(self, tier, entry):
        """Get label of following interval."""
        try:
            for i, e in enumerate(tier.entries):
                if e.start == entry.start and e.end == entry.end:
                    if i < len(tier.entries) - 1:
                        return tier.entries[i+1].label
                    break
        except:
            pass
        return None
    
    def _check_voicing(self, sound, start, end):
        """Check if interval contains voiced frames."""
        try:
            pitch = sound.to_pitch()
            times = pitch.xs()
            values = pitch.selected_array['frequency']
            mask = (times >= start) & (times <= end)
            return bool(np.any(values[mask] > 0))
        except:
            return False
    
    def process_interval(self, row, sound, entry, tier, feature_config):
        """Process a single interval and extract all selected features."""
        row["preceding_label"] = self._get_preceding_label(tier, entry)
        row["following_label"] = self._get_following_label(tier, entry)
        row["has_voicing"] = self._check_voicing(sound, entry.start, entry.end)
        row["start_time"] = entry.start
        row["end_time"] = entry.end
        row["duration"] = entry.end - entry.start
        
        for cat_key, cat_config in feature_config.items():
            requires_voicing = cat_config.get('requires_voicing', False)
            
            if requires_voicing and not row["has_voicing"]:
                continue
            
            for feature_name in cat_config['features']:
                func, is_contour, needs_voicing = self._get_feature_function(feature_name)
                
                if func is None:
                    continue
                
                if needs_voicing and not row["has_voicing"]:
                    continue
                
                try:
                    if is_contour:
                        step = cat_config.get('contour_step', 10)
                        contour_values = self.extract_contour_values(
                            func, sound, entry.start, entry.end, step, feature_name
                        )
                        for col_name, value in contour_values.items():
                            if col_name in row:
                                row[col_name] = value
                    else:
                        value = func(sound, entry.start, entry.end)
                        if feature_name in row:
                            row[feature_name] = value
                except Exception as e:
                    pass
    
    def run_with_rich_progress(self, file_pairs, tier_filters, output_path):
        """Run extraction with Rich progress bar."""
        # First, count total intervals to process for accurate progress
        print("\n" + "="*60)
        print("EXTRACTION IN PROGRESS")
        print("="*60)
        
        # Count total intervals
        print("\nScanning for total intervals...")
        total_intervals = 0
        for wav_path, tg_path in file_pairs:
            try:
                tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
                for tier_name, interval_filter in tier_filters.items():
                    if tier_name in tg.tierNames:
                        tier = tg.getTier(tier_name)
                        for entry in tier.entries:
                            if interval_filter.matches(entry.label):
                                total_intervals += 1
            except:
                pass
        
        if total_intervals == 0:
            print("⚠ No intervals to process.")
            return 0
        
        print(f"Found {total_intervals} intervals to process.")
        
        all_rows = []
        processed = 0
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Extracting features...", 
                total=total_intervals
            )
            
            for wav_path, tg_path in file_pairs:
                progress.update(task, description=f"[cyan]Processing: {wav_path.name}")
                
                try:
                    sound = parselmouth.Sound(str(wav_path))
                    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
                except Exception as e:
                    progress.console.log(f"[red]Error loading {wav_path.name}: {e}")
                    continue
                
                for tier_name, interval_filter in tier_filters.items():
                    if tier_name not in tg.tierNames:
                        continue
                    
                    tier = tg.getTier(tier_name)
                    interval_type = detect_interval_type(tier_name)
                    filter_desc = get_filter_description(interval_filter)
                    
                    feature_config = self.feature_selector.get_config_for(
                        tier_name, filter_desc, interval_type
                    )
                    
                    if not feature_config:
                        continue
                    
                    entries_to_process = []
                    for entry in tier.entries:
                        if interval_filter.matches(entry.label):
                            entries_to_process.append(entry)
                    
                    if interval_filter.is_full_tier() and entries_to_process:
                        start = min(e.start for e in tier.entries)
                        end = max(e.end for e in tier.entries)
                        entries_to_process = [type('Entry', (), {
                            'start': start, 'end': end, 'label': f"FULL_TIER_{tier_name}"
                        })]
                    
                    for entry in entries_to_process:
                        row = self.csv_config.create_empty_row()
                        row["file_name"] = wav_path.name
                        row["tier_name"] = tier_name
                        row["interval_label"] = entry.label
                        row["interval_type"] = interval_type
                        
                        self.process_interval(row, sound, entry, tier, feature_config)
                        all_rows.append(row)
                        
                        processed += 1
                        progress.update(task, advance=1, description=f"[cyan]Processed {processed}/{total_intervals}")
        
        # Save results
        if all_rows:
            df = pd.DataFrame(all_rows)
            df = df[self.csv_config.all_columns]
            df.to_csv(output_path, index=False, float_format='%.3f')
            
            console.print(f"\n[bold green]✓ EXTRACTION COMPLETE![/bold green]")
            console.print(f"  Total intervals processed: {len(all_rows)}")
            console.print(f"  Output saved to: {output_path}")
        else:
            console.print("\n[bold yellow]⚠ No data extracted. Check your filters and selections.[/bold yellow]")
        
        return len(all_rows)
    
    def run_simple(self, file_pairs, tier_filters, output_path):
        """Run extraction with simple progress output (when Rich not available)."""
        all_rows = []
        total_files = len(file_pairs)
        
        print("\n" + "="*60)
        print("EXTRACTION IN PROGRESS")
        print("="*60)
        
        for file_idx, (wav_path, tg_path) in enumerate(file_pairs, 1):
            print(f"\n[{file_idx}/{total_files}] Processing: {wav_path.name}")
            
            try:
                sound = parselmouth.Sound(str(wav_path))
                tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
            except Exception as e:
                print(f"  Error loading file: {e}")
                continue
            
            for tier_name, interval_filter in tier_filters.items():
                if tier_name not in tg.tierNames:
                    print(f"  Warning: Tier '{tier_name}' not found, skipping")
                    continue
                
                tier = tg.getTier(tier_name)
                interval_type = detect_interval_type(tier_name)
                filter_desc = get_filter_description(interval_filter)
                
                feature_config = self.feature_selector.get_config_for(
                    tier_name, filter_desc, interval_type
                )
                
                if not feature_config:
                    print(f"  No features selected for '{tier_name}', skipping")
                    continue
                
                entries_to_process = []
                for entry in tier.entries:
                    if interval_filter.matches(entry.label):
                        entries_to_process.append(entry)
                
                if interval_filter.is_full_tier() and entries_to_process:
                    start = min(e.start for e in tier.entries)
                    end = max(e.end for e in tier.entries)
                    entries_to_process = [type('Entry', (), {
                        'start': start, 'end': end, 'label': f"FULL_TIER_{tier_name}"
                    })]
                
                if not entries_to_process:
                    continue
                
                print(f"  Processing {len(entries_to_process)} intervals from '{tier_name}'")
                
                for entry in entries_to_process:
                    row = self.csv_config.create_empty_row()
                    row["file_name"] = wav_path.name
                    row["tier_name"] = tier_name
                    row["interval_label"] = entry.label
                    row["interval_type"] = interval_type
                    
                    self.process_interval(row, sound, entry, tier, feature_config)
                    all_rows.append(row)
                    
                    self.progress_counter += 1
                    if self.progress_counter % 100 == 0:
                        print(f"    Processed {self.progress_counter} intervals so far...")
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            df = df[self.csv_config.all_columns]
            df.to_csv(output_path, index=False, float_format='%.3f')
            print(f"\n{'='*60}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*60}")
            print(f"Total intervals processed: {len(all_rows)}")
            print(f"Output saved to: {output_path}")
        else:
            print("\n⚠ No data extracted. Check your filters and selections.")
        
        return len(all_rows)
    
    def run(self, file_pairs, tier_filters, output_path):
        """Main run method - uses Rich if available, otherwise simple."""
        if RICH_AVAILABLE:
            return self.run_with_rich_progress(file_pairs, tier_filters, output_path)
        else:
            return self.run_simple(file_pairs, tier_filters, output_path)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    
    # Rich welcome banner
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]MADE - Master Acoustic Feature Extraction Controller[/bold cyan]\n"
            "[dim]Version 3.0 | Comprehensive Phonetic Analysis[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*60)
        print("MADE - Master Acoustic Feature Extraction Controller")
        print("="*60)
    
    # -------------------------------------------------------------------------
    # 1. DIRECTORY SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 1: Directory Selection")
    print("-"*40)
    
    search_dir = get_user_input("Enter directory containing WAV and TextGrid files", os.getcwd())
    search_path = Path(search_dir)
    
    if not search_path.exists():
        print(f"Error: Directory '{search_path}' does not exist.")
        return
    
    print(f"  ✓ Using directory: {search_path}")
    
    # -------------------------------------------------------------------------
    # 2. FILE DISCOVERY
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 2: File Discovery")
    print("-"*40)
    
    print("Scanning directory tree for files...")
    
    all_wavs = {f.stem: f for f in search_path.rglob("*") if f.suffix.lower() == ".wav"}
    all_tgs = {f.stem: f for f in search_path.rglob("*") if f.suffix.lower() == ".textgrid"}
    
    file_pairs = []
    for stem, wav_path in all_wavs.items():
        if stem in all_tgs:
            file_pairs.append((wav_path, all_tgs[stem]))
    
    if not file_pairs:
        print(f"\nError: No matching WAV/TextGrid pairs found in {search_path}.")
        print(f"  WAV files found: {len(all_wavs)}")
        print(f"  TextGrid files found: {len(all_tgs)}")
        print("\nMake sure your files have matching base names (e.g., 'audio1.wav' and 'audio1.TextGrid')")
        return
    
    print(f"\n  ✓ Found {len(file_pairs)} matching file pairs")
    print(f"  First few files: {', '.join([p[0].name for p in file_pairs[:3]])}")
    if len(file_pairs) > 3:
        print(f"  ... and {len(file_pairs) - 3} more")
    
    # -------------------------------------------------------------------------
    # 3. TIER SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 3: Tier Selection")
    print("-"*40)
    print("Note: Using first TextGrid file to display available tiers")
    print("      (Assumes all TextGrids have similar structure)")
    
    first_tg_path = file_pairs[0][1]
    tier_selector = TierSelector(first_tg_path)
    selected_tiers = tier_selector.select_tiers()
    
    if not selected_tiers:
        print("Error: No tiers selected. Exiting.")
        return
    
    print(f"\n  ✓ Selected {len(selected_tiers)} tier(s): {', '.join(selected_tiers)}")
    
    # -------------------------------------------------------------------------
    # 4. INTERVAL FILTER CONFIGURATION
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 4: Interval Filter Configuration")
    print("-"*40)
    
    tier_filters = {}
    for tier_name in selected_tiers:
        interval_filter = IntervalFilter(tier_name).configure()
        tier_filters[tier_name] = interval_filter
    
    # -------------------------------------------------------------------------    # 5. FEATURE SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 5: Feature Selection")
    print("-"*40)
    
    feature_selector = FeatureSelector()
    
    for tier_name, interval_filter in tier_filters.items():
        filter_desc = get_filter_description(interval_filter)
        interval_type = detect_interval_type(tier_name)
        feature_selector.select_features(tier_name, filter_desc, interval_type)
    
    # Display selection summary with Rich if available
    if RICH_AVAILABLE:
        summary_table = Table(title="Feature Selection Summary", border_style="green")
        summary_table.add_column("Configuration", style="cyan")
        summary_table.add_column("Details", style="white")
        
        for (tier_name, filter_desc, interval_type), categories in feature_selector.selections.items():
            cat_details = []
            for cat_key, cat_config in categories.items():
                cat_name = feature_selector.FEATURE_MAP[cat_key]['name']
                cat_details.append(f"  • {cat_name}: {len(cat_config['features'])} features")
                if cat_config.get('contour_step'):
                    cat_details.append(f"    (contours at {cat_config['contour_step']}% steps)")
            summary_table.add_row(
                f"{tier_name}\n[{filter_desc}]\n({interval_type})",
                "\n".join(cat_details)
            )
        console.print(summary_table)
    else:
        print("\n" + "="*70)
        print("COMPLETE FEATURE SELECTION SUMMARY")
        print("="*70)
        print(feature_selector.get_selection_summary())
    
    # -------------------------------------------------------------------------
    # 6. OUTPUT CONFIGURATION
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 6: Output Configuration")
    print("-"*40)
    
    raw_output = get_user_input("Enter the name for the output CSV file", "MADE_acoustic_results.csv")
    output_csv = raw_output if raw_output.lower().endswith('.csv') else raw_output + ".csv"
    print(f"  ✓ Output will be saved to: {output_csv}")
    
    # -------------------------------------------------------------------------
    # 7. CONFIRMATION
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Input directory:    {search_path}")
    print(f"File pairs found:   {len(file_pairs)}")
    print(f"Selected tiers:     {len(selected_tiers)}")
    for tier_name, interval_filter in tier_filters.items():
        filter_type_name = IntervalFilter.FILTER_TYPES[interval_filter.filter_type]
        print(f"  - {tier_name}: {filter_type_name}")
    print(f"Feature configs:    {len(feature_selector.selections)}")
    print(f"Output file:        {output_csv}")
    
    print("\n" + "="*60)
    if not get_boolean_choice("Proceed with extraction?", default=True):
        print("Extraction cancelled.")
        return
    
    # -------------------------------------------------------------------------
    # 8. EXTRACTION
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("STEP 7: Extraction")
    print("-"*40)
    
    csv_config = CSVOutputConfig(feature_selector)
    engine = ExtractionEngine(feature_selector, csv_config)
    num_rows = engine.run(file_pairs, tier_filters, output_csv)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ MADE completed successfully![/bold green] Processed {num_rows} rows.")
    else:
        print(f"\n✓ MADE completed successfully! Processed {num_rows} rows.")


if __name__ == "__main__":
    main()
