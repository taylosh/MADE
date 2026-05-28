MADE: Modular Acoustic Data Extractor (v3.0.0)
================================================================================
Comprehensive User Guide & Documentation
--------------------------------------------------------------------------------
MADE is a modular, high-performance toolkit designed for phoneticians and speech 
researchers requiring comprehensive acoustic feature extraction from labeled 
speech corpora. It prioritizes accessibility through an interactive, terminal-based 
user interface while maintaining flexibility for advanced users via modular 
extractor libraries.

Features:
- Recursive WAV/TextGrid pairing across nested directory structures
- 26 feature categories spanning pitch, formants, voice quality, MFCC, and context
- Customizable interval filtering (regex, substring, exact match, full tier)
- Interactive tier selection with partial name matching
- Contour extraction with user-selectable time steps
- Structured CSV output with metadata preservation
- Rich progress bar with ETA for long-running extractions
--------------------------------------------------------------------------------
Table of Contents

1. Introduction & Privacy
2. Installation & Dependencies
3. Quick Start Guide
4. Core Modules (./libs/)
5. User Workflow (Step-by-Step)
   - Step 1: Directory & File Discovery
   - Step 2: Tier Selection
   - Step 3: Interval Filtering
   - Step 4: Feature Selection
   - Step 5: Output Configuration
   - Step 6: Extraction & Progress Monitoring
6. Feature Categories Reference
7. Output CSV Structure
8. Extending MADE (Custom Extractors)
9. Hardware Considerations
10. Citing this Work
11. Acknowledgements & Third-Party Citations
--------------------------------------------------------------------------------
1. Introduction & Privacy

MADE is built to ensure that sensitive research audio data is processed entirely 
on your local machine with no external data transmission.

Local Processing: All acoustic extraction uses Praat via the parselmouth 
Python bindings. No audio leaves your workstation.

Data Handling: Extracted features are saved locally as CSV files. Original 
audio and TextGrid files remain unmodified. No telemetry or usage data is 
collected.

File Structure Requirements: MADE expects paired files with matching base names:
- audiofile1.wav + audiofile1.TextGrid
- interview_01.wav + interview_01.TextGrid

The tool recursively searches all subdirectories, making it ideal for 
organized corpus workflows.
--------------------------------------------------------------------------------
2. Installation & Dependencies

Prerequisites:
- Python 3.8 or higher
- Conda (recommended) or venv for environment management

Create and activate a dedicated environment:

```
bash
conda create -n made python=3.10
conda activate made
Install core dependencies:

bash
pip install parselmouth praatio pandas numpy
Optional (for beautiful progress bars):

bash
pip install rich
Verification: Run the MADE controller to confirm all modules load correctly.

bash
python MADE_v3.py
The script will automatically create the ./libs/ directory structure on first run.
```


Quick Start Guide

For users wanting immediate extraction with sensible defaults:

Organize your files:

text
./corpus/
├── speaker1/
│   ├── recording1.wav
│   ├── recording1.TextGrid
│   └── ...
└── speaker2/
    └── ...

Run MADE:

python MADE_v3.py
Follow the interactive prompts:

Enter corpus directory (or press Enter for current directory)

Select tiers by name, partial match, or number

Choose interval filter (e.g., "non-empty intervals" for phone tiers)

Select feature categories (enter "0" for all, or comma-separated numbers)

Choose specific features within each category

Specify output CSV filename

Review configuration summary and confirm extraction.

Monitor progress via Rich progress bar (if installed) or simple text output.

Locate results in your specified CSV file.

Core Modules (./libs/)

MADE uses a modular architecture where each acoustic domain has a dedicated
extractor library. These modules can be used independently or called by the
master controller.

Module	Functions
pitch_extractor.py	F0 statistics (mean, SD, min, max, median), pitch contours, voicing detection
formant_extractor.py	F1-F5 frequencies, B1-B5 bandwidths, A1-A5 amplitudes, formant modeler predictions
intensity_extractor.py	dB statistics, intensity contours, min/max timing
voice_quality_extractor.py	HNR, NHR, CPPS, H1-H2, jitter (local/RAP/PPQ5/DDP), shimmer (local/APQ3/APQ5/APQ11/DDA), pulse/period analysis, voice breaks
timing_extractor.py	Duration (phone, word, preceding, following), VOT, closure/release, zero-crossing rate, PVI
spectral_extractor.py	Spectral centroid, spread, skewness, kurtosis, peak, slope, band energy (0-500, 500-1k, 1k-2k, 2k-4k, 4k-8k), LTAS measures, sibilant resegmentation
mfcc_extractor.py	MFCC coefficients 1-12, delta, delta-delta, contour extraction
context_extractor.py	Preceding/following labels (phone, word, syllable), stress patterns, syllable structure, PVI
Extensibility: To add a new feature, implement the extraction function in its
domain module and register it in the ExtractionEngine._get_feature_function()
dictionary.

User Workflow (Step-by-Step)

Step 1: Directory & File Discovery

MADE recursively scans the specified directory for all .wav and .TextGrid files,
pairing them by base name (filename without extension).

Displays count of found file pairs

Lists first few files for verification

Assumes TextGrid structure is consistent across all files

Step 2: Tier Selection

Users can identify tiers using five methods:

Exact name match (e.g., "phones", "words")

Name starts with (e.g., "phone" matches "phones", "phone_tier")

Name ends with (e.g., "tier" matches "phone_tier", "word_tier")

Name contains (e.g., "seg" matches "segments", "segment_tier")

Tier number (from displayed list)

Multiple tiers can be selected (e.g., process both "phones" and "words" tiers
in the same run). Each tier gets its own filter and feature configuration.

Step 3: Interval Filtering

For each selected tier, users specify which intervals to extract:

Filter	Description
1. All intervals	Includes empty labels
2. Non-empty	Skips blank intervals
3. Exact match	Only labels matching user-provided strings
4. Contains substring	Labels containing specified text
5. Regex pattern	Labels matching regular expression
6. Full tier	Single interval from tier start to end
Filters 3-5 support multiple values (add until user stops).

Step 4: Feature Selection

MADE organizes features into 26 categories, displayed in two columns:

Vowel-specific (1-7):

Pitch (F0 statistics & contours)

Formant Frequencies (F1-F5)

Formant Bandwidths (B1-B5)

Formant Amplitudes (A1-A5)

Formant Modeler Predictions

Voice Quality (HNR, jitter, shimmer, CPPS, etc.)

Nasalization (A1-P0, A1-P1, F1-FP0, spectral peaks)

Fricative/Sibilant (8-11):
8. Spectral Moments (centroid, spread, skewness, kurtosis)
9. Band Energy (0-500Hz to 4k-8kHz)
10. LTAS Measures (A1, P0, P1, FP0, FP1)
11. Sibilant Resegmentation

Stop/Plosive (12-14):
12. Voice Onset Time (VOT)
13. Closure & Release Duration
14. Release Burst Spectrum

Universal (15-20):
15. Nasal Spectral Measures
16. Intensity (dB statistics & contours)
17. Duration (phone, preceding, following)
18. Zero-Crossing Rate
19. MFCC (coefficients 1-12, contour)
20. MFCC Deltas & Delta-Deltas

Suprasegmental (21-23):
21. Word-Level Duration
22. PVI (Pairwise Variability Index)
23. Syllable Structure (count, stress, onset/coda type)

Contextual (24-26):
24. Phonetic Context (preceding/following phones, stress)
25. Lexical Context (preceding/following words, window)
26. Syllable Context (preceding/following syllables, stress)

Selection workflow:

Enter category numbers (comma-separated) or "0" for all

For each category, select specific features or "0" for all

For contour features, specify step percentage (default 10% = 11 points)

Step 5: Output Configuration

Users specify the output CSV filename (default: MADE_acoustic_results.csv).

Output columns are automatically ordered:
1-10: Fixed metadata columns
11+: Dynamic feature columns (in FEATURE_MAP order, 1-26)

Step 6: Extraction & Progress Monitoring

After confirmation, MADE begins extraction with real-time progress display:

Rich progress bar (if installed) shows:

Current file being processed

Progress percentage

Time elapsed

Estimated time remaining

Simple text fallback shows:

File-by-file progress

Interval count every 100 rows

Extraction proceeds row by row: each row = one interval × one tier × one file.

Feature Categories Reference

Complete list of available features with descriptions:

PITCH (Category 1)

PitchMean: Mean fundamental frequency (Hz)

PitchSD: Standard deviation of pitch (Hz)

PitchMedian: Median pitch (Hz)

PitchRange: Max - min pitch (Hz)

PitchMin/Max: Extremal pitch values (Hz)

PitchMinTime/MaxTime: Timing of extremal pitch (s)

PitchMidpoint: Pitch at temporal midpoint (Hz)

PitchContour: Time-varying pitch (user-selectable points)

FORMANT FREQUENCIES (Category 2)

F1-F5 Mean/SD/Range/Max/Midpoint: Formant statistics (Hz)

F1-F5 Contour: Time-varying formant trajectories

FORMANT BANDWIDTHS (Category 3)

B1-B5 Mean/SD: Bandwidth statistics (Hz)

B1-B5 Contour: Time-varying bandwidth trajectories

FORMANT AMPLITUDES (Category 4)

A1-A5 Mean: Formant peak amplitudes (dB)

A1-A5 Contour: Time-varying amplitude trajectories

FORMANT MODELER (Category 5)

F1-F5 Predicted: Model-predicted formant values

FormantModelR2_F1-F5: Model fit quality (R²)

FormantResidualSS: Residual sum of squares

VOICE QUALITY (Category 6)

HNR: Harmonics-to-noise ratio (dB)

NHR: Noise-to-harmonics ratio (inverse)

CPPS: Cepstral peak prominence smoothed (dB)

H1H2: H1-H2 amplitude difference (dB)

JitterLocal/RAP/PPQ5/DDP: Period perturbation measures

ShimmerLocal/APQ3/APQ5/APQ11/DDA: Amplitude perturbation measures

JitterContour/ShimmerContour: Time-varying perturbation

PulseCount, PeriodCount, MeanPeriod, PeriodSD: Glottal pulse analysis

UnvoicedFraction, VoiceBreakCount, VoiceBreakDegree: Voicing statistics

MeanAutocorrelation: Periodicity strength

SubharmonicRatio: Subharmonic energy proportion

NASALIZATION (Category 7)

A1P0, A1P1, F1FP0: Spectral measures for nasality

SpectralPeaks40dB: Peak count within 40dB of maximum

SPECTRAL MOMENTS (Category 8)

SpectralCentroid: Center of gravity (Hz)

SpectralSpread: Standard deviation of spectrum (Hz)

SpectralSkewness: Asymmetry of spectral distribution

SpectralKurtosis: Peakedness of spectral distribution

SpectralPeak: Frequency of maximum amplitude (Hz)

SpectralSlope: Tilt of spectral envelope (dB/Hz)

All moments available as contours

BAND ENERGY (Category 9)

BandEnergy_0_500, _500_1k, _1k_2k, _2k_4k, _4k_8k: Energy in frequency bands (dB)

BandEnergyDiff_1500_2500_3500_5500: Band energy difference

LTAS (Category 10)

LTAS_A1: Amplitude of first harmonic (dB)

LTAS_P0/P1: Amplitude of peaks below F1 and between F1-F2 (dB)

LTAS_FP0/FP1: Frequencies of those peaks (Hz)

SIBILANT RESEGMENTATION (Category 11)

SibilantStart/End: Resegmented boundaries (s)

SibilantDuration: Duration of sibilant (s)

BandDiffMaxTime: Peak of band energy difference (s)

VOT (Category 12)

VOT: Voice onset time (s)

VOTStart: Release onset (s)

VOTEnd: Voicing onset (s)

CLOSURE & RELEASE (Category 13)

ClosureDuration, ReleaseDuration: Stop phase durations (s)

ReleaseAmplitudeRatio: Burst amplitude relative to closure

ClosureBurstCount, ReleaseBurstCount: Number of transient events

BURST SPECTRUM (Category 14)

BurstSpectralCentroid/Spread/Skewness/Kurtosis: Spectral moments of release

NASAL SPECTRAL (Category 15)

NasalBandEnergyDiff: Low-vs-high frequency energy difference

NasalSpectralPeak: Primary nasal resonance (Hz)

INTENSITY (Category 16)

IntensityMean/SD/Range/Min/Max/Midpoint: dB statistics

IntensityMinTime/MaxTime: Timing of extremal intensity

IntensityDifference: Max - min intensity (dB)

IntensityContour: Time-varying intensity

DURATION (Category 17)

PhoneDuration: Interval duration (s)

PrecedingPhoneDuration, FollowingPhoneDuration: Adjacent interval durations (s)

ZERO-CROSSING RATE (Category 18)

ZeroCrossingCount: Number of zero crossings

ZeroCrossingRate: Crossings per second

MFCC (Category 19)

MFCC_01_12: 12 Mel-frequency cepstral coefficients

MFCCContour: Time-varying MFCC trajectories

MFCC DELTAS (Category 20)

MFCC_Delta_01_12: First derivative of MFCCs

MFCC_DeltaDelta_01_12: Second derivative of MFCCs

WORD-LEVEL DURATION (Category 21)

WordDuration: Duration of word (s)

LeftWordDuration, RightWordDuration: Adjacent word durations (s)

PVI (Category 22)

PVI: Pairwise Variability Index (rhythm metric)

NucleusDuration: Syllable nucleus duration (s)

LastNucleusDuration: Previous nucleus duration for PVI calculation

SYLLABLE STRUCTURE (Category 23)

SyllableCount: Number of syllables in word

FootCount: Number of metrical feet

StressPattern: String of stress levels (e.g., "101")

CurrentStress: 0=unstressed, 1=primary, 2=secondary

OnsetType, CodaType: "none", "single", or "cluster"

WordFinalPosition, PrepausalPosition: Boundary flags (1/0)

PHONETIC CONTEXT (Category 24)

PrecedingPhone, FollowingPhone: Adjacent phone labels

PrecedingPhone2, FollowingPhone2: Two positions away

PrecedingPhoneStress, FollowingPhoneStress: Stress levels of adjacent phones

LEXICAL CONTEXT (Category 25)

PrecedingWord, FollowingWord: Adjacent word labels

PrecedingWordCount, FollowingWordCount: Number of words before/after

ContextWindow: Space-separated window (e.g., "word1 word2 [target] word3")

SYLLABLE CONTEXT (Category 26)

PrecedingSyllable, FollowingSyllable: Adjacent syllable labels

CurrentSyllableStress: Stress level of current syllable

Output CSV Structure

Each row represents ONE interval from ONE tier from ONE file.

Fixed Columns (always present, columns 1-10):

Column	Description
file_name	Source WAV filename
tier_name	TextGrid tier name
interval_label	Label from TextGrid
interval_type	phone / word / syllable / unknown
preceding_label	Label of previous interval (or "boundary")
following_label	Label of next interval (or "boundary")
has_voicing	Boolean (True/False) - contains voiced frames
start_time	Interval start time (seconds)
end_time	Interval end time (seconds)
duration	Interval duration (seconds)
Dynamic Columns (position varies based on selections):

Single-value features: appear as single columns (e.g., "PitchMean")

Contour features: appear as multiple columns (e.g., "PitchContour_onset",
"PitchContour_10pct", ..., "PitchContour_offset")

Column ordering: Fixed columns first, then dynamic columns in FEATURE_MAP
order (categories 1-26). Within each category, features appear in the order
defined in FEATURE_MAP.

Missing data: Features that cannot be extracted (e.g., pitch on unvoiced
intervals, formants on stops) remain as empty/NaN cells. This preserves
the rectangular CSV structure without requiring placeholder values.

Example row (abbreviated):
file_name, tier_name, interval_label, interval_type, preceding_label,
following_label, has_voicing, start_time, end_time, duration, PitchMean,
PitchContour_onset, PitchContour_50pct, F1Mean, ...

speaker1.wav, phones, AA, phone, None, B, True, 1.234, 1.456, 0.222,
125.4, 120.2, 126.1, 720.5, ...

Extending MADE (Custom Extractors)

Researchers can add new acoustic measures by following this protocol:

Choose or create the appropriate domain module in ./libs/:

Add new function to existing module (e.g., spectral_extractor.py)

Or create new module for novel feature types

Implement extraction function with standard signature:

python
def extract_custom_feature(sound, start_time, end_time, **kwargs):
    """
    Args:
        sound: parselmouth.Sound object
        start_time: float (seconds)
        end_time: float (seconds)
        **kwargs: optional parameters (e.g., step for contours)
    
    Returns:
        For single value: float or int
        For contour: dict {percentage: value}
    """
    # Implementation using parselmouth/Praat calls
    return result
Register function in ExtractionEngine._get_feature_function():

python
"CustomFeature": (custom_extractor.extract_custom_feature, False, requires_voicing),
"CustomContour": (custom_extractor.extract_custom_contour, True, requires_voicing),
Add to FEATURE_MAP in FeatureSelector class:

python
"27": {
    "name": "Custom Category",
    "features": {
        "CustomFeature": "Description of feature",
        "CustomContour": "Contour description",
    },
    "has_contour": True,
    "requires_voicing": True/False,
}
Test with sample files to validate output.

The modular architecture ensures that new features integrate seamlessly with
the existing selection, filtering, and CSV output systems.

Hardware Considerations

MADE is CPU-bound and does not require GPU acceleration for typical usage.

Minimum Requirements:

4GB RAM (8GB recommended for large corpora)

Any modern CPU (Intel Core i5 or equivalent)

100MB disk space for code + output storage

Performance Notes:

Extraction speed depends on interval count and feature complexity

Contour extraction (especially formants and MFCC) is more computationally intensive

Pitch and intensity extraction are relatively fast

Large files (>1 hour) may take several minutes per file

The Rich progress bar adds minimal overhead

Optimization Tips:

Use interval filters to exclude empty or irrelevant intervals

Select only needed feature categories (avoid "all" for large corpora)

Process smaller batches of files by organizing into subdirectories

For extremely large corpora (>1000 files), consider splitting across machines

C-Modules: Unlike RePAST, MADE does not currently include C-accelerated
modules, as Praat operations via parselmouth are already optimized. Future
releases may include compiled extensions for spectral analysis.

Automatic Fallback: MADE degrades gracefully; if a specific feature extraction
fails (e.g., formants on unvoiced segment), the corresponding cell remains empty
and extraction continues. No single feature failure halts the entire process.

Citing this Work

If you use MADE in your research, please cite it as follows:

Taylor, S. P. (2026). MADE: Modular Acoustic Data Extractor (Version 3.0.0)
[Computer software]. https://github.com/taylosh/MADE

BibTeX entry:

bibtex
@software{Taylor_MADE_Modular_Acoustic_2026,
  author = {Taylor, S. P.},
  title = {MADE: Modular Acoustic Data Extractor},
  version = {3.0.0},
  year = {2026},
  url = {https://github.com/taylosh/MADE}
}
Reproducibility Note: When publishing research using MADE, include:

The version number of MADE used

A list of selected feature categories and specific features

The contour step percentages used

The interval filtering criteria applied

A sample of your output CSV (first 10 rows and column headers)

This information ensures that other researchers can precisely replicate
your acoustic feature extraction methodology.

Acknowledgements & Third-Party Citations

While the orchestration logic, interactive CLI framework, and modular
extractor architecture in this repository are original works, MADE functions
as a Python-based orchestration hub for the following foundational technologies:

Acoustic Analysis: Powered by Praat via the parselmouth Python bindings
(Boersma & Weenink, 2024) for all signal processing and feature extraction.

Audio/TextGrid I/O: Uses praatio (Gorman, 2024) for robust TextGrid parsing
and manipulation across formats.

Data Handling: Employs pandas (McKinney, 2010) for structured CSV output
and NumPy (Harris et al., 2020) for numerical operations.

Progress Display: Optional integration with Rich (Will McGugan, 2025) for
beautiful terminal progress bars and enhanced user experience.

For academic transparency and reproducibility, exact dependency versions are
documented in the requirements.txt file included with this release.

MADE stands on the shoulders of these excellent tools, and users are encouraged
to cite them alongside MADE in their research publications.

Privacy & Security Notice

MADE processes all audio and TextGrid data strictly locally on the researcher's
machine. No audio data, TextGrid content, or extracted features are ever
transmitted to external servers. There is no telemetry, usage tracking, or
automatic update checking.

The only external connectivity occurs when users manually install dependencies
via pip or conda. Once installed, MADE operates entirely offline.

This local-first architecture protects sensitive participant data and maintains
compliance with IRB, GDPR, and HIPAA protocols regarding the handling of
biometric voice data and personally identifiable information (PII) that may
be present in transcriptions.

Researchers should treat their TextGrid files and extracted CSV outputs as
sensitive data and apply appropriate access controls and encryption for
storage and transmission.
