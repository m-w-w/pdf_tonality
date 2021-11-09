# pdf_tonality

A python package for determining the proportion of in-key notes, as used in [Weiss & Peretz (2021)](https://psyarxiv.com/xev3w/) to analyze sung improvisations. Other common methods (e.g., Krumhansl-Kessler algorithm implemented in the [MIDI Toolbox](https://github.com/miditoolbox/1.1/blob/master/miditoolbox/kkkey.m)) require notes to be rounded to the nearest semitone (12 semitones/levels), which distorts intervals, and assumes that the performer is in tune. The approach here uses a more continuous probability density function (PDF, 1200 cents/levels) representing the tonal hierarchy, and makes no assumptions about tuning except for octave equivalence. Hence, a tonic can be returned to the nearest cent.

![](https://github.com/m-w-w/pdf_tonality/blob/main/figures/KK_v_PDF.png)

The package includes several tiers of functions:

Basic:
- build_diatonic_PDF(): Build a probability density function evaluated at each cent representing diatonic intervals
- MIDI_to_transpose_array(): Convert MIDI pitch values to cents and transpose across the octave in steps of 1 cent, then change to pitch classes from C=0 (resolution 1 cent)
- nll_tonality(): Compute negative log likelihood of values given evaluated points PDF
- optimal_shift(): Calculate mean of negative log likelihood array, weighting pitches optionally
- min_idx_to_tonic(): Calculate the tonic from the transposition minimum index (0-1199), where 0=C.
- notes_in_key_major_minor(): Calculate proportion of in-key notes for major or minor profiles.
- generate_random_pitches(): Generate a sequence of random pitch values matching the length of another sequence.

Convenience:
- run_PDF(): Convenience function to build a model under some PDF distribution.
- PDF_major_minor(): A convenience function that calculates the tonic using the best fit from either the major or minor settings, wrapping run_PDF().
- PDF_null_notes(): A convenience function that calculates a null distribution of models using PDF_major_minor(), but generating random pitches using generate_random_pitches().
- z_score_null_notes(): Convert the proportion of notes that are tonal in the actual performance into a z-score relative to a null distribution of proportions.

Plotting and saving:
- plot_model_output(): Plot the results of run_PDF().
- save_model_info(): Save information generated by run_PDF().
- plot_PIKN_distribution(): Visualize proportion of in-key notes value in the null distribution.

# installation

A virtual environment can be created using Anaconda that installs all necessary dependencies:

```conda create -n test_pdft pip python=3.7```
```conda activate test_pdft```

Then cd or provide the full path to the setup.py file and pip install:

```pip install ~/pdf_tonality```

From within this virtual environment, you can now load the library, for example `from pdf_tonality import pdf_tonality as pdft`.

# Examples

Two examples are provided in the folder `pdf_tonality/examples`. To run them from within the virtual environment:

```python ~/pdf_tonality/examples/example1.py```

```python ~/pdf_tonality/examples/example2.py```
