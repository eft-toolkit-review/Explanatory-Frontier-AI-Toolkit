# 03_measure: construct measurement (Step 2a)

This folder is the measurement step (Step 2a): 
turning the language, voice and face features of a clip into one estimate of perceived trustworthiness( or other alternative constructs like EI, confidence, charisma...

Purpose. Estimate the researcher-defined construct from behavior, out of sample, and show which behavioral information carries the estimate.

## Files
labels.py/: corresponding to Technical Appendix A2.1 Criterion labels (illustrative code). 

splits.py/: corresponding to Technical Appendix A2.2 Splits (illustrative code) with two evaluation modes (Mode S as single split and Mode K as K fold split)

gmfn_pilot_code/ ：contains A2.3 model, A2.4 benchmark models(Pilot pipeline code). It is the GMFN code used in the pilot, together with the benchmark models it was compared with (early-fusion LSTM, TFN, LMF, MFN). It evaluates trained models on a feature file. See the README inside for the list of models and the pilot results. 

gate.py/：corresponding to Technical Appendix A2.4 Validity gate (illustrative code). an illustrative decision rule as an example.

## Installation
Before running the scripts, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Scipy
- Numpy
- Pandas
- Statsmodels
- Scikit-learn
