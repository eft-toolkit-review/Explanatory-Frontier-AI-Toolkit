# 01_ingest: corpus collection and quality control (Step 1, A1.1)

**Purpose.** Turn public videos of executives into a set of clips that have the same length and the
same checks applied, before any construct-specific analysis.

This folder documents the collection protocol.

## Protocol (Technical Appendix A1.1)

- **Unit of analysis.** One clip of 135 seconds per listed-company CEO. Because each CEO contributes
  one clip, every split by clip is also a split by CEO.
  
- **Sources.** as our pilot study, we collected 1,276 candidate videos from YouTube, CNBC, Fox Business, The Wall Street Journal and
  Yahoo Finance.
  
- **Six-stage pipeline.** (1) separate audio from video (MoviePy); (2) verify face presence
  (OpenFace 3.0); (3) language check (BERT); (4) generate and validate transcripts; (5) confirm a
  single active speaker (active speaker detection); (6) standardize the duration.
  
- **Exclusions.** Strict selection:  videos shorter than 135 seconds are excluded (some researchers might choose zero-padded),
- so that clip length is the same across the sample. Where a CEO had several videos, the first qualifying one was
  kept.
  
- **Audit.** Three trained coders audited a random 20% of the final sample (203 clips) for visual
  clarity, audio quality and transcript accuracy, and found no material concerns.
  
- **Result.** 1,016 clips for pilot study.


