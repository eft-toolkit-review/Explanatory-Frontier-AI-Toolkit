# Explanatory Frontier AI Toolkit

**Making the Explanatory Frontier of Upper-Echelons Theory Observable: An AI-Enabled Architecture for Measurement, Causal Discovery, and Theory Expansion**

> **Development-stage research artifact (Stage 1).
> This repository documents a proposed scientific workflow and its first prototype
> the measurement step for CEO trustworthiness,
> together with the protocol machinery that fixes every analytic choice before it is used.
> Modules are at different stages, and each is labelled below.
> Note: A design description is **not** evidence that a module has been implemented or validated.

How to read the status labels used in this repository:

| Label | Meaning |
|---|---|
| **Verified here** | Checked by tests or a run included in this repository (on synthetic or simulated data) |
| **Provided, not yet run here** | Code is included but has not been executed in this repository's test setup |
| **Reported, not reproducible here** | A result stated in the appendix whose data are not in this repository |
| **Planned** | Designed; no code in this release |

## 1. The scientific question

Latent constructs, such as perceived trustworthiness, charisma, confidence and authenticity, 
which are hard to observe repeatedly in natural settings. Video, audio and transcripts. We ask: 
What natural behavior does the construct actually capture? 
Which parts of that behavioral footprint are causally consequential? 
Where does the construct’s explanatory reach hold, weaken, or fail? 
And what stable, outcome-relevant behavioral structure remains outside it? 

We call the boundary revealed by these questions the explanatory frontier of theory.

The **explanatory frontier** is the empirically inspectable boundary of a construct's current behavioral
and causal explanatory reach. 

## 2. Architecture

Testbed and Stage of Development 
       We demonstrate the architecture on CEOs, an unusually demanding and theoretically consequential testbed. Upper-echelons theory rests on latent executive characteristics that are difficult to manipulate and rarely observed repeatedly at scale (Hambrick & Mason, 1984). We have already assembled the empirical infrastructure: 1,276 videos of listed-company CEOs from major public and business-media platforms, processed through a six-stage quality-control pipeline. After verifying face visibility, audio and transcript quality, single-speaker status, and standardized duration, the final corpus contains 1,016 CEO videos. Three trained coders independently audited a random 20% and identified no material quality concerns.

**Architecture flow:**

Step 1: Turn Naturalistic Behavior Traces into Shared Behavioral Substrate 

Step 2: Theory-Driven Measurement and Residualized Profiling

Step 2a: Theory-Driven Measurement

Step 2b: Construct the Theory's Negative Space Before Seeing Causal Outcomes

Step 3: Use Generative AI to Create Naturalistic Counterfactuals

Step 4: Force Computational Explanations to Survive Causal Intervention

Step 5: Diagnose Where the Construct's Explanatory Reach Ends


## 3. Implementation status
03_measure: GMFN pilot code and protocol utilities




## 4. Licence and third-party material

The code in this repository is released under the MIT licence (`LICENSE`). Where parts of it come from
other public projects, and which tools are deliberately not included, is explained in `NOTICE.md`.
Licences of third-party code, model weights and datasets must be checked separately; no single licence
is asserted for the complete dependency stack.
