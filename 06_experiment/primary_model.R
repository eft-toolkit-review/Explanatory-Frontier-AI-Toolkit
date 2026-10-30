# A5.3 Primary model (E16), illustrative.
#
#   Y_pj = b0 + b1 * D_pj + u_p + w_j + v_j * D_pj + e_pj
#
# Y: rating of trustworthiness (1-7). D: 0 = low version, 1 = high version. u_p: participant, w_j: clip,
# v_j: clip-specific treatment effect. Estimation by REML with Satterthwaite degrees of freedom
# (lme4 + lmerTest). Test of b1 is two-sided at alpha = .05.
#
# If the model is singular or does not converge, terms are dropped in a fixed order: first the
# random-effect correlation, then the slope variance. Never on the basis of the test of b1.

library(lme4)
library(lmerTest)

random <- c(
  maximal  = "(1 | participant) + (1 + D | clip)",
  no_corr  = "(1 | participant) + (1 | clip) + (0 + D | clip)",
  no_slope = "(1 | participant) + (1 | clip)"
)

# Fit "Y ~ <fixed> + <random structure>", moving down the list only when the model is singular or warns.
# Also used by the reopening model (07_reopen/reopen_model.R).
fit_mixed <- function(fixed, d) {
  for (structure in names(random)) {
    problem <- FALSE
    fit <- withCallingHandlers(
      lmer(as.formula(paste("Y ~", fixed, "+", random[[structure]])), data = d, REML = TRUE),
      warning = function(w) { problem <<- TRUE; invokeRestart("muffleWarning") }
    )
    if (!problem && !isSingular(fit)) break
  }
  list(structure = structure, fit = fit)
}

fit_primary <- function(d, outcome = "Y") {
  # d: one row per rating, with columns participant, clip, D and the outcome
  d$Y <- d[[outcome]]
  r <- fit_mixed("D", d)
  list(structure = r$structure,                              # which random-effects structure was kept
       b1 = summary(r$fit)$coefficients["D", ],              # estimate, SE, df (Satterthwaite), t, p
       fit = r$fit)
}

# The same model is fitted to each rival rating (authenticity, competence, warmth, charisma,
# confidence) for the specificity analysis:
#   lapply(c("authenticity", "competence", "warmth", "charisma", "confidence"),
#          function(r) fit_primary(d, outcome = r))
