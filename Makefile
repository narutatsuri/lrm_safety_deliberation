# Reproduce the figures and tables of "the refusal valley".
#
# Each target runs a thin script under reproduce/ that loads the library + the
# downloaded intermediate artifacts and writes a figure/table into figures/.
# All targets below are CPU-only (minutes). The from-scratch GPU/training
# pipeline lives under pipeline/ and is documented in README.md.
#
#   make env        # install the core (CPU) environment
#   make artifacts  # download intermediate artifacts from the HuggingFace Hub
#   make all        # regenerate every figure and table
#   make help       # list individual targets

PY ?= python
REPRODUCE = $(PY) reproduce

.PHONY: help env artifacts all figures tables clean

help:
	@echo "Setup:"
	@echo "  make env                       install core (CPU) dependencies"
	@echo "  make artifacts                 download intermediate artifacts (HuggingFace)"
	@echo "Reproduce everything:"
	@echo "  make all | make figures | make tables"
	@echo "Figures:"
	@echo "  make fig-refusal-valley        fig:refusal_valley  (+ -supplementary)"
	@echo "  make fig-prefill-decision      fig:prefill_decision"
	@echo "  make fig-extended-thinking     fig:asr_orr_extended_thinking"
	@echo "  make fig-nothink-to-think      fig:no_think_to_think (+ -lenient)"
	@echo "  make fig-within-prefix         fig:within_prefix_variance"
	@echo "  make fig-thinking-causal       fig:thinking_causal (+ -bysplit)"
	@echo "  make fig-defense-pareto        fig:defense_pareto"
	@echo "  make fig-training-slopegraph   fig:training_slopegraph"
	@echo "Tables:"
	@echo "  make tab-base-asr-orr          tab:base_asr_orr"
	@echo "  make tab-defenses-asr-orr      tab:defenses_asr_orr"
	@echo "  make tab-auroc-ci              tab:auroc_ci"
	@echo "  make tab-iaa                   tab:iaa"
	@echo "  make tab-inference-suppression tab:inference_defense_suppression"

env:
	$(PY) -m pip install -e .

artifacts:
	$(PY) -m lrm_safety_deliberation.fetch_artifacts

# --- Figures ---------------------------------------------------------------
fig-refusal-valley:
	$(REPRODUCE)/fig_refusal_valley.py
	$(REPRODUCE)/fig_refusal_valley_supplementary.py
fig-prefill-decision:
	$(REPRODUCE)/fig_prefill_decision.py
fig-extended-thinking:
	$(REPRODUCE)/fig_extended_thinking.py
fig-nothink-to-think:
	$(REPRODUCE)/fig_nothink_to_think.py
	$(REPRODUCE)/fig_nothink_to_think_lenient.py
fig-within-prefix:
	$(REPRODUCE)/fig_within_prefix_variance.py
fig-thinking-causal:
	$(REPRODUCE)/fig_thinking_causal.py
fig-defense-pareto:
	$(REPRODUCE)/fig_defense_pareto.py
fig-training-slopegraph:
	$(REPRODUCE)/fig_training_slopegraph.py

# --- Tables ----------------------------------------------------------------
tab-base-asr-orr:
	$(REPRODUCE)/tab_base_asr_orr.py
tab-defenses-asr-orr:
	$(REPRODUCE)/tab_defenses_asr_orr.py
tab-auroc-ci:
	$(REPRODUCE)/tab_auroc_ci.py
tab-iaa:
	$(REPRODUCE)/tab_iaa.py
tab-inference-suppression:
	$(REPRODUCE)/tab_inference_defense_suppression.py

figures: fig-refusal-valley fig-prefill-decision fig-extended-thinking \
         fig-nothink-to-think fig-within-prefix fig-thinking-causal \
         fig-defense-pareto fig-training-slopegraph
tables: tab-base-asr-orr tab-defenses-asr-orr tab-auroc-ci tab-iaa tab-inference-suppression
all: figures tables

clean:
	rm -rf figures/*.pdf figures/*.png figures/*.tex
