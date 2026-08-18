# Guardian OS V9.0 — Safety Intelligence Engine

V9.0 consolidates Guardian's validated intelligence layers into a single, read-only Safety Intelligence state.

Unified layers:
- Perception
- Behaviour / signal quality
- Learned fatigue model
- Personalisation / Passport Validation
- Explainable Decision Engine
- Decision / Near-Miss / Visual Evidence memory
- Predictive Guardian

V9.0 adds:
- one `safety_intelligence` state in the Intelligence API;
- explicit TRUSTED / GUARDED / LIMITED / STANDBY trust-chain state;
- standard reason-code records with layer, severity and blocking semantics;
- explicit layer dependencies;
- one central safety contract;
- a compact Intelligence overview with progressive disclosure;
- a human-readable Why? section;
- technical details on demand;
- a small deterministic `guardian-cag-context-v1` object for a later CAG/RAG copilot.

V9.0 does not retrain the fatigue model, change calibration, change Passport Validation, change Perception Confidence, change Decision Memory sampling, or alter Guardian alerts. It introduces no LLM dependency.
