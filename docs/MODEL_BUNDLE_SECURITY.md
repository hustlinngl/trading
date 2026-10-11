# Model bundle security and migration

## Trust boundary

A model bundle contains Python joblib artifacts. Joblib uses Python pickle-based
serialization, so loading a bundle from an untrusted party can execute arbitrary
code. Treat executable model files as trusted program inputs, not as passive data.

The supported serving path is InferenceBundle.load(): it checks symbol/timeframe
identity, semantic fingerprints, deployment provenance, artifact hashes, readiness
evidence and the presence of the required artifacts before deserializing them.
Do not bypass this gate to load an externally supplied bundle. Direct calls to
AdaptiveEngine.load_bundle() are for assets produced by a trusted local training
run only.

## Version 0.9.46 migration

This release changes the model target from a one-direction label to independent
position-success labels. It adds short_signal_model.joblib,
short_analog_memory.joblib and short_meta_policy.joblib. Model and deployment
semantic fingerprints have changed deliberately; older bundles must be retrained,
then pass the ordinary clean holdout, specialist and deployment readiness checks.

The short model's p_up field is the calibrated probability that a short position
is profitable under the short label/exit semantics. It is exposed as p_short at
the serving boundary. Never reinterpret the long head as a short head via 1-p_up.
Short publication remains disabled by default and still requires supported
derivative market type, configured borrow cost, funding/data safety and all
independent signal gates.

A successful code build or test run is not evidence of positive trading edge.
Do not lower publication gates merely to increase signal frequency.
