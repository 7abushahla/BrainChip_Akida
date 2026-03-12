# CNN2SNN toolkit

The Brainchip CNN2SNN toolkit provides a means to convert Convolutional Neural
Networks (CNN) that were trained using Deep Learning methods to a low-latency 
and low-power Spiking Neural Network (SNN) for use with the Akida Runtime.

## Local Modifications in This Copy

This repository contains a **slightly modified** copy of `cnn2snn` 2.19.1, with
extra helper functionality added on top of the original toolkit. The core
conversion and quantization behaviour is unchanged; the additions are primarily
for **diagnostics and Akida v1 constraint checking**.

### 1. `check_model_compatibility_all`

In `cnn2snn/converter.py` we added:

- `check_model_compatibility_all(model, device=None, input_dtype="uint8")`

Compared to the original `check_model_compatibility`, which stops on the **first**
error, `check_model_compatibility_all`:

- Runs all checks independently and accumulates **all issues** into a list.
- Tags each issue with a prefix to indicate where it was found:
  - `[PRELIMINARY]` – ONNX/v1 mismatch, device/version mismatch.
  - `[STRUCTURE]` – per-layer structural and ordering problems.
  - `[QUANTIZATION]` – QuantizeML quantization errors.
  - `[CONVERSION]` – CNN2SNN / QuantizeML → Akida conversion errors.
  - `[INPUT]` – static Akida v1 input-shape constraints.
  - `[DENSE]` – static Akida v1 Dense fan-in constraints.
  - `[MAPPING]` – hardware mapping problems on the target device.

It also pretty-prints a short report to stdout and returns the raw list of
issues for programmatic inspection.

### 2. Static Akida v1 Constraint Checks

To make compatibility checking line up with the Akida v1 hardware constraints
documented in this repo, we added:

- `_check_static_v1_constraints(model)` in `cnn2snn/converter.py`

This helper is called from `check_model_compatibility_all` and enforces:

- **Input dimensions** (for `channels_last` models):
  - Width ≤ 256 pixels
  - Height ≥ 5 pixels
  - Channels ∈ {1, 3}
- **Dense fan-in**:
  - Each `Dense` layer must have total input features ≤ 57,334

Violations are reported as `[INPUT] ...` or `[DENSE] ...` entries in the issues
list.

### 3. Extended Structural Diagnostics

In `cnn2snn/compatibility_checks.py` we added:

- `_collect_sequential_issues(model)` – an internal helper that walks a
  Sequential model and collects all structural issues without raising on the
  first error. This is used by `check_model_compatibility_all` to provide a
  more complete structural report.

The original public APIs (`convert`, `check_model_compatibility`, etc.) are
kept intact and behave as in the upstream 2.19.1 release. The new helpers are
intended for development and debugging of Akida v1-compatible models and do
not change the runtime conversion path.
