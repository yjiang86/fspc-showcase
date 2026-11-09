# Full Self-Patch Clamp: Engineering Showcase

A compact source-review subset of a physical autopatcher integration project.
It connects typed camera observations, pipette positioning, electrophysiology,
pressure, cleaning and attempt records through explicit adapter interfaces.
This is a supporting engineering project, not evidence of an unattended
autonomous experiment or a completed scientific result.

**Lifecycle: development paused. Examples are offline and mock-only.**

## Problem And Architecture

Patching requires observations and actions from several independently timed
devices. The engineering problem is to keep controller state, observations,
action records and failure outcomes understandable when individual subsystems
are incomplete or unavailable.

```text
Camera owner / frame producer
            |
 shared-frame reader + observation validity
            |
 Baseline controller ---- typed attempt and action records
      |       |       |                    |
 manipulator ephys  pressure         offline review/export
      |
 cleaning interface / second-arm handoff
```

| Component | Included evidence | Limit |
| --- | --- | --- |
| Camera | Shared-frame reader/publisher, metadata checks and synthetic tests | No camera SDK, native preview binary or new camera measurement |
| Positioning | Manipulator interfaces and existing Sutter transport source | No device opened or motion performed for this delivery |
| Dual-arm workflow | Separate positioning and cleaning interfaces | Not a demonstrated, fully coordinated dual-arm autonomous experiment |
| DIC and electrode localization | Observation/label schemas, evaluation requirements and split/export tests | No images, detector weights or new localization accuracy claim |
| Controller | State transitions, observer hooks and terminal pressure release | Mock outcomes do not establish biological or rig success |

The original integration baseline is hybrid: motion and camera components have
source-level integration, while focus, pressure, cleaning and other stages retain
substitute, mock or human-handoff boundaries. This subset does not reproduce a
physical setup or independently re-verify earlier hardware observations.

## Inspect The Code

- `src/autopatcher/core/baseline.py`: controller and terminal failure behavior.
- `src/autopatcher/adapters/interfaces.py`: physical-system interfaces.
- `src/autopatcher/adapters/shared_memory_camera.py`: observation transport.
- `src/autopatcher/models/domain.py`: camera, action and attempt contracts.
- `src/autopatcher/vision_eval/`: offline labels, split/export and report contracts.
- `src/autopatcher/hardware_safety.py`: existing live-intent checks.
- `tests/`: selected original tests plus showcase-boundary checks.

## Run Without Hardware

Python 3.11 or later is required. The included examples and tests need only the
standard library and can run directly from the repository root:

```shell
python examples/mock_baseline.py
python examples/vision_contract.py
python -m unittest discover -s tests -v
```

The first example runs the existing controller with synthetic adapters and
reports the controller's internal outcome label, counts and final pressure.
Even an internal success label is a mock state-machine result, not a patch
experiment. The second prints required evaluation fields without supplying
invented scores. Actual check results are in [VALIDATION.md](VALIDATION.md).

Packaging metadata is included for a conventional local installation:

```shell
python -m pip install --no-deps --no-build-isolation .
```

Installation requires an existing compatible setuptools build backend. It does
not install vendor drivers. Runtime artifact locations are user-supplied through
`FSP_RUNTIME_ROOT`; no machine-specific deployment configuration is included.
The Sutter DLL default, `DRIVER_DIRECTORY/ftd2xx.dll`, is a documented
placeholder rather than an installation path; an actual user-owned driver
location would have to be supplied explicitly for separate hardware work.

## Scope, Contributions And Rights

This delivery is a new, independently rooted selection, not a mirror of the
research repository or its historical branches. See [SCOPE.json](SCOPE.json)
for its closed file set and exclusions, [CONTRIBUTIONS.md](CONTRIBUTIONS.md)
for the human/agent attribution limits, and
[RIGHTS_AND_ATTRIBUTION.md](RIGHTS_AND_ATTRIBUTION.md) for unresolved licensing.

## Still Unverified

- End-to-end unattended patching and biological outcomes.
- Coordinated real dual-arm operation and calibrated device timing.
- Image-level DIC generalization and electrode-localization accuracy.
- Real pressure, focus and cleaning operation for this delivery.
- Physical safety acceptance, complete source licensing and external sharing clearance.

No training, cluster workload, service restart or instrument operation is part
of these examples. Source inspection, installation, offline execution,
scientific acceptance and public visibility are separate states.
