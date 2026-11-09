# Validation Status

Local offline checks were run on this exact selected source with Python 3.12.
Original project test totals are not imported as results for this subset.

| Command | Observed result |
| --- | --- |
| `python -m unittest discover -s tests -v` | PASS: 36 tests, including selected original controller, shared-frame, runtime-path and vision-contract/export tests |
| `python examples/mock_baseline.py` | PASS: 5 actions, 12 snapshots, terminal mock pressure 0.0 mbar |
| `python examples/vision_contract.py` | PASS: prints three existing task contracts without invented benchmark scores |
| Package installation | NOT RUN: the isolated verification interpreter has no setuptools build backend; no dependency download was performed |

The mock controller reports its existing `whole_cell_success` internal label.
This is not a biological success: the output explicitly reports
`physical_patch_success=NOT_TESTED` and `clock_calibration=NOT_TESTED`.

Internal verification retained exact command outcomes and checked that the
original source branch, index and dirty-file identities were unchanged.
Those internal paths and logs are intentionally not included here.

## Not Established

- Live camera acquisition, serial motion, dual-arm coordination or device timing.
- Real pressure, focus, cleaning or electrophysiology behavior for this delivery.
- DIC model accuracy, real pipette localization or closed-loop biological yield.
- Package installation or public-release permission.

Remote synchronization and visibility are recorded separately by the delivery
owner; the offline checks above do not certify either state.

The fixed Sutter installation default was replaced only in this subset with
the documented relative driver placeholder. Driver loading and physical
behavior after that configuration change were not exercised.
