# Energy measurement protocol

Reviewer concern: the benchmark lacked direct power/energy measurement.
`src/energy_measurement.py` + `src/run_energy_measurement.py` add it where
the platform supports it, and are explicit where it does not. **No energy
value is ever estimated, modeled, or fabricated.**

## Supported measurement backends

| Backend | Platform | What it measures | Status |
| --- | --- | --- | --- |
| `intel_rapl` | Linux with powercap sysfs (`/sys/class/powercap/intel-rapl:*`) | CPU package energy counters (µJ), summed across packages, wraparound-corrected | Primary |
| NVML (`pynvml`) | Any with NVIDIA driver | GPU total-energy counter | **Supplementary only** — the benchmark is CPU-only; reported for completeness if a GPU exists but never part of the main claims. `pynvml` is an optional dependency. |
| `none` (fallback) | Windows / macOS / RAPL not readable | Wall time, mean CPU utilization, RSS | Honest fallback |

## Fallback semantics (e.g. Windows)

When no direct counter is readable, every output row records:

```
direct_energy_available = false
cpu_energy_joules = NA
gpu_energy_joules = NA
energy_per_query_joules = NA
notes = Direct package-energy measurement unavailable on this platform; reporting timing and utilization only.
```

Downstream tables must treat these rows as *timing-only*; the claim-support
audit (`src/claim_support_audit.py`) caps energy claims accordingly.

## Protocol rules

1. Run on an otherwise idle machine; energy counters are system-wide, so
   background load contaminates RAPL readings.
2. On Linux, RAPL sysfs may need permissions:
   `sudo chmod -R a+r /sys/class/powercap/intel-rapl*` (or run with sudo).
3. Warmup queries (100) run **before** the measured window.
4. The workload is a fixed, seeded query stream (i2i: sampled item vectors;
   u2i: seeded mean-of-5-items history proxies — recorded in `notes`).
5. Report energy per query (`cpu_energy_joules / queries`) alongside wall
   time; never divide NA.
6. Pair every run with `capture_hardware.py` output from the same machine.

## Command

```bash
python src/run_energy_measurement.py --datasets ml-1m ml-20m goodbooks --queries 5000 --seed 42
```

Outputs: `results/energy/energy_measurement_all.csv`,
`results/paper_tables/energy_measurement_summary.{csv,md,tex}`.

## Interpretation limits

- RAPL covers the CPU package (and RAM on some SKUs), not wall power; no
  PSU/at-the-wall claim is supported.
- Cross-machine energy comparisons are invalid; compare methods only within
  one captured hardware profile.
- Windows results support *relative timing* statements only, never energy
  statements.
