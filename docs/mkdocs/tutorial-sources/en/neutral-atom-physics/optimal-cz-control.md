---
title: "Defining your own Rydberg CZ pulse"
description: "Author a phase-modulated global pulse for two Rydberg atoms with the public atom emulator, scan its two parameters, and read off the conditional phase of the resulting two-qubit gate."
figure_alts:
  - "Laser phase of the selected pulse next to the phases it imprints on the four collective two-atom states, with the pi value a CZ gate needs marked."
  - "Conditional phase of the two-atom gate versus pulse duration for several phase-modulation amplitudes, with the pi line that a CZ gate requires highlighted."
---

# Defining your own Rydberg CZ pulse

Two atoms driven by a single global laser acquire a conditional phase from the
Rydberg blockade: the doubly excited state is shifted by \(B=C_6/d^6\), so the
collective single-excitation state \(|W\rangle=(|gr\rangle+|rg\rangle)/\sqrt2\),
the antisymmetric state \(|A\rangle=(|gr\rangle-|rg\rangle)/\sqrt2\), and
\(|rr\rangle\) each accumulate a different phase. FatQat gives you the simulator;
the pulse is yours to define. This tutorial authors a phase-modulated global
pulse, scans its two parameters with the public two-level atom emulator, and
reads off the conditional phase that decides whether the pulse is a CZ gate.

```python
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

document = fq.emulator.load_model_document("atom2level.reference")
model = fq.emulator.Atom2LevelModel.from_document(document)
omega = 2 * np.pi * 5.0                       # global Rabi frequency (rad/us)
spacing = 2.21                                # atom spacing (um), B/Omega = 50
arrangement = fq.emulator.AtomArrangement.chain(num_sites=2, spacing=spacing)
backend = fq.emulator.Atom2LevelEmulator(model, arrangement=arrangement, method="unitary")
blockade = abs(document["parameters"]["c6"]) / spacing**6
print(f"blockade B = {blockade:.0f} rad/us, B/Omega = {blockade / omega:.0f}")
```

## Measure the gate a pulse produces

A pulse is a waveform on the drive channel: a constant amplitude and a laser
phase we are free to shape. The emulator returns the propagator of the full
two-atom space, ordered as \(|gg\rangle,|gr\rangle,|rg\rangle,|rr\rangle\).
Because the drive is global and symmetric, that propagator maps the collective
states to themselves up to phases, and the gate is a phase gate whose
conditional phase is \(\Phi_{rr}-\Phi_W-\Phi_A\). A CZ gate is reached when
this equals \(\pi\) up to single-qubit rotations.

```python
def run_pulse(amplitude, duration, samples=401):
    """Return the 4x4 gate of one phase-modulated global pulse."""
    times = np.linspace(0.0, duration, samples)
    phase = amplitude * np.sin(2 * np.pi * times / duration)
    waveform = fq.emulator.SampledWaveform(
        tuple(times), tuple(omega * np.exp(-1j * phase))
    )
    program = fq.Program(2)
    program.add(
        ops.PulseOperation(
            duration, (fq.emulator.PulseControl(model.control.drive(), waveform),)
        )
    )
    return backend.run(program).result().get_unitary()[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]


def collective_phases(gate):
    """Return (Phi_W, Phi_A, Phi_rr) of a symmetric two-atom propagator."""
    phi_w = np.angle(0.5 * (gate[1, 1] + gate[1, 2] + gate[2, 1] + gate[2, 2]))
    phi_a = np.angle(0.5 * (gate[1, 1] - gate[1, 2] - gate[2, 1] + gate[2, 2]))
    return float(phi_w), float(phi_a), float(np.angle(gate[3, 3]))


gate = run_pulse(amplitude=0.0, duration=0.3)
phi_w, phi_a, phi_rr = collective_phases(gate)
print("unmodulated pulse: Phi_W = %.3f, Phi_A = %.3f, Phi_rr = %.3f" % (phi_w, phi_a, phi_rr))
print("conditional phase: %.3f rad" % (phi_rr - phi_w - phi_a))
```

## Scan the two parameters

The pulse has two free numbers: how strongly the phase is modulated, and how
long the pulse lasts. One emulator call takes a fraction of a second, so we can
scan a small grid and look for the conditional phase a CZ gate needs.

```python
amplitudes = np.linspace(0.0, 3.0, 7)
durations = np.linspace(0.1, 0.5, 9)
conditional = np.zeros((amplitudes.size, durations.size))

for row, amplitude in enumerate(amplitudes):
    for column, duration in enumerate(durations):
        gate = run_pulse(float(amplitude), float(duration))
        phi_w, phi_a, phi_rr = collective_phases(gate)
        conditional[row, column] = phi_rr - phi_w - phi_a

distance = np.abs(np.mod(conditional - np.pi + np.pi, 2 * np.pi) - np.pi)
best = np.unravel_index(int(np.argmin(distance)), conditional.shape)
best_amplitude = float(amplitudes[best[0]])
best_duration = float(durations[best[1]])
print(f"closest to pi: amplitude {best_amplitude:.2f}, duration {best_duration:.2f} us")
print(f"conditional phase there: {conditional[best]:.3f} rad (pi = 3.142)")
```

## The selected pulse

```python
gate = run_pulse(best_amplitude, best_duration)
phi_w, phi_a, phi_rr = collective_phases(gate)
times = np.linspace(0.0, best_duration, 401)
phase = best_amplitude * np.sin(2 * np.pi * times / best_duration)

figure, (pulse_axis, phase_axis) = plt.subplots(1, 2, figsize=(9.5, 3.6))
pulse_axis.plot(times, phase)
pulse_axis.set_xlabel("time (us)")
pulse_axis.set_ylabel("laser phase (rad)")
pulse_axis.set_title(f"Selected pulse: a={best_amplitude:.2f}, T={best_duration:.2f} us")

labels = [r"$|gg\rangle$", r"$|W\rangle$", r"$|A\rangle$", r"$|rr\rangle$"]
values = [0.0, phi_w, phi_a, phi_rr - phi_w - phi_a]
phase_axis.bar(labels, values, color=["0.7", "tab:blue", "tab:green", "tab:red"])
phase_axis.axhline(np.pi, color="0.4", linestyle=":", label=r"$\pi$ (CZ)")
phase_axis.set_ylabel("accumulated phase (rad)")
phase_axis.set_title("Phases of the collective states")
phase_axis.legend()
figure.tight_layout()
plt.show()
```

## Where the CZ condition is met

```python
figure, axis = plt.subplots(figsize=(6.4, 4.0))
for row, amplitude in enumerate(amplitudes):
    axis.plot(durations, conditional[row], marker="o", markersize=3, label=f"a={amplitude:.1f}")
axis.axhline(np.pi, color="0.3", linestyle="--", label=r"$\pi$ (CZ)")
axis.set_xlabel("pulse duration (us)")
axis.set_ylabel(r"conditional phase $\Phi_{rr}-\Phi_W-\Phi_A$ (rad)")
axis.set_title("Scanning a two-parameter pulse family")
axis.legend(fontsize="small", ncols=2)
figure.tight_layout()
plt.show()
```

The scan shows how the conditional phase moves with the pulse: the duration sets
how much phase the blockade accumulates, and the phase modulation shifts the
collective sectors relative to each other. From here you can refine the grid,
add parameters to the waveform, or hand the same objective to an optimizer —
the workflow stays the same: define the pulse, run it with the emulator, read
the gate, and decide.