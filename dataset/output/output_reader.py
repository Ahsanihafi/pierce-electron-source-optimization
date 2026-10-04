import re
import numpy as np


def read_current(filename):
    """
    Read CST current output.

    For each sample:
      - first column = gun iteration
      - second column = current
    Returns the value from the latest iteration.
    """

    sample_ids = []
    final_iterations = []
    currents = []

    current_sample_id = None
    last_iteration = None
    last_value = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # New sample block
            if line.startswith("#Parameters"):
                if last_value is not None:
                    sample_ids.append(current_sample_id)
                    final_iterations.append(last_iteration)
                    currents.append(last_value)

                current_sample_id = None
                last_iteration = None
                last_value = None
                continue

            # Get sample number from E-SOURCE (N)
            if "E-SOURCE" in line:
                match = re.search(r"E-SOURCE\s*\((\d+)\)", line)
                if match:
                    current_sample_id = int(match.group(1))
                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) >= 2:
                try:
                    last_iteration = int(float(parts[0]))
                    last_value = float(parts[1])
                except ValueError:
                    pass

    # Save final sample
    if last_value is not None:
        sample_ids.append(current_sample_id)
        final_iterations.append(last_iteration)
        currents.append(last_value)

    return (
        np.array(sample_ids, dtype=int),
        np.array(final_iterations, dtype=int),
        np.array(currents, dtype=float)
    )


def read_last_monitor(filename):
    """
    Read CST beam-monitor output such as emittance or envelope.

    For each sample:
      - first column = beam monitor position
      - second column = quantity
    Returns the value at the LAST beam monitor.
    """

    sample_ids = []
    monitor_positions = []
    values = []

    current_sample_id = None
    last_position = None
    last_value = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # New sample block
            if line.startswith("#Parameters"):
                if last_value is not None:
                    sample_ids.append(current_sample_id)
                    monitor_positions.append(last_position)
                    values.append(last_value)

                current_sample_id = None
                last_position = None
                last_value = None
                continue

            # Extract sample number, e.g.
            # Emittance_x (123)
            if "Real Part" in line:
                match = re.search(r"\((\d+)\)", line)
                if match:
                    current_sample_id = int(match.group(1))
                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) >= 2:
                try:
                    last_position = float(parts[0])
                    last_value = float(parts[1])
                except ValueError:
                    pass

    # Save final sample
    if last_value is not None:
        sample_ids.append(current_sample_id)
        monitor_positions.append(last_position)
        values.append(last_value)

    return (
        np.array(sample_ids, dtype=int),
        np.array(monitor_positions, dtype=float),
        np.array(values, dtype=float)
    )


# ============================================================
# FILES
# ============================================================

current_file = "current200.txt"
emittance_file = "emittance200.txt"
envelope_file = "envelope200.txt"


# ============================================================
# READ DATA
# ============================================================

current_ids, final_iterations, current = read_current(current_file)

emittance_ids, emittance_positions, emittance = read_last_monitor(
    emittance_file
)

envelope_ids, envelope_positions, envelope = read_last_monitor(
    envelope_file
)

expected_ids = np.arange(1, 201)

missing_current = np.setdiff1d(expected_ids, current_ids)
missing_emittance = np.setdiff1d(expected_ids, emittance_ids)
missing_envelope = np.setdiff1d(expected_ids, envelope_ids)

print("Missing current IDs:", missing_current)
print("Missing emittance IDs:", missing_emittance)
print("Missing envelope IDs:", missing_envelope)
# ============================================================
# SANITY CHECK
# ============================================================

if not np.array_equal(current_ids, emittance_ids):
    raise ValueError(
        "Current and emittance sample IDs do not match!"
    )

if not np.array_equal(current_ids, envelope_ids):
    raise ValueError(
        "Current and envelope sample IDs do not match!"
    )


# ============================================================
# COMBINE OUTPUTS
# ============================================================

# Each row:
# [current, emittance, envelope]

Y = np.column_stack([
    current,
    emittance,
    envelope
])


# ============================================================
# PRINT SUMMARY
# ============================================================

print("Number of samples:", len(current_ids))

print("\nFirst few samples:")
print("ID | Iter | Current | Emittance | Envelope")

for i in range(min(10, len(current_ids))):
    print(
        f"{current_ids[i]:3d} | "
        f"{final_iterations[i]:4d} | "
        f"{current[i]: .8e} | "
        f"{emittance[i]: .8e} | "
        f"{envelope[i]: .8e}"
    )

print("\nCombined output array Y:")
print(Y)

print("\nY shape:", Y.shape)

np.savetxt("output_electron.csv", Y, delimiter=',')