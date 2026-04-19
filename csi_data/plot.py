import numpy as np
import matplotlib.pyplot as plt


# ----------- PARSE ONE LINE -----------
def parse_line(line):
    try:
        parts = line.strip().split(',')

        # Extract timestamp (last value)
        timestamp = float(parts[-1])

        # Extract RSSI (4th column)
        rssi = int(parts[3])

        # Extract CSI data inside [ ... ]
        csi_start = line.find('[')
        csi_end = line.find(']')

        if csi_start == -1 or csi_end == -1:
            return None

        csi_str = line[csi_start+1:csi_end]
        csi = np.array([int(x) for x in csi_str.split()])

        return timestamp, rssi, csi

    except:
        return None


# ----------- CSI → AMPLITUDE -----------
def get_amplitude(csi):
    I = csi[0::2]
    Q = csi[1::2]
    amp = np.sqrt(I**2 + Q**2)
    return amp


# ----------- LOAD DATA -----------
timestamps = []
mean_amplitudes = []
subcarrier_amplitudes = []

subcarrier_index = 20  # you can change this

with open("csi.csv", "r") as f:
    for line in f:
        parsed = parse_line(line)
        if parsed is None:
            continue

        t, rssi, csi = parsed

        # Convert to amplitude
        amp = get_amplitude(csi)

        # Store mean amplitude (overall signal behavior)
        mean_amp = np.mean(amp)

        # Store single subcarrier (more sensitive to movement)
        if len(amp) > subcarrier_index:
            sub_amp = amp[subcarrier_index]
        else:
            sub_amp = 0

        timestamps.append(t)
        mean_amplitudes.append(mean_amp)
        subcarrier_amplitudes.append(sub_amp)
        break


# ----------- TIME NORMALIZATION -----------
t0 = timestamps[0]
time_axis = [t - t0 for t in timestamps]


# # ----------- PLOT 1: MEAN AMPLITUDE -----------
# plt.figure()
# plt.plot(time_axis, mean_amplitudes)
# plt.xlabel("Time (seconds)")
# plt.ylabel("Mean CSI Amplitude")
# plt.title("CSI Mean Amplitude Over Time")
# plt.grid()


# # ----------- PLOT 2: SINGLE SUBCARRIER -----------
# plt.figure()
# plt.plot(time_axis, subcarrier_amplitudes)
# plt.xlabel("Time (seconds)")
# plt.ylabel(f"Subcarrier {subcarrier_index} Amplitude")
# plt.title("CSI Subcarrier Variation (Movement Detection)")
# plt.grid()


# plt.show()