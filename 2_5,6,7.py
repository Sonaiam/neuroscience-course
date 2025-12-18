import mne
import numpy as np
import matplotlib.pyplot as plt

# === Define participant groups ===
cont_files = ['513_cleaned.set', '568 cleaned.set', '532 cleaned.set', '529 cleaned.set', '534 cleaned.set'] 

# === define function for zscore and se === #
def zscore(x):
    return (x - np.mean(x)) / np.std(x)
def SE(x,n):
    return (np.std(x)) / np.sqrt(n)

# === define function for recognition of reward === #
def load_a(files, label):
    evokeds = []
    min_len = None
    counts=[]
    for file in files:
        print(f'Loading {file}...')

        try:
            epochs = mne.read_epochs_eeglab(file)
            epochs.pick_channels(['FZ', 'CZ', 'PZ'])
            epochs = epochs.copy().apply_function(zscore)
            selected = epochs[['keypad2/104', 'keypad1/104', 'keypad1/keypad1/104', 'keypad2/keypad2/104']]
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue

        if len(selected) == 0:
            print(f"⚠️ No usable epochs in {file}")
            continue

        data = selected.get_data().mean(axis=0)
        counts.append(len(selected))

        if min_len is None or data.shape[1] < min_len:
            min_len = data.shape[1]

        evokeds.append(data)

    if len(evokeds) == 0:
        raise ValueError(f"❌ No usable data in {label}")

    # Trim all to shortest length
    evokeds = [e[:, :min_len] for e in evokeds]
    return np.stack(evokeds), selected.times[:min_len] , counts # return data and trimmed time vector


def load_p(files, label):
    evokeds = []
    min_len = None
    counts=[]
    for file in files:
        print(f'Loading {file}...')

        try:
            epochs = mne.read_epochs_eeglab(file)
            epochs.pick_channels(['FZ', 'CZ', 'PZ'])
            epochs = epochs.copy().apply_function(zscore)
            selected = epochs[['keypad2/94', 'keypad1/94', 'keypad1/keypad1/94', 'keypad2/keypad2/94']]
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue

        if len(selected) == 0:
            print(f"⚠️ No usable epochs in {file}")
            continue

        data = selected.get_data().mean(axis=0)
        counts.append(len(selected))
        if min_len is None or data.shape[1] < min_len:
            min_len = data.shape[1]

        evokeds.append(data)

    if len(evokeds) == 0:
        raise ValueError(f"❌ No usable data in {label}")

    # Trim all to shortest length
    evokeds = [e[:, :min_len] for e in evokeds]
    return np.stack(evokeds), selected.times[:min_len] , counts # return data and trimmed time vector


# === Load both groups ===

data2, times2, counts2 = load_a(cont_files, "control")
data4, times4, counts4 = load_p(cont_files, "control")



# === Compute grand averages ===
grand_avg2 = data2.mean(axis=0) # shape: (n_channels, n_times)
grand_avg4 = data4.mean(axis=0)

# standard error across subjects (axis=0)
se2 = data2.std(axis=0) / np.sqrt(len(data2))
se4 = data4.std(axis=0) / np.sqrt(len(data4))


# === Plot CZ channel from both groups ===
ch_names = ['FZ', 'CZ', 'PZ']
ch = ch_names.index("CZ")

plt.plot(times2, grand_avg2[ch, :], label='Reward', color='blue', linewidth=2)
plt.fill_between(times2,
                 grand_avg2[ch, :] - se2[ch, :],
                 grand_avg2[ch, :] + se2[ch, :],
                 color='blue', alpha=0.3)

plt.plot(times4, grand_avg4[ch, :], label='Punishment', color='red', linewidth=2)
plt.fill_between(times4,
                 grand_avg4[ch, :] - se4[ch, :],
                 grand_avg4[ch, :] + se4[ch, :],
                 color='red', alpha=0.3)

plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.title('ERP Comparison at CZ Control subjects')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


