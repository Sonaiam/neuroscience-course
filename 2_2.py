import mne
import numpy as np
import matplotlib.pyplot as plt

def zscore(x):
    return (x - np.mean(x)) / np.std(x)
def SE(x):
    return (np.std(x)) / np.sqrt(x)

epochs = mne.io.read_epochs_eeglab('594 cleaned.set')

selected_channels = ['FZ', 'CZ', 'PZ']
# Note: The pick_channels() function is legacy. New code should use epochs.pick(...)
epochs_selected = epochs.copy().pick_channels(selected_channels)

epochs_zscored = epochs_selected.copy().apply_function(zscore)

evoked = epochs_zscored.average()
evoked.plot()

epochs=epochs_zscored

punishment_keys = [k for k in epochs.event_id if '/94' in k]
achievement_keys = [k for k in epochs.event_id if '/104' in k]

# Select channels of interest: FZ and CZ
epochs.pick(['FZ', 'CZ'])

# Extract epochs for each event type using the event labels
punishment_epochs = epochs[punishment_keys]
achievement_epochs = epochs[achievement_keys]

# Select the first five epochs (trials) for each event type
punishment_epochs_5 = punishment_epochs[:10]
achievement_epochs_5 = achievement_epochs[:10]

# For each event type, pick the channel FZ
punishment_FZ = punishment_epochs_5.copy().pick_channels(['FZ'])
achievement_FZ = achievement_epochs_5.copy().pick_channels(['FZ'])

punishment_CZ = punishment_epochs_5.copy().pick_channels(['CZ'])
achievement_CZ = achievement_epochs_5.copy().pick_channels(['CZ'])

# Compute the averaged evoked responses
evoked_punishment_FZ = punishment_FZ.average()
evoked_achievement_FZ = achievement_FZ.average()

evoked_punishment_CZ = punishment_CZ.average()
evoked_achievement_CZ = achievement_CZ.average()

# Create a dictionary to compare evoked responses
compare_dict = {
    'Punishment': evoked_punishment_FZ,
    'Achievement': evoked_achievement_FZ
}

# Plot the comparison for channel FZ
mne.viz.plot_compare_evokeds(compare_dict, picks=['FZ'], title='FZ: Punishment vs. Achievement DEPRESSED SUBJECT', show=True)

compare_dict = {
    'Punishment': evoked_punishment_CZ,
    'Achievement': evoked_achievement_CZ
}

mne.viz.plot_compare_evokeds(compare_dict, picks=['CZ'], title='CZ: Punishment vs. Achievement DEPRESSED SUBJECT', show=True)

