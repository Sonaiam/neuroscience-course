import mne
import numpy as np
import os
from scipy.signal import stft
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# مسیرها
data_dir = r"C:\Users\Sons"
output_dir = r'C:\Users\Sons'
os.makedirs(output_dir, exist_ok=True)

# تنظیمات عمومی
TARGET_CHANNEL = 'FZ'
FS = 250
NPERSEG = 128
NOVERLAP = 64
NTRIALS = 15

# شناسه سوژه‌ها
healthy_ids = ['513', '568', '532', '529', '534']
depressed_ids = ['598', '558', '587', '594', '597']

def extract_subject_id(filename):
    return filename.split()[0]

# --- توابع بارگذاری اپوک‌ها ---

def load_reward_epochs(file_path):
    """دریافت اپوک‌های مربوط به پاداش"""
    epochs = mne.read_epochs_eeglab(file_path)
    selected = epochs[['keypad2/104', 'keypad1/104', 'keypad1/keypad1/104', 'keypad2/keypad2/104']]
    
    if TARGET_CHANNEL not in selected.ch_names:
        raise ValueError(f"{TARGET_CHANNEL} not in {file_path}")
    
    ch_idx = selected.ch_names.index(TARGET_CHANNEL)
    data = selected.get_data()[:, ch_idx, :]
    return data, selected.info['sfreq']

def load_punishment_epochs(file_path):
    """دریافت اپوک‌های مربوط به مجازات"""
    epochs = mne.read_epochs_eeglab(file_path)
    selected = epochs[['keypad2/94', 'keypad1/94', 'keypad1/keypad1/94', 'keypad2/keypad2/94']]
    
    if TARGET_CHANNEL not in selected.ch_names:
        raise ValueError(f"{TARGET_CHANNEL} not in {file_path}")
    
    ch_idx = selected.ch_names.index(TARGET_CHANNEL)
    data = selected.get_data()[:, ch_idx, :]
    return data, selected.info['sfreq']

# --- پردازش STFT ---

def compute_stft_tensor(data, sfreq):
    _, _, Zxx_sample = stft(np.zeros(NPERSEG), fs=sfreq, nperseg=NPERSEG, noverlap=NOVERLAP)
    n_freq, n_time = Zxx_sample.shape
    n_epochs = min(NTRIALS, data.shape[0])
    
    tensor = np.zeros((n_epochs, n_freq, n_time))
    for i in range(n_epochs):
        segment = data[i]
        if len(segment) < NPERSEG:
            segment = np.pad(segment, (0, NPERSEG - len(segment)))
        else:
            segment = segment[:NPERSEG]
        _, _, Zxx = stft(segment, fs=sfreq, nperseg=NPERSEG, noverlap=NOVERLAP)
        tensor[i] = np.abs(Zxx)**2
    return tensor

# --- ساخت تانسورها برای گروه‌ها ---

def process_group(subject_ids, event_type):
    tensors = []
    for file in os.listdir(data_dir):
        if not file.endswith('.set'):
            continue
        subject_id = extract_subject_id(file)
        if subject_id not in subject_ids:
            continue
        try:
            file_path = os.path.join(data_dir, file)
            if event_type == 'reward':
                data, sfreq = load_reward_epochs(file_path)
            else:
                data, sfreq = load_punishment_epochs(file_path)
            tensor = compute_stft_tensor(data, sfreq)
            if tensor is not None:
                tensors.append(tensor)
                print(f"✓ {subject_id} ({event_type})")
        except Exception as e:
            print(f"✗ خطا در {subject_id} ({event_type}): {e}")
    try:
        result = np.stack(tensors)
        print(f"✅ تانسور {event_type} برای {len(tensors)} نفر ساخته شد. سایز: {result.shape}")
        return result
    except Exception as e:
        print(f"⛔ خطا در ساخت نهایی تانسور {event_type}: {e}")
        return None

# --- ساخت داده‌ها ---

reward_healthy = process_group(healthy_ids, 'reward')
reward_depressed = process_group(depressed_ids, 'reward')
punish_healthy = process_group(healthy_ids, 'punishment')
punish_depressed = process_group(depressed_ids, 'punishment')

# --- ذخیره داده‌ها ---

def save_tensor(tensor, name):
    if tensor is not None:
        path = os.path.join(output_dir, name + '.npy')
        np.save(path, tensor)
        print(f"💾 ذخیره شد: {path} — شکل: {tensor.shape}")

save_tensor(reward_healthy, 'reward_healthy')
save_tensor(reward_depressed, 'reward_depressed')
save_tensor(punish_healthy, 'punish_healthy')
save_tensor(punish_depressed, 'punish_depressed')

# --- رسم نمودارها ---

def plot_tensor_diff(tensor_1, tensor_2, title1, title2, suptitle, filename):
    if tensor_1 is None or tensor_2 is None:
        print(f"⛔ داده‌ای برای {suptitle} وجود ندارد.")
        return
    
    avg_1 = tensor_1.mean(axis=(0,1))
    avg_2 = tensor_2.mean(axis=(0,1))
    diff = avg_2 - avg_1

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(avg_1, aspect='auto', cmap='jet')
    plt.title(title1)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(avg_2, aspect='auto', cmap='jet')
    plt.title(title2)
    plt.xlabel('Time')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(diff, aspect='auto', cmap='bwr')
    plt.title(f'Diff: {title2} - {title1}')
    plt.xlabel('Time')
    plt.colorbar()

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

plot_tensor_diff(
    reward_healthy, reward_depressed,
    'Reward - Healthy', 'Reward - Depressed',
    'Power Changes - Reward',
    'reward_comparison.png'
)

plot_tensor_diff(
    punish_healthy, punish_depressed,
    'Punishment - Healthy', 'Punishment - Depressed',
    'Power Changes - Punishment',
    'punishment_comparison.png'
)

print("\n🎉 پردازش کامل شد.")
