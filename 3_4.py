
import mne
import numpy as np
import os
from scipy.signal import stft
import warnings
import matplotlib.pyplot as plt

# غیرفعال کردن هشدارها
warnings.filterwarnings('ignore')

# تنظیمات اصلی
data_dir = r"C:\Users\Sons"
output_dir = r'C:\Users\Sons'
os.makedirs(output_dir, exist_ok=True)

# پارامترهای پردازش
TARGET_CHANNEL = 'FZ'  # کانال هدف
FS = 250               # فرکانس نمونه‌برداری (بررسی شود)
NPERSEG = 128          # طول هر قطعه برای STFT (کاهش یافته)
NOVERLAP = 64          # همپوشانی برای STFT
NTRIALS = 15           # تعداد آزمون‌ها از هر سوژه

# لیست فایل‌ها
set_files = [f for f in os.listdir(data_dir) if f.endswith('.set')]

# دسته‌بندی سوژه‌ها
healthy_ids = ['513', '568', '532' ,'529', '534']                                # سوژه‌های سالم
depressed_ids = ['598', '558', '587', '594', '597']  # سوژه‌های افسرده

def extract_subject_id(filename):
    """استخراج شناسه سوژه از نام فایل"""
    return filename.split()[0]

def load_and_process(file_path):
    """بارگذاری و پردازش هر فایل EEG"""
    try:
      
        try:
            data = mne.io.read_epochs_eeglab(file_path)
            data_type = 'epochs'
        except:
            data = mne.io.read_raw_eeglab(file_path, preload=True)
            data_type = 'raw'
        
        # انتخاب کانال
        selected_channel = TARGET_CHANNEL if TARGET_CHANNEL in data.ch_names else data.ch_names[0]
        ch_idx = data.ch_names.index(selected_channel)
        
        # استخراج داده‌ها
        if data_type == 'epochs':
            ch_data = data.get_data()[:, ch_idx, :]
        else:
            ch_data = data.get_data()[ch_idx, :]
        
        return {
            'sfreq': data.info['sfreq'],
            'data': ch_data,
            'type': data_type,
            'channel': selected_channel
        }
    except Exception as e:
        print(f"خطا در پردازش {file_path}: {str(e)}")
        return None

# پردازش تمام فایل‌ها
all_data = {}
for file in set_files:
    subject_id = extract_subject_id(file)
    file_path = os.path.join(data_dir, file)
    processed = load_and_process(file_path)
    if processed is not None:
        all_data[subject_id] = processed
        print(f"پردازش {file} با موفقیت انجام شد. کانال: {processed['channel']}")

def create_power_tensor(subjects_list, data_dict):
    """ساخت تانسور توان طیفی"""
    if not subjects_list:
        return None
    
    # تعیین ابعاد خروجی STFT
    _, _, Zxx_sample = stft(np.zeros(NPERSEG), fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    n_freq, n_time = Zxx_sample.shape
    
    # ساخت تانسور خالی
    tensor = np.zeros((len(subjects_list), NTRIALS, n_freq, n_time))
    
    for i, subj in enumerate(subjects_list):
        subject_data = data_dict[subj]
        data = subject_data['data']
        sfreq = subject_data['sfreq']
        
        if subject_data['type'] == 'epochs':
            # انتخاب تصادفی trials
            n_available = data.shape[0]
            trials = np.random.choice(n_available, size=min(NTRIALS, n_available), replace=False)
            
            for j, trial_idx in enumerate(trials):
                segment = data[trial_idx]
                # تطبیق طول داده
                if len(segment) < NPERSEG:
                    segment = np.pad(segment, (0, NPERSEG - len(segment)))
                else:
                    segment = segment[:NPERSEG]
                
                _, _, Zxx = stft(segment, fs=sfreq, nperseg=NPERSEG, noverlap=NOVERLAP)
                tensor[i,j] = np.abs(Zxx)**2
        else:
            # تقسیم داده raw به بخش‌های مساوی
            segment_length = NPERSEG
            n_segments = len(data) // segment_length
            n_segments = min(n_segments, NTRIALS)
            
            for j in range(n_segments):
                start = j * segment_length
                segment = data[start: start + segment_length]
                _, _, Zxx = stft(segment, fs=sfreq, nperseg=NPERSEG, noverlap=NOVERLAP)
                tensor[i,j] = np.abs(Zxx)**2
                
    return tensor


# ساخت تانسورها
healthy_subjects = [subj for subj in healthy_ids if subj in all_data]
depressed_subjects = [subj for subj in depressed_ids if subj in all_data]

healthy_tensor = create_power_tensor(healthy_subjects, all_data)
depressed_tensor = create_power_tensor(depressed_subjects, all_data)

# ذخیره نتایج
if healthy_tensor is not None:
    np.save(os.path.join(output_dir, 'healthy_power.npy'), healthy_tensor)
    print(f"\nتانسور سالم ذخیره شد. ابعاد: {healthy_tensor.shape}")

if depressed_tensor is not None:
    np.save(os.path.join(output_dir, 'depressed_power.npy'), depressed_tensor)
    print(f"تانسور افسرده ذخیره شد. ابعاد: {depressed_tensor.shape}")

# نمایش نتایج
def plot_power_comparison():
    """نمایش گرافیکی مقایسه توان طیفی"""
    if healthy_tensor is not None and depressed_tensor is not None:
        plt.figure(figsize=(15,6))
        
        plt.subplot(131)
        plt.imshow(healthy_tensor.mean(axis=(0,1)), aspect='auto', cmap='jet')
        plt.title('control subject ')
        plt.xlabel('time')
        plt.ylabel('frequency')
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(depressed_tensor.mean(axis=(0,1)), aspect='auto', cmap='jet')
        plt.title('depressed subject')
        plt.xlabel('time')
        plt.colorbar()
        
        plt.subplot(133)
        diff = depressed_tensor.mean(axis=(0,1)) - healthy_tensor.mean(axis=(0,1))
        plt.imshow(diff, aspect='auto', cmap='coolwarm')
        plt.title('depressed mines control')
        plt.xlabel('time')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'power_comparison.png'))
        plt.show()

plot_power_comparison()
print("\nپردازش کامل شد!")


