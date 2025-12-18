import mne
import numpy as np
import matplotlib.pyplot as plt

# === فایل‌های EEG شرکت‌کننده‌ها ===
files = ['594 cleaned.set', '598 cleaned.set', '587 cleaned.set', '597 cleaned.set', '558 cleaned.set']

# === نوع رویدادها برای پاداش و مجازات ===
reward_ids = ['keypad2/104', 'keypad1/104', 'keypad1/keypad1/104', 'keypad2/keypad2/104']
punish_ids = ['keypad2/94', 'keypad1/94', 'keypad1/keypad1/94', 'keypad2/keypad2/94']

# === پارامترهای تبدیل زمان-فرکانس (TFR) ===
frequencies = np.linspace(4, 40, 30)
n_cycles = frequencies / 2.  # تعداد سیکل‌ها برای موجک مورله

# === لیست برای ذخیره قدرت ERSP ===
reward_powers = []
punish_powers = []

# === حلقه پردازش هر فایل ===
for file in files:
    print(f"Processing {file}")
    epochs = mne.read_epochs_eeglab(file, preload=True)

    # نرمال‌سازی z-score در زمان برای هر اپک
    data = epochs.get_data()
    zscored = (data - data.mean(axis=2, keepdims=True)) / data.std(axis=2, keepdims=True)
    epochs._data = zscored  # هشدار: تغییر مستقیم داده‌ها

    # استخراج اپک‌های مرتبط با پاداش و مجازات
    try:
        reward_epochs = epochs[reward_ids]
        punish_epochs = epochs[punish_ids]
    except Exception as e:
        print(f"⚠️ Skipping {file} due to missing events: {e}")
        continue

    # انتخاب ۱۵ اپک تصادفی (در صورت کافی بودن)
    rng = np.random.default_rng(seed=42)
    if len(reward_epochs) >= 15:
        reward_epochs = reward_epochs[rng.choice(len(reward_epochs), 15, replace=False)]
    if len(punish_epochs) >= 15:
        punish_epochs = punish_epochs[rng.choice(len(punish_epochs), 15, replace=False)]

    # محاسبه توان ERSP با موجک مورله بدون میانگین‌گیری
    reward_power = mne.time_frequency.tfr_morlet(
        reward_epochs, freqs=frequencies, n_cycles=n_cycles,
        use_fft=True, return_itc=False, average=False
    )
    punish_power = mne.time_frequency.tfr_morlet(
        punish_epochs, freqs=frequencies, n_cycles=n_cycles,
        use_fft=True, return_itc=False, average=False
    )

    # میانگین‌گیری درون‌فردی برای تبدیل به AverageTFR
    reward_power_avg = reward_power.average()
    punish_power_avg = punish_power.average()

    # نرمال‌سازی با پایه‌ی قبل از رویداد (۵۰۰- میلی‌ثانیه تا ۰)
    reward_power_avg.apply_baseline(baseline=(-0.5, 0), mode='logratio')
    punish_power_avg.apply_baseline(baseline=(-0.5, 0), mode='logratio')

    reward_powers.append(reward_power_avg)
    punish_powers.append(punish_power_avg)

# === میانگین گروهی (grand average) ===
reward_avg = mne.grand_average(reward_powers)
punish_avg = mne.grand_average(punish_powers)

# === استخراج داده کانال CZ برای نمایش ===
ch_index = reward_avg.info['ch_names'].index('CZ')
reward_cz = reward_avg.data[ch_index]
punish_cz = punish_avg.data[ch_index]

# === رسم نمودار ERSP در کانال CZ ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(reward_cz, aspect='auto', origin='lower',
           extent=[reward_avg.times[0], reward_avg.times[-1], frequencies[0], frequencies[-1]])
plt.title('Reward ERSP at CZ')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power Ratio')

plt.subplot(1, 2, 2)
plt.imshow(punish_cz, aspect='auto', origin='lower',
           extent=[punish_avg.times[0], punish_avg.times[-1], frequencies[0], frequencies[-1]])
plt.title('Punishment ERSP at CZ')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power Ratio')

plt.tight_layout()
plt.show()

