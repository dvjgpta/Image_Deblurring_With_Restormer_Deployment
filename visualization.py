# # plot_restormer_metrics.py
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator

# # ======== CONFIG ========
# log_dir = '/mnt/DATA/EE22B013/Btech_project/Model/experiments/Restormer_GOPRO/log/train_Restormer_GOPRO_20251017-174456.log'  # your TB log folder
# train_loss_tag = 'Train Loss'
# val_loss_tag   = 'Val Loss'
# psnr_tag       = 'PSNR'  # or 'train/PSNR' if you logged training PSNR
# # ========================

# # Load TensorBoard logs
# ea = event_accumulator.EventAccumulator(log_dir)
# ea.Reload()

# # ======== Extract Scalars ========
# print("Available scalar tags:", ea.Tags()['scalars'])

# # Train Loss
# if train_loss_tag in ea.Tags()['scalars']:
#     train_loss_events = ea.Scalars(train_loss_tag)
#     train_losses = [e.value for e in train_loss_events]
# else:
#     train_losses = None

# # Validation Loss
# if val_loss_tag in ea.Tags()['scalars']:
#     val_loss_events = ea.Scalars(val_loss_tag)
#     val_losses = [e.value for e in val_loss_events]
# else:
#     val_losses = None

# # PSNR
# if psnr_tag in ea.Tags()['scalars']:
#     psnr_events = ea.Scalars(psnr_tag)
#     psnr_vals = [e.value for e in psnr_events]
# else:
#     psnr_vals = None

# epochs = range(1, len(train_losses) + 1 if train_losses else len(psnr_vals) + 1)

# # ======== PLOT LOSS ========
# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# if train_losses:
#     plt.plot(epochs, train_losses, label='Train Loss', marker='o')
# if val_losses:
#     plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Epoch vs Loss")
# plt.grid(True)
# plt.legend()

# # ======== PLOT PSNR ========
# plt.subplot(1,2,2)
# if psnr_vals:
#     plt.plot(range(1, len(psnr_vals)+1), psnr_vals, label='PSNR', marker='s', color='green')
#     plt.xlabel("Epochs")
#     plt.ylabel("PSNR (dB)")
#     plt.title("Epoch vs PSNR")
#     plt.grid(True)
#     plt.legend()
# else:
#     plt.text(0.5, 0.5, 'No PSNR values logged', horizontalalignment='center', verticalalignment='center')

# plt.tight_layout()
# plt.show()


import re
import matplotlib.pyplot as plt
import os


log_file = '/mnt/DATA/EE22B013/Btech_project/Model/experiments/Restormer_GOPRO/log/train_Restormer_GOPRO_20251017-174456.log'  

# Lists to store values
train_losses = []
val_losses = []
psnr_vals = []
iterations = []

# Regex patterns
train_pattern = re.compile(r"Train Loss: ([\d\.]+)")
val_pattern   = re.compile(r"Val Loss: ([\d\.]+)")
psnr_pattern  = re.compile(r"PSNR: ([\d\.]+)")
iter_pattern  = re.compile(r"\[Iter (\d+)\]")

# Read log file
with open(log_file, 'r') as f:
    for line in f:
        iter_match = iter_pattern.search(line)
        train_match = train_pattern.search(line)
        val_match = val_pattern.search(line)
        psnr_match = psnr_pattern.search(line)

        if iter_match:
            iter_num = int(iter_match.group(1))
            iterations.append(iter_num)

        if train_match:
            train_losses.append(float(train_match.group(1)))
        if val_match:
            val_losses.append(float(val_match.group(1)))
        if psnr_match:
            psnr_vals.append(float(psnr_match.group(1)))

# Plot Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(iterations[:len(train_losses)], train_losses, label='Train Loss', marker='o')
plt.plot(iterations[:len(val_losses)], val_losses, label='Val Loss', marker='x')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.grid(True)
plt.legend()

# Plot PSNR
plt.subplot(1,2,2)
plt.plot(iterations[:len(psnr_vals)], psnr_vals, label='PSNR', marker='s', color='green')
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("PSNR over Iterations")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
