import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def log_reader(file_path):
    # Read the file content
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize list for storing data
    data = []

    # Iterate through the lines to extract relevant information
    for i, line in enumerate(lines):
        if line.startswith("Iter: ") or line.startswith("Warm up iter:"):  # Iter: 202050/300000
            # remove /n
            line = line.replace("\n", "")
            iter_num, total_iter = line.split(' ')[-1].split('/')
            iter_num = int(iter_num)
            total_iter = int(total_iter)
            if line.startswith("Warm up iter:"):
                warm_up_total_iter = total_iter
                warm_up = True
            elif line.startswith("Iter: "):
                iter_num += warm_up_total_iter
                warm_up = False

        elif line.startswith("Loss:  "):  # 'Loss:  motion 0.8200070858001709, commit 0.04169795662164688, vel 0.3412715792655945, ergo 24.689285278320312, ave_REBA 1.1957859992980957\n'
            elements = re.split(' |, ', line)
            elements = [e for e in elements if e]
            motion = float(elements[2])
            commit = float(elements[4])
            vel = float(elements[6])
            ergo = float(elements[8])
            ave_reba = float(elements[10])
            try:
                max_reba = float(elements[12])
            except:
                max_reba = 0


        elif line.startswith("Total loss"):  #'Total loss 1.2383697032928467, loss before 0.9914768934249878, ergo% 0.19936925172805786\n'
            elements = re.split(' |, ', line)
            elements = [e for e in elements if e]
            total_loss = float(elements[2])
            loss_before = float(elements[5])
            ergo_percentage = float(elements[7])
            try:
                ergo_weight = float(elements[10])
            except:
                ergo_weight = 0

        elif line.startswith("###########"):
            row = [iter_num, total_iter, motion, commit, vel, ergo, ave_reba, total_loss, loss_before, ergo_percentage, max_reba, ergo_weight]
            data.append(row)
    header = ["iter_num", "total_iter", "motion", "commit", "vel", "ergo", "ave_reba", "total_loss", "loss_before", "ergo_percentage"]

    data = np.array(data)
    print(f"{file_path} finished")
    return header, data


def log_FID_reader(file_path):
    # Read the file content
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize list for storing data
    data = []

    # Iterate through the lines to extract relevant information
    for i, line in enumerate(lines):
        if "INFO --> 	 Eva. Iter " in line:  # INFO --> 	 Eva. Iter 300000, FID:  0.8200070858001709
            match = re.search(r"Eva\. Iter (\d+) :.*, FID\. ([\d.]+)", line)
            iter_num = int(match.group(1))
            fid = float(match.group(2))
            data.append([iter_num, fid])
    header = ["iter_num", "fid"]

    data = np.array(data)
    print(f"{file_path} finished")
    return header, data

# Define the file path

#### 2025-02-04 experiment
# dir_path = "output_slurm"
# ergo_file = "eval_log_3.txt"
# og_file = "eval_log_3_no_ergo.txt"

# #### 2025-02-11 experiment
# num = 71
# ergo_file = f"/Users/leyangwen/Downloads/VQVAE_tests/VQVAE_{num}/eval_log.txt"
dir_path = "output_slurm"
og_file = "eval_log_3_no_ergo.txt"


ergo_file = "/Users/leyangwen/Downloads/VQVAE_R3_3/VQVAE_R3_8/eval_log.txt"

# 1e-9 too lttle
# 1e-8 going down slowly
# 1e-7 going down nicely


# file_path = os.path.join(dir_path, ergo_file)
header, ergo_results = log_reader(os.path.join(dir_path, ergo_file))
_, og_results = log_reader(os.path.join(dir_path, og_file))

min_epoch = min(ergo_results.shape[0], og_results.shape[0])
ergo_results = ergo_results[:min_epoch]
og_results = og_results[:min_epoch]

header_FID, ergo_FID = log_FID_reader(os.path.join(dir_path, ergo_file))
_, og_FID = log_FID_reader(os.path.join(dir_path, og_file))
min_epoch = min(ergo_FID.shape[0], og_FID.shape[0])
ergo_FID = ergo_FID[:min_epoch]
og_FID = og_FID[:min_epoch]


# Plot the results


## ave_reba over iter
plt.figure()
plt.plot(ergo_results[:, 0], ergo_results[:, 6], label="w. Ergo Loss")
plt.plot(og_results[:, 0], og_results[:, 6], label="Unmodified")
plt.xlabel("Iteration")
plt.ylabel("Motion REBA Score")
plt.legend()
plt.title("Average REBA over Iteration")
plt.show()

#
# ## recon loss over iter
plt.figure()
plt.plot(ergo_results[:, 0], ergo_results[:, 7], label="w. Ergo Loss")
plt.plot(og_results[:, 0], og_results[:, 7], label="Unmodified")
plt.xlabel("Iteration")
plt.ylabel("Reconstruction Loss")
plt.legend()
plt.ylim([0, 2])
plt.title("Reconstruction Loss over Iteration")
plt.show()
#
# ## total loss over iter
# plt.figure()
# plt.plot(ergo_results[:, 0], ergo_results[:, 7], label="w. Ergo Loss")
# plt.plot(og_results[:, 0], og_results[:, 7], label="Unmodified")
# plt.xlabel("Iteration")
# plt.ylabel("Total Loss")
# plt.legend()
# plt.ylim([0, 2])
# plt.title("Total Loss over Iteration")
# plt.show()
#
# ## commit loss over iter
# plt.figure()
# plt.plot(ergo_results[:, 0], ergo_results[:, 3], label="w. Ergo Loss")
# plt.plot(og_results[:, 0], og_results[:, 3], label="Unmodified")
# plt.xlabel("Iteration")
# plt.ylabel("Commit Loss")
# plt.legend()
# plt.title("Commit Loss over Iteration")
# plt.show()
#
#
# ## vel loss over iter
# plt.figure()
# plt.plot(ergo_results[:, 0], ergo_results[:, 4], label="w. Ergo Loss")
# plt.plot(og_results[:, 0], og_results[:, 4], label="Unmodified")
# plt.xlabel("Iteration")
# plt.ylabel("Velocity Loss")
# plt.legend()
# plt.ylim([0, 0.75])
# plt.title("Velocity Loss over Iteration")
# plt.show()
#
## ergo loss percentage over iter
plt.figure()
plt.plot(ergo_results[:, 0], ergo_results[:, 9], label="w. Ergo Loss")
plt.plot(og_results[:, 0], og_results[:, 9], label="Unmodified")
plt.xlabel("Iteration")
plt.ylabel("Ergo Loss Percentage")
plt.legend()
plt.ylim([0, 1])
plt.title("Ergo Loss Percentage over Iteration")
plt.show()


## FID over iter
plt.figure()
plt.plot(ergo_FID[:, 0], ergo_FID[:, 1], label="w. Ergo Loss")
plt.plot(og_FID[:, 0], og_FID[:, 1], label="Unmodified")
plt.xlabel("Iteration")
plt.ylabel("FID")
plt.legend()
# plt.ylim([0, 3])
plt.title("FID over Iteration")
plt.show()

# max_reba
plt.figure()
plt.plot(ergo_results[:, 0], ergo_results[:, 10], label="w. Ergo Loss")
plt.xlabel("Iteration")
plt.ylabel("Max REBA Score")
plt.legend()
plt.title("Max REBA over Iteration")
plt.show()

# ergo_weight
plt.figure()
plt.plot(ergo_results[:, 0], ergo_results[:, 11], label="w. Ergo Loss")
plt.xlabel("Iteration")
plt.ylabel("Ergo Weight")
plt.legend()
plt.title("Ergo Weight over Iteration")
plt.show()
