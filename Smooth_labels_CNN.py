import torch
import h5py 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from scipy.signal import savgol_filter
import os
import matplotlib as mpl

# Set the font settings for plotting
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Set the figure size based on the textwidth
fig_width = 404.02908/72.27
fig_height = fig_width * (mpl.rcParams['figure.figsize'][1] / mpl.rcParams['figure.figsize'][0])

# Update the default figure size
mpl.rcParams['figure.figsize'] = (fig_width, fig_height)

def interpolate_nan(array_like):
    """
    function to interpolate nan values in the predicted values
    """
    array = array_like.copy()

    nans = np.isnan(array)

    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return torch.tensor(array)

if __name__ == "__main__":

    filepath = "path/to/raw_data"
    values_pth = "path/to/predicted image wise labels" 

    window_size_savgol_filter = 240//12-1     # window size of the savgol filter
    window_size = 60//12                   # window size for the running window
    poly_order = 1                        # order of the polynomial for the savgol filter
    all_length = []

    threshhold = 0.48560428619384766             # from KMeans lower limit of 2nd class

    to_inches = 0.393701
    size = 10*4*to_inches

    fig, axes = plt.subplots(4,4,figsize=(size,size*9/16))
    ax = axes.flatten()

    with h5py.File(filepath, 'r') as file:
        group_names = list(file.keys())  
    for i in tqdm(len(group_names)):
        with h5py.File(filepath, 'r') as file:
            group = file[group_names[i]]
            harp_number = group_names[i]
            continuum = torch.tensor(np.array(group['continuum']))
            shape = (continuum.shape[1], continuum.shape[2])

        continuum = torch.nan_to_num(continuum, nan=0.0, posinf=0.0, neginf=0.0)

        value_harp = torch.load(os.path.join(values_pth, f'{harp_number}.pt'))

        # since there are nan values in list, I will extrapolate these values
        value_harp = interpolate_nan(value_harp.numpy())
        
        try:
            smoothed_values = savgol_filter(value_harp, window_size_savgol_filter, poly_order)
        except:
            smoothed_values = savgol_filter(value_harp, window_size_savgol_filter//2, poly_order)

        labels = [1 if x >= threshhold else 0 for x in smoothed_values]

        smoothed_labels = torch.full(value_harp.shape, 2)

        for j in range(0,len(smoothed_values), window_size):
            if np.nanmean(smoothed_values[j:j+window_size])>=threshhold:
                smoothed_labels[j:j+window_size] = 1
            else:
                smoothed_labels[j:j+window_size] = 0

        ax[i].plot(value_harp, marker='.', markersize=1, linestyle='--', color='tab:blue', linewidth=1, alpha=0.5, label="CNN-output")
        ax[i].plot(smoothed_values, linewidth=1, color='blue', label="smoothed curve")
        ax[i].plot(labels, linestyle='--', color='tab:purple', linewidth=1, label="labels")
        ax[i].plot(smoothed_labels, linestyle='-', color='tab:red', linewidth=1, label="smoothed labels")
        ax[i].axhline(threshhold, color='black', linestyle=':', linewidth=0.5)
        ax[i].set_xticks([])
        ax[i].tick_params(axis='y', which='major', labelsize=11)
        # ax[i].set_ylim(-0.1,2.1)
        # ax[i].text(len(value_harp), 1, f'idx = {i}', fontsize=12, horizontalalignment='right')
        # ax[i].text(len(value_harp), 0.95, f'shape = {shape}', fontsize=12,horizontalalignment='right')
        # total_length += continuum.shape[0]
        
    legend_handles = [
    Line2D([0], [0], marker='.', markersize=1, linestyle='--', color='tab:blue', linewidth=1, alpha=0.5),
    Line2D([0], [0], linewidth=1, color='blue'),
    Line2D([0], [0], linestyle='--', color='tab:purple', linewidth=1),
    Line2D([0], [0], linestyle='-', color='tab:red', linewidth=1)
    ]
    legend_labels = ["predicted values", "smoothed values", "labels", "smoothed labels"]
    ax[-1].legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=1, fontsize = 11)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    plt.tight_layout()
    plt.savefig(f"smoothed_labels_{window_size_savgol_filter}_{window_size}.pdf", bbox_inches='tight')
    plt.close()