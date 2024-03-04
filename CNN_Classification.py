import h5py
import torch
import numpy as np
from tqdm import tqdm
from Sunspot_decetction_model import Sunspot_CNN

if __name__ == "__main__":
    filepath = "path/to/raw_data/to/be/labeled"
    model_path = "path/to/pretrained/model"
    save_path ="path/where/to/save/image-wise-labels"

    model = Sunspot_CNN()
    state_dict = torch.load(model_path)    # 2nd model
    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()

    from scipy.optimize import curve_fit

    def f(x, a, b, c):
        """
        function to fit a parabola to the average intensity
        """
        return a+b*x+c*x**2
    
    with h5py.File(filepath, 'r') as file:
        group_names = list(file.keys())                                                 # These are the different HARP numbers

    for i in tqdm(range(1,2)):
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # loading the data
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        with h5py.File(filepath, 'r') as file:
            group = file[group_names[i]]                                                # accesing a random Harp Number
                
            continuum = torch.tensor(np.array(group['continuum']))                      # get the continuum data
            continuum = torch.nan_to_num(continuum, nan=0.0, posinf=0.0, neginf=0.0)    # set all nan values to 0

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # normalize the continuum, with polynomial of degree 2
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        I_column = torch.mean(continuum, dim = 1)                                       # average over the y-axis
        fit_params = []
        fit_covs = []
        x = np.arange(I_column.shape[1])
        for r in range(I_column.shape[0]):
            p, c = curve_fit(f, x, I_column[r])                                         # fit a parabola to the average intensity
            fit_params.append(p)
            fit_covs.append(c)

        mean_quiet = torch.zeros((continuum.shape[0], continuum.shape[2]))
        for k in range(continuum.shape[0]):
            mean_quiet[k,:] = torch.tensor(f(x,*fit_params[k]))                         # extrapolate the parabola to the whole image [images, width]

        normalized_continuum = torch.div(continuum, mean_quiet[:,None,:])               # normalize the continuum shape: [images, height, width]
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # predict the values with pretrained CNN
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        values = []
        with torch.no_grad():
            n = 0
            for x in normalized_continuum:
                x = x.to('cuda')
                x = x.unsqueeze(0).expand(3,-1,-1).unsqueeze(0)
                value = model(x).unsqueeze(0).item()
                print(value)
                values.append(value)
        value = torch.tensor(values)
        # torch.save(value, f"{save_path}/{group_names[i]}.pt")