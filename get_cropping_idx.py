from Data.DataSet import NEW_HMI_Dataset
import torch
from tqdm import tqdm
import json

def get_indices(DataSet):
    cropped_id = []
    for n, (data,_) in enumerate(tqdm(DataSet)):
        d = torch.zeros((data.shape[1],2),dtype = torch.int)
        indices = torch.full((data.shape[1],2),0,dtype = torch.int)
        for i in range(data.shape[1]):
            image = data[0,i,:,:]
            if torch.any(image>33):
                y,x = torch.where(image>33)
                d[i,0] = int(torch.max(x)-torch.min(x))
                d[i,1] = int(torch.max(y)-torch.min(y))
                indices[i,0] = int((torch.max(x)+torch.min(x))/2)
                indices[i,1] = int((torch.max(y)+torch.min(y))/2)
            else:
                indices[i,0] = indices[i-1,0]
                indices[i,1] = indices[i-1,1]
        dx = 2*(torch.max(d[:,0])//2)
        dy = 2*(torch.max(d[:,1])//2)
        idx, idy = indices.T

        start_y = idy - dy / 2
        end_y = idy + dy / 2
        start_x = idx - dx / 2
        end_x = idx + dx / 2

        end_y = torch.where(start_y<0,dy,end_y).to(dtype=torch.int)
        start_y = torch.where(start_y<0,0,start_y).to(dtype=torch.int)

        end_x = torch.where(start_x <0,dx,end_x).to(dtype=torch.int)
        start_x = torch.where(start_x<0,0,start_x).to(dtype=torch.int)

        start_y = torch.where(end_y>data.shape[2],data.shape[2]-dy,start_y).to(dtype=torch.int)
        end_y = torch.where(end_y>data.shape[2],data.shape[2],end_y).to(dtype=torch.int)

        start_x = torch.where(end_x>data.shape[3],data.shape[3]-dx,start_x).to(dtype=torch.int)
        end_x = torch.where(end_x>data.shape[3],data.shape[3],end_x).to(dtype=torch.int)

        cropped_id.append([start_y,end_y,start_x,end_x])

    return cropped_id

def save_indices_to_json(cropped_id, filename):
    with open(filename, 'w') as json_file:
        json.dump(cropped_id, json_file)

def load_indices_from_json(filename):
    with open(filename, 'r') as json_file:
        cropped_id = json.load(json_file)
    return cropped_id

if __name__ == "__main__":
    Bitmaps = NEW_HMI_Dataset(mode='Final',type = 'bitmap', crop = False, set='new')
    cropped_indices = get_indices(Bitmaps)
    cropped_id_python = [[item.tolist() for item in sublist] for sublist in cropped_indices]
    save_indices_to_json(cropped_id_python, '/fast/witmerj/cropping_indices_CNN/cropped_indices_Final.json')

    # Bitmaps = NEW_HMI_Dataset(mode='test',type = 'bitmap', crop = False, set='new')
    # cropped_indices = get_indices(Bitmaps)
    # cropped_id_python = [[item.tolist() for item in sublist] for sublist in cropped_indices]
    # save_indices_to_json(cropped_id_python, '/fast/witmerj/cropping_indices_CNN/cropped_indices_test_5.json')