import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import pickle
from scipy.stats import norm

def setup_gpus():
    if dist.is_available():
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(dist.get_rank())

def get_model_name(flags, fine_tune=False, add_string=""):
    model_name = f'PET_{flags.dataset}_{flags.num_layers}_{"local" if flags.local else "nolocal"}_{"layer_scale" if flags.layer_scale else "nolayer_scale"}_{"simple" if flags.simple else "token"}_{"fine_tune" if fine_tune else "baseline"}_{flags.mode}{add_string}.pth'
    return model_name

def load_pickle(folder, f):
    file_name = os.path.join(folder, 'histories', f.replace(".pth", ".pkl"))
    with open(file_name, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict

def revert_npart(npart, name='30'):
    stats = {'30': (29.03636, 2.7629626),
             '49': (21.66242333, 8.86935969),
             '150': (49.398304, 20.772636),
             '279': (57.28675, 29.41252836)}
    mean, std = stats[name]
    return np.round(npart * std + mean).astype(np.int32)

class DataLoader:
    def __init__(self, path, batch_size=512, rank=0, size=1, **kwargs):
        self.path = path
        self.batch_size = batch_size
        self.rank = rank
        self.size = size

        self.mean_part = [0.0, 0.0, -0.0278, 1.8999407, -0.027, 2.244736, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.std_part = [0.215, 0.215, 0.070, 1.2212526, 0.069, 1.2334691, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet = [6.18224920e+02, 0.0, 1.2064709e+02, 3.94133173e+01]
        self.std_jet = [106.71761, 0.88998157, 40.196922, 15.096386]

        self.part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($1 - p_{Trel}$)', 'log($p_{T}$)', 'log($1 - E_{rel}$)', 'log($E$)', '$\Delta$R']
        self.jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet Mass [GeV]', 'Multiplicity']

    def pad(self, x, num_pad):
        return np.pad(x, pad_width=((0, 0), (0, 0), (0, num_pad)), mode='constant', constant_values=0)

    def data_from_file(self, file_path, preprocess=False):
        with h5py.File(file_path, 'r') as file:
            data_chunk = file['data'][:]
            mask_chunk = data_chunk[:, :, 2] != 0
            
            jet_chunk = file['jet'][:]
            label_chunk = file['pid'][:]

            if preprocess:
                data_chunk = self.preprocess(data_chunk, mask_chunk)
                data_chunk = self.pad(data_chunk, num_pad=self.num_pad)
                jet_chunk = self.preprocess_jet(jet_chunk)
                
            points_chunk = data_chunk[:, :, :2]
            
        return [data_chunk, points_chunk, mask_chunk, jet_chunk], label_chunk

    def make_eval_data(self):
        X = self.preprocess(self.X, self.mask).astype(np.float32)
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        dataset = TensorDataset(
            torch.tensor(X),
            torch.tensor(X[:, :, :2]),
            torch.tensor(self.mask.astype(np.float32)),
            torch.tensor(jet),
            torch.zeros((self.jet.shape[0], 1))
        )
                        
        return DataLoader(dataset, batch_size=self.batch_size), torch.tensor(self.y)

    def make_tfdata(self):
        X = self.preprocess(self.X, self.mask).astype(np.float32)
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        dataset = TensorDataset(
            torch.tensor(X),
            torch.tensor(X[:, :, :2]),
            torch.tensor(self.mask.astype(np.float32)),
            torch.tensor(jet),
            torch.tensor(self.y)
        )

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def load_data(self, path, batch_size=512, rank=0, size=1, nevts=None):
        with h5py.File(self.path, 'r') as f:
            self.X = f['data'][rank:nevts:size]
            self.y = f['pid'][rank:nevts:size]
            self.jet = f['jet'][rank:nevts:size]
            self.mask = self.X[:, :, 2] != 0

        self.nevts = f['data'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

    def preprocess(self, x, mask):                
        num_feat = x.shape[-1]
        return mask[:, :, None] * (x[:, :, :num_feat] - self.mean_part[:num_feat]) / self.std_part[:num_feat]

    def preprocess_jet(self, x):        
        return (x - self.mean_jet) / self.std_jet

    def revert_preprocess(self, x, mask):                
        num_feat = x.shape[-1]        
        new_part = mask[:, :, None] * (x[:, :, :num_feat] * self.std_part[:num_feat] + self.mean_part[:num_feat])
        new_part[:, :, 2] = np.minimum(new_part[:, :, 2], 0.0)
        return new_part

    def revert_preprocess_jet(self, x):
        new_x = self.std_jet * x + self.mean_jet
        new_x[:, -1] = np.round(new_x[:, -1])
        new_x[:, -1] = np.clip(new_x[:, -1], 1, self.num_part)
        return new_x

class EicPythiaDataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [-6.57722423e-01, -1.32635604e-04, -1.35429178, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.std_part = [1.43289689, 0.95137615, 1.49257704, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.mean_jet = [6.48229788, -2.52708796, 21.66242333]
        self.std_jet = [2.82288916, 0.4437837, 8.86935969]
        
        self.part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($p_{Trel}$)', 'charge', 'is proton', 'is neutron', 'is kaon',
                           'is pion', 'is neutrino', 'is muon', 'is electron', 'is photon', 'is pi0']
        self.jet_names = ['electron $p_T$ [GeV]', 'electron $\eta$', 'Multiplicity']

        self.load_data(path, batch_size, rank, size)
        self.y = torch.zeros((self.X.shape[0], 1))
        self.num_feat = self.X.shape[2]
        self.num_classes = self.y.shape[1]
        self.num_pad = 0
        self.steps_per_epoch = None
        self.files = [path]

    def add_noise(self, x, shape=None):
        if shape is None:
            noise = torch.rand(x.shape[0]) * 0.6 - 0.3
            x[:, -1] += noise.unsqueeze(1)
        else:
            noise = torch.rand(shape) * 0.6 - 0.3
            x[:, :, 4:] += noise.unsqueeze(1)
        return x

    def preprocess_jet(self, x):
        new_x = self.add_noise(x.clone())
        return (new_x - self.mean_jet) / self.std_jet

    def preprocess(self, x, mask):                
        num_feat = x.shape[-1]
        new_x = self.add_noise(x.clone(), x[:, :, 4:].shape)
        return mask.unsqueeze(-1) * (new_x[:, :, :num_feat] - self.mean_part[:num_feat]) / self.std_part[:num_feat]

    def revert_preprocess(self, x, mask):                
        num_feat = x.shape[-1]        
        new_part = mask.unsqueeze(-1) * (x[:, :, :num_feat] * self.std_part[:num_feat] + self.mean_part[:num_feat])
        pids = torch.zeros_like(new_part[:, :, 4:])
        max_indices = torch.argmax(new_part[:, :, 4:], dim=-1)
        pids.scatter_(-1, max_indices.unsqueeze(-1), 1)
        new_part[:, :, 4:] = pids
        return new_part

class JetNetDataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1, big=False):
        super().__init__(path, batch_size, rank, size)
        if big:
            self.mean_part = [0.0, 0.0, -0.0217, 1.895, -0.022, 2.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [0.115, 0.115, -0.054, 1.549, 0.054, 1.57, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [1.0458962e+03, 3.6804923e-03, 9.4020386e+01, 4.9398304e+01]
            self.std_jet = [123.23525, 0.7678173, 43.103817, 20.772703]
        else:
            self.mean_part = [0.0, 0.0, -0.035, 2.791, -0.035, 3.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.std_part = [0.09, 0.09, 0.067, 1.241, 0.067, 1.26, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.mean_jet = [1.0458962e+03, 3.6804923e-03, 9.4020386e+01, 2.9036360e+01]
            self.std_jet = [123.23525, 0.7678173, 43.103817, 2.76302]
            
    def add_noise(self, x):
        noise = torch.rand(x.shape[0]) * 1.0 - 0.5
        x[:, -1] += noise.unsqueeze(1)
        return x
            
    def preprocess_jet(self, x):
        new_x = self.add_noise(x.clone())
        return (new_x - self.mean_jet) / self.std_jet

    def load_data(self, path, batch_size, rank, size):
        super().load_data(path, batch_size, rank, size)
        self.big = big        
        self.num_pad = 6
        self.num_feat = self.X.shape[2] + self.num_pad
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = [path]

class LHCODataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1, mjjmin=2300, mjjmax=5000, nevts=-1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [0.0, 0.0, -0.019, 1.83, -0.019, 2.068, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.std_part = [0.26, 0.26, 0.066, 1.452, 0.064, 1.46, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.mean_jet = [1.28724651e+03, -4.81260266e-05, 0.0, 2.05052711e+02, 5.72253125e+01]
        self.std_jet = [244.15460668, 0.74111563, 1.0, 151.10313677, 29.44343823]

        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.nevts = f['jet'].shape[0] if nevts < 0 else nevts

        self.size = size

        with h5py.File(self.path, 'r') as f:
            self.jet = torch.tensor(f['jet'][rank:int(self.nevts):size])
            self.X = torch.tensor(f['data'][rank:int(self.nevts):size])
        
        self.X = self.X * (self.X[:, :, :, 3:4] > -0.0)

        if 'pid' in h5py.File(self.path, 'r'):
            with h5py.File(self.path, 'r') as f:
                self.raw_y = torch.tensor(f['pid'][rank:int(self.nevts):size])
        else:
            self.raw_y = self.get_dimass(self.jet)
            
        self.y = self.prep_mjj(self.raw_y)

        self.mask = self.X[:, :, :, 2] != 0
        self.batch_size = batch_size
        
        self.num_part = self.X.shape[2]
        self.num_pad = 6
        self.num_feat = self.X.shape[3] + self.num_pad
        self.num_jet = self.jet.shape[2]
        self.num_classes = 1
        self.steps_per_epoch = None
        self.files = [path]
        self.label = torch.zeros((self.y.shape[0], 1))

    def LoadMjjFile(self, folder, file_name, use_SR, mjjmin=2300, mjjmax=5000):    
        with h5py.File(os.path.join(folder, file_name), "r") as h5f:
            mjj = torch.tensor(h5f['mjj'][:])

        mask = self.get_mjj_mask(mjj, use_SR, mjjmin, mjjmax)
        mjj = self.prep_mjj(mjj)
        return mjj[mask]
        
    def prep_mjj(self, mjj, mjjmin=2300, mjjmax=5000):
        new_mjj = (mjj - mjjmin) / (mjjmax - mjjmin)
        new_mjj = 2 * new_mjj - 1.0
        new_mjj = torch.stack([new_mjj, torch.ones_like(new_mjj)], -1)
        return new_mjj.float()

    def revert_mjj(self, mjj, mjjmin=2300, mjjmax=5000):
        x = (mjj[:, 0] + 1.0) / 2.0        
        x = x * (mjjmax - mjjmin) + mjjmin
        return x
        
    def get_dimass(self, jets):
        jet_e = torch.sqrt(jets[:, 0, 3]**2 + jets[:, 0, 0]**2 * torch.cosh(jets[:, 0, 1])**2)
        jet_e += torch.sqrt(jets[:, 1, 3]**2 + jets[:, 1, 0]**2 * torch.cosh(jets[:, 1, 1])**2)
        jet_px = jets[:, 0, 0] * torch.cos(jets[:, 0, 2]) + jets[:, 1, 0] * torch.cos(jets[:, 1, 2])
        jet_py = jets[:, 0, 0] * torch.sin(jets[:, 0, 2]) + jets[:, 1, 0] * torch.sin(jets[:, 1, 2])
        jet_pz = jets[:, 0, 0] * torch.sinh(jets[:, 0, 1]) + jets[:, 1, 0] * torch.sinh(jets[:, 1, 1])
        mjj = torch.sqrt(torch.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
        return mjj
    
    def get_mjj_mask(self, mjj, use_SR, mjjmin, mjjmax):
        if use_SR:
            mask_region = (mjj > 3300) & (mjj < 3700)
        else:
            mask_region = ((mjj < 3300) & (mjj > mjjmin)) | ((mjj > 3700) & (mjj < mjjmax))
        return mask_region

    def pad(self, x, num_pad):
        return torch.nn.functional.pad(x, (0, num_pad, 0, 0, 0, 0), mode='constant', value=0)

    def make_eval_data(self):
        X = self.preprocess(self.X, self.mask).float()
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).float()

        dataset = torch.utils.data.TensorDataset(
            X,
            X[:, :, :, :2],
            self.mask.float(),
            jet,
            self.y[:, 0],
            torch.zeros((self.jet.shape[0], 2, 1)),
        )
                        
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size), self.label

    def add_noise(self, x):
        noise = torch.rand(x.shape[0]) * 1.0 - 0.5
        x[:, :, -1] += noise.unsqueeze(1)
        return x
    
    def make_tfdata(self, classification=False):
        X = self.preprocess(self.X, self.mask).float()
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.add_noise(self.jet)
        jet = self.preprocess_jet(jet).float()

        if classification:
            y = torch.cat([self.label, self.w], -1)
        else:
            y = self.y

        dataset = torch.utils.data.TensorDataset(
            X,
            X[:, :, :, :2],
            self.mask.float(),
            self.y[:, 0],
            jet,
            y
        )
        
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def combine(self, datasets, use_weights=False):        
        for dataset in datasets:
            self.nevts += dataset.nevts
            self.X = torch.cat([self.X, dataset.X], 0)
            self.mask = torch.cat([self.mask, dataset.mask], 0)
            self.jet = torch.cat([self.jet, dataset.jet], 0)
            self.label = torch.cat([self.label, torch.ones((dataset.y.shape[0], 1))], 0)
            self.y = torch.cat([self.y, dataset.y], 0)
            if use_weights:
                self.w = torch.cat([self.w, torch.ones((dataset.y.shape[0], 1))], 0)
        if use_weights:
            self.X, self.mask, self.jet, self.label, self.y, self.w = self.shuffle(self.X, self.mask, self.jet, self.label, self.y, self.w)
        else:
            self.X, self.mask, self.jet, self.label, self.y = self.shuffle(self.X, self.mask, self.jet, self.label, self.y)

    @staticmethod
    def shuffle(*args):
        permutation = torch.randperm(args[0].shape[0])
        return [arg[permutation] for arg in args]

    def data_from_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            data_chunk = torch.tensor(file['data'][:])
            N, J, P, F = data_chunk.shape
            mask_chunk = data_chunk[:, :, :, 2] != 0  

            jet_chunk = torch.tensor(file['jet'][:])
            label_chunk = self.get_dimass(jet_chunk)
                        
            data_chunk = self.preprocess(data_chunk, mask_chunk)
            data_chunk = self.pad(data_chunk, num_pad=self.num_pad)
            jet_chunk = self.preprocess_jet(jet_chunk)
            points_chunk = data_chunk[:, :, :, :2]            
            data_chunk = data_chunk.reshape(N*J, P, -1)
            jet_chunk = jet_chunk.reshape(N*J, -1)
            
        return [data_chunk, points_chunk, mask_chunk, jet_chunk], label_chunk

    def preprocess_jet(self, x):
        new_x = x.clone()
        new_x[:, :, 2] = torch.from_numpy(norm.ppf(0.5 * (1.0 + x[:, :, 2].numpy() / np.pi)))
        return (new_x - torch.tensor(self.mean_jet)) / torch.tensor(self.std_jet)

    def preprocess(self, x, mask):        
        num_feat = x.shape[-1]
        return mask.unsqueeze(-1) * (x[:, :, :, :num_feat] - torch.tensor(self.mean_part[:num_feat])) / torch.tensor(self.std_part[:num_feat])

    def revert_preprocess(self, x, mask):                
        num_feat = x.shape[-1]
        new_part = mask.unsqueeze(-1) * (x[:, :, :, :num_feat] * torch.tensor(self.std_part[:num_feat]) + torch.tensor(self.mean_part[:num_feat]))
        new_part[:, :, :, 2] = torch.min(new_part[:, :, :, 2], torch.tensor(0.0))
        return new_part
        
    def revert_preprocess_jet(self, x):
        new_x = torch.tensor(self.std_jet) * x + torch.tensor(self.mean_jet)
        new_x[:, :, 2] = torch.pi * (2 * torch.from_numpy(norm.cdf(new_x[:, :, 2].numpy())) - 1.0)
        new_x[:, :, 2] = torch.clamp(new_x[:, :, 2], -torch.pi, torch.pi)
        new_x[:, :, -1] = torch.round(new_x[:, :, -1])
        new_x[:, :, -1] = torch.clamp(new_x[:, :, -1], 2, self.num_part)
        return new_x

class TopDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512, rank=0, size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size, rank, size)
        self.num_pad = 6
        self.num_feat = self.X.shape[2] + self.num_pad
        
        self.y = torch.eye(2)[self.y.long()]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = [path]

class ToyDataLoader(DataLoader):    
    def __init__(self, nevts, batch_size=512, rank=0, size=1):
        super().__init__(nevts, batch_size, rank, size)

        self.nevts = nevts
        self.X = torch.cat([
            torch.randn(self.nevts, 15, 13),
            torch.randn(self.nevts, 15, 13) + 1
        ])
        self.jet = torch.cat([
            torch.randn(self.nevts, 4),
            torch.randn(self.nevts, 4) + 1
        ])
        self.mask = self.X[:, :, 2] != 0
        self.y = torch.cat([torch.ones(self.nevts), torch.zeros(self.nevts)])        
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad
        
        self.y = torch.eye(2)[self.y.long()]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = None

class TauDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512, rank=0, size=1, nevts=None):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [0.0, 0.0, -4.68198519e-02, 2.20178221e-01, -7.48168704e-02, 2.56480441e-01, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.std_part = [0.03927566, 0.04606768, 0.25982114, 0.82466037, 0.7541279, 0.86455974, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.mean_jet = [6.16614813e+01, 2.05619964e-03, 3.52885518e+00, 4.28755680e+00]
        self.std_jet = [34.22578952, 0.68952567, 4.54982729, 3.20547624]

        self.load_data(path, batch_size, rank, size, nevts=nevts)

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad
        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = [path]

class AtlasDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512, rank=0, size=1, is_small=False):
        super().__init__(path, batch_size, rank, size)
        self.mean_jet = [1.73933684e+03, 4.94380870e-04, 2.21667582e+02, 5.52376512e+01]
        self.std_jet = [9.75164004e+02, 8.31232765e-01, 2.03672420e+02, 2.51242747e+01]
        
        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.nevts = int(4e6) if is_small else f['data'].shape[0]
            
        with h5py.File(self.path, 'r') as f:
            self.X = torch.tensor(f['data'][rank:self.nevts:size])
            self.y = torch.tensor(f['pid'][rank:self.nevts:size])
            self.w = torch.tensor(f['weights'][rank:self.nevts:size])
            self.jet = torch.tensor(f['jet'][rank:self.nevts:size])
        self.mask = self.X[:, :, 2] != 0

        self.batch_size = batch_size
        
        self.num_part = self.X.shape[1]
        self.num_pad = 6

        self.num_feat = self.X.shape[2] + self.num_pad
        self.num_jet = self.jet.shape[1]
        self.num_classes = 1
        self.steps_per_epoch = None
        self.files = [path]

    def make_tfdata(self):
        X = self.preprocess(self.X, self.mask).float()
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).float()

        dataset = torch.utils.data.TensorDataset(
            X,
            X[:, :, :2],
            self.mask.float(),
            jet,
            torch.stack([self.y, self.w], -1)
        )

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

class H1DataLoader(DataLoader):    
    def __init__(self, path, batch_size=512, rank=0, size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [0.031, 0.0, -0.10, -0.23, -0.10, 0.27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.std_part = [0.35, 0.35, 0.178, 1.2212526, 0.169, 1.17, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet = [19.15986358, 0.57154217, 6.00354102, 11.730992]
        self.std_jet = [9.18613789, 0.80465287, 2.99805704, 5.14910232]
        
        self.load_data(path, batch_size, rank, size)
                
        self.y = torch.eye(2)[self.y.long()]        
        self.num_pad = 5
        self.num_feat = self.X.shape[2] + self.num_pad
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = [path]

class OmniDataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_jet = [2.25826286e+02, 1.25739745e-03, 1.83963520e+01, 1.88828832e+01]
        self.std_jet = [90.39824296, 1.34598289, 10.73467645, 8.45697634]

        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.X = torch.tensor(f['reco'][rank::size])
            self.Y = torch.tensor(f['gen'][rank::size])        

        self.weight = torch.ones(self.X.shape[0])
        
        with h5py.File(self.path, 'r') as f:
            self.nevts = f['reco'].shape[0]
        self.num_part = self.X.shape[1]
        self.num_pad = 0

        self.num_feat = self.X.shape[2] + self.num_pad
        self.num_jet = 4
        self.num_classes = 1
        self.steps_per_epoch = None
        self.files = [path]

        with h5py.File(self.path, 'r') as f:
            self.reco = self.get_inputs(self.X, torch.tensor(f['reco_jets'][rank::size]))
            self.gen = self.get_inputs(self.Y, torch.tensor(f['gen_jets'][rank::size]))
            self.high_level_reco = torch.tensor(f['reco_subs'][rank::size])
            self.high_level_gen = torch.tensor(f['gen_subs'][rank::size])

    def get_inputs(self, X, jet):
        mask = X[:, :, 2] != 0
        
        time = torch.zeros((mask.shape[0], 1))
        X = self.preprocess(X, mask).float()
        X = self.pad(X, num_pad=self.num_pad)
        jet = self.preprocess_jet(jet).float()
        coord = X[:, :, :2]
        return [X, coord, mask, jet, time]

    def data_from_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            X = torch.tensor(file['reco'][:])
            reco = self.get_inputs(X, torch.tensor(file['reco_jets'][:]))
            label_chunk = torch.ones(X.shape[0])
                        
        return reco, label_chunk

class QGDataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size, rank, size)        
        self.y = torch.eye(2)[self.y.long()]
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = [path]

class CMSQGDataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size, rank, size)
        self.y = torch.eye(2)[self.y.long()]
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None
        self.files = [path]

class JetClassDataLoader(DataLoader):
    def __init__(self, path, batch_size=512, rank=0, size=1, chunk_size=5000, **kwargs):
        super().__init__(path, batch_size, rank, size)
        self.chunk_size = chunk_size

        all_files = [os.path.join(self.path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.files = np.array_split(all_files, self.size)[self.rank]

        self.get_stats(all_files)

    def get_stats(self, file_list):
        self.nevts = len(file_list) * 100000 // 5
        with h5py.File(file_list[0], 'r') as f:
            self.num_part = f['data'].shape[1]
            self.num_feat = f['data'].shape[2]
            self.num_jet = 4
            self.num_classes = f['pid'].shape[1]
        self.steps_per_epoch = self.nevts // self.size // self.batch_size
        self.num_pad = 0
        
    def single_file_generator(self, file_path):
        with h5py.File(file_path, 'r') as file:
            data_size = file['data'].shape[0]
            for start in range(0, data_size, self.chunk_size):
                end = min(start + self.chunk_size, data_size)
                jet_chunk = torch.tensor(file['jet'][start:end])
                mask_particle = jet_chunk[:, -1] > 1
                jet_chunk = jet_chunk[mask_particle]
                data_chunk = torch.tensor(file['data'][start:end]).float()
                data_chunk = data_chunk[mask_particle]
                mask_chunk = data_chunk[:, :, 2] != 0  
                
                label_chunk = torch.tensor(file['pid'][start:end])
                label_chunk = label_chunk[mask_particle]
                data_chunk = self.preprocess(data_chunk, mask_chunk).float()
                jet_chunk = self.preprocess_jet(jet_chunk).float()
                points_chunk = data_chunk[:, :, :2]
                for j in range(data_chunk.shape[0]):                        
                    yield ({
                        'input_features': data_chunk[j],
                        'input_points': points_chunk[j],
                        'input_mask': mask_chunk[j],
                        'input_jet': jet_chunk[j]
                    }, label_chunk[j])
                    
    def interleaved_file_generator(self):
        random.shuffle(self.files)
        generators = [self.single_file_generator(fp) for fp in self.files]
        round_robin_generators = itertools.cycle(generators)

        while True:
            try:
                next_gen = next(round_robin_generators)
                yield next(next_gen)
            except StopIteration:
                break

    def make_tfdata(self):
        def collate_fn(batch):
            inputs = {
                'input_features': torch.stack([item[0]['input_features'] for item in batch]),
                'input_points': torch.stack([item[0]['input_points'] for item in batch]),
                'input_mask': torch.stack([item[0]['input_mask'] for item in batch]),
                'input_jet': torch.stack([item[0]['input_jet'] for item in batch])
            }
            labels = torch.stack([item[1] for item in batch])
            return inputs, labels

        dataset = torch.utils.data.IterableDataset.from_iterable(self.interleaved_file_generator)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

# Utility functions

def configure_optimizers(flags, train_loader, lr_factor=1.0):
    scale_lr = flags.lr * np.sqrt(dist.get_world_size())
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=flags.epoch * train_loader.nevts // flags.batch // dist.get_world_size(),
        eta_min=scale_lr / lr_factor / flags.lr_factor
    )
    optimizer = optim.Lion(
        model.parameters(),
        lr=scale_lr / lr_factor,
        weight_decay=flags.wd * lr_factor,
        betas=(flags.b1, flags.b2)
    )
    return optimizer, lr_schedule