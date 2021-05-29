import numpy as np
from tqdm import tqdm

data = np.load('nor_traces_maxmin.npy')


new = []
for x in tqdm(data):
    #noise = np.random.normal(0, 0.01, 400)
    noise = np.random.normal(0, 0.0065, 400)

    new.append(x+noise)


new = np.array(new)

np.save('nor_traces_with_noise_sigma0.0065.npy',new)
