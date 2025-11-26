import numpy as np
import matplotlib.pyplot as plt
import numba

from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
from toolkit import get_transition_matrix,get_transition_matrix_PBC,get_transition_matrix_PBC_fast


class full_diffusion_dataset(Dataset):
    def __init__(self, data=None, r=None, t_size=None, schedule=None,PBC=False):

        # hyperparams
        self.r = r  # Hopping rate in each direction
        self.t_size = t_size
        self.schedule = schedule
        self.PBC=PBC

        # datasets
        if data == 'celebA':
            self.ims = np.load('data/celebA_64_64.npy')
            self.ims = np.concatenate([self.ims, self.ims[:,:,::-1,]], axis=0) #augment
        elif data == "mnist":
            dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
            self.ims = np.array(dataset.data[..., None]).astype(int)
        elif data == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
            self.ims = np.array(dataset.data).astype(int)
            self.ims = np.concatenate([self.ims, self.ims[:,:,::-1,]], axis=0) #augment
        elif data == "cifar_gray":
            dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
            self.ims = np.array(dataset.data).astype(int)[..., 0:1]
        else:
            raise ValueError(f"unknown dataset: {data}")

        # dataset shapes
        self.num_ims, self.size, _ , self.in_ch = self.ims.shape
        self.im_size = self.ims.shape[1:-1]
       
        self.dims = len(self.im_size)
        self.size_sqrd = self.size**2

        # Full diffusion internals
        self.out_ch = 2 ** self.dims * self.in_ch
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # FTMs
        if self.PBC:
            self.tk, self.dtk, self.ftp = get_transition_matrix_PBC_fast(self.r, self.t_size, self.size, self.schedule)
        else:
            self.tk, self.dtk, self.ftp = get_transition_matrix(self.r, self.t_size, self.size, self.schedule)

        # ftp is shape (size_sqrd,size_sqrd,t_size)
        self.cumulative_ftp = np.cumsum(self.ftp, axis=0)


    def __len__(self):
        return 1_000_000

    def transpose(self, im):
        return im.transpose( [2,0,1] ).astype(np.float32)

    def forward_backward(self, img, prob, cprob, rnd_seed=None):
        """

        :param img: image, size (L,L,n_channels)
        :param probs: transition probabilities as (out, in)
        :param cprobs: cumulative transition probabilities as (out, in)
        :param rnd_seed: random seed
        :return:
        """

        if rnd_seed:
            np.random.seed(rnd_seed)

        # get outputs:
        all_forward_img = np.zeros((*self.im_size, self.in_ch), dtype=img.dtype)
        all_reverse = np.zeros((self.out_ch, *self.im_size), dtype=np.float32)
        
        for i in range(self.in_ch):
        
            random_numbers = np.random.random(img[:,:,i].sum())  # Generate all random numbers at once

            forward_channel, backward_channel = forward_backward_sample(img[...,i].flatten(), 
                                                                        prob, 
                                                                        cprob, 
                                                                        random_numbers, 
                                                                        self.size,
                                                                        self.PBC)
        
            all_forward_img[..., i] = forward_channel.reshape(*self.im_size)
            all_reverse[i * 4 : (i + 1) * 4,] = (self.r*backward_channel).astype(np.float32)
 
        return all_forward_img, all_reverse

    def __get_image__(self, ind_im, ind_t, seed=None):

        img = self.ims[ind_im,]
        transition = self.ftp[..., ind_t]
        cumulative_transition = self.cumulative_ftp[..., ind_t]
        output, reverse_r = self.forward_backward(img, transition, cumulative_transition, rnd_seed=seed)

        output = self.transpose(output)
        
        return output, reverse_r

    def __getitem__(self, idx):

        # pick indices for image and t
        ind_im = np.random.randint(low=0, high=self.num_ims)
        ind_t  = np.random.randint(low=1, high=self.t_size)

        output, reverse_r = self.__get_image__(ind_im, ind_t) # output is (in_ch,im_size)
        ind_t = np.asarray(ind_t)[None,]
        time = ind_t / self.t_size
        time = torch.as_tensor(time, dtype=torch.float32)
        timeDiff = torch.as_tensor(self.dtk[ind_t], dtype=torch.float32)

        return output, reverse_r, time, timeDiff
        


@numba.jit(parallel=False)
def forward_backward_sample(flat_image, prob, cprob, random_numbers, image_length, PBC):
    """

    Sample SanLinLubb transition probability matrix for independent pixels.

    :param flat_image: single-channel image as shape (n_pixels,)
    :param prob: transition probability as (n_pixels, n_pixels)
    :param cprob: cumulative transition probability as (n_pixels, n_pixels)
    :param random_numbers: block of random numbers (n_particles_total,)
    :param image_length: L=length of y-direction in image
    :return: forward_img (n_pixels), reverse_rates (4,n_pixels//L,L)
    """

    n_pixels = flat_image.size
    # N.L. channel shenings
    if (PBC==1) and (prob.shape != (n_pixels, 1) or cprob.shape != (n_pixels, 1)):
        print('correct, here')
        raise ValueError("Shapes are not equal")
    elif (PBC==0) and (prob.shape != (n_pixels, n_pixels) or cprob.shape != (n_pixels, n_pixels)):
        print('incorrect, here')
        raise ValueError("Shapes are not equal")

    noisy_image = np.zeros_like(flat_image)
    reverse_rates = np.zeros((4, image_length, image_length), dtype=random_numbers.dtype)

    # hard coded for RGB?
    directions = [(0, 1, 0), (1, -1, 0), (2, 0, 1), (3, 0, -1)]
    
    index = 0  
    nonzero_locations = np.flatnonzero(flat_image)


    for location in nonzero_locations:
        num_particles = flat_image[location]

        if PBC:
            cumulative_prob_vector = cprob[:,0]
            transition_prob_vector = prob[:,0]
        else:
            cumulative_prob_vector = cprob[:,location]  # Shape: (n_pixels,)
            transition_prob_vector = prob[:,location]   # Shape: (n_pixels,)

        # Load randome number sequence from pre-generated sequence
        these_random = random_numbers[index:index + num_particles]
        index += num_particles  

        jump_locations = np.searchsorted(cumulative_prob_vector, these_random) # inverse transform sampling 

        if PBC:
            
            x0,y0 = divmod(location, image_length)
            
            for jump_location in jump_locations:

                # Transition probability for the final location
                prob_target = transition_prob_vector[jump_location]

                x, y = divmod(jump_location, image_length)
                x, y = (x0+x)% image_length, (y0+y)% image_length # shifted 2D location
                jump_location = x*image_length + y  # shifted 1D location index

                noisy_image[jump_location] += 1  # Place the particle

                # neighbors
                for d_i, dx, dy in directions:
                    xp, yp = (x + dx - x0)% image_length, (y + dy -y0) % image_length  # shifted to relative location to x0, y0
                    neighbor_location = xp * image_length + yp
                    prob_neighbor = transition_prob_vector[neighbor_location]
                    reverse_rates[d_i, x, y] += prob_neighbor / prob_target  # technically not reverse rate, an overall transition rate constant self.r will be multipled outside this jitted function
                    
        else:

            for jump_location in jump_locations:
                noisy_image[jump_location] += 1  # Place the particle

                # Transition probability for the final location
                prob_target = transition_prob_vector[jump_location]

                # not sure if python or numpy is faster
                x, y = divmod(jump_location, image_length)

                # neighbors
                for d_i, dx, dy in directions:
                    xp, yp = x + dx, y + dy
                    
                    # Boundary check
                    if 0 <= xp < image_length and 0 <= yp < image_length:
                        neighbor_location = xp * image_length + yp
                        prob_neighbor = transition_prob_vector[neighbor_location]
                        reverse_rates[d_i, x, y] += prob_neighbor / prob_target  # technically not reverse rate, an overall transition rate constant self.r will be multipled outside this jitted function

    return noisy_image, reverse_rates
