"""
ImplantSimulateDataset.py

MIT License

Copyright (c) 2020 Alexandros Benetatos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
DESCRIPTION

A class used to simulate the percepted image of a person with a retinal implant for each sample of a Dataset

"""

import os
import torch
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from skimage.transform import resize

class ImplantSimulateDataset():
    def __init__(self, implant, trainset, testset, dataset_name, model, base_data_dir,
                 train_work_samples=None, test_work_samples=None):
        if hasattr(trainset, 'data') and hasattr(testset, 'data'):
            self.out_size = np.array(trainset.data[0]).squeeze().shape
        else:
            raise TypeError("Only pytorch dataset objects with 'data' attribute are supported for trainset and testset")

        if str(type(implant)).split('.')[-1][:-2] == 'ArgusII':
            self.implant_name = 'ArgusII'
        else:
            self.implant_name = implant.name

        self.implant       = implant
        self.dataset_name  = dataset_name
        self.model_name    = str(type(model)).split('.')[-1][:-2]
        self.model         = model
        self.trainset      = trainset
        self.testset       = testset
        self.base_data_dir = base_data_dir

        np.random.seed(77) # for predictable subsets using numpy.random.choice

        self.work_with_subset(train_work_samples, test_work_samples)

        self.calculate_zipped_args()

    def change_implant(self, implant):
        self.implant_name = implant.name
        self.implant      = implant

        self.calculate_and_create_path_names()
        self.calculate_zipped_args() #maybe not

    def change_model(self, model):
        self.model_name = str(type(model)).split('.')[-1][:-2]
        self.model      = model

        self.calculate_and_create_path_names()
        self.calculate_zipped_args() #maybe not

    def calculate_and_create_path_names(self):
        self.percept_path       = os.path.join(self.base_data_dir, self.dataset_name,
                                               'percept', self.model_name+'-'+self.implant_name)
        self.percept_path_test  = os.path.join(self.percept_path, 'test')
        self.percept_path_train = os.path.join(self.percept_path, 'train')

        if not os.path.exists(self.percept_path):
            os.makedirs(self.percept_path)
        if not os.path.exists(self.percept_path_test):
            os.makedirs(self.percept_path_test)
        if not os.path.exists(self.percept_path_train):
            os.makedirs(self.percept_path_train)

    def work_with_subset(self, train_work_samples=None, test_work_samples=None):
        def stratify_subset(dataset, samples):
            labels            = np.array(dataset.targets)
            labels_number     = len(np.unique(labels))
            samples_per_label = int(samples/labels_number)

#             whr_subset = np.concatenate([np.argwhere(labels==i).flatten()[:samples_per_label]
#                                          for i in range(labels_number)]).flatten()
            np.random.seed(77) # for predictable subsets using numpy.random.choice
            whr_subset = np.concatenate([np.random.choice(np.argwhere(labels==i).flatten(),
                                                          samples_per_label, replace=False)
                                         for i in range(labels_number)]).squeeze()

            subset = np.zeros((len(whr_subset), dataset.data[0].shape[0], dataset.data[0].shape[1]))
            for idx, whr in enumerate(whr_subset):
                subset[idx] = dataset[whr][0]

            return (subset, labels[whr_subset])

        if train_work_samples is not None:
            self.dataset_name  = self.dataset_name+'_'+str(train_work_samples)
            self.work_trainset = stratify_subset(self.trainset, train_work_samples)
        else:
            self.dataset_name  = self.dataset_name+'_all'
            self.work_trainset = (np.array(self.trainset.data), np.array(self.trainset.targets))

        if test_work_samples is not None:
            self.dataset_name = self.dataset_name+'_'+str(test_work_samples)
            self.work_testset = stratify_subset(self.testset, test_work_samples)
        else:
            self.dataset_name = self.dataset_name+'_all'
            self.work_testset = (np.array(self.testset.data), np.array(self.testset.targets))

        if self.dataset_name.endswith('_all_all'):
            self.dataset_name = self.dataset_name.split('_all_all')[0]

        self.calculate_and_create_path_names()

    def min_max_scaling(self, dat):
        dat = np.array(dat)
        return (dat-dat.min())/(dat.max()-dat.min())

    # transform an image to a stimuli for the implant's electrodes
    def img2stim(self, img, tight_crop=False):
        """
            Scaled crop out completely black area around the image. By scaled crop we mean
            that we crop in the scale of the implant we want the electrode's to stimulate
        """
        img = np.array(img).squeeze()
        (img_h, img_w) = img.shape # (y, x)

        (implnt_h, implnt_w) = self.implant.earray.shape # (y, x)

        # find where the image is completely black and cut out that area
        if tight_crop:
            npwhr = np.where(img > 0.05)
            crop_ymin  = min(npwhr[0])
            crop_ymax  = max(npwhr[0])
            crop_xmin  = min(npwhr[1])
            crop_xmax  = max(npwhr[1])
        else:
            crop_ymin  = 0
            crop_ymax  = img_h-1
            crop_xmin  = 0
            crop_xmax  = img_w-1

        ymin  = crop_ymin
        ymax  = crop_ymax
        xmin  = crop_xmin
        xmax  = crop_xmax

        if (implnt_h/implnt_w < (ymax-ymin)/(xmax-xmin)):
            stim = np.zeros(((ymax-ymin+1), int(np.ceil((ymax-ymin+1)*implnt_w/implnt_h))))
            xcenter = (stim.shape[1]-1)/2
            xcenter_dist = (xmax-xmin)/2
            xmin  = max(0, int(np.floor(xcenter - xcenter_dist)))
            xmax  = min(stim.shape[1]-1, int(np.floor(xcenter + xcenter_dist)))
            stim[:, xmin:xmax+1] = img[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]
        else:
            stim = np.zeros((int(np.ceil((xmax-xmin+1)*implnt_h/implnt_w)), (xmax-xmin+1)))
            ycenter = (stim.shape[0]-1)/2
            ycenter_dist = (ymax-ymin)/2
            ymin  = max(0, int(np.floor(ycenter - ycenter_dist)))
            ymax  = min(stim.shape[0]-1, int(np.floor(ycenter + ycenter_dist)))
            stim[ymin:ymax+1, :] = img[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]

        # resize the image to fully fit the implant to have the maximum
        # use of the electrodes and flatten it for use in the library
        return resize(stim, self.implant.earray.shape).flatten()

    # take the flattened stimuli (from img2stim) and return an 2-D array representing and image
    def img2implant_img(self, img, tight_crop=False):
        return self.min_max_scaling(np.reshape(self.img2stim(img, tight_crop), self.implant.earray.shape))

    def perc2img(self, percept, padding=0):
        """
            padding: percentage of padding around the image [%]
        """
#         data  = percept.data.squeeze()
#         npwhr = np.where(data > 0.01)

#         ymin = min(npwhr[0])
#         ymax = max(npwhr[0])
#         xmin = min(npwhr[1])
#         xmax = max(npwhr[1])

#         data = resize(data[ymin:ymax, xmin:xmax], self.out_size)
#         data = self.min_max_scaling(data)

#         return Image.fromarray(np.uint8(data*255))
#         # return torch.from_numpy(resize(data[ymin:ymax, xmin:xmax], self.out_size))

        e_radius = list(self.implant.earray.electrodes.items())[0][1].r
        electrodes_list = list(self.implant.earray.electrodes.items())

        xmin = min(electrode[1].x for electrode in electrodes_list)
        xmax = max(electrode[1].x for electrode in electrodes_list)
        ymin = min(electrode[1].y for electrode in electrodes_list)
        ymax = max(electrode[1].y for electrode in electrodes_list)

        percept_y_size = 8000
        percept_y_pxls = percept.data.squeeze().shape[0]
        percept_x_pxls = percept.data.squeeze().shape[1]

        perc_ymin = max(int(round(((100+2*padding)/100)*percept_y_pxls*ymin/percept_y_size) + \
                            percept_y_pxls/2), 0)
        perc_xmin = max(int(round(((100+2*padding)/100)*percept_y_pxls*xmin/percept_y_size) + \
                            percept_x_pxls/2), 0)

        perc_ymax = min(int(round(((100+2*padding)/100)*percept_y_pxls*ymax/percept_y_size) + \
                            percept_y_pxls/2), percept_y_pxls)
        perc_xmax = min(int(round(((100+2*padding)/100)*percept_y_pxls*xmax/percept_y_size) + \
                            percept_x_pxls/2), percept_x_pxls)

        out_percept = percept.data.squeeze()
        out_percept = self.min_max_scaling(out_percept[perc_ymin:perc_ymax, perc_xmin:perc_xmax])

        return Image.fromarray(np.uint8(out_percept*255))

    def calculate_zipped_args(self):
        # exclude files that have already been simulated - valid (size > 0) image files (.png) 
        def filter_function(path, file):
            return file.endswith('.png') and os.path.getsize(os.path.join(path, file)) > 0

        def zip_args(dataset, path):
            all_files = os.listdir(os.path.abspath(path))
            excl_file_numbers = []

            if all_files is not None:
                ex_dataset_files  = list(filter(lambda file: filter_function(path, file), all_files))
                excl_file_numbers = [int(dataset_file.split('-')[0]) for dataset_file in ex_dataset_files]

            data   = dataset[0]
            labels = dataset[1]

            return [[d, t.item(), i]
                    for i, (d, t), in enumerate(zip(data, labels))
                    if i not in excl_file_numbers
                   ]

        self.zipped_test_args  = zip_args(self.work_testset , self.percept_path_test)
        self.zipped_train_args = zip_args(self.work_trainset, self.percept_path_train)

    def print_info(self, plot=True):
        if plot:
            self.implant.plot_on_axon_map()
        print(self)

    def __str__(self):
        return self.__repr__()+"\n" + \
               f"Implant Name      : {self.implant_name}\n" + \
               f"Model   Name      : {self.model_name}\n"   + \
               f"Dataset Name      : {self.dataset_name}\n" + \
               f"Output  Directory : {self.percept_path}\n" + \
               f"Number of train samples to simulate: {len(self.zipped_train_args)}\n" + \
               f"Number of test  samples to simulate: {len(self.zipped_test_args)}"

    def one_loop(self, img, label, idx, tight_crop=False, path=None, padding=0):
        # squeeze image array
        img = np.array(img).squeeze()
        # calculate the optimal implant's electodes stimulation (reduce the number of inactive electrodes)
        self.implant.stim = self.img2stim(img, tight_crop)
        # predict the visual perception from the implant
        percept = self.model.predict_percept(self.implant)
        # crop the visual perception around the implant area with padding around it
        percept = self.perc2img(percept, padding = padding)
        # save the produced image
        if path is not None:
            percept.save(os.path.join(path, f'{idx}-{label}.png'), compress_level=0)
###         torch.save(img, os.path.join(path, f'{idx}-{label}.pt'))
        return percept

    def one_train_loop(self, img, label, idx, tight_crop=False, padding=0):
        self.one_loop(img, label, idx, tight_crop, self.percept_path_train, padding=padding)

    def one_test_loop( self, img, label, idx, tight_crop=False, padding=0):
        self.one_loop(img, label, idx, tight_crop, self.percept_path_test,  padding=padding)

    def samples_visualize(self, num_of_samples=20, tight_crop=False, show_output=True, padding=0):
        np.random.seed(77) # for predictable subsets using numpy.random.choice
        indexes = np.random.choice(len(self.work_trainset[0]), num_of_samples, replace=False)
        images  = np.array(self.work_trainset[0][indexes])
        labels  = np.array(self.work_trainset[1][indexes])

        num_of_rows = np.ceil(num_of_samples/2) if show_output else np.ceil(num_of_samples/3)

        fig = plt.figure(figsize=(16, 3*num_of_rows))
        for index in range(1, num_of_samples + 1):
            plt.subplot(num_of_rows, 6, 3*index-2 if show_output else 2*index-1)
            plt.axis('off')
            plt.imshow(self.min_max_scaling(images[index-1]), cmap='gray')
            plt.title('input - '+str(int(labels[index-1])))

            plt.subplot(num_of_rows, 6, 3*index-1 if show_output else 2*index)
            plt.axis('off')
            stimuli = self.img2implant_img(images[index-1], tight_crop)
            plt.imshow(stimuli, cmap='gray')
            plt.title('stimuli - '+str(int(labels[index-1])))

            if show_output:
                plt.subplot(num_of_rows, 6, 3*index)
                plt.axis('off')
                percept = np.array(self.one_loop(images[index-1], 5, 5, tight_crop, path=None, padding=padding))
                plt.imshow(percept, cmap='gray')
                plt.title('padded output - '+str(int(labels[index-1])))

        plt.subplots_adjust(hspace=0.3, wspace=0)

    @staticmethod
    def create_dataset(path=None, out_path=None, save=True, return_dataset=False, output=True, p_bar=True):
        # function to "create" one sample from its name and path
        def one_sample(path, sample_name):
            # keep only those with size greater than zero '0'
            if os.path.getsize(os.path.join(path, sample_name)) <= 0:
                return None
            # extract the sample labels from each name
            sample_label = int(sample_name.split('-')[1].split('.')[0])
            # load sample data
            img_fp = Image.open(os.path.join(path, sample_name))
            sample_data = img_fp.copy()
            img_fp.close()

            return sample_data, sample_label

        # function to "create" a samples list from a folder path
        def multiple_samples(path):
            if output:
                print(f"Create samples list from path: {path}")
            samples_names  = os.listdir(path)
            samples_names  = list(filter(lambda file: file.endswith('.png'), samples_names))
            samples_labels = []
            samples_data   = []
            if p_bar==True:
                iterator = tqdm(samples_names)
            else:
                iterator = samples_names
            for sample_name in iterator:
                sample = one_sample(path, sample_name)
                if sample is None:
                    print(os.path.join(path, sample_name)+" has zero size - recalculate")
                    continue
                sample_data, sample_label = sample
                samples_labels.append(sample_label)
                samples_data.append(sample_data)

            return samples_data, samples_labels

        if path is None:
            path = self.percept_path

        train_samples_data, train_samples_labels = multiple_samples(os.path.join(path, 'train'))
        test_samples_data,  test_samples_labels  = multiple_samples(os.path.join(path, 'test' ))

        if save:
            if out_path is None:
                out_path = os.path.join(self.base_data_dir, self.dataset_name,
                                        'processed', self.model_name+'-'+self.implant_name)

            # create path if not already there
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # save labels Tensors in their respective files
            torch.save(train_samples_labels, os.path.join(out_path, 'train_set_labels.pt'))
            torch.save(test_samples_labels,  os.path.join(out_path, 'test_set_labels.pt'))

            # save data list in their respective pickles
            with open(os.path.join(out_path, 'train_set_data.pk'), "wb") as f:
                pickle.dump(train_samples_data, f)

            with open(os.path.join(out_path, 'test_set_data.pk' ), "wb") as f:
                pickle.dump(test_samples_data,  f)

        if output:
            return ((train_samples_data, train_samples_labels),
                    (test_samples_data, test_samples_labels))