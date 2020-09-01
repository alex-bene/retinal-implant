import pickle
import torch
import os

class MNISTpulse2percept(torch.utils.data.Dataset):
	def __init__(self, path, val_perc=0, train=True, val=False, transform=None):
		if not os.path.exists(path):
			raise NameError("Path does not exist. Run create_dataset first")

		if (train or val):
			with open(os.path.join(path, 'train_set_data.pk'), "rb") as f:
				self.data = pickle.load(f)
			# self.data    = torch.load(os.path.join(path, 'train_set_data.pt'  ))
			self.targets = torch.load(os.path.join(path, 'train_set_labels.pt'))
		else:
			with open(os.path.join(path, 'test_set_data.pk' ), "rb") as f:
				self.data = pickle.load(f)
			# self.data    = torch.load(os.path.join(path, 'test_set_data.pt'   ))
			self.targets = torch.load(os.path.join(path, 'test_set_labels.pt' ))

		self.transform = transform

		# get dataset dimensions
		data_size  = len(self.data)

		# set validation set percentage (wrt the training set size)
		train_size = round((1 - val_perc) * data_size)

		if val:
			# Reserve val_size samples for validation
			self.data    = self.data[   train_size:]
			self.targets = self.targets[train_size:]
		elif train:
			self.data    = self.data[   :train_size]
			self.targets = self.targets[:train_size]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		label  = int(self.targets[idx])

		if self.transform:
			sample = self.transform(sample)

		return [sample, label]

		# # select the number of classes
		# cifar100_classes_url = self._select_classes_number(classes_number)
		# team_classes         = pd.read_csv(cifar100_classes_url, sep=',', header=None)
		# CIFAR100_LABELS_LIST = pd.read_csv('https://pastebin.com/raw/qgDaNggt', sep=',', header=None).astype(str).values.tolist()[0]

		# self.our_index   = team_classes.iloc[team_seed,:].values.tolist()
		# self.our_classes = self._select_from_list(CIFAR100_LABELS_LIST, self.our_index)
		# index = self._get_ds_index(self.dataset.targets, self.our_index)

		# x_ds = np.asarray(self._select_from_list(self.dataset.data,    index))
		# y_ds = np.asarray(self._select_from_list(self.dataset.targets, index))

		# self.subset_target_to_CIFAR100_target = {i: target for i, target in enumerate(self.our_index)}
		# self.CIFAR100_target_to_subset_target = {target: i for i, target in enumerate(self.our_index)}

		# data_size, img_rows, img_cols = self.data.shape
		# self.sample_shape = (img_rows, img_cols)
