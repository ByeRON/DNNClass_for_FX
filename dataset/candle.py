import os.path
import pickle
import numpy as np
import pandas as pd

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/candledata.pkl"

file_name  = "kebinGBPJPY_M1.txt"
valid_date = "2013-02-01 00:00:00"

def comvert_dataframe(file_name):
	file_path = dataset_dir + '/' + file_name

	print("Converting " + file_name + " to pandas Dataframe")
	df = pd.read_table(file_path, 
			names=("date","time","open", "high", "low", "close", "yield"),
			dtype = {"date" : "object", "time" : "object"}
		)
	
	df["time"] = pd.to_datetime(df["date"].str.cat(df["time"], sep=' '))
	df.index = df["time"]
	df = df.drop(["date", "yield", "time"], axis=1)

	print("Done")
	return df

def extract_valid(df):
	df = df[valid_date:]
	return df

def init_candle():
	dataset = comvert_dataframe(file_name)

	#extract valid data fpr kebbin
	dataset = extract_valid(dataset)

	print("Creating pickle file ...")
	with open(save_file, 'wb') as f:
		pickle.dump(dataset, f, -1)
	print("Done")
	
def reshape_candle(data, start, end, sticks):
	arr = data.values[start:end].reshape(-1, 4 * sticks)
	return arr

def make_mask(data_size, mask_size):
	return np.random.choice(data_size, mask_size, replace=False)

def resample_candle(data, rate):
	data = data.resample(rate).agg(
		{
			"open" : "first",
			"high" : "max"  ,
			"low"  : "min"  ,
			"close": "last"
		}
	)
	data = data.dropna()
	return data[["open", "high", "low", "close"]]

def make_sequential_data(data, in_sticks, out_sticks):
	x_ = np.empty((0, 4 * in_sticks))
	t_ = np.empty((0, 4 * out_sticks))

	itr = data.shape[0] - (in_sticks + out_sticks - 1)
	for i in range(itr):
		#start = i
		#end   = start + in_sticks
		#arr = data.values[start:end].reshape(-1, 4 * in_sticks)
		arr = reshape_candle(data, i, i + in_sticks, in_sticks)
		#x_train = np.append(x_train,np.array([tmp[0]]), axis=0)
		x_ = np.concatenate( [x_, np.array([arr[0]]) ], axis=0)

		#start = i + in_sticks
		#end   = start + out_sticks
		#arr = data.values[start:end].reshape(-1, 4 * out_sticks)
		arr = reshape_candle(data, i + in_sticks, i + in_sticks + out_sticks, out_sticks)
		t_ = np.concatenate( [t_, np.array([arr[0]]) ], axis=0)
	return (x_ , t_)

def make_xtdata(x_, t_, prob):
	dataset = {}

	train_mask = make_mask(x_.shape[0], int(x_.shape[0] * prob * 100) // 100)
	test_mask  = make_mask(x_.shape[0], int(x_.shape[0] * (1 - prob) * 100) // 100)

	dataset["x_train"] = x_[train_mask]
	dataset["t_train"] = t_[train_mask]
	dataset["x_test"]  = x_[test_mask]
	dataset["t_test"]  = t_[test_mask]

	return dataset

def normalize_candle(dataset, norm):
	dataset["x_train"] = dataset["x_train"] / norm
	dataset["t_train"] = dataset["t_train"] / norm
	dataset["x_test"] = dataset["x_test"] / norm
	dataset["t_test"] = dataset["t_test"] / norm

	return dataset

def load_candle(normalize=True, flatten=True, one_hot_label=False, rate="1H", in_sticks=1, out_sticks=1):

	if not os.path.exists(save_file):
		init_candle()

	with open(save_file, 'rb') as f:
		data = pickle.load(f)
	
	"""
	data = data.resample(rate).agg(
		{
			"open" : "first",
			"high" : "max"  ,
			"low"  : "min"  ,
			"close": "last"
		}
	)
	data = data.dropna()
	data = data[["open", "high", "low", "close"]]
	
	print(data)
	"""
	if rate != "1min":
		data = resample_candle(data, rate)

	"""
	x_ = np.empty((0, 4 * in_sticks))
	t_ = np.empty((0, 4 * out_sticks))

	itr = data.shape[0] - (in_sticks + out_sticks - 1)
	for i in range(itr):
		#start = i
		#end   = start + in_sticks
		#arr = data.values[start:end].reshape(-1, 4 * in_sticks)
		arr = reshape_candle(data, i, i + in_sticks, in_sticks)
		#x_train = np.append(x_train,np.array([tmp[0]]), axis=0)
		x_ = np.concatenate( [x_, np.array([arr[0]]) ], axis=0)

		#start = i + in_sticks
		#end   = start + out_sticks
		#arr = data.values[start:end].reshape(-1, 4 * out_sticks)
		arr = reshape_candle(data, i + in_sticks, i + in_sticks + out_sticks, out_sticks)
		t_ = np.concatenate( [t_, np.array([arr[0]]) ], axis=0)
	"""
	print("make dataset ...")
	(x_, t_) = make_sequential_data(data, in_sticks, out_sticks)

	"""
	train_mask_size = x_.shape[0] * 8 // 10
	train_mask = np.random.choice(x_.shape[0], train_mask_size, replace=False)
	test_mask_size = x_.shape[0] - train_mask_size
	test_mask = np.random.choice(x_.shape[0], test_mask_size, replace=False)
	"""
	"""
	train_mask = make_mask(x_.shape[0], x_.shape[0] * 8 // 10)
	test_mask  = make_mask(x_.shape[0], x_.shape[0] * 2 // 10)

	dataset = {}

	dataset["x_train"] = x_[train_mask]
	dataset["t_train"] = t_[train_mask]
	dataset["x_test"]  = x_[test_mask]
	dataset["t_test"]  = t_[test_mask]
	"""
	dataset = make_xtdata(x_, t_, 0.8)

	if normalize:
		norm = data["high"].max()
		dataset = normalize_candle(dataset, norm)

	if one_hot_label:
		pass

	print("Done!")
	return (dataset['x_train'], dataset['t_train']), (dataset['x_test'], dataset['t_test']) 

if __name__ == "__main__":
	load_candle()
