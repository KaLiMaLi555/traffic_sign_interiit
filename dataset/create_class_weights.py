import numpy as np
import math
import argparse
import os

def get_class_dist(path):

	path = os.path.join(path,'train')
	_, dirs, _ = next(os.walk(path))
	class_dist = {}

	for dir in dirs:
	  _, _, files = next(os.walk(os.path.join(path,dir)))
	  class_dist[dir] = len(files)
		
	return class_dist

def create_class_weight_log(labels_dict,mu=0.15):
	total = np.sum(list(labels_dict.values()))
	keys = labels_dict.keys()
	class_weight = []
	
	for key in keys:
		score = math.log(mu*total/float(labels_dict[key]))
		score = score if score > 1.0 else 1.0
		class_weight.append(score)
	
	return class_weight

def create_class_weight_average(labels_dict):
	total = np.sum(list(labels_dict.values()))
	keys = labels_dict.keys()
	class_weight = []
	
	for key in keys:
		score = labels_dict[key] * len(keys) / total
		score = score if score > 1.0 else 1.0
		class_weight.append(score)
	
	return class_weight

parser = argparse.ArgumentParser( description="training script for InterIIT Trafic Sign Recognition" )
parser.add_argument("--data-dir", type=str, default='../dataset/GTSRB', help="path to the dataset directory")
parser.add_argument("--weights", type=str, default='average', help="options available: average/log")
parser.add_argument("--save-dir",type=str, default='../config/', help="path to save class weights")

args = parser.parse_args().initialize()

labels_dict = get_class_dist(args.data_dir)

if args.weights == 'average':
	class_weight = create_class_weight_average(labels_dict)
elif args.weights == 'log' :
	class_weight = create_class_weight_log(labels_dict)

np.save(os.path.join(args.save_dir,'class_weights.npy'),np.array(class_weight))
