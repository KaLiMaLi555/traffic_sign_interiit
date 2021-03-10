import numpy as np
import math

# labels_dict : {ind_label: count_label}
# mu : parameter to tune 

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = []
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        x=(score if score > 1.0 else 1.0)
        class_weight.append(x)
    
    np.save('../config/class_weights.npy',np.array(class_weight))

# random labels_dict
labels_dict = {0: 2813, 1: 78, 2: 2814, 3: 78, 4: 7914, 5: 248, 6: 7914, 7: 248}

create_class_weight(labels_dict)