import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from data_loading import data_prep

dataset = data_prep(language='Ojibwe', text_type="mono")
train,test = dataset.prepare_data()
train = [item for item in train]
test = [item for item in test]

print(train[0])
