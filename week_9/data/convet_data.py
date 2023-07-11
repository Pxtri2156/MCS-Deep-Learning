
import os
import csv

train_path="/workspace/tripx/MCS/deep_learning/week_9/data/train"
validation_path="/workspace/tripx/MCS/deep_learning/week_9/data/validation"

label_dic = {'Bill_Gate':0, 
             'Mark_Zuckerberg': 1}

# train
train_target_path = train_path + "/train.csv"

train_data = []

for label in os.listdir(train_path):
    sub_path = os.path.join(train_path, label)
    int_label = label_dic[label]
    for fi_name in os.listdir(sub_path):
        img_path = os.path.join(sub_path, fi_name)
        train_data.append([img_path, int_label])
        
# save data
with open(train_target_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train_data)


# validation
validation_target_path = validation_path + "/validation.csv"
validation_data = []

for label in os.listdir(validation_path):
    sub_path = os.path.join(validation_path, label)
    int_label = label_dic[label]
    for fi_name in os.listdir(sub_path):
        img_path = os.path.join(sub_path, fi_name)
        validation_data.append([img_path, int_label])
        
# save data
with open(validation_target_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(validation_data)
