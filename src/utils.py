import torch
import copy

def create_train_data(file_path):
    """
    prepare the data for training
    """
    data_org = torch.load(file_path, map_location='cpu')
    data_train = copy.deepcopy(data_org)

    return data_train,data_org

def random_split_masks(data, num_train=140, num_val=500, num_test=1000, seed=32):

    """
    randomly set the data for train, val, test with certain number
    """
    torch.manual_seed(seed)  

    num_nodes = data.num_nodes
    all_indices = torch.randperm(num_nodes)  

    # index rearrange
    train_idx = all_indices[:num_train]
    val_idx = all_indices[num_train:num_train + num_val]
    test_idx = all_indices[num_train + num_val:num_train + num_val + num_test]


    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data 

def load_response(file_path,data_name):
    """
    load the response form LLM with certain prompt strategy
    """
    data = torch.load(file_path, map_location='cpu')
    result_data = data[data_name]

    return result_data

def get_result(s,class_map):
    """
    return the class [0,7) and it's confidence score from the string returned by LLM
    """
    l=0
    while(s[l]!=":"):
        l+=1
    l+=2
    if s[l-1]!='"':
        l+=1

    r=l
    while(s[r]!='"'):
        r+=1
    label_str=s[l:r]
    #print(label_str)
    conf=0
    for i in s:
        if ord(i)>=ord('0') and ord(i)<=ord('9'):
            conf=conf*10+ord(i)-ord('0')

    if label_str in class_map:
        return class_map[label_str],conf

def get_inf_from_response(consistency_data,class_map,data_org):
    """
    get all the information from data
    """
    #unknown_labels = set()
    result=[] # (index, (label, confidecnce score))
    for i, item in enumerate(consistency_data):
        if not item or not isinstance(item, list) or not any(s.strip() for s in item):
            continue
        if get_result(item[0],class_map)!=None:
            result.append([i,get_result(item[0],class_map)])
    print('suucessfully read the data')
    total=0
    op=0

    mask_list=[]
    label_list=[]
    anao=[0 for i in range(len(consistency_data))] 
    conf_list=[]
    for i in range(len(result)):
        idx=result[i][0]
        anao[idx]=result[i][1][1]/100
        conf_list.append(result[i][1][1]/100)
        label=result[i][1][0]
        mask_list.append(idx)
        label_list.append(label)
        if data_org.y[idx].item()==label:
            op+=1
        total+=1
    print(f'accuracy of all the annotations: {(100*op/total):.2f} %') # the accuracy of response compared with the ground truth

    annotations=torch.tensor(anao)
    mask_list_tensor=torch.tensor(mask_list)
    label_list_tensor=torch.tensor(label_list)

    #print(f'unknown_labels: {unknown_labels}')
    return mask_list_tensor,label_list_tensor,annotations,conf_list

def change_trainmask_and_label(mask_list,label_list,data):
    """
    change the mask of data
    """
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[mask_list] = True
    data.train_mask = train_mask
    data.y[data.train_mask] = label_list