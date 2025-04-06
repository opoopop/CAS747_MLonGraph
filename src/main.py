from utils import create_train_data,random_split_masks,load_response,get_inf_from_response,change_trainmask_and_label
import argparse
from model import GCN2,train,test,test_final
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from postfilter import compute_cluster_rank,post_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN-LLM args")
    parser.add_argument("--post_filter", type=bool, default=False, help="use post-filtering or not")
    parser.add_argument("--prompt", type=str, default='few_shot', help="the stagety of prompt")
    parser.add_argument("--filter_num", type=int, default=300, help="the number of nodes that leave after post filtering")

    args = parser.parse_args()
            # load train , test data
    data_path="data/cora_fixed_sbert.pt"
    data_org,data_train = create_train_data(data_path)
    data_org = random_split_masks(data_org)
    data_train = random_split_masks(data_train)
    # load response
    response_path='LLM_GNN_data/cora_openai.pt' 
    consistency_data = load_response(response_path,args.prompt)

    # process response
    class_map = { # a mapping from the responses
    'rule_Learning': 0,
    'neural_networks': 1,
    "case_based": 2,
    'genetic_algorithms': 3,
    'theory': 4,
    "reinforcement_learning": 5,
    "probabilistic_methods": 6
    }
    mask_list_tensor,label_list_tensor,annotations,conf_list=get_inf_from_response(consistency_data,class_map,data_org)
    change_trainmask_and_label(mask_list_tensor,label_list_tensor,data_train)
    train_nodes = data_train.train_mask.sum().item()
    print(f'finish preparing the data and response from LLM using {args.prompt} prompt, {train_nodes} nodes in total')
    
    if args.post_filter==True:
        print('start post filtering process')
        cluster_list=compute_cluster_rank(data_train)
        A,B=post_filter(args.filter_num,mask_list_tensor,label_list_tensor,conf_list,annotations,cluster_list,data_train.num_nodes)
        change_trainmask_and_label(A,torch.tensor(B),data_train)
        print(f'{args.filter_num} nodes left')

    print('start train GNN')
    model = GCN2(num_layers=2, input_dim=data_train.num_node_features, hidden_dimension=64, 
    num_classes=7, dropout=0.5, norm=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(1, 80):
        loss = train(data_train,model,optimizer)
        train_acc, val_acc = test(data_train,model)
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
    
    #print(f'Number of training nodes: {train_nodes}')
    print(f'final accuracy of GNN: {test_final(data_org,model)*100:.1f} %')
    content=f'final accuracy of GNN on Cora with prompt {args.prompt}: {test_final(data_org,model)*100:.1f} %'
    with open('result/result.txt', 'w') as f:
        f.write(content)






