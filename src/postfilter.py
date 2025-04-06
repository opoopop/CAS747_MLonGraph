import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.datasets import Planetoid
from sklearn.metrics.pairwise import euclidean_distances
import faiss
import numpy as np
import torch
from collections import Counter
from scipy.stats import entropy

def get_rank_score(rank_list,n,m):
    result=[0 for i in range(n)]
    for i in range(m):
        k=rank_list[i]
        result[k]=100*(m-i)/m
    return result
def compute_rC_density(node_features: torch.Tensor, cluster_centers: torch.Tensor, labels: torch.Tensor):
    """
    cal the C-density score

        node_features: [N, D] node feature
        cluster_centers: [K, D] cluster centure
        labels: [N] the label

    """
    assigned_centers = cluster_centers[labels]        
    distances = torch.norm(node_features - assigned_centers, dim=1)  
    c_density = 1 / (1 + distances)  
    return c_density


def compute_cluster_rank(data, num_clusters=7):
    """
    implement k-means then get c_density score , then return the ranking
    """

    x = data.x.detach().cpu().float()
    n, d = x.shape

    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=20, verbose=False, seed=42)
    kmeans.train(x.numpy())


    _, I = kmeans.index.search(x.numpy(), 1)  # I: [N, 1]
    labels = torch.tensor(I.squeeze(), dtype=torch.long)  # [N]

    centers = torch.tensor(kmeans.centroids, dtype=x.dtype)  # [K, D]
    c_density = compute_rC_density(x, centers, labels)  # [N]
    sorted_idx = torch.argsort(c_density, descending=True)  # [N]

    rank_list = sorted_idx.tolist()
    return get_rank_score(rank_list, data.num_nodes,data.num_nodes)

def compute_entropy_from_labels(labels):
    """
    cal  Shannon entropy
    labels: List[int] or 1D tensor
    """
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    counter = Counter(labels)
    probs = np.array([v / len(labels) for v in counter.values()])
    return entropy(probs, base=np.e)  # Shannon entropy in natural log base


def compute_rCOE(selected_nodes, annotations, return_rank=True,all_number=2708):
    """
    cal every node's COE score return ranking
    """
    if isinstance(selected_nodes, torch.Tensor):
        selected_nodes = selected_nodes.tolist()

    selected_labels = annotations[selected_nodes]
    base_entropy = compute_entropy_from_labels(selected_labels)

    coe_scores = {}
    for node in selected_nodes:
        # subset apart from one node
        remaining_nodes = [n for n in selected_nodes if n != node]
        remaining_labels = annotations[remaining_nodes]
        new_entropy = compute_entropy_from_labels(remaining_labels)
        coe_scores[node] = new_entropy - base_entropy

    if return_rank:
        sorted_nodes = sorted(coe_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_nodes = [node for node, _ in sorted_nodes]
        return get_rank_score(ranked_nodes,all_number,len(ranked_nodes))

    return coe_scores

def get_rank_score(rank_list,n,m):
    result=[-1 for i in range(n)]
    for i in range(m):
        k=rank_list[i]
        result[k]=100*(m-i)/m
    return result

def get_confidence_rank_score(selected_nodes,conf,all_number):
    result=[]
    for i in range(len(selected_nodes)):
        result.append((selected_nodes[i],conf[i]))

    sorted_result=sorted(result, key=lambda x: x[1], reverse=True)
    rank_list = [node for node, _ in sorted_result]
    #print(len(rank_list))
    score_list=get_rank_score(rank_list,all_number,len(rank_list))

    return score_list
def get_filter_out_index(mask_list,label_list,conf_list,annotations,cluster_list,all_number):
    """
    get score= c-density-score+cond_score+COE_score
    """
    conf_score=get_confidence_rank_score(mask_list,conf_list,all_number)
    COE_score=compute_rCOE(mask_list,annotations,all_number)

    op=-1
    val=-1
    for i in range(len(COE_score)):
        if conf_score[i]==-1:
            continue
        score=0.25*conf_score[i]+0.5*COE_score[i]+0.25*cluster_list[i]
        if op==-1 or score<val:
            val=score
            op=i
    return op


def post_filter(final_number,mask_list,label_list,conf_list,annotations,cluster_list,all_number):
    """
    return the list with the certain number(final_number) [mask_list,label_list]
    using score= c-density-score+cond_score+COE_score

    """
    if final_number>=len(mask_list):
        print('No nodes will be filtered out')
        return [mask_list,label_list]
    process_num=0
    for k in range(len(mask_list)-final_number):
        process_num+=1
        j=get_filter_out_index(mask_list,label_list,conf_list,annotations,cluster_list,all_number)
        if process_num%40==0:
            print(f'process {process_num} nodes')

        
        #print(j)
        new_label_list=[]
        new_conf_list=[]
        new_mask_list=[]
        for i in range(len(mask_list)):
            if mask_list[i]==j:
                #print(mask_list[i],j)
                continue
            new_label_list.append(label_list[i])
            new_conf_list.append(conf_list[i])
            new_mask_list.append(mask_list[i])
        
        mask_list = new_mask_list
        label_list = new_label_list
        conf_list = new_conf_list
    
    return [mask_list,label_list]