# -*- coding: utf-8 -*-
"""
Util file includes utility functions
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, hstack
import math
import sys
#from WLVectorizer import WLVectorizer
#import time

def create_graph(adjacency_path=None):
    AM = np.loadtxt(adjacency_path)
    N = AM.shape[0]  
    g = nx.Graph()        
    g.add_nodes_from(range(N), label = "")
    g.graph['node_order'] = range(N)

    list_edges = []
    for u in range(N-1):
        for v in range(u+1,N):
            w = AM[u,v]
            if w != 0.0:
                list_edges.append((u, v))
    g.add_edges_from(list_edges, label="")
    return g  

def node_labeling(g=None, label_path=None): 
    
    list_labels = load_list_from_file(label_path)
    
    dict_node_label = {}
    for idx in range(len(g.nodes())):
        dict_node_label[idx] = list_labels[idx]
        
    #nx.set_node_attributes(g,'label', dict_node_label)
    nx.set_node_attributes(g,name = 'label', values = dict_node_label)
    
def list_files_in_folder(folder_path):    
    """
    Return: A list of the file names in the folder
    """
          
    list = listdir(folder_path)
    onlyfiles = [ f for f in list  if isfile(join(folder_path,f)) ]
    return onlyfiles 

def list_folder_in_folder(folder_path):    
    """
    Return: A list of the file names in the folder
    """
          
    list = listdir(folder_path)
    onlyfiles = [ f for f in list  if not isfile(join(folder_path,f)) ]
    return onlyfiles 
    
def load_list_from_file(file_path):
    """
    Return: A list saved in a file
    """
    
    f = open(file_path,'r')
    listlines = [line.rstrip() for line in f.readlines()]
    f.close()
    return listlines

def hops_neighbors(graph=None, source=None, n_hop=None):
    hop_step = []
    temp_list_nodes = [source]
    hop_step.append(temp_list_nodes)
    
    accumulative_set = set([source])
    for hop in range(1, n_hop+1):
        current_nodes = []
        for u in temp_list_nodes:
            neighbors = graph.neighbors(u)
            current_nodes.extend(neighbors)
        hop_step.append(list(set(current_nodes) - accumulative_set))
        accumulative_set = accumulative_set.union(set(current_nodes))
        temp_list_nodes = current_nodes
    return hop_step

def get_dict_hops_neighbors(graph=None, n_hop=None):
    dict_hops_neighbors = {}
    for n in graph.nodes():
        dict_hops_neighbors[n] = hops_neighbors(graph=graph, source=n, n_hop=n_hop)
    return dict_hops_neighbors

def get_deepgk_grammatrix(graphs=None, n_iter=None, n_hop=None):
    WLvect= WLVectorizer(r=n_iter)
    iters_features = WLvect.transform(graphs)
    list_feature_matrix = []
    for idx in range(len(graphs)):
        M = iters_features[0][idx]
        for iter_id in range(1, n_iter+1):
            M+= iters_features[iter_id][idx]
        list_feature_matrix.append(M)

    N = len(graphs)
    list_graph_vectors = []
    
    for g_idx, g in enumerate(graphs):
        #print g_idx
        dict_hops_neighbors = get_dict_hops_neighbors(graph=g, n_hop=n_hop)
        
        list_pairwise_matrix = []
        
        # initialize the matrix with node 0 (the node id starts from 1, so we need to minus 1)
        hop_vec_0 = list_feature_matrix[g_idx][0,:]
        for hop in range(1, n_hop+1):
            nodes = [n - 1 for n in dict_hops_neighbors[1][hop]]
            hop_vec = csr_matrix(list_feature_matrix[g_idx][nodes,:].sum(axis=0))
            
            H_temp = hop_vec_0.T*hop_vec
            list_pairwise_matrix.append(H_temp)
        
        # Loop all nodes on the graphs
        for n_idx in range(2, len(g.nodes())+1):
            hop_vec_0 = list_feature_matrix[g_idx][n_idx-1,:]
            
            for hop in range(1, n_hop+1):
                nodes = [n - 1 for n in dict_hops_neighbors[n_idx][hop]]
                hop_vec = csr_matrix(list_feature_matrix[g_idx][nodes,:].sum(axis=0))
                H_temp = hop_vec_0.T*hop_vec
                list_pairwise_matrix[hop-1]+= H_temp
        #print "I am here"
        H = hstack(list_pairwise_matrix)

        #col_vec = H.T.reshape((H.shape[0]*H.shape[1],1), order='C')
        
        list_graph_vectors.append(H)

    G = np.zeros((N, N))    
    
    for idx1 in range(N):
        for idx2 in range(idx1, N):
            G[idx1, idx2] = G[idx2, idx1] = list_graph_vectors[idx1].multiply(list_graph_vectors[idx2]).sum()
 
    # Normalize grammatrix
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0    
    return G

def get_deepgk_grammatrix_cas(graphs=None, n_iter=None, n_hop=None):
    WLvect= WLVectorizer(r=n_iter)
    iters_features = WLvect.transform(graphs)
    list_feature_matrix = []
    for idx in range(len(graphs)):
        M = iters_features[0][idx]
        for iter_id in range(1, n_iter+1):
            M+= iters_features[iter_id][idx]
        list_feature_matrix.append(M)

    N = len(graphs)
    list_graph_vectors = []
    
    for g_idx, g in enumerate(graphs):
        #print g_idx
        dict_hops_neighbors = get_dict_hops_neighbors(graph=g, n_hop=n_hop)
        
        list_pairwise_matrix = []
        
        # initialize the matrix with node 0 (the node id starts from 1, so we need to minus 1)
        hop_vec_0 = list_feature_matrix[g_idx][0,:]
        for hop in range(1, n_hop+1):
            nodes = dict_hops_neighbors[0][hop]
            hop_vec = csr_matrix(list_feature_matrix[g_idx][nodes,:].sum(axis=0))
            
            H_temp = hop_vec_0.T*hop_vec
            list_pairwise_matrix.append(H_temp)
        
        # Loop all nodes on the graphs
        for n_idx in range(1, len(g.nodes())):
            hop_vec_0 = list_feature_matrix[g_idx][n_idx,:]
            
            for hop in range(1, n_hop+1):
                nodes = dict_hops_neighbors[n_idx][hop]
                hop_vec = csr_matrix(list_feature_matrix[g_idx][nodes,:].sum(axis=0))
                H_temp = hop_vec_0.T*hop_vec
                list_pairwise_matrix[hop-1]+= H_temp
        #print "I am here"
        H = hstack(list_pairwise_matrix)

        #col_vec = H.T.reshape((H.shape[0]*H.shape[1],1), order='C')
        
        list_graph_vectors.append(H)

    G = np.zeros((N, N))    
    
    for idx1 in range(N):
        for idx2 in range(idx1, N):
            G[idx1, idx2] = G[idx2, idx1] = list_graph_vectors[idx1].multiply(list_graph_vectors[idx2]).sum()
 
    # Normalize grammatrix
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0    
    return G
    
def sum_sparse_vectors(list_sparse_vectors=None):
    vector_sum = list_sparse_vectors[0]
    N = len(list_sparse_vectors)
    
    for idx in range(1,N):
        vector_sum = vector_sum + list_sparse_vectors[idx]
    
    return vector_sum
    
def speeding_up(feature_matrix=None, dict_node_neighbors=None, n_hop=None):
    dict_node_feature = {}
    #print feature_matrix.shape
    for n in dict_node_neighbors:
        dict_temp = {}
        dict_temp[0] = feature_matrix[n,:]
        #print "neighbors hood ", dict_node_neighbors[n]
        dict_temp[1] = csr_matrix(feature_matrix[dict_node_neighbors[n],:].sum(axis=0))
        dict_node_feature[n] = dict_temp
    for hop in range(2, n_hop+1):
        for n in dict_node_neighbors:
            list_sparse_vectors = [dict_node_feature[v][hop-1] for v in dict_node_neighbors[n]]
            dict_node_feature[n][hop] = sum_sparse_vectors(list_sparse_vectors)
    
    return dict_node_feature

def get_deepwl_nk_grammatrix_speedup(dict_node_feature=None, n_hop=None):
    
    """ separately compute between hops """
    N = len(dict_node_feature)
    G = np.zeros((N, N)) 
   
    # Loop over every node couple
    for u in range(N):
        #print u
        #sys.stdout.flush()        
        u_0 = dict_node_feature[u][0]
        for v in range(u, N):
            v_0 = dict_node_feature[v][0]
            dot_0 = u_0.multiply(v_0).sum()
            G[u, v] = dot_0
            
            if dot_0 != 0:
                for hop_id in range(1, n_hop+1):
                    dot_hop = dict_node_feature[u][hop_id].multiply(dict_node_feature[v][hop_id]).sum()
                    G[u, v]+= dot_0*dot_hop
                                   
            G[v, u] = G[u,v]
    # Normalize kernel matrix G   
    #print "Starting normalizing"
    sys.stdout.flush()
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0
                
    return G    
    
def get_deepwl_nk_grammatrix(graph=None, feature_matrix=None, n_hop=None):
    
    """ separately compute between hops """
    N = len(graph.nodes())
    G = np.zeros((N, N)) 
    dict_hops_neighbors = get_dict_hops_neighbors(graph=graph, n_hop=n_hop)
   
    # Loop over every node couple
    for u in range(N):
        #print u
        #sys.stdout.flush()        
        u_vec = feature_matrix[u,:]
        for v in range(u, N):
            v_vec = feature_matrix[v,:]
            dot_value_check = u_vec.multiply(v_vec).sum()
            G[u, v] = dot_value_check
            
            if dot_value_check != 0:
                for hop_id in range(1, n_hop+1):
                    u_vec_sum = csr_matrix(feature_matrix[dict_hops_neighbors[u][hop_id],:].sum(axis=0))
                    v_vec_sum = csr_matrix(feature_matrix[dict_hops_neighbors[v][hop_id],:].sum(axis=0))
                    dot_value_sum = u_vec_sum.multiply(v_vec_sum).sum()
                    G[u, v]+= dot_value_check*dot_value_sum
                                   
                G[v, u] = G[u,v]
    # Normalize kernel matrix G   
    #print "Starting normalizing"
    sys.stdout.flush()
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0
                
    return G    

def deepwl_gk(g1=None, g2=None, feature_matrix1=None, feature_matrix2=None, n_hop=None):
    n1 = len(g1.nodes())
    n2 = len(g2.nodes())
    dict_hops_neighbors_u = get_dict_hops_neighbors(graph=g1, n_hop=n_hop)
    dict_hops_neighbors_v = get_dict_hops_neighbors(graph=g2, n_hop=n_hop)
    k = 0.0
    
    for u in range(1, n1+1):
        u_vec = feature_matrix1[u-1,:]
        for v in range(1, n2+1):
            v_vec = feature_matrix2[v-1,:]
            dot_value_check = u_vec.multiply(v_vec).sum()         
            
            if dot_value_check !=0:
                k+= dot_value_check
                for hop_id in range(1, n_hop+1):
                    u_vec_sum = csr_matrix(feature_matrix1[dict_hops_neighbors_u[u][hop_id],:].sum(axis=0))
                    v_vec_sum = csr_matrix(feature_matrix2[dict_hops_neighbors_v[v][hop_id],:].sum(axis=0))
                    dot_value_sum = u_vec_sum.multiply(v_vec_sum).sum()
                    k+= dot_value_check*dot_value_sum
    return k

def get_deepwl_gk_grammatrix_speedup(graphs=None, n_iter=None, n_hop=None):
    
    WLvect= WLVectorizer(r=n_iter)
    iters_features = WLvect.transform(graphs)
    list_feature_matrix = []
    for idx in range(len(graphs)):
        M = iters_features[0][idx]
        for iter_id in range(1, n_iter+1):
            M+= iters_features[iter_id][idx]
        list_feature_matrix.append(M)
    
    dict_graph_node_hopfeatures = {}
    for g_idx, g in enumerate(graphs):
        dict_node_hop_neighbors = get_dict_hops_neighbors(graph=g, n_hop=n_hop)
        dict_node_hop_feature = {}
        
        for n in g.nodes():
            dict_hop_feature = {}
            dict_hop_feature[0] = list_feature_matrix[g_idx][n-1,:]
            for hop_id in range(1, n_hop+1):
                nodes = [v-1 for v in dict_node_hop_neighbors[n][hop_id]]
                dict_hop_feature[hop_id] = csr_matrix(list_feature_matrix[g_idx][nodes,:].sum(axis=0))
            
            dict_node_hop_feature[n] = dict_hop_feature
        
        dict_graph_node_hopfeatures[g_idx] = dict_node_hop_feature   
                    
    N = len(graphs)
    
    G = np.zeros((N, N))
    
    for g_idx1 in range(N):
        for g_idx2 in range(g_idx1, N):
            value = 0
            
            for n1 in dict_graph_node_hopfeatures[g_idx1]:
                for n2 in dict_graph_node_hopfeatures[g_idx2]:
                    dot_0 = dict_graph_node_hopfeatures[g_idx1][n1][0].multiply(dict_graph_node_hopfeatures[g_idx2][n2][0]).sum()
                    dot_temp = 0
                    if dot_0 !=0:
                        for hop_id in range(1, n_hop+1):
                            dot_temp+= dict_graph_node_hopfeatures[g_idx1][n1][hop_id].multiply(dict_graph_node_hopfeatures[g_idx2][n2][hop_id]).sum()
                        value+= dot_temp*dot_0
            G[g_idx1, g_idx2] = G[g_idx2, g_idx1] = value
    
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0
        
    return G

def get_deepwl_gk_grammatrix(graphs=None, list_feature_matrix=None, n_hop=None):
    N = len(graphs)
    
    G = np.zeros((N, N))
    
    for g_idx1 in range(1, N+1):
        for g_idx2 in range(g_idx1, N+1):
            G[g_idx1-1, g_idx2-1] = deepwl_gk(g1=graphs[g_idx1], g2=graphs[g_idx2], feature_matrix1=list_feature_matrix[g_idx1-1],
                                          feature_matrix2=list_feature_matrix[g_idx2-1], n_hop=n_hop)
            G[g_idx2-1, g_idx1-1] = G[g_idx1-1, g_idx2-1]
    
    for idx1 in range(N):
        for idx2 in range(idx1+1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0
        
    return G
    
def extract_submatrix(row_indices, col_indices, A):
    """ Extracting a submatrix from  matrix A
    
    Parameter:
    row_indices: row index list that we want to extract
    col_indices: Column index list that we want to extract
    A: Matrix
    
    Return:
    submatrix of A
    """

    len_row = len(row_indices)
    len_col = len(col_indices)
    
    M = np.zeros((len_row,len_col))
    for order1, idx_row in enumerate(row_indices):
        for order2, idx_col in enumerate(col_indices):
            M[order1,order2] = A[idx_row,idx_col]
    
    return M

def kcore_decompose_graph(graph=None, max_deg=None):
    g = graph.copy()
    high_degree_nodes = []
    low_degree_nodes = []
    m = min([e[1] for e in g.degree()])
    t = max(max_deg,m)
    
    for n in g.nodes():
        if len(list(g.neighbors(n))) > t:
            high_degree_nodes.append(n)
        else:
            low_degree_nodes.append(n)
    g_high_degree0 = g.subgraph(high_degree_nodes)
    n_nodes = len(high_degree_nodes)
    
    g_low_degree0 = g.subgraph(low_degree_nodes)
    
    g_union = g_low_degree0.copy()

    while n_nodes > 0:        
        high_degree_nodes = []
        low_degree_nodes = []
        m = min([e[1] for e in g_high_degree0.degree()])
        t = max(max_deg,m)
        
        for n in g_high_degree0.nodes():
            if len(list(g_high_degree0.neighbors(n))) > t:      
                high_degree_nodes.append(n)
            else:
                low_degree_nodes.append(n)
        g_high_degree = g_high_degree0.subgraph(high_degree_nodes)
        n_nodes = len(high_degree_nodes)
        
        g_low_degree = g_high_degree0.subgraph(low_degree_nodes)
        
        g_union = nx.union(g_union, g_low_degree)

        g_high_degree0 = g_high_degree
        
    return g_union
        