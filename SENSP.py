#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8
import time
import sys
import math
import random
from cgi import test

import numpy as np
import scipy.io
from numpy import loadtxt
import pandas as pd
from collections import Counter


from numba import  typeof
from numba.experimental import jitclass



from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import *
from sklearn.metrics import f1_score


import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

# Get the input name of the dataset
graphName = sys.argv[1]


# Class to read the graph - dataset
class ReadGraph():
    graphs = 1  			# total numbers of graphs / layers
    nodes = 0				# total number of nodes in a graph
    atts = 0				# total number of node attributes (if available) and node labels
    
    dim = 32				# dimensionality
    iterations = 10			# number of iterations
    extraiter = 2			# number of extra iterations
    
    clusters = 3			# total number of clusters
	
    node_attr = {}			# all node attributes (if available) and node labels
    adj = []				# all adjacency matrices of input graphs/layers
    gt = []					# ground truth data
    train_idx = []
    
    labels = []             # all the labels
    samples = 2             # number of sample data we need for the adjacency matrix for the negative sampling layer
    dim_cluster = 2         # the dimensionality for the clustering task
    name = ''               # name of the dataset
    
    
    def __init__(self, name):
        self.name = name
        self.readData()
        self.printD()
        
    def load_adj_neg(self, num_nodes, samples):
        col = np.random.randint(0, num_nodes, size=num_nodes * samples)
        row = np.repeat(range(num_nodes), samples)
        index = np.not_equal(col,row)
        col = col[index]
        row = row[index]
        new_col = np.concatenate((col,row),axis=0)
        new_row = np.concatenate((row,col),axis=0)
        data = np.ones(new_col.shape[0])
        adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
        adj_neg = adj_neg.todense(order='F')

        return adj_neg 
    
    def readACM(self):
        self.graphs = 5
        
        self.dim = 32
        self.clusters = 3
        
        self.iterations = 100
        self.extraiter = 20
        self.samples = 52
        self.dim_cluster = 2
        
        extraPath = 'data/'
        path = extraPath + "ACM/"
        adj1 = scipy.io.loadmat(path + "PAP.mat")['PAP']
        adj2 = scipy.io.loadmat(path + "PLP.mat")['PLP']

        
        self.nodes = adj1.shape[0]
        self.labels = scipy.io.loadmat(path + 'label.mat')['label']
        self.node_attr = scipy.io.loadmat(path + "feature.mat")['feature']
        train_ids = np.sort(loadtxt(path + 'train_ids_1.txt')).astype(int)
        
        
        self.gt = np.loadtxt(path + "ground_truth.txt")
        self.atts = len(self.node_attr[0]) 
        
        gt_ = np.zeros((self.nodes,3))
        for i in train_ids:
            gt_[int(i)][0] = self.labels[int(i)][0]
            gt_[int(i)][1] = self.labels[int(i)][1]
            gt_[int(i)][2] = self.labels[int(i)][2]
            
            
        
        self.node_attr = np.concatenate((self.node_attr, gt_), axis=1)
        self.atts = len(self.node_attr[0])
        self.train_idx = train_ids
        
        adj12 = adj1 + adj2 
        
        adj1s = np.zeros((self.nodes, self.nodes))
        for j in range(adj1.shape[1]):
            edges = np.asarray(np.where(adj1[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj1[e]>0)[0])
                    for e_ in edges_:
                        adj1s[j][e_] += 1
                        
        
        adj1s_ = sp.coo_matrix(adj1s)
        adj1s_ = adj1s_.todense(order='F')
        adj1s_ = np.array(adj1s_)
        
        adj2s = np.zeros((self.nodes, self.nodes))
        for j in range(adj2.shape[1]):
            edges = np.asarray(np.where(adj2[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj2[e]>0)[0])
                    for e_ in edges_:
                        adj2s[j][e_] += 1
        
        adj2s_ = sp.coo_matrix(adj2s)
        adj2s_ = adj2s_.todense(order='F')
        adj2s_ = np.array(adj2s_)
        
        adj12s_ = adj1s_ + adj2s_ 
        

        neg1 = self.load_adj_neg(self.nodes, self.samples)
        neg_m1 = np.array(neg1)
        
        
        for i in range(self.nodes):
            neg_m1[i][i] = 0
            for j in range(self.nodes):
                if neg_m1[i][j] > 0 and adj12[i][j] > 0:
                    neg_m1[i][j] = 0
                elif neg_m1[i][j] > 0 and adj12s_[i][j] > 0:
                    neg_m1[i][j] = 0
                    
        adj_neg = np.array(neg_m1)
        
        self.adj = [adj1,adj2, adj1s_, adj2s_, adj_neg]
    

    def readIMDB(self):
        self.graphs = 5
        
        self.dim = 32
        self.clusters = 3
        
        self.iterations = 100 
        self.extraiter = 20
        self.samples = 83
        self.dim_cluster = 2
        
        extraPath = 'data/'
        path = extraPath + "IMDB/"
        adj1 = scipy.io.loadmat(path + "imdb.mat")['MDM']
        adj2 = scipy.io.loadmat(path + "imdb.mat")['MAM']
        
        self.nodes = adj1.shape[0]

        mat = scipy.io.loadmat(path + "imdb.mat")
        self.labels = mat['label']
        
        self.node_attr = scipy.io.loadmat(path + "imdb.mat")['feature']
        self.atts = len(self.node_attr[0])
        self.gt = np.loadtxt(path + "ground_truth.txt")
        
        train_ids = np.sort(loadtxt(path + 'train_ids_1.txt')).astype(int)

        gt_ = np.zeros((self.nodes,3))
        
        for i in range(0, len(train_ids)):
            gt_[int(train_ids[i])][0] = self.labels[int(train_ids[i])][0]
            gt_[int(train_ids[i])][1] = self.labels[int(train_ids[i])][1]
            gt_[int(train_ids[i])][2] = self.labels[int(train_ids[i])][2]
            
        self.node_attr = np.concatenate((self.node_attr, gt_), axis=1)
        self.atts = len(self.node_attr[0])
        self.train_idx = train_ids     
        adj12 = adj1 + adj2   
        
        adj1s = np.zeros((self.nodes, self.nodes))
        for j in range(adj1.shape[1]):
            edges = np.asarray(np.where(adj1[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj1[e]>0)[0])
                    for e_ in edges_:
                        adj1s[j][e_] += 1             
        
        adj1s_ = sp.coo_matrix(adj1s)
        adj1s_ = adj1s_.todense(order='F')
        adj1s_ = np.array(adj1s_)
        
        adj2s = np.zeros((self.nodes, self.nodes))
        for j in range(adj2.shape[1]):
            edges = np.asarray(np.where(adj2[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj2[e]>0)[0])
                    for e_ in edges_:
                        adj2s[j][e_] += 1
        
        adj2s_ = sp.coo_matrix(adj2s)
        adj2s_ = adj2s_.todense(order='F')
        adj2s_ = np.array(adj2s_)
        
        adj12s_ = adj1s_ + adj2s_ 
        
        neg1 = self.load_adj_neg(self.nodes, self.samples)
        neg_m1 = np.array(neg1)
        
        for i in range(self.nodes):
            neg_m1[i][i] = 0
            for j in range(self.nodes):
                if neg_m1[i][j] > 0 and adj12[i][j] > 0:
                    neg_m1[i][j] = 0
                elif neg_m1[i][j] > 0 and adj12s_[i][j] > 0:
                    neg_m1[i][j] = 0
                    
        adj_neg = np.array(neg_m1)
        
        self.adj = [adj1,adj2, adj1s_, adj2s_, adj_neg]


    def readDBLP(self):
        self.graphs = 9
        
        self.dim = 128
        self.clusters = 3 
		
        self.iterations = 100
        self.extraiter = 20
        self.samples = 8
        self.dim_cluster = 2
        
        extraPath = 'data/'
        path = extraPath + "DBLP/"
        
        adj1e = scipy.io.loadmat(path + "coauthor_mat.mat")['coauthor_mat']
        adj2e = scipy.io.loadmat(path + "apnet_mat.mat")['apnet_mat']
        adj3e = scipy.io.loadmat(path + "citation_mat.mat")['citation_mat']
        adj4e = scipy.io.loadmat(path + "cocitation_mat.mat")['cocitation_mat']
        
        adj1e = adj1e + np.eye(adj1e.shape[0])
        adj2e = adj2e + np.eye(adj2e.shape[0])
        adj3e = adj3e + np.eye(adj3e.shape[0])
        adj4e = adj4e + np.eye(adj4e.shape[0])
        
        print("Number of edges: ", np.sum(adj1e), "; ", np.sum(adj2e), "; ", np.sum(adj3e), "; ", np.sum(adj4e))
        
        adj1 = np.zeros((adj1e.shape[0], adj1e.shape[0]), order='F')
        adj2 = np.zeros((adj2e.shape[0], adj2e.shape[0]), order='F')
        adj3 = np.zeros((adj3e.shape[0], adj3e.shape[0]), order='F')
        adj4 = np.zeros((adj4e.shape[0], adj4e.shape[0]), order='F')
        for i in range(0, adj1.shape[0]):
            for j in range(0, adj1.shape[1]):
                adj1[i][j] = adj1e[i][j]
                adj2[i][j] = adj2e[i][j]
                adj3[i][j] = adj3e[i][j]
                adj4[i][j] = adj4e[i][j]
        
        self.nodes = adj1.shape[0]
        self.gt = np.loadtxt(path + "ground_truth.txt")      

        self.node_attr = np.zeros((adj1.shape[0], int(np.max(self.gt) + 1)))
        self.atts = self.node_attr.shape[1]
        
        train_ids = np.sort(loadtxt(path + 'train_ids_.txt')).astype(int)

        for i in train_ids:
            if self.gt[int(i)] == 0:
                self.node_attr[int(i)][0] = 1.0
            elif self.gt[int(i)] == 1:
                self.node_attr[int(i)][1] = 1.0
            elif self.gt[int(i)] == 2:
                self.node_attr[int(i)][2] = 1.0

        self.train_idx = train_ids          
        
        adj1234 = adj1 + adj2 + adj3 + adj4
        
        adj1s = np.zeros((self.nodes, self.nodes))
        for j in range(adj1.shape[1]):
            edges = np.asarray(np.where(adj1[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj1[e]>0)[0])
                    for e_ in edges_:
                        adj1s[j][e_] += 1
                        
        adj1s_ = sp.coo_matrix(adj1s)
        adj1s_ = adj1s_.todense(order='F')
        adj1s_ = np.array(adj1s_)
        
        adj2s = np.zeros((self.nodes, self.nodes))
        for j in range(adj2.shape[1]):
            edges = np.asarray(np.where(adj2[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj2[e]>0)[0])
                    for e_ in edges_:
                        adj2s[j][e_] += 1
        
        adj2s_ = sp.coo_matrix(adj2s)
        adj2s_ = adj2s_.todense(order='F')
        adj2s_ = np.array(adj2s_)
        
        adj3s = np.zeros((self.nodes, self.nodes))
        for j in range(adj3.shape[1]):
            edges = np.asarray(np.where(adj3[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj3[e]>0)[0])
                    for e_ in edges_:
                        adj3s[j][e_] += 1
        
        adj3s_ = sp.coo_matrix(adj3s)
        adj3s_ = adj3s_.todense(order='F')
        adj3s_ = np.array(adj3s_)
        
        adj4s = np.zeros((self.nodes, self.nodes))
        for j in range(adj4.shape[1]):
            edges = np.asarray(np.where(adj4[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj4[e]>0)[0])
                    for e_ in edges_:
                        adj4s[j][e_] += 1
        
        adj4s_ = sp.coo_matrix(adj4s)
        adj4s_ = adj4s_.todense(order='F')
        adj4s_ = np.array(adj4s_)
        
        adj1234s_ = adj1s_ + adj2s_ + adj3s_ +adj4s_
        
        neg1 = self.load_adj_neg(self.nodes, self.samples)
        neg_m1 = np.array(neg1)
        
        for i in range(self.nodes):
            for j in range(self.nodes):
                if neg_m1[i][j] > 0 and adj1234[i][j] > 0:
                    neg_m1[i][j] = 0
                elif neg_m1[i][j] > 0 and adj1234s_[i][j] > 0:
                    neg_m1[i][j] = 0   
                    
        adj_neg = np.array(neg_m1)
        
        self.adj = [adj1, adj2, adj3, adj4, adj1s_, adj2s_, adj3s_, adj4s_, adj_neg]

         
    def readFREEBASE(self):
        self.graphs = 7
        
        self.dim = 16
        self.clusters = 3
        
        self.iterations = 100
        self.extraiter = 20
        self.samples = 14
        self.dim_cluster = 2
        
        extraPath = 'data/'
        path = extraPath + "FB/"
        data = scipy.io.loadmat(path + 'freebase.mat')
        
        adj_edge1 = np.array(data['mam'].todense())
        adj_edge2 = np.array(data['mdm'].todense())
        adj_edge3 = np.array(data['mwm'].todense())
        
        adj1 = np.zeros((adj_edge1.shape[0], adj_edge1.shape[1]), order='F')
        adj2 = np.zeros((adj_edge2.shape[0], adj_edge2.shape[1]), order='F')
        adj3 = np.zeros((adj_edge3.shape[0], adj_edge3.shape[1]), order='F')
        
        for i in range(0, adj1.shape[0]):
            for j in range(0, adj1.shape[1]):
                adj1[i][j] = adj_edge1[i][j]
                adj2[i][j] = adj_edge2[i][j]
                adj3[i][j] = adj_edge3[i][j]

        adj123 = adj1 + adj2 + adj3
        
        self.nodes = adj1.shape[0]
        
        self.labels = data['label']
        self.gt = np.loadtxt(path + "ground_truth.txt") 
        
        self.node_attr = np.zeros((adj1.shape[0], int(np.max(self.gt) + 1)))
        self.atts = self.node_attr.shape[1]
        
        train_ids = np.sort(loadtxt(path + 'train_ids_.txt')).astype(int)

        for i in train_ids:
            self.node_attr[int(i)] = self.labels[int(i)]
            
        self.train_idx = train_ids          

        adj1s = np.zeros((self.nodes, self.nodes))
        for j in range(adj1.shape[1]):
            edges = np.asarray(np.where(adj1[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj1[e]>0)[0])
                    for e_ in edges_:
                        adj1s[j][e_] += 1
                        
        adj1s_ = sp.coo_matrix(adj1s)
        adj1s_ = adj1s_.todense(order='F')
        adj1s_ = np.array(adj1s_)
        
        adj2s = np.zeros((self.nodes, self.nodes))
        for j in range(adj2.shape[1]):
            edges = np.asarray(np.where(adj2[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj2[e]>0)[0])
                    for e_ in edges_:
                        adj2s[j][e_] += 1
        
        adj2s_ = sp.coo_matrix(adj2s)
        adj2s_ = adj2s_.todense(order='F')
        adj2s_ = np.array(adj2s_)
        
        adj3s = np.zeros((self.nodes, self.nodes))
        for j in range(adj3.shape[1]):
            edges = np.asarray(np.where(adj3[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj3[e]>0)[0])
                    for e_ in edges_:
                        adj3s[j][e_] += 1
        
        adj3s_ = sp.coo_matrix(adj3s)
        adj3s_ = adj3s_.todense(order='F')
        adj3s_ = np.array(adj3s_)
        
        adj123s_ = adj1s_ + adj2s_ + adj3s_

        neg1 = self.load_adj_neg(self.nodes, self.samples)
        neg_m1 = np.array(neg1)
        
        for i in range(self.nodes):
            for j in range(self.nodes):
                if neg_m1[i][j] > 0 and adj123[i][j] > 0:
                    neg_m1[i][j] = 0
                elif neg_m1[i][j] > 0 and adj123s_[i][j] > 0:
                    neg_m1[i][j] = 0  
                    
        adj_neg = np.array(neg_m1)
        
        self.adj = [adj1, adj2, adj3, adj1s_, adj2s_, adj3s_, adj_neg]
        

    def readFLICKR(self):
        self.graphs = 5
        
        self.dim = 64
        self.clusters = 7 
		
        self.iterations = 100
        self.extraiter = 20
        self.samples = 84
        self.dim_cluster = 2
        
        extraPath = 'data/'
        path = extraPath + "FLICKR/"
        
        adj1 = scipy.io.loadmat(path + "layer0mat.mat")['layer0mat']
        adj2 = scipy.io.loadmat(path + "layer1mat.mat")['layer1mat']
        
        self.nodes = adj1.shape[0]
        self.gt = np.loadtxt(path + "ground_truth.txt")      

        self.node_attr = np.zeros((adj1.shape[0], int(np.max(self.gt) + 1)))
        self.atts = self.node_attr.shape[1]
        
        train_ids = np.sort(loadtxt(path + 'train_ids_.txt')).astype(int)

        for i in train_ids:
            if self.gt[int(i)] == 0:
                self.node_attr[int(i)][0] = 1.0
            elif self.gt[int(i)] == 1:
                self.node_attr[int(i)][1] = 1.0
            elif self.gt[int(i)] == 2:
                self.node_attr[int(i)][2] = 1.0
            elif self.gt[int(i)] == 3:
                self.node_attr[int(i)][3] = 1.0
            elif self.gt[int(i)] == 4:
                self.node_attr[int(i)][4] = 1.0
            elif self.gt[int(i)] == 5:
                self.node_attr[int(i)][5] = 1.0
            elif self.gt[int(i)] == 6:
                self.node_attr[int(i)][6] = 1.0

        self.train_idx = train_ids          
        
        adj12 = adj1 + adj2 
        
        adj1s = np.zeros((self.nodes, self.nodes))
        for j in range(adj1.shape[1]):
            edges = np.asarray(np.where(adj1[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj1[e]>0)[0])
                    for e_ in edges_:
                        adj1s[j][e_] += 1
                                
        adj1s_ = sp.coo_matrix(adj1s)
        adj1s_ = adj1s_.todense(order='F')
        adj1s_ = np.array(adj1s_)
        
        adj2s = np.zeros((self.nodes, self.nodes))
        for j in range(adj2.shape[1]):
            edges = np.asarray(np.where(adj2[j]>0)[0])
            for e in edges:
                if e!=j:
                    edges_ = np.asarray(np.where(adj2[e]>0)[0])
                    for e_ in edges_:
                        adj2s[j][e_] += 1
        
        adj2s_ = sp.coo_matrix(adj2s)
        adj2s_ = adj2s_.todense(order='F')
        adj2s_ = np.array(adj2s_)
        
        adj12s_ = adj1s_ + adj2s_         
        
        neg1 = self.load_adj_neg(self.nodes, self.samples)
        neg_m1 = np.array(neg1)
        
        for i in range(self.nodes):
            for j in range(self.nodes):
                if neg_m1[i][j] > 0 and adj12[i][j] > 0:
                    neg_m1[i][j] = 0
                elif neg_m1[i][j] > 0 and adj12s_[i][j] > 0:
                    neg_m1[i][j] = 0
                    
        adj_neg = np.array(neg_m1)
        
        self.adj = [adj1, adj2, adj1s_, adj2s_, adj_neg]
        
        
    def readData(self):
        if(self.name == 'imdb'):
            self.readIMDB()
        elif(self.name == 'acm'):
            self.readACM()
        elif(self.name == 'dblp'):
            self.readDBLP()
        elif(self.name == 'freebase'):
            self.readFREEBASE()
        elif(self.name == 'flickr'):
            self.readFLICKR()

    def printD(self):
        print("#graphs = ", self.graphs)
        print("#nodes = ", self.nodes)
        print("#atts = ", self.atts)
        print("#dim = ", self.dim)
        print("#dim cluster = ", self.dim_cluster)
        print("Negative Samples = ", np.sum(self.adj[-1]))
        print("#iterations = ", self.iterations)
        print("#extra iterations = ", self.extraiter)



# Compute the weighting factors for each layer, each node, each node attribute (if available), and each node label        
class mixedSpectral():
    num_g = 0				# total number of graphs/layers
    weighted = False		# weighted or unweighted edges
    attributes = []			# node attributes 
    numAtt = 0				# total number of node attributes and node labels
    
    
    num_objects = 0			# total number of nodes
    num_cat = 0				# total number of categorical values of node attributes and node labels
    
    
    startIndex = []			# starting index of node attributes and labels
    catCount = {}			# counting of categories for each categorical node attribute and node labels
    
    weightFactors = []		# weighting factors for all graphs/layers and node attributes/labels
    sumWeights = []			# sum weights of each node from all available graphs/layers
    adj = []				# all adjacency matrices of input graphs/layers
    
    clusters = 3			# numbers of clusters
    
    
    def __init__(self, num_g, adj, num_nodes, weighted, attributes, numAtt, clusters):

        self.num_g = num_g
        self.weighted = weighted
        self.attributes = attributes
        self.numAtt = numAtt
        self.adj = adj
        self.clusters = clusters
        self.num_objects = num_nodes
        self.countCat = {}

        for i in range(0, self.numAtt):
            c = Counter(attributes[:,i])
            self.countCat[i] = c
            self.num_cat += len(c.keys())

        placesBefore = 0
        self.startIndex = np.zeros(self.numAtt)
        for i in range(1, len(self.startIndex)):
            placesBefore += len(self.countCat[i-1].keys())
            self.startIndex[i] = placesBefore

        self.weightFactors = np.zeros(self.num_g + self.numAtt)
        overallWeight = np.zeros(self.num_g + self.numAtt)
        maxWeight = 0.0
        maxIndex = -1
        
        for i in range(0, self.num_g):
            overallWeight[i] = len(np.asarray(np.where(self.adj[i]>0)[0]))  
            if overallWeight[i] > maxWeight:
                maxWeight = overallWeight[i]
                maxIndex = i
   
        for i in range(0, self.numAtt):
            for j in range(0, len(self.countCat[i].keys())):
                overallWeight[self.num_g + i] += self.countCat[i][j]
                
                if overallWeight[self.num_g + i] > maxWeight:
                    maxWeight = overallWeight[self.num_g + i]
                    maxIndex = self.num_g + i 

        for i in range(0, self.numAtt):
            if i < self.numAtt - self.clusters:
                overallWeight[self.num_g + i] += self.countCat[i][1]
            else:
                overallWeight[self.num_g + i] += self.countCat[i][1] * self.clusters

        for i in range(0, len(self.weightFactors)):
            self.weightFactors[i] = overallWeight[maxIndex] / overallWeight[i]
          
        self.sumWeights = np.zeros(self.num_objects)
        for i in range(0, self.num_objects):
            for j in range(0, self.num_g):
                self.sumWeights[i] += (len(np.asarray(np.where(self.adj[j][i]>0)[0])) * self.weightFactors[j])
            for k in range(0, self.numAtt):
                if self.attributes[i][k] > -1:
                    self.sumWeights[i] += (self.weightFactors[self.num_g + k])


dataset = ReadGraph(graphName)

graphs = dataset.graphs
node_attr = dataset.node_attr
num_atts = dataset.atts
num_nodes = dataset.nodes
adj = dataset.adj
clusters = dataset.clusters
dim = dataset.dim
iterations = dataset.iterations
extra = dataset.extraiter

m = mixedSpectral(graphs, adj, num_nodes , False, node_attr, num_atts, clusters)

startIndex = m.startIndex
sumWeights = m.sumWeights
weightFactors = m.weightFactors

countCat = m.countCat
count = np.array(list(countCat.items()))

countAtt = np.zeros((len(count), len(count[0])))
cat_count = int(countAtt.shape[0]*countAtt.shape[1])             

for i in range(0, countAtt.shape[0]):
    for j in range(0, countAtt.shape[1]):
        countAtt[i][j] = count[i][1][j]
num_cat = countAtt.shape[0]*count.shape[1]
    
emb = np.zeros((num_nodes,33))
gt = dataset.gt

for k in range(clusters):
    for i in countCat:
        for j in countCat[i]:
            if j < 0:
                del(countCat[i][j])
                break
                
count = np.array(list(countCat.items()))

countAtt = np.zeros((len(count), len(count[0])))
cat_count = int(countAtt.shape[0]*countAtt.shape[1])             

for i in range(0, countAtt.shape[0]):
    for j in range(0, countAtt.shape[1]):
        countAtt[i][j] = count[i][1][j]
num_cat = countAtt.shape[0]*count.shape[1]

train_idx = dataset.train_idx

spec = [
        ('clusters', typeof(clusters)),
        ('attributes', typeof(node_attr)),
        ('attributeLabelNumber', typeof(num_atts)),
        ('countCat',typeof(countAtt)),
        ('attributeLabelEmbedding', typeof(emb)),
        ('numNodes', typeof(num_nodes)),
        ('nodeEmbedding', typeof(emb)),
        ('startIndex', typeof(startIndex)),
        ('d', typeof(dim)),
        ('iterr', typeof(iterations)),
        ('extraiter', typeof(extra)),    
        ('num_g', typeof(graphs)),
        ('adj', typeof(adj)),
        ('weightFactors', typeof(weightFactors)),
        ('sumWeights', typeof(sumWeights)),
        ('cat_count', typeof(cat_count)), 
        ('train_idx', typeof(train_idx))
    ]


# Numba class - compute the node embeddings, node attribute embeddings, and node labels embeddings
@jitclass(spec)
class Embeddings(object):  
    def __init__(self, num_g, adj, numNodes, attributes, numAtt, d, iterr, extraiter, cat_count, countCat, startIndex, weightFactors, sumWeights, clusters, train_idx):
        self.num_g = num_g
        self.adj = adj
        self.attributeLabelNumber = numAtt
        self.d = d
        self.iterr = iterr
        self.extraiter = extraiter 
        self.startIndex = startIndex
                
        self.numNodes = numNodes
        self.attributes = attributes
        self.countCat = countCat
        self.cat_count = cat_count
		
        self.weightFactors = weightFactors
        self.sumWeights = sumWeights
             
        self.clusters = clusters
        self.nodeEmbedding = np.zeros((self.numNodes, self.d))
        self.attributeLabelEmbedding = np.zeros((self.cat_count, self.d))
        
        self.train_idx = train_idx
        
        self.initLoop() 
        

    # The objective function of our proposed method
    def Objective(self):
        dist = 0.0
        cost = np.zeros(self.num_g + self.attributeLabelNumber)
        for i in range(0, self.num_g):
            if i < self.num_g-1: # here we consider all given pairs and the 2-hop proximity pairs
                dist = 0.0
                for j in range(0, self.numNodes):
                    edges = np.asarray(np.where(self.adj[i][j]>0)[0])
                    for e in edges:
                        if e != j:
                            for l in range(0, self.d):
                                dist += self.weightFactors[i] * ((self.nodeEmbedding[j][l] - self.nodeEmbedding[int(e)][l]) ** 2) / self.sumWeights[j]
                            cost[i] += dist
            else: # here we consider all negative pairs
                dist = 0.0
                for j in range(0, self.numNodes):
                    edges = np.asarray(np.where(self.adj[i][j]>0)[0])
                    for e in edges:
                        if e != j:
                            for l in range(0, self.d):
                                dist += self.weightFactors[i] * ((self.nodeEmbedding[j][l] - self.nodeEmbedding[int(e)][l]) ** 2) / self.sumWeights[j]
                            cost[i] -= dist

        for i in range(0, self.numNodes):
            for j in range(0, self.attributeLabelNumber-self.clusters): # here we consider only node attributes
                if self.attributes[i][j]>-1:
                    dist = 0.0
                    for l in range(0, self.d):   
                        dist += self.weightFactors[self.num_g + j] * ((self.nodeEmbedding[i][l] - self.attributeLabelEmbedding[int(self.startIndex[j]) + int(self.attributes[i][j])][l]) ** 2) 
                    cost[self.num_g + j] += dist
            
            if i in self.train_idx: # here we consider only labeled nodes
                for j in range(self.attributeLabelNumber-self.clusters, self.attributeLabelNumber):
                    if self.attributes[i][j]==1:
                        dist = 0.0
                        for l in range(0, self.d):   
                            dist += self.weightFactors[self.num_g + j] * ((self.nodeEmbedding[i][l] - self.attributeLabelEmbedding[int(self.startIndex[j]) + int(self.attributes[i][j])][l]) ** 2) 
                        cost[self.num_g + j] += dist
                    elif self.attributes[i][j]==0:
                        dist = 0.0
                        for l in range(0, self.d):   
                            dist += self.weightFactors[self.num_g + j] * ((self.nodeEmbedding[i][l] - self.attributeLabelEmbedding[int(self.startIndex[j]) + int(self.attributes[i][j])][l]) ** 2) 
                        cost[self.num_g + j] -= dist

        sumCost = 0.0
        for i in range(0, len(cost)):
            sumCost += cost[i]        
        return sumCost    
    
    # We initialize randomly node, node attribute and class label embeddings
    def initLoop(self):  
        random.seed(0)                     
        for i in range(0, self.numNodes):
            for j in range(0, self.d):
                self.nodeEmbedding[i][j] = (random.random()/10) 
                
            for j in range(0, self.d):    
                for k in range(0, self.attributeLabelNumber):
                    if self.attributes[i][k] > -1:
                        self.attributeLabelEmbedding[int(self.startIndex[k]) + int(self.attributes[i][k])][j] +=  self.nodeEmbedding[i][j] / self.countCat[k][int(self.attributes[i][k])]
              
    # Method to apply the modified Gram Schmidt orthonomalization 
    def modifiedGramSchmidt(self, newCoord):
        for j in range(0, self.d):
            for i in range(0, j):
                skalarprod = 0.0
                self_i = 0.0
                proj_vi_vj = 0.0
                for l in range(0, self.numNodes):
                    skalarprod += newCoord[l][i] * newCoord[l][j]
                    self_i += newCoord[l][i] * newCoord[l][i]
                for l in range(0, self.numNodes):
                    proj_vi_vj = (skalarprod / self_i) * newCoord[l][i]
                    newCoord[l][j] = newCoord[l][j] - proj_vi_vj
            
            norm_j = 0.0
            
            for l in range(0, self.numNodes):
                norm_j += newCoord[l][j] * newCoord[l][j]
                
            norm_j = math.sqrt(norm_j)
            
            for l in range(0, self.numNodes):
                newCoord[l][j] = newCoord[l][j] /  norm_j

        return newCoord
     
    # Update the node embeddings    
    def updateNodeCoordinates(self):
        newCoord = np.zeros((self.numNodes, self.d))
        
        for i in range(0, self.numNodes):
            for j in range(0, self.num_g): 
                if j < self.num_g-1: # considering only given layers, and the 2-hop proximity
                    edges = np.asarray(np.where(self.adj[j][i]>0)[0])
                    for e in edges:
                        if e!=i:
                            for l in range(0, self.d):
                                newCoord[i][l] +=  self.weightFactors[j] * 1 * self.nodeEmbedding[int(e)][l] / (self.sumWeights[i])
                else: # considering only negative pairs
                    edges = np.asarray(np.where(self.adj[j][i]>0)[0])
                    for e in edges:
                        if e!=i:
                            for l in range(0, self.d):
                                newCoord[i][l] -=  self.weightFactors[j] * 1 * self.nodeEmbedding[int(e)][l] / (self.sumWeights[i])
            
            for k in range(0, self.attributeLabelNumber-self.clusters): # considering only node attributes
                for l in range(0, self.d):
                    if self.attributes[i][k] > -1:
                        newCoord[i][l] += self.weightFactors[int(self.num_g + k)] * self.attributeLabelEmbedding[int(self.startIndex[k]) + int(self.attributes[i][k])][l]/ self.sumWeights[i]
            
            if i in self.train_idx: # considering only labeled nodes
                for k in range(self.attributeLabelNumber-self.clusters, self.attributeLabelNumber):
                    if self.attributes[i][k] == 1: # if the labeled node belongs to this class k
                        for l in range(0, self.d):
                            newCoord[i][l] += self.weightFactors[int(self.num_g + k)] * self.attributeLabelEmbedding[int(self.startIndex[k]) + int(self.attributes[i][k])][l]/ self.sumWeights[i]
                    elif self.attributes[i][k] == 0: # if the labeled node does not belong to this class k
                        for l in range(0, self.d):
                            newCoord[i][l] -= self.weightFactors[int(self.num_g + k)] * self.attributeLabelEmbedding[int(self.startIndex[k]) + int(self.attributes[i][k])][l]/ self.sumWeights[i]
         
        # the last step of update is orthonormalization 
        
        self.nodeEmbedding = self.modifiedGramSchmidt(newCoord) 
        return self.nodeEmbedding

    # Update the node attribute and class label emebddings
    def updateAttributeCoordinates(self):
        newCoord = np.zeros((self.attributeLabelEmbedding.shape[0], self.d))
    
        for i in range(0, self.numNodes):
            for k in range(0, self.attributeLabelNumber-self.clusters): # update only node attributes
                if self.attributes[i][k] > -1:
                    for j in range(0, self.d):
                        newCoord[int(self.startIndex[k]) + int(self.attributes[i][k])][j] += (self.nodeEmbedding[i][j] / self.countCat[k][int(self.attributes[i][k])])                       
            if i in self.train_idx: 
                for k in range(self.attributeLabelNumber-self.clusters, self.attributeLabelNumber): # update only class labels
                    if self.attributes[i][k] == 1:
                        for j in range(0, self.d):
                            newCoord[int(self.startIndex[k]) + int(self.attributes[i][k])][j] += (self.nodeEmbedding[i][j] / self.countCat[k][int(self.attributes[i][k])])
                    elif self.attributes[i][k] == 0:
                        for j in range(0, self.d):
                            newCoord[int(self.startIndex[k]) + int(self.attributes[i][k])][j] -= (self.nodeEmbedding[i][j] / self.countCat[k][int(self.attributes[i][k])])                  
        self.attributeLabelEmbedding = newCoord
        return self.attributeLabelEmbedding
    
    def run(self):            
        minCost = self.Objective()
        threshold = 0.05
        iteration = 0
        minCostID = 0

        updateObjCoord2 = self.updateNodeCoordinates()
        updateCatCoord2 = self.updateAttributeCoordinates()

        converged = False 
        while converged != True or iteration < self.iterr:
            updateObjCoord1 = self.updateNodeCoordinates()    
            updateCatCoord1 = self.updateAttributeCoordinates()

            actualCost = self.Objective()

            # To have an output for each iteration you can uncomment the following line
            # print("Iteration = ", iteration, " actualCost = ", actualCost, " minCost = ", minCost, " mincostID = ", minCostID)

            if actualCost < minCost-threshold:
                minCost = actualCost
                minCostID = iteration
                updateObjCoord2 = updateObjCoord1
                updateCatCoord2 = updateCatCoord1
            else:
                if minCostID + self.extraiter > iteration:
                    converged = False
                else:
                    self.nodeEmbedding = updateObjCoord2
                    self.attributeLabelEmbedding = updateCatCoord2
                    converged = True
    
            if iteration>1000:
                self.nodeEmbedding = updateObjCoord2
                self.attributeLabelEmbedding = updateCatCoord2
                converged = True
                print("Terminated after 1000 iterations")
            iteration += 1
        print("Final iteration: ", iteration, " ; Best iteration: ", minCostID)
        print()
        print("Iteration = ", iteration, " actualCost = ", actualCost, " minCost = ", minCost, " mincostID = ", minCostID)
        print()



def classification(X, y, test_ids, val_ids, train_ids):    
    X_test = np.zeros((len(test_ids), X.shape[1]))
    y_test = [0 for i in range(0, len(test_ids))]

    X_val = np.zeros((len(val_ids), X.shape[1]))
    y_val = [0 for i in range(0, len(val_ids))]

    X_train = np.zeros((len(train_ids), X.shape[1]))
    y_train = [0 for i in range(0, len(train_ids))]

    for t in range(len(test_ids)):
        X_test[t] = X[int(test_ids[t])]
        y_test[t] = y[int(test_ids[t])]

    for v in range(len(val_ids)):
        X_val[v] = X[int(val_ids[v])]
        y_val[v] = y[int(val_ids[v])]

    for tr in range(len(train_ids)):
        X_train[tr] = X[int(train_ids[tr])]
        y_train[tr] = y[int(train_ids[tr])]

    mi_f1, ma_f1 = 0, 0    
    ma_best, mi_best = 0.0, 0.0
    ma_list = []
    mi_list = []
    clf = LogisticRegressionCV(max_iter=3000, cv = 10, class_weight='balanced', solver='sag')

    for r in range(0, 10):

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        ma_f1 = f1_score(y_val, y_pred, average="macro")
        mi_f1 = f1_score(y_val, y_pred, average="micro")

        if mi_f1 > mi_best:
            mi_best = mi_f1
            y_pred = clf.predict(X_test)

            ma_f1_best = f1_score(y_test, y_pred, average="macro")
            mi_f1_best = f1_score(y_test, y_pred, average="micro")
        
    ma_list.append(ma_f1_best)
    mi_list.append(mi_f1_best)
    print("Solver = sag")

    print('MA_F1: ', ma_f1_best, "; MA_list = ", ma_list)
    print("MI_F1: ", mi_f1_best, "; MI_list = ", mi_list)
    return np.mean(ma_list), np.mean(mi_list), 'sag';


def run_kmeans(X, y, k, test_ids):
    X_test = np.zeros((len(test_ids), X.shape[1]))
    y_test = [0 for i in range(0, len(test_ids))]

    for t in range(len(test_ids)):
        X_test[t] = X[int(test_ids[t])]
        y_test[t] = y[int(test_ids[t])]
        
    estimator = KMeans(n_clusters=k, n_init=10)

    best_nmi_ = 0
    best_ari_ = 0 
    best_ami_ = 0
    best_dim_ = 2
    for iter in range(2, 12):
        estimator.fit(X_test[:,:int(iter)])
        y_pred = estimator.predict(X_test[:,:int(iter)])
        nmi = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(y_test, y_pred)
        ami = adjusted_mutual_info_score(y_test, y_pred)
        if nmi>best_nmi_:
            best_nmi_ = nmi
            best_ami_ = ami
            best_ari_ = ari
            best_dim_ = iter

    return best_nmi_, best_ari_, best_ami_, best_dim_;
    

def run_kmeans_test_centroid(X, y, k, centroids, test_ids):
    X_test = np.zeros((len(test_ids), X.shape[1]))
    y_test = [0 for i in range(0, len(test_ids))]

    for t in range(len(test_ids)):
        X_test[t] = X[int(test_ids[t])]
        y_test[t] = y[int(test_ids[t])]

    estimator = KMeans(n_clusters=k, init=centroids, n_init=1)
    estimator.fit(X_test)
    y_pred = estimator.predict(X_test)

    nmi = normalized_mutual_info_score(y_pred, y_test, average_method='arithmetic')
    ari = adjusted_rand_score(y_pred, y_test)
    ami = adjusted_mutual_info_score(y_pred, y_test)
            
    return nmi, ari, ami;


def cluster_centroids(nodeEmb, attEmb, gt, clusters, test_ids):
    labelEmb = attEmb[-(clusters*2):,:]
    centroids = np.zeros((clusters,nodeEmb.shape[1]))
    c = 0
    for i in range(1,clusters*2,2):
        for j in range(0,nodeEmb.shape[1]):
            centroids[c][j] = labelEmb[i][j]
        c+=1
    
    nmi1, ari1, ami1, dim1  = run_kmeans(nodeEmb, gt, clusters, test_ids)
    
    nmi4, ari4, ami4 = run_kmeans_test_centroid(nodeEmb, gt, clusters, centroids, test_ids)

    print("Without Centorids")
    print()
    print("(Best Dim) NMI = ", nmi1, " ARI = ", ari1, " AMI = ", ami1, " Dim = ", dim1)

    print("With Centroids Clustering ALL DIMS:")
    print("(All Dims) NMI = ", nmi4, " ARI = ", ari4, " AMI = ", ami4)
    print()
    
    return nmi1, ari1, ami1, dim1, nmi4, ari4, ami4;
    

def run_kmeans_org(X, y, k, test_ids):   
    X_test = np.zeros((len(test_ids), X.shape[1]))
    y_test = [0 for i in range(0, len(test_ids))]

    for t in range(len(test_ids)):
        X_test[t] = X[int(test_ids[t])]
        y_test[t] = y[int(test_ids[t])]
        
    estimator = KMeans(n_clusters=k, n_init=10)

    NMI_list = []
    ARI_list = []
    AMI_list = []
    for i in range(10):
        estimator.fit(X_test)
        y_pred = estimator.predict(X_test)
        nmi = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(y_test,y_pred)
        ami = adjusted_mutual_info_score(y_test, y_pred)
        NMI_list.append(nmi)
        ARI_list.append(ari)
        AMI_list.append(ami)

    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    mean_a = np.mean(ARI_list)
    std_a = np.std(ARI_list)
    mean_ami = np.mean(AMI_list)
    std_ami = np.std(AMI_list)
    print('[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    print('[Clustering] ARI: {:.4f} | {:.4f}'.format(mean_a, std_a))
    print('[Clustering] AMI: {:.4f} | {:.4f}'.format(mean_ami, std_ami))


def run_kmeans_o(X, y, k, test_ids):
    X_test = np.zeros((len(test_ids), X.shape[1]))
    y_test = [0 for i in range(0, len(test_ids))]

    for t in range(len(test_ids)):
        X_test[t] = X[int(test_ids[t])]
        y_test[t] = y[int(test_ids[t])]
        
    estimator = KMeans(n_clusters=k, n_init=10)

    NMI_list = []
    ARI_list = []
    AMI_list = []
    for i in range(10):
        estimator.fit(X_test)
        y_pred = estimator.predict(X_test)
        nmi = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
        ari = adjusted_rand_score(y_test, y_pred)
        ami = adjusted_mutual_info_score(y_test, y_pred)

        NMI_list.append(nmi)
        ARI_list.append(ari)
        AMI_list.append(ami)

    mean_nmi = np.mean(NMI_list)
    std_nmi = np.std(NMI_list)
    mean_ari = np.mean(ARI_list)
    std_ari = np.std(ARI_list)
    mean_ami = np.mean(AMI_list)
    std_ami = np.std(AMI_list)
    print('[Clustering] NMI: {:.4f} | {:.4f}'.format(mean_nmi, std_nmi))
    print('[Clustering] ARI: {:.4f} | {:.4f}'.format(mean_ari, std_ari))
    print('[Clustering] AMI: {:.4f} | {:.4f}'.format(mean_ami, std_ami))
    return mean_nmi, mean_ari, mean_ami;


def store_result(X, A, graphName):
	test_ids = np.sort(loadtxt('data/'+graphName.upper()+'/test_ids_16.txt'))
	val_ids = np.sort(loadtxt('data/'+graphName.upper()+'/val_ids_16.txt'))
	train_ids = np.sort(loadtxt('data/'+graphName.upper()+'/train_ids_16.txt'))

	X_test = np.zeros((len(test_ids), X.shape[1]))
	X_val = np.zeros((len(val_ids), X.shape[1]))
	X_train = np.zeros((len(train_ids), X.shape[1]))
	
	for t in range(len(test_ids)):
		X_test[t] = X[int(test_ids[t])]
		
	for v in range(len(val_ids)):
		X_val[v] = X[int(val_ids[v])]
		
	for tr in range(len(train_ids)):
		X_train[tr] = X[int(train_ids[tr])]
		
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_attribute_label_emb.txt', A)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_test_emb.txt', X_test)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_train_emb.txt', X_train)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_val_emb.txt', X_val)
	np.savetxt('data/' + graphName.upper() +'/' + graphName + '_emb.txt', X)


class_ma = []
class_mi = []

clus_nmi1 = []
clus_ari1 = []
clus_ami1 = []

clus_nmi2 = []
clus_ari2 = []
clus_ami2 = []


print('Graph name: ', graphName)
dataset = ReadGraph(graphName)

if graphName == 'acm':
    k = 100
elif graphName == 'imdb':
    k = 15 
print('K = ', k)

for train_sample in [1,2,3,4,5]: #[8,11,16,17,19] (ACM)  [10,14,16,18,20] (IMDB)
    print()
    print("Train Sample = ", train_sample)
    path = "data/" + graphName.upper() +'/'

    if graphName == 'acm':
        dataset.labels = scipy.io.loadmat(path + 'label.mat')['label']
        dataset.node_attr = scipy.io.loadmat(path + "feature.mat")['feature']
        train_ids = np.sort(loadtxt(path + 'train_ids_'+str(train_sample)+'.txt')).astype(int)
        test_ids = list(np.sort(loadtxt(path + 'test_ids_'+str(train_sample)+'.txt')).astype(int))
        dataset.gt = np.loadtxt(path + "ground_truth.txt")
        dataset.atts = len(dataset.node_attr[0]) 
        
        gt_ = np.zeros((dataset.nodes,3))
        for i in train_ids:
            gt_[int(i)][0] = dataset.labels[int(i)][0]
            gt_[int(i)][1] = dataset.labels[int(i)][1]
            gt_[int(i)][2] = dataset.labels[int(i)][2]
            for j in train_ids:
                if dataset.gt[int(i)] != dataset.gt[j]:
                    dataset.adj[-1][int(i)][int(j)] = 1
                    dataset.adj[-1][int(j)][int(i)] = 1
    
    elif graphName =='imdb':
        mat = scipy.io.loadmat(path + "imdb.mat")
        dataset.labels = mat['label']
        
        dataset.node_attr = scipy.io.loadmat(path + "imdb.mat")['feature']
        dataset.atts = len(dataset.node_attr[0])
        dataset.gt = np.loadtxt(path + "ground_truth.txt")
        test_ids = np.sort(loadtxt(path + 'test_ids_'+str(train_sample)+'.txt')).astype(int)
        train_ids = np.sort(loadtxt(path + 'train_ids_'+str(train_sample)+'.txt')).astype(int)
        gt_ = np.zeros((dataset.nodes,3))
        
        for i in range(0, len(train_ids)):
            gt_[int(train_ids[i])][0] = dataset.labels[int(train_ids[i])][0]
            gt_[int(train_ids[i])][1] = dataset.labels[int(train_ids[i])][1]
            gt_[int(train_ids[i])][2] = dataset.labels[int(train_ids[i])][2]
            for j in range(0, len(train_ids)):
                if dataset.gt[int(train_ids[i])] != dataset.gt[int(train_ids[j])]:
                    dataset.adj[-1][int(train_ids[i])][int(train_ids[j])] = 1
                    dataset.adj[-1][int(train_ids[j])][int(train_ids[i])] = 1

    dataset.node_attr = np.concatenate((dataset.node_attr, gt_), axis=1)
    dataset.atts = len(dataset.node_attr[0])
    dataset.train_idx = train_ids     
    
    graphs = dataset.graphs
    node_attr = dataset.node_attr
    num_atts = dataset.atts
    num_nodes = dataset.nodes
    adj = dataset.adj
    clusters = dataset.clusters
    dim = dataset.dim
    iterations = dataset.iterations
    extra = dataset.extraiter
    train_idx = dataset.train_idx
    dim_cluster = dataset.dim_cluster

    m = mixedSpectral(graphs, adj, num_nodes , False, node_attr, num_atts, clusters)

    startIndex = m.startIndex
    sumWeights = m.sumWeights
    weightFactors = m.weightFactors
    countCat = m.countCat

    count = np.array(list(countCat.items()))
    countAtt = np.zeros((len(count), len(count[0])))
    cat_count = int(countAtt.shape[0]*countAtt.shape[1]) 

    for i in range(0, countAtt.shape[0]):
        for j in range(0, countAtt.shape[1]):
            countAtt[i][j] = count[i][1][j]
    
    num_cat = countAtt.shape[0]*count.shape[1]
    gt = dataset.gt
    
    for kk in range(clusters):
        for i in countCat:
            for j in countCat[i]:
                if j < 0:
                    del(countCat[i][j])
                    break
    
    count = np.array(list(countCat.items()))
    countAtt = np.zeros((len(count), len(count[0])))
    cat_count = int(countAtt.shape[0]*countAtt.shape[1])             
    
    for i in range(0, countAtt.shape[0]):
        for j in range(0, countAtt.shape[1]):
            countAtt[i][j] = count[i][1][j]
    
    num_cat = countAtt.shape[0]*count.shape[1]
    
    w_tot = 0
    for i in range(1, clusters+1):
        w_tot += count[-i][1][1]
    
    for i in range(1, clusters+1):
        weightFactors[-i] = w_tot/ count[-i][1][1]
    
    start = time.time()

    objectEmbeddings = Embeddings(graphs, adj, num_nodes,  node_attr, num_atts, dim, iterations, extra, num_cat, countAtt, startIndex, weightFactors, sumWeights, clusters, train_idx)

    minCost = objectEmbeddings.Objective()
    threshold = 0.05
    iteration = 0
    minCostID = 0
    updateObjCoord2 = objectEmbeddings.updateNodeCoordinates()
    updateCatCoord2 = objectEmbeddings.updateAttributeCoordinates()
    converged = False 

    while converged != True or iteration < objectEmbeddings.iterr:
        updateObjCoord1 = objectEmbeddings.updateNodeCoordinates()    
        updateCatCoord1 = objectEmbeddings.updateAttributeCoordinates()

        actualCost = objectEmbeddings.Objective()
        # print("Iteration = ", iteration, " actualCost = ", actualCost, " minCost = ", minCost, " mincostID = ", minCostID)
        
        if actualCost < minCost-threshold:
            minCost = actualCost
            minCostID = iteration
            updateObjCoord2 = updateObjCoord1
            updateCatCoord2 = updateCatCoord1
        
        else:
            if minCostID + objectEmbeddings.extraiter > iteration:
                converged = False
        
            else:
                objectEmbeddings.nodeEmbedding = updateObjCoord2
                objectEmbeddings.attributeLabelEmbedding = updateCatCoord2
                converged = True
        
        if iteration % 5 == 0:
            new_train_set = []
            new_train_labels = []
            labelEmb = objectEmbeddings.attributeLabelEmbedding[-(clusters*2):,:]
            centroids_ = np.zeros((clusters,objectEmbeddings.attributeLabelEmbedding.shape[1]))
            c = 0
        
            for i in range(1,clusters*2,2):
                for j in range(0,objectEmbeddings.attributeLabelEmbedding.shape[1]):
                    centroids_[c][j] = labelEmb[i][j]
                c+=1
            
            X_t = np.concatenate((objectEmbeddings.nodeEmbedding, centroids_))
            k_ = k + int(iteration%5) 
            
            nbrs = NearestNeighbors(n_neighbors=k_, algorithm='auto').fit(X_t)
            
            point = centroids_
            
            # Find the k nearest neighbors
            distances, indices = nbrs.kneighbors(point)
            
            # Extract the k nearest neighbors
            for c in range(0,clusters):
                test = 0
                for i in indices[c]:
                    if i < num_nodes:
                        if i in test_ids:
                            test+=1
                            if test < 15 + int(iteration%5):
                                new_train_set.append(i)
                                new_train_labels.append(c)
                                test_ids.remove(i)
            
            for i in range(0, len(new_train_set)):
                if new_train_labels[i] == 0:
                    objectEmbeddings.attributes[int(new_train_set[i])][-3] = 1
                elif new_train_labels[i] == 1:
                    objectEmbeddings.attributes[int(new_train_set[i])][-2] = 1
                elif new_train_labels[i] == 2:
                    objectEmbeddings.attributes[int(new_train_set[i])][-1] = 1
            
            new_train = np.concatenate((train_ids, new_train_set))
            new_train = np.sort(new_train.astype(np.int64))
            objectEmbeddings.train_idx = new_train
        
        if iteration>1000:
            objectEmbeddings.nodeEmbedding = updateObjCoord2
            objectEmbeddings.attributeLabelEmbedding = updateCatCoord2
            converged = True
            
        iteration += 1
    
    print()      

    nodeEmb = objectEmbeddings.nodeEmbedding
    attEmb = objectEmbeddings.attributeLabelEmbedding
    
    end = time.time()
    print("Time : ", end-start)
    
    print()
    
    print("\nClassification Task:")
    extraPath = 'data/'
    test_ids = np.sort(loadtxt(extraPath+graphName.upper()+ '/test_ids_'+str(train_sample)+'.txt')).astype(int)
    val_ids = np.sort(loadtxt(extraPath+graphName.upper()+'/val_ids_'+str(train_sample)+'.txt')).astype(int)
    train_ids = np.sort(loadtxt(extraPath+graphName.upper()+'/train_ids_'+str(train_sample)+'.txt')).astype(int)
    
    ma_ac, mi_ac, solv = classification(nodeEmb, gt, test_ids, val_ids, train_ids)  
    class_ma.append(ma_ac)
    class_mi.append(mi_ac)
    print()
    
    print("\nNode Clustering: ")
    nmi1, ari1, ami1, dim1, nmi4, ari4, ami4 = cluster_centroids(nodeEmb, attEmb, gt, clusters,  test_ids)
    clus_nmi1.append(nmi1)
    clus_ari1.append(ari1)
    clus_ami1.append(ami1)
    clus_nmi2.append(nmi4)
    clus_ari2.append(ari4)
    clus_ami2.append(ami4)
    print()

    print("Clustering with dim_cluster = ", dim_cluster)
    run_kmeans_org(nodeEmb[:,:dim_cluster], gt, clusters, test_ids)
    print()
    nmi, ari, ami = run_kmeans_o(nodeEmb[:,:dim_cluster], gt, clusters, test_ids)
    print()

    # store_result(nodeEmb, attEmb, graphName)

print("Classification Mean: ", k)
print("MA-F1 = ", np.mean(class_ma), " std = ", np.std(class_ma), "; MI-F1 = ", np.mean(class_mi), " std = ", np.std(class_mi))
print("Clustering Mean 1: ", k)
print("NMI = ", np.mean(clus_nmi1), " std = ",np.std(clus_nmi1), "; ARI = ", np.mean(clus_ari1), " std = ", np.std(clus_ari1), "; AMI = ", np.mean(clus_ami1), " std = ", np.std(clus_ami1))
print("Clustering Mean 2: ", k)
print("NMI = ", np.mean(clus_nmi2), " std = ",np.std(clus_nmi2), "; ARI = ", np.mean(clus_ari2), " std = ", np.std(clus_ari2), "; AMI = ", np.mean(clus_ami2), " std = ", np.std(clus_ami2))