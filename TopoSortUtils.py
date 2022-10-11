#!/usr/bin/env python
# coding: utf-8

# In[67]:


from Common import DiGraph
import numpy as np
import pandas as pd
import copy, gc, os

import networkx as nx
import matplotlib.pyplot as plt


class TopoOrder(DiGraph):
    def __init__(self, graph_dict=None):
        super(TopoOrder, self).__init__(graph_dict)
        self.graph = graph_dict
        self.res_ls = []
        
    def topoSort(self):
        in_degrees = copy.deepcopy(self.in_degrees())
        # out_degrees = copy.deepcopy(self.out_degrees())
        Q = [u for u in in_degrees if in_degrees[u] == 0] # 筛选入度为0的顶点     
        Seq = []
        
        while Q:         
            u = Q.pop()       #默认从最后一个删除         
            Seq.append(u)         
            for v in self.graph[u]:             
                in_degrees[v] -= 1  #移除子节点的入边
                if in_degrees[v] == 0:        
                    Q.append(v) #再次加入下一次迭代生成的入度为0的顶点
        if len(Seq) == len(self.all_vertices()): #输出的顶点数是否与图中的顶点数相等
            return Seq     
        else:
            print("G is not the directed acyclic graph.")
            return None
            
            
    def topoAllSorts(self): 

        # Create a vector to store indegrees of all 
        # vertices. Initialize all indegrees as 0. 
        in_degree = dict((u,0) for u in self.graph)

        # Traverse adjacency lists to fill indegrees of 
        # vertices.  This step takes O(V+E) time 
        for i in self.graph:
            for j in self.graph[i]:
                in_degree[j] += 1

        # Create an queue and enqueue all vertices with 
        # indegree 0 
        queue = []
        for i in self._graph_dict:
            if in_degree[i] == 0:
                queue.append(i)

        self.process_queue(queue[:], copy.deepcopy(in_degree), [], 0, self.res_ls)
        return self.res_ls
        
    def process_queue(self, queue, in_degree, top_order, cnt, res_ls):
        if queue:
            # We have multiple possible next nodes, generate all possbile variations
            for u in queue:

                # create temp copies for passing to process_queue
                curr_top_order = top_order + [u]
                curr_in_degree = copy.deepcopy(in_degree)
                curr_queue = queue[:]
                curr_queue.remove(u)

                # Iterate through all neighbouring nodes 
                # of dequeued node u and decrease their in-degree 
                # by 1 
                for i in self.graph[u]:
                    curr_in_degree[i] -= 1
                    # If in-degree becomes zero, add it to queue 
                    if curr_in_degree[i] == 0:
                        curr_queue.append(i)
                
                self.process_queue(curr_queue, curr_in_degree, curr_top_order, cnt + 1, res_ls)  # continue recursive
        
        elif cnt != len(self.all_vertices()):
            print("There exists a cycle in the graph")
        else:
            
            #Print topological order
            res_ls.append(top_order)
            #print(top_order)



