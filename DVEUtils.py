#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Common import DiGraph
from TopoSortUtils import TopoOrder
import copy
import numpy as np
import pandas as pd


# In[2]:


class DVE(DiGraph):
    def __init__(self, graph_dict=None):
        super(DVE, self).__init__(graph_dict)
        self.graph = graph_dict
        self.Grc = DiGraph(copy.deepcopy(self._graph_dict))
        self.topoOrder = TopoOrder(self._graph_dict).topoSort()
        
    def get_min_graph(self, u, topoOrder=None):
        if topoOrder is None:
            topoOrder = self.topoOrder
        
        tp = topoOrder
        S = self.children(u)
        S = [val for val in tp if val in S] # 按照拓扑序列进行排序 u的子节点
        Dc = [] #存储亏边集
        
        for v in S:
            #if self.Grc.is_min_alpha_child_order(self, u, v)
            # 添加亏边集
            Grc_temp = copy.deepcopy(self.Grc)
            u_parents = self.Grc.parents(u)
            for up in u_parents:
                self.Grc.add_edges([[up,v]])
                if [up,v] not in Dc:
                    Dc.append([up,v])
                   
            v_parents = list(set(self.Grc.parents(v)))
            v_parents.remove(u)   # 去除掉 v 的父节点 u
            for vp in v_parents:
                self.Grc.add_edges([[vp,u]])
                if [vp,u] not in Dc:
                    Dc.append([vp,u])
                    
            self.Grc.reverse_edge(u,v)
            
        Dc_list = [v for v in Dc if v not in self.all_edges()]
        Dc_list = [[p,c] for p, c in Dc_list if [c,p] not in self.all_edges()]
        
        self.Grc.delete_vertex(u)
        return self.Grc, Dc_list
                
        """
        Grc = copy.deepcopy(self._graph_dict)
        u_parents = self.parents(u) 
        # 添加亏边集Dc
        for v in S:
            for up in u_parents:
                self.Grc.add_edge([[up,v]])
            # 每个v点的父节点集
            v_parents = self.parents(v)
            for vp in v_parents:
                self.Grc.add_edge([[vp,u]])
                
        # 反转（u,vi）      
        for v in S:
            self.Grc.reverse_vertices(u, v)
        
        # 删掉顶点u和相关边
        self.Grc.delte_vertex(u)
        return self.Grc._graph_dict
        """
    """    
    def is_min_alpha_child_order(self, u, v):
        result = copy.deepcopy(self._graph_dict[u])
        for child in result:
            if len(self.find_all_paths(u, v)) >= 2: # exist a directed path between (u,v)
                return False
            else:
                return True

    """

