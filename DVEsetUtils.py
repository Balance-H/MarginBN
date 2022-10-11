#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Common import DiGraph
from TopoSortUtils import TopoOrder
from DVEUtils import DVE
import copy
import numpy as np
import pandas as pd
from functools import reduce


# In[2]:


class DVEset(DiGraph):
    def __init__(self, graph_dict=None):
        super(DVEset, self).__init__(graph_dict)
        self.graph = DiGraph(copy.deepcopy(self._graph_dict))
        self.Gve = DiGraph(copy.deepcopy(self._graph_dict))
        self.G_plus = DiGraph(copy.deepcopy(self._graph_dict))
        self.G_Ac = DiGraph(copy.deepcopy(self._graph_dict))
        self.topoOrder = TopoOrder(self._graph_dict).topoSort()
        self.DVE = DVE
        
    def get_set_min_graph(self, B, topoOrder=None):
        if topoOrder is None:
            topoOrder = self.topoOrder

        tp = topoOrder
        S = [val for val in tp if val in B] # B's nodes topological sorts
        S.sort(reverse=True) # Reverse S,  α(b1) > α(b2) > · · ·> α(bq)
        F_alpha_B = []
        count = 0
        for i in S:
            if count==0: # First iteration the Gve must be original graph.
                Gve = copy.deepcopy(self.Gve)
            Gve, F_alpha_i = self.DVE(Gve._graph_dict).get_min_graph(i, topoOrder)
            for p, c in F_alpha_i:
                F_alpha_B.append([p,c])
            count +=1
        self.G_plus.add_edges(F_alpha_B)
        return self.G_plus, Gve, F_alpha_B
    
    def get_MinIMap(self, A, topoOrder=None):
        if topoOrder is None:
            topoOrder = self.topoOrder
        G_plus_A, Gve_A, F_alpha_A = self.get_set_min_graph(A, topoOrder)
        for a in A:
            self.G_Ac.delete_vertex(a)
        Dc_A = [[p, c] for p,c in Gve_A.all_edges() if [p,c] not in self.G_Ac.all_edges()]
        G_alpha_Ac = Gve_A
        return G_alpha_Ac, Dc_A



