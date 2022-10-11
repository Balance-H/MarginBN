#!/usr/bin/env python
# coding: utf-8


import copy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

class DiGraph(object):
    
    _directed = True
    
    def __init__(self, graph_dict=None):
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict
        self.res_ls = []
    
    def edges(self, vertice):
        """ returns a list of all the edges of a vertice"""
        return self._graph_dict[vertice]
        
    def all_vertices(self):
        vertices = list(self._graph_dict)
        vertices_cp = copy.deepcopy(vertices)
        for i in vertices_cp:
            for j in self._graph_dict[i]:
                vertices.append(j)
        """ returns the vertices of a graph as a set that drop duplicates vertices """
        return list(set(vertices))

    def all_edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()
    
    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for parent in self._graph_dict:
            for child in self._graph_dict[parent]:
                #if [neighbour, vertex] not in edges:
                edges.append([parent, child])
        return edges
    
    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []
    
    def delete_vertex(self, vertex):
        if vertex in self._graph_dict:
            del self._graph_dict[vertex]
            
        for parent in self._graph_dict:
            for child in self._graph_dict[parent]:
                if child == vertex:
                    self._graph_dict[parent].remove(child)
    
    def add_edges(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        for parent, child in edge:
            if parent not in self._graph_dict:
                self._graph_dict[parent] = []
            elif child not in self._graph_dict[parent]:
                self._graph_dict[parent].append(child)
            else:
                return None
                #print("{}-->{} has already exist in the graph!".format(parent, child))
    
    def delete_edges(self, edge):
        for parent, child in edge:
            if parent in self._graph_dict and child in self._graph_dict[parent]:
                self._graph_dict[parent].remove(child)
            else:
                return None
    
    def reverse_vertices(self, u, v): # u 和 v 位置对调
    
        self._graph_dict["v_tempppp"] = self._graph_dict.pop(u)
        self._graph_dict[u] = self._graph_dict.pop(v)
        self._graph_dict[v] = self._graph_dict.pop("v_tempppp")
        
        P_ = copy.deepcopy(list(self._graph_dict.keys()))
        for p in list(self._graph_dict.keys()):
            if u in self._graph_dict[p] and v not in self._graph_dict[p]:
                self._graph_dict[p] = [v if i==u else i for i in self._graph_dict[p]]
                P_.remove(p)
            if u in self._graph_dict[p] and v in self._graph_dict[p]:
                P_.remove(p)
                
        for q in P_:
            if v in self._graph_dict[q] and u not in self._graph_dict[q]:
                self._graph_dict[q] = [u if i==v else i for i in self._graph_dict[q]]
    
    def reverse_edge(self, u, v): # 有向边方向的反转
        """
        u-->v transform to v-->u
        params:
        u: the v's parnet node
        v: the u's child node
        """
        self._graph_dict[u].remove(v)
        self._graph_dict[v].append(u)
    
    def parents(self, vertex):
        parents = []
        for i in self._graph_dict:
            for j in self._graph_dict[i]:
                if j == vertex:
                    parents.append(i)
                else:
                    continue
        parents = list(set(parents))
        #if len(parents) == 0:
            #print("{} has not any parent nodes.".format(vertex))
        return parents

    def children(self, vertex):
        if vertex in self._graph_dict:
            return self._graph_dict[vertex]
        else:
            empty_list = []
            return empty_list
            #print("{} has not any child nodes.".format(vertex))

    def neighbors(self, vertices):
        if isinstance(vertices, str):
            vertices = [vertices]
        neighbors_list = []
        for v in vertices:
            neighbors_list.extend(list(set(self.parents(v)).union(set(self.children(v)))))
        neighbors_list = list(set(neighbors_list) - set(vertices))
        """get the neighbors of a vertex"""
        return neighbors_list
    
    def ancestors(self, vertices):
        """
        consider a set of vertex, return the ancestors vertices of the vertices 
        params vertices: given a set of vertex, which is a list, e.g. ['1','2']
        """
        U = list(set(self.all_vertices()) - set(vertices)) # not given vertices u
        ancestors = copy.deepcopy(self.all_vertices()) # Init ancestors

        for u in U:
            for v in vertices:
                # if u-->v, u is one of the ancestors of vertices
                # else, drop u
                if self.find_path(u, v) != None:
                    break
            else:
                ancestors.remove(u)

        return ancestors
        
    def descendants(self, vertices):
        """
        consider a set of vertex, return the descendants vertices of the vertices 
        params vertices: given a set of vertex, which is a list, e.g. ['1','2']
        """
        U = list(set(self.all_vertices()) - set(vertices)) # not given vertices u
        ancestors = copy.deepcopy(self.all_vertices()) # Init ancestors

        for u in U:
            for v in vertices:
                # if v-->u, u is one of the descendants of vertices
                # else, drop u
                if self.find_path(v, u) != None:
                    break
            else:
                ancestors.remove(u)

        return ancestors       
    
    def inducedSubgraph(self, vertices):
        """
        consider a set of vertex, generate the graph-dict of induced subgraph
        params vertices: given a set of vertex, which is a list
        """
        subgraph = copy.deepcopy(self._graph_dict)

        U = list(set(self.all_vertices()) - set(vertices))  # not given vertices u
        # remove the parent vertices which is not conclude in given vertices
        for u in U:
            if u in subgraph.keys() and u not in vertices:
                del subgraph[u]
        
        for k in subgraph.keys():
            subgraph[k] = list(set(subgraph[k]).intersection(set(vertices)))
        return subgraph

    def topologicalOrder(self):
        """
        generate a DAG's topological order.
        """
        in_degrees = copy.deepcopy(self.in_degrees())
        # out_degrees = copy.deepcopy(self.out_degrees())
        Q = [u for u in in_degrees if in_degrees[u] == 0] # 筛选入度为0的顶点     
        Seq = []
        
        while Q:         
            u = Q.pop()       #默认从最后一个删除         
            Seq.append(u)         
            for v in self._graph_dict[u]:             
                in_degrees[v] -= 1  #移除子节点的入边
                if in_degrees[v] == 0:        
                    Q.append(v) #再次加入下一次迭代生成的入度为0的顶点
        if len(Seq) == len(self.all_vertices()): #输出的顶点数是否与图中的顶点数相等
            return Seq     
        else:
            print("G is not the directed acyclic graph.")
            return None
        
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
                for i in self._graph_dict[u]:
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

    def entireTopogicalOrders(self):
        """
        generate entire topological orders of a DAG's.
        """
        # Create a vector to store indegrees of all 
        # vertices. Initialize all indegrees as 0. 
        in_degree = dict((u,0) for u in self._graph_dict)

        # Traverse adjacency lists to fill indegrees of 
        # vertices.  This step takes O(V+E) time 
        for i in self._graph_dict:
            for j in self._graph_dict[i]:
                in_degree[j] += 1

        # Create an queue and enqueue all vertices with 
        # indegree 0 
        queue = []
        for i in self._graph_dict:
            if in_degree[i] == 0:
                queue.append(i)

        self.process_queue(queue[:], copy.deepcopy(in_degree), [], 0, self.res_ls)
        return self.res_ls

    # Return the DAG's indegrees result
    def in_degrees(self):
        result = {}
        for v in self.all_vertices():
            result[v] = len(self.parents(v))
        return result

    # Return the DAG's outdegrees result
    def out_degrees(self):
        result = {}
        for v in self.all_vertices():
            result[v] = len(self.children(v))
        return result
    
    def dict2matrix(self):
        """ convert graph to adjacencyMatrix """
        if self._graph_dict == None:
            df = [[]]
        else:
            vertex_list = list(self.all_vertices())
            n = len(vertex_list)
            df = pd.DataFrame(np.zeros((n,n),dtype=int), index=vertex_list, columns=vertex_list)
            edges = []
            for i in self.all_edges():
                df.loc[i[0],i[1]] = 1
        return df

    #def matrix2dict(self):

    def digraphPlot(self):
        graph = nx.DiGraph()
        for i in self.all_vertices():
            graph.add_node(i)
        edges = []
        for j in self.all_edges():
            graph.add_edge(j[0],j[1])
        pos = nx.layout.fruchterman_reingold_layout(graph)
        nx.draw_networkx(graph, pos=pos)
    
    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj
    
    def __next__(self):
        """ allows us to iterate over the vertices """
        return next(self._iter_obj)

    def __str__(self):
        res = "vertices: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res
    
    def find_path(self, start_vertex, end_vertex, path=None):
        """ find a path from start_vertex to end_vertex 
            in graph """
        if path == None:
            path = []
        graph = self._graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, 
                                               end_vertex, 
                                               path)
                if extended_path: 
                    return extended_path
        return None

    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        """ find all paths from start_vertex to 
            end_vertex in graph """
        graph = self._graph_dict 
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths: 
                    paths.append(p)
        return paths
    
    def markov_blanket(self, vertex):
        res = [] #initilization
        parent_set = self.parents(vertex)
        child_set = self.children(vertex)
        common_parent_set = []
        for c in child_set:
            common_parent_set.extend(self.parents(c))
        res.extend(parent_set)
        res.extend(child_set)
        res.extend(common_parent_set)
        return list(set(res))
   

    def BFS(graph, s):
        #创建队列
        queue = []
        #将起始点s放入队列，假设起始点为‘A’
        queue.append(s)
        #set():创建一个无序不重复元素集，可进行关系测试，删除重		复数据,还可以计算交集、差集、并集
        seen = set()
        #'A'我们已经见过，放入seen
        seen.add(s)
        #当队列不是空的时候
        while len(queue) > 0:
            #将队列的第一个元素读出来，即‘A’
            vertex = queue.pop(0)
         #graph['A']就是A的相邻点：['B','C'],将其储存到nodes
            nodes = graph[vertex]
            #遍历nodes中的元素，即['B','C']
            for w in nodes:
                #如果w没见过
                if w not in seen:
                    queue.append(w)
                    #加入seen表示w我们看见过了
                    seen.add(w)
            print(vertex)


    def connected_components(self):
        
        def BFS(graph, s):
            #创建队列
            walk = set()
            queue = []
            #将起始点s放入队列，假设起始点为‘A’
            queue.append(s)
            #set():创建一个无序不重复元素集，可进行关系测试，删除重		复数据,还可以计算交集、差集、并集
            seen = set()
            #'A'我们已经见过，放入seen
            seen.add(s)
            #当队列不是空的时候
            while len(queue) > 0:
                #将队列的第一个元素读出来，即‘A’
                vertex = queue.pop(0)
             #graph['A']就是A的相邻点：['B','C'],将其储存到nodes
                nodes = graph[vertex]
                #遍历nodes中的元素，即['B','C']
                for w in nodes:
                    #如果w没见过
                    if w not in seen:
                        queue.append(w)
                        #加入seen表示w我们看见过了
                        seen.add(w)
                walk.add(vertex)
            return walk
        
        double_edge_graph = DiGraph(self._graph_dict)
        for e in self.all_edges():
            double_edge_graph.add_edges([e[::-1]]) # ADD double directed edg
        total_components_nodes = []
        nodes = set(self.all_vertices())
        while(nodes):
            r = nodes.pop()
            seen = BFS(double_edge_graph._graph_dict, r)
            total_components_nodes.append(seen)
            nodes = nodes - seen
            
        return total_components_nodes
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        