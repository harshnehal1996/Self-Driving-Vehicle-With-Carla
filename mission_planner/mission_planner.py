import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class MissionPlanner:
    def __init__(self, world_map):
        self.graph = nx.DiGraph()
        self.edges = {}
        self.vertex = {}
        self.wmap = world_map

    def add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def visualize_graph(self):
        plt.figure(figsize=(40, 30), dpi=80, facecolor='w', edgecolor='k')
        nx.draw_networkx(self.graph)
        plt.show()

    def __get_dist(self, u, v):
        u = u.transform.location
        v = v.transform.location
        l = (u.x - v.x) ** 2
        l += (u.y - v.y) ** 2
        l += (u.z - v.z) ** 2
        return l ** 0.5

    def __get_road_length(self, u, v, max_depth=1000):
        length = 0
        while max_depth >= 0:
            prev = u
            y = u.next(2)
            if len(y) > 1:
                for r in y:
                    if r.road_id == v.road_id:
                        return length + self.__get_dist(r, prev)
                return -1
            else:
                r = y[0]
                if r.road_id == v.road_id:
                    return length + self.__get_dist(r, prev)
                elif r.road_id == u.road_id:
                    length += self.__get_dist(r, prev) 
                    u = r
                else:
                    return -1
            max_depth -= 1
        return -1

    def __get_road_length_prev(u, v, max_depth=1000):
        length = 4
        
        y = v.previous(2)
        if len(y) > 1:
            c = False
            for r in y:
                if r.road_id == u.road_id:
                    c = True
                    v = r
                    break
            if not c:
                return -1
        else:
            r = y[0]
            if r.road_id != u.road_id:
                return -1
            v = r
                    
        while max_depth >= 0:
            prev = v
            y = v.previous(2)
            if len(y) > 1:
                c = False
                for r in y:
                    if r.road_id == u.road_id:
                        length += self.__get_dist(r, prev)
                        v = r
                        c = True
                        break
                if not c:
                    return length
            else:
                r = y[0]
                if r.road_id == u.road_id:
                    length += self.__get_dist(r, prev) 
                    v = r
                else:
                    return length
            max_depth -= 1
        return -1

    def build_graph(self):
        top = self.wmap.get_topology()
        self.graph.clear()
        self.edges.clear()
        self.vertex.clear()
        
        for i in range(len(top)):
            u = top[i][0]
            v = top[i][1]
            if self.edges.__contains__((u.road_id, v.road_id)):
                continue
            self.vertex[u.road_id] = u
            self.vertex[v.road_id] = v
            d1 = self.__get_road_length(u, v)
            d2 = self.__get_road_length_prev(u, v)
            if d1 != -1 or d2 != -1:
                self.add_edge(u.road_id, v.road_id)
                self.edges[(u.road_id, v.road_id)] = d1 if d1!=-1 else d2
            else:
                self.add_edge(u.road_id, v.road_id)
                self.edges[(u.road_id, v.road_id)] = self.__get_dist(u, v)

    def __distance(self, v1, v2):
        return self.__get_dist(self.vertex[v1], self.vertex[v2])

    def __assign_weight(self, u, v, attr):
        return self.edges[(u, v)]

    def get_shortest_path(self, u, v):
        return nx.astar_path(self.graph, source=u.road_id, target=v.road_id, heuristic=self.__distance, weight=self.__assign_weight)

    def visualize_shortest_path(self, u, v):
        path = self.get_shortest_path(u, v)
        options_1 = {"edgecolors": "tab:gray", "node_size": 1000, "alpha": 0.9}
        options_2 = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 0.9}
        pos = nx.spring_layout(self.graph, seed=31139452)
        plt.figure(figsize=(40, 30), dpi=80, facecolor='w', edgecolor='k')
        nx.draw(self.graph, pos, nodelist=[k for k, v in self.vertex.items()], node_color='tab:blue', **options_2)
        nx.draw(self.graph, pos, nodelist=[path[0]], node_color='tab:green', **options_1)
        nx.draw(self.graph, pos, nodelist=[path[-1]], node_color='black', **options_1)
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=[(u, v) for u, v in zip(path[:-1], path[1:])],
            width=8,
            alpha=1,
            edge_color="tab:red",
        )

        plt.tight_layout()
        plt.axis("off")
        green_patch = mpatches.Patch(color='green', label='source vertex')
        black_patch = mpatches.Patch(color='black', label='target vertex')
        plt.legend(handles=[green_patch, black_patch], prop={'size' : 37})
        plt.show()

