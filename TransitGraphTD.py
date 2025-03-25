import rustworkx as rx
import geopandas as gpd
import json
import shapely
import os
from tqdm import tqdm
from scipy.spatial import KDTree
import numpy as np
from functools import lru_cache
from TransitGraphStatic import EnhTransitGraph

class PermTransitGraph:
    def __init__(self, trip_info_dir, stops: gpd.GeoDataFrame, k=4, d=0.002):
        graph=rx.PyDiGraph()
        # add nodes
        stops['node_idx']=None
        stops['type']=None
        for i, stop in stops.iterrows():
            if stops['wc_access'] is not None:
                wc=stop['wc_access']
            else:
                wc='yes'
            node_idx = graph.add_node({'stop_id': stop['stoppoint_id'], 'name': stop['stoppoint_name'], 'wheelchair': wc, 'lat': stop['geometry'].y, 'lon': stop['geometry'].x})
            stops.at[i, 'node_idx'] = node_idx
        
        # add static edges
        for filename in tqdm(os.listdir(trip_info_dir), total=len(os.listdir(trip_info_dir))):
            if not filename.endswith(".json"):
                continue
            
            ref = filename.split(".")[0]
            #print(ref)
            if len(ref)<3:
                self.type='bus'
            elif len(ref)>2 and ref[0]=='2':
                self.type='bus'
            elif len(ref)>2 and ref[0]=='8':
                self.type='tram'

            
            file_path = os.path.join(trip_info_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    route_data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Ошибка чтения файла {filename}")
                    continue
            for direction in ["fwd", "bkwd"]:
                stoppoints = route_data.get(f"{direction}Stoppoints", [])  # Защита от KeyError
                for i in range(len(stoppoints)-1):
                    parent=stoppoints[i]['stoppointId']
                    child=stoppoints[i+1]['stoppointId']
                    #print(parent, child)
                    parent_idx=stops.loc[stops['stoppoint_id']==parent]['node_idx'].values[0]
                    child_idx=stops.loc[stops['stoppoint_id']==child]['node_idx'].values[0]
                    
                    parent_name=stops.loc[stops['stoppoint_id']==parent, 'stoppoint_name']
                    child_name=stops.loc[stops['stoppoint_id']==child, 'stoppoint_name']
                    edge_idx=graph.add_edge(parent_idx, child_idx, {'type': self.type, 'ref': ref, 'side': direction, 'from': parent, 'to': child, 'leg_name': f'{parent_name}->{child_name}'})
        print(f'initialized PermTransitGraph')
        self.graph=graph 

        stop_nodes_idx=[node for node in self.graph.node_indices()]

        stop_nodes=[(idx, self.graph[idx]['lat'], self.graph[idx]['lon']) for idx in stop_nodes_idx]

        stop_node_ids, stop_lats, stop_lons = zip(*stop_nodes)
        self.stop_kdtree = KDTree(np.c_[stop_lats, stop_lons])

        for node in self.graph.node_indices():
            lat, lon = self.graph[node]['lat'], self.graph[node]['lon']
            nearest_stops = self.get_kn_stops_node_idx((lat, lon), k, d)
            for stop in nearest_stops:
                self.graph.add_edge(node, stop, {'type': 'interchange', 'traveltime': 2, 'from': node, 'to': stop})
                print(f'{self.graph[node]['name']}->{self.graph[stop]['name']}')
        print(f'initialized interchanges')


        

    @lru_cache(maxsize=None)  # Кэширование результатов
    def get_kn_stops_node_idx(self, point, k, d):
        """
        Найти ближайшие k остановок к точке в пределах d

        :param point: кортеж формата (lon, lat)
        :return: list nearest_idx
        """
        _, nearest_idxs = self.stop_kdtree.query(point, k=k, distance_upper_bound=d)
        #print(nearest_idxs)
        if len(nearest_idxs)>0:
            #print(nearest_idxs, self.stop_node_ids)
            return list(set([nearest_idx for nearest_idx in nearest_idxs if self.graph.has_node(nearest_idx)]))
        else:
            return []
    
class KjaTransitGraph:
    def __init__(self, graph: EnhTransitGraph, stops: gpd.GeoDataFrame, k=4, d=0.002):
        self.graph=graph.graph
        stop_nodes_idx=[node for node in self.graph.node_indices()]

        stop_nodes=[(idx, self.graph[idx]['lat'], self.graph[idx]['lon']) for idx in stop_nodes_idx]

        stop_node_ids, stop_lats, stop_lons = zip(*stop_nodes)
        self.stop_kdtree = KDTree(np.c_[stop_lats, stop_lons])
    @lru_cache(maxsize=None)  # Кэширование результатов
    def get_kn_stops_node_idx(self, point, k, d):
        """
        Найти ближайшие k остановок к точке в пределах d

        :param point: кортеж формата (lon, lat)
        :return: list nearest_idx
        """
        _, nearest_idxs = self.stop_kdtree.query(point, k=k, distance_upper_bound=d)
        #print(nearest_idxs)
        if len(nearest_idxs)>0:
            #print(nearest_idxs, self.stop_node_ids)
            return list(set([nearest_idx for nearest_idx in nearest_idxs if self.graph.has_node(nearest_idx)]))
        else:
            return []
          
    