import rustworkx as rx
import json
import shapely
import osmium
from scipy.spatial import KDTree
import numpy as np
from geopy.distance import geodesic
import geopandas as gpd


class TransitGraph:
    def __init__(self, trips=None, stops=None, s2s=None, speed=27, type='subway', interchanges=None, wc_mode=False, keep_limited=True):
        self.type=type
        self.speed=speed
        self.s2s=[]
        self.stops=[]
        self.trips=[]
        with open(s2s, mode='r', encoding='utf-8') as f:
            self.s2s.extend(json.loads(f.read()))
        with open(stops, mode='r', encoding='utf-8') as f:
            self.stops.extend(json.loads(f.read()))
        with open(trips, mode='r', encoding='utf-8') as f:
            self.trips.extend(json.loads(f.read()))
        
        if interchanges is not None:
            with open(interchanges, mode='r', encoding='utf-8') as f:
                self.interchanges=json.loads(f.read())
        else:
            self.interchanges=None
        graph=rx.PyDiGraph()
        # Add node indices in stops, add nodes in graph
        for stop in self.stops:
            node_idx=graph.add_node({'name': stop['stop_name'], 'id': stop['stop_id'], 'type': self.type, 'geom': stop['stop_shape'], 'wc_access': stop['wheelchair'], 'lat': shapely.from_wkt(stop['stop_shape']).y, 'lon': shapely.from_wkt(stop['stop_shape']).x})
            stop['node_idx']=node_idx
        # Add edge indices in s2s, add edges in graph
        for i in range(len(self.s2s)):
            segment=self.s2s[i]
            parent=self.find_idx_by_id(segment['from'], self.stops)
            child=self.find_idx_by_id(segment['to'], self.stops)
            edge_idx=graph.add_edge(parent, child, {'type': self.type,
                                                    'traveltime': self.set_traveltime(float(segment['length']), speed),
                                                    'trip_id': segment['trip_id'], 
                                                    'ref': segment['trip_ref'], 
                                                    'geom': segment['shape'], 
                                                    'length': float(segment['length']),
                                                    'from': parent,
                                                    'to': child,
                                                    'leg_name': f'{self.find_name_by_id(segment['from'], self.stops)}-{self.find_name_by_id(segment['to'], self.stops)}'})
            self.s2s[i]['edge_idx']=edge_idx
        self.graph=graph
        if wc_mode:
            self.edge_contractor(keep_limited=keep_limited)
        
    # Учитывать limited ключи в тэге wheelchair
    def edge_contractor(self, keep_limited=True, logger=True):
        if logger:
            print(f'Graph, before contraction: |V|={len(self.graph.nodes())}, |E|={len(self.graph.edges())}')
        for trip in self.trips:
            sequence=trip['stop_sequence']
            trip_id=trip['route_id']
            sequence_access_flags=[self.find_access_by_id(s, self.stops) for s in sequence]
            sequence_node_ids=[self.find_idx_by_id(s, self.stops) for s in sequence]
            sequence_edge_ids=[self.find_edge_idx_by_stops(sequence[i], sequence[i+1], trip_id, self.s2s) for i in range(len(sequence)-1)] #Elen = Vlen-1
            
            sequence_edge_ids.append(None)
            sequence_zipped=list(zip(sequence, sequence_access_flags, sequence_node_ids, sequence_edge_ids))
            sequence_new=[]
            # find contracted stop sequence
            for i in range(len(sequence_zipped)):
                s, a, n, e = sequence_zipped[i]
                # /!\ Допилить логику на кросс-платформенных пересадках: если wheelchair=limited & wheelchair:description=cross-platform, тогда добавить кросс-платформенную пересадку и последующую за ней для смены направления
                if [s, a, n, e] not in sequence_new:
                    if a==1: # if accessed by wc
                        sequence_new.append([s, a, n, e])
                    elif a==2:
                        sequence_new.append([s, a, n, e])
                        sn, an, nn, en = sequence_zipped[i+1]
                        sequence_new.append([sn, an, nn, en])
                    else: pass
            if sequence_new!=sequence_zipped: # if contracted stop sequence contains less stops
                # remove edges
                for e in sequence_edge_ids:
                    if e is not None:
                        self.graph.remove_edge_from_index(e['edge_idx'])                
                # remove nodes?
                #self.graph.remove_nodes_from(sequence_node_ids)
                
                # update geometry of contracted stop sequence and add contracted edges
                print(sequence_zipped)
                print(sequence_new)
                for i in range(len(sequence_new)-1):
                    parent=sequence_new[i]
                    child=sequence_new[i+1]
                    print(parent[0], child[0])
                    total_geom=[parent[-1]['shape']]
                    start_index = next(index for index, stop in enumerate(sequence_zipped) if stop[0] == parent)
                    for j in range(start_index+1, len(sequence_zipped)):
                        total_geom.append(shapely.from_wkt(sequence_zipped[j][-1]['shape']))
                        if sequence_zipped[j][0]==child[0]:
                            break
                    merged_line=shapely.union_all(total_geom)
                    sequence_new[i][-1]['geom']=merged_line.wkt
                    sequence_new[i][-1]['length']=merged_line.length
                    sequence_new[i][-1]['from']=parent[i]['stop_id']
                    sequence_new[i][-1]['to']=child[i]['stop_id']

                    self.graph.add_edge(parent, child, {'type': self.type,
                                                    'traveltime': self.set_traveltime(float(sequence_new[i][-1]['length']), self.speed),
                                                    'trip_id': sequence_new[i][-1]['trip_id'], 
                                                    'ref': sequence_new[i][-1]['trip_ref'], 
                                                    'geom': sequence_new[i][-1]['shape'], 
                                                    'length': float(sequence_new[i][-1]['length']),
                                                    'from': sequence_new[i][-1]['from'],
                                                    'to': sequence_new[i][-1]['to'],
                                                    'leg_name': f'{self.find_name_by_id(sequence_new[i][-1]['from'], self.stops)}-{self.find_name_by_id(sequence_new[i][-1]['to'], self.stops)}'})
        if logger:
            print(f'Graph, after contraction: |V|={len(self.graph.nodes())}, |E|={len(self.graph.edges())}')
    def set_traveltime(self, length, speed):
        ms=(speed*1000)/60
        traveltime=round(length/ms, 3)
        return traveltime
    
    def find_edge_idx_by_stops(self, efrom, eto, trip_id, s2s):
        for segment in s2s:
            if trip_id==segment['trip_id'] and efrom==segment['from'] and eto==segment['to']:
                return segment
    def find_idx_by_id(self, stop_id, stops):
        for stop in stops:
            if stop_id==str(stop['stop_id']):
                return stop['node_idx']
    def find_name_by_id(self, stop_id, stops):
        for stop in stops:
            if stop_id==str(stop['stop_id']):
                return stop['stop_name']
    def find_access_by_id(self, stop_id, stops, keep_limited=True):
        for stop in stops:
            if stop_id==str(stop['stop_id']):
                if stop['wheelchair']=='no':
                    return 0
                else:
                    if keep_limited:
                        if stop['wheelchair']=='cross-platform':
                            return 2
                        else:
                            return 1
                    else:
                        return 1

class PedestrianGraphHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.edges = []
        self.nodes = {}

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)
        #print(self.nodes[n.id])

    def way(self, w):
        # Filter ways based on pedestrian tags
        # /!\ Add condition if include steps in pedestrian graph

        # Красноярск хреново затегирован для пешехода, поэтому придется использовать весь граф
        #if any(tag.k in ["highway"] and tag.v in ["footway", "path", "pedestrian", "sidewalk", "living_street", 'service', "steps"] for tag in w.tags):
        if any(tag.k in ["highway"] for tag in w.tags):
            for i in range(len(w.nodes) - 1):
                self.edges.append((w.nodes[i].ref, w.nodes[i + 1].ref))
class PedestrianGraph():
    def __init__(self, pbf, walking=2):
        handler = PedestrianGraphHandler()
        try:
            handler.apply_file(pbf, locations=True)
            nodes = handler.nodes
            edges = handler.edges
            self.graph=rx.PyDiGraph()
            node_idx = {node_id: self.graph.add_node({'name': 'None', 'id': 'null', 'type': 'pedestrian', 
                                                      'geom': shapely.to_wkt(shapely.geometry.Point(lon, lat)),
                                                      'wc_access': 'yes', 'lat': lat, 'lon': lon}) for node_id, (lat, lon) in nodes.items()}
            for src, dst in edges:
                if src in node_idx and dst in node_idx:
                    src_coords = nodes[src]
                    dst_coords = nodes[dst]
                    distance = geodesic(src_coords, dst_coords).meters
                    self.graph.add_edge(node_idx[src], node_idx[dst], {'type': 'pedestrian', 'traveltime': distance / ((walking*1000)/60), 
                                                                       'geom': shapely.geometry.LineString([src_coords[::-1], dst_coords[::-1]]).wkt, 
                                                                       'length': distance, 'from': src, 'to': dst})
                    self.graph.add_edge(node_idx[dst], node_idx[src], {'type': 'pedestrian', 'traveltime': distance / ((walking*1000)/60), 
                                                                       'geom': shapely.geometry.LineString([dst_coords[::-1], src_coords[::-1]]).wkt, 
                                                                       'length': distance, 'from': dst, 'to': src})

            # Находим изолированные вершины (без входящих и исходящих рёбер)
            isolated_nodes = [
                node for node in self.graph.node_indices()
                if self.graph.in_degree(node) == 0 and self.graph.out_degree(node) == 0
            ]

            # Удаляем изолированные вершины
            for node in isolated_nodes:
                self.graph.remove_node(node)
        except:
            raise UserWarning('Something wrong with pbf file')

class EnhTransitGraph:
    def __init__(self, graphs, pedestrian, walking=3):
        self.walking=walking
        # Сначала объединить транспортные графы в Enh
        if len(graphs)==1:
            self.graph=graphs[0]
            raise UserWarning("Only one transitgraph provided")
        else:
            self.graph=rx.digraph_union(graphs[0], graphs[1])
            for i in range(2, len(graphs)):
                print(i)
                self.graph=rx.digraph_union(self.graph, graphs[i])

        # Потом объединить Enh и Pedestrian
        self.graph=rx.digraph_union(self.graph, pedestrian)
        
        # После объединения вершины получили новые node_idx, отфильтровать между собой type=pedestrian и остальные type
        stop_nodes_idx=[]
        pedestrian_nodes_idx=[]
        for node in self.graph.node_indices():
            data=self.graph[node]
            if data['type']=='pedestrian':
                pedestrian_nodes_idx.append(node)
            else:
                stop_nodes_idx.append((node, self.graph[node]))
        
        # Добавить коннекторы
        pedestrian_nodes=[(idx, self.graph[idx]['lat'], self.graph[idx]['lon']) for idx in pedestrian_nodes_idx]
        pedestrian_node_ids, pedestrian_lats, pedestrian_lons = zip(*pedestrian_nodes)
        kdtree = KDTree(np.c_[pedestrian_lats, pedestrian_lons])
        
        # 2. Поиск ближайших вершин пешеходного графа к остановкам транспортного графа
        for stop_node_id, stop_data in stop_nodes_idx:
            stop_coords = (shapely.from_wkt(stop_data['geom']).y, shapely.from_wkt(stop_data['geom']).x)

            # Поиск ближайшей вершины пешеходного графа
            _, nearest_idx = kdtree.query(stop_coords)
            nearest_pedestrian_node_id = pedestrian_node_ids[nearest_idx]
            #print(stop_node_id, nearest_pedestrian_node_id)
            # Добавление рёбер пересадки (в обе стороны)
            self.graph.add_edge(stop_node_id, nearest_pedestrian_node_id, {"type": "connector", 'traveltime': 0.1, 'from': stop_node_id, 'to': nearest_pedestrian_node_id})
            self.graph.add_edge(nearest_pedestrian_node_id, stop_node_id, {"type": "connector", 'traveltime': 0.1, 'from': nearest_pedestrian_node_id, 'to': stop_node_id})
    def vizard(self, type='stations'):
        if type=='stations':
            stations=[]
            for node in self.graph.node_indices():
                data=self.graph[node]
                data['node_idx']=node
                if data['type']!='pedestrian':
                    stations.append(data)
            stations_gdf=gpd.GeoDataFrame(stations)
            stations_gdf['geom']=stations_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
            stations_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
            return stations_gdf
