import rustworkx as rx
import json
import shapely
import osmium
from scipy.spatial import KDTree
import numpy as np
from geopy.distance import geodesic
import geopandas as gpd
from functools import lru_cache


class TransitGraph:
    def __init__(self, trips=None, stops=None, s2s=None, speed=27, type='subway', wc_mode=False, keep_limited=True):
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

        graph=rx.PyDiGraph()
        # Add node indices in stops, add nodes in graph
        for stop in self.stops:
            if self.type == 'bus':
                stop['wheelchair']='yes'
            node_idx=graph.add_node({'name': stop['stop_name'], 
                                     'id': stop['stop_id'], 
                                     'type': self.type, 
                                     'geom': stop['stop_shape'], 
                                     'wc_access': stop['wheelchair'], 
                                     'lat': shapely.from_wkt(stop['stop_shape']).y, 
                                     'lon': shapely.from_wkt(stop['stop_shape']).x})
            stop['node_idx']=node_idx
        # Add edge indices in s2s, add edges in graph
        for i in range(len(self.s2s)):
            segment=self.s2s[i]
            parent=self.find_idx_by_id(segment['from'], self.stops)
            child=self.find_idx_by_id(segment['to'], self.stops)
            #print(f'{parent}-->{child}')
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
    # Контрактор не работает так, как надо. Необходимо в RAPTOR добавить фильтр остановок которые недоступны для МГН  
    def edge_contractor(self, keep_limited=True):
        print(f'{self.type} graph, before contraction: |V|={len(self.graph.nodes())}, |E|={len(self.graph.edges())}')
        
        for trip in self.trips:
            print(trip['route_id'])
            sequence = trip['stop_sequence']
            trip_id = trip['route_id']
            
            # Получаем данные о вершинах и рёбрах
            sa = [self.find_access_by_id(s, self.stops) for s in sequence]
            sn = [self.find_idx_by_id(s, self.stops) for s in sequence]
            se = [self.find_edge_idx_by_stops(self.find_idx_by_id(sequence[i], self.stops), self.find_idx_by_id(sequence[i+1], self.stops), trip_id) for i in range(len(sequence)-1)]
            se.append(None)

            # Проверяем атрибут wheelchair
            if trip['wheelchair'] == 'yes':
                pass
            elif trip['wheelchair'] == 'limited' and keep_limited:
                pass
            else:
                print(f'deleting trip {trip_id}: {trip["wheelchair"]} tag')
                for edge in se:
                    if edge is not None:
                        # Удаляем ребро по индексам вершин и trip_id
                        from_node_idx = self.find_idx_by_id(edge['from'], self.stops)
                        to_node_idx = self.find_idx_by_id(edge['to'], self.stops)
                        edge_idx = self.find_edge_by_trip_id(from_node_idx, to_node_idx, trip_id)
                        if edge_idx is not None:
                            self.graph.remove_edge_by_index(edge_idx)
                continue

            # Формируем список для контракции
            sz = list(zip(range(len(sequence)), sequence, sa, sn, se))
            sz_new = []
            for j in range(len(sz)):
                i, s, a, n, e = sz[j]
                if keep_limited:
                    if a > 0:
                        sz_new.append([i, s, a, n, e])
                else:
                    if a > 1:
                        sz_new.append([i, s, a, n, e])

            # Контракция рёбер
            if len(sz_new) != len(sz):
                for j in range(len(sz_new)-1):
                    start, end = sz_new[j], sz_new[j+1]
                    
                    if start[0] + 1 == end[0]:  # Соседние вершины
                        pass
                    else:  # Есть пропущенные вершины
                        deleting_edges = []
                        for k in range(start[0], end[0]):
                            edge = sz[k][-1]
                            if edge is not None:
                                deleting_edges.append(edge)
                                # Удаляем ребро по индексам вершин и trip_id
                                from_node_idx = edge['from']
                                to_node_idx = edge['to']
                                edge_idx = self.find_edge_by_trip_id(from_node_idx, to_node_idx, trip_id)
                                if edge_idx is not None:
                                    self.graph.remove_edge_from_index(edge_idx)

                        # Создаем новое ребро
                        print(deleting_edges)
                        traveltime = sum(e['traveltime'] for e in deleting_edges)
                        length = sum(e['length'] for e in deleting_edges)
                        geom = shapely.ops.linemerge([shapely.from_wkt(e['geom']) for e in deleting_edges]).wkt
                        ref = deleting_edges[0]['ref']
                        trip_id = deleting_edges[0]['trip_id']
                        lfrom = start[-2]  # Индекс вершины начала
                        lto = end[-2]      # Индекс вершины конца
                        leg_name = f'{self.graph[lfrom]["name"]}-{self.graph[lto]["name"]}'
                        
                        edict = {
                            'type': self.type,
                            'traveltime': traveltime,
                            'trip_id': trip_id,
                            'ref': ref,
                            'geom': geom,
                            'length': length,
                            'from': lfrom,  # Индекс вершины
                            'to': lto,      # Индекс вершины
                            'leg_name': leg_name
                        }
                        self.graph.add_edge(lfrom, lto, edict)

        # Удаление изолированных вершин
        for node in self.graph.node_indices():
            out_d = self.graph.out_degree(node)
            if out_d == 0:
                if keep_limited and (self.graph[node]['wc_access'] in ['limited', 'yes']):
                    pass
                elif not keep_limited and self.graph[node]['wc_access'] == 'yes':
                    pass
                else:
                    self.graph.remove_node(node)

        print(f'{self.type} graph, after contraction: |V|={len(self.graph.nodes())}, |E|={len(self.graph.edges())}')

    def set_traveltime(self, length, speed):
        ms=(speed*1000)/60
        traveltime=round(length/ms, 3)
        return traveltime
    
    def find_edge_idx_by_stops(self, efrom, eto, trip_id):
        #print(efrom, eto)
        for edge in self.graph.edge_indices():
            segment=self.graph.get_edge_data_by_index(edge)
            
            if trip_id==segment['trip_id'] and efrom==segment['from'] and eto==segment['to']:
                return segment
    def find_edge_by_trip_id(self, from_node_idx, to_node_idx, trip_id):
        """Найти ребро по индексам вершин и trip_id."""
        for edge_idx in self.graph.edge_indices():
            edge_data = self.graph.get_edge_data_by_index(edge_idx)
            if (edge_data['from'] == from_node_idx and
                edge_data['to'] == to_node_idx and
                edge_data['trip_id'] == trip_id):
                return edge_idx
        return None
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
                        if stop['wheelchair']=='cross-platform' or stop['wheelchair']=='limited':
                            return 2
                        else:
                            return 1
                    else:
                        return 2

class PedestrianGraphHandler(osmium.SimpleHandler):
    def __init__(self, tags):
        super().__init__()
        self.highway_filter = tags
        self.edges = []
        self.nodes = {}

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)
        #print(self.nodes[n.id])

    def way(self, w):
        # Красноярск хреново затегирован для пешехода, поэтому придется использовать весь граф
        #if any(tag.k in ["highway"] and tag.v in ["footway", "path", "pedestrian", "sidewalk", "living_street", 'service', "steps"] for tag in w.tags):
        if any(tag.k in ["highway"] and tag.v in self.highway_filter for tag in w.tags):
            for i in range(len(w.nodes) - 1):
                self.edges.append((w.nodes[i].ref, w.nodes[i + 1].ref))
class PedestrianGraph():
    def __init__(self, pbf, tags: list, walking=2):
        handler = PedestrianGraphHandler(tags)
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
    def __init__(self, graphs, pedestrian=None, walking=3):
        """Инициализация объединенных транспортных и пешеходного графов
        
        :param graphs: список графов `rx.PyDiGraph`
        :param pedestrian: пешеходный граф
        :param walking: скорость пешехода, км/ч
        """
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

        stop_nodes_idx=[]
        stop_nodes_idx2=[]
        for node in self.graph.node_indices():
            data=self.graph[node]
            if data['type']!='pedestrian':
                stop_nodes_idx2.append(node)
                stop_nodes_idx.append((node, self.graph[node]))
        stop_nodes=[(idx, shapely.from_wkt(self.graph[idx]['geom']).y, shapely.from_wkt(self.graph[idx]['geom']).x) for idx in stop_nodes_idx2]
        stop_node_ids, stop_lats, stop_lons = zip(*stop_nodes)
        stop_kdtree = KDTree(np.c_[stop_lats, stop_lons])
        # Сохранить К дерево остановок для поиска 
        self.stop_kdtree = stop_kdtree
        self.stop_node_ids=stop_node_ids

        if pedestrian is not None:
            # Потом объединить Enh и Pedestrian
            self.graph=rx.digraph_union(self.graph, pedestrian)
            
            # После объединения вершины получили новые node_idx, отфильтровать между собой type=pedestrian и остальные type
            
            pedestrian_nodes_idx=[]
            for node in self.graph.node_indices():
                data=self.graph[node]
                if data['type']=='pedestrian':
                    pedestrian_nodes_idx.append(node)
                
            
            # Добавить коннекторы
            pedestrian_nodes=[(idx, self.graph[idx]['lat'], self.graph[idx]['lon']) for idx in pedestrian_nodes_idx]
            pedestrian_node_ids, pedestrian_lats, pedestrian_lons = zip(*pedestrian_nodes)
            ped_kdtree = KDTree(np.c_[pedestrian_lats, pedestrian_lons])
            
            # Сохранить К дерево пешеходного графа для поиска ближайших вершин
            self.ped_kdtree=ped_kdtree
            self.pedestrian_node_ids=pedestrian_node_ids

            # Поиск ближайших вершин пешеходного графа к остановкам транспортного графа
            for stop_node_id, stop_data in stop_nodes_idx:
                stop_coords = (shapely.from_wkt(stop_data['geom']).y, shapely.from_wkt(stop_data['geom']).x)

                # Поиск ближайшей вершины пешеходного графа
                _, nearest_idx = ped_kdtree.query(stop_coords)
                nearest_pedestrian_node_id = pedestrian_node_ids[nearest_idx]
                #print(stop_node_id, nearest_pedestrian_node_id)
                # Добавление рёбер пересадки (в обе стороны)
                self.graph.add_edge(stop_node_id, nearest_pedestrian_node_id, {"type": "connector", 'traveltime': 0.1, 'from': stop_node_id, 'to': nearest_pedestrian_node_id})
                self.graph.add_edge(nearest_pedestrian_node_id, stop_node_id, {"type": "connector", 'traveltime': 0.1, 'from': nearest_pedestrian_node_id, 'to': stop_node_id})

    def init_interchanges(self, k=4, d=0.003):
        """ Инициализация ребер пересадки в объединенном транспортном графе. 
        Строит ребра `type=interchange` между видами транспорта (subway, commuter, tram, bus) до 2k раз

        :param k: количество ближайших остановок
        :param d: ширина поиска в градусах (0.003 по умолчанию)"""
        # Dirty interchanges
        tram_nodes = [node for node in self.graph.node_indices() if self.graph[node]['type']=='tram']
        subway_nodes = [node for node in self.graph.node_indices() if self.graph[node]['type']=='subway']
        commuter_nodes = [node for node in self.graph.node_indices() if self.graph[node]['type']=='commuter']
        bus_nodes = [node for node in self.graph.node_indices() if self.graph[node]['type']=='bus']
        
        if len(tram_nodes)>0: 
            for tram_node in tram_nodes:
                tlon, tlat = shapely.from_wkt(self.graph[tram_node]['geom']).xy
                tram_point=(tlat[0], tlon[0])
                nearest_stops = self.get_kn_stops_node_idx(tram_point, k, d)
                #print(nearest_stops)
                for stop in nearest_stops:
                    #if self.graph[stop]['type']!='tram':
                    #print(f'init tram {tram_node}, {stop}')
                    #print(f'init tram {self.graph[tram_node]['name']}', end=' ')
                    #print(f'-- {self.graph[stop]['name']}')
                    self.graph.add_edge(tram_node, stop, {'type': 'interchange', 'traveltime': 1, 'from': tram_node, 'to': stop})
                    self.graph.add_edge(stop, tram_node, {'type': 'interchange', 'traveltime': 1, 'from': stop, 'to': tram_node})
        if len(subway_nodes)>0:
            for sw_node in subway_nodes:
                swlon, swlat = shapely.from_wkt(self.graph[sw_node]['geom']).xy
                sw_point=(swlat[0], swlon[0])
                nearest_stops = self.get_kn_stops_node_idx(sw_point, k, d)
                for stop in nearest_stops:
                    if self.graph[stop]['type']!='subway':
                        print(f'init subway {self.graph[sw_node]['name']}--{self.graph[stop]['name']}')
                        self.graph.add_edge(sw_node, stop, {'type': 'interchange', 'traveltime': 10, 'from': sw_node, 'to': stop})
                        self.graph.add_edge(stop, sw_node, {'type': 'interchange', 'traveltime': 10, 'from': stop, 'to': sw_node})
                    else:
                        self.graph.add_edge(sw_node, stop, {'type': 'interchange', 'traveltime': 7, 'from': sw_node, 'to': stop})
                        self.graph.add_edge(stop, sw_node, {'type': 'interchange', 'traveltime': 7, 'from': stop, 'to': sw_node})
        if len(commuter_nodes)>0:
            for c_node in commuter_nodes:
                clon, clat = shapely.from_wkt(self.graph[c_node]['geom'])
                c_point=(clat[0], clon[0])
                nearest_stops = self.get_kn_stops_node_idx(c_point, k, d)
                for stop in nearest_stops:
                    #print(f'init commuter {self.graph[c_node]['name']}--{self.graph[stop]['name']}')
                    self.graph.add_edge(c_node, stop, {'type': 'interchange', 'traveltime': 10, 'from': c_node, 'to': stop})
                    self.graph.add_edge(stop, c_node, {'type': 'interchange', 'traveltime': 10, 'from': stop, 'to': c_node})
        if len(bus_nodes)>0: 
            for b_node in bus_nodes:
                blon, blat = shapely.from_wkt(self.graph[b_node]['geom']).xy
                b_point=(blat[0], blon[0])
                nearest_stops = self.get_kn_stops_node_idx(b_point, k, d)
                
                for stop in nearest_stops:
                    #if self.graph[stop]['type']!='tram':
                    #print(f'init tram {self.graph[tram_node]['name']}--{self.graph[stop]['name']}')
                    if self.graph[stop]['type']=='bus':
                        self.graph.add_edge(b_node, stop, {'type': 'interchange', 'traveltime': 1, 'from': b_node, 'to': stop})
                        self.graph.add_edge(stop, b_node, {'type': 'interchange', 'traveltime': 1, 'from': stop, 'to': b_node})
    @lru_cache(maxsize=None)  # Кэширование результатов
    def get_nearest_node_idx(self, point):
        """
        Найти ближайшую пешеходную вершину к точке

        :param point: кортеж формата (lon, lat)
        :return: nearest_idx
        """
        _, nearest_idx = self.ped_kdtree.query(point)
        if self.graph.has_node(nearest_idx):
            return nearest_idx
    
    @lru_cache(maxsize=None)  # Кэширование результатов
    def get_kn_stops_node_idx(self, point, k, d):
        """
        Найти ближайшие k остановок к точке в пределах d

        :param point: кортеж формата (lon, lat)
        :return: list nearest_idx
        """
        _, nearest_idxs = self.stop_kdtree.query(point, k=k, distance_upper_bound=d)
        if len(nearest_idxs)>0:
            #print(nearest_idxs, self.stop_node_ids)
            return list(set([nearest_idx for nearest_idx in nearest_idxs if self.graph.has_node(nearest_idx)]))
        else:
            return []
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
class Deprecated:
    # Учитывать limited ключи в тэге wheelchair
    def edge_contractor2(self, keep_limited=True, logger=True):
        if logger:
            print(f'Graph, before contraction: |V|={len(self.graph.nodes())}, |E|={len(self.graph.edges())}')
        for trip in self.trips:
            
            sequence=trip['stop_sequence']
            trip_id=trip['route_id']
            print(trip_id)

            sequence_access_flags=[self.find_access_by_id(s, self.stops) for s in sequence]
            sequence_node_ids=[self.find_idx_by_id(s, self.stops) for s in sequence]
            sequence_edge_ids=[self.find_edge_idx_by_stops(sequence[i], sequence[i+1], trip_id, self.s2s) for i in range(len(sequence)-1)] #Elen = Vlen-1
            
            sequence_edge_ids.append(None)
            sequence_zipped=list(zip(sequence, sequence_access_flags, sequence_node_ids, sequence_edge_ids))
            sequence_new=[]
            if trip['wheelchair'] == 'yes':
                pass
            elif trip['wheelchair'] == 'limited' and keep_limited:
                pass
            else:
                print(f'deleting trip {trip_id}: {trip["wheelchair"]} tag')
                for e in sequence_edge_ids:
                    if e is not None:
                        self.graph.remove_edge_from_index(e['edge_idx'])
                continue
            
            # find contracted stop sequence
            for i in range(len(sequence_zipped)):
                s, a, n, e = sequence_zipped[i]
                # /!\ Допилить логику на кросс-платформенных пересадках: если wheelchair=limited & wheelchair:description=cross-platform, тогда добавить кросс-платформенную пересадку и последующую за ней для смены направления
                if [s, a, n, e] not in sequence_new:
                    if a==1: # if accessed by wc
                        sequence_new.append((s, a, n, e))
                    elif a==2:
                        sequence_new.append((s, a, n, e))
                        #sn, an, nn, en = sequence_zipped[i+1]
                        #sequence_new.append([sn, an, nn, en])
                    else: pass
            if sequence_new!=sequence_zipped: # if contracted stop sequence contains less stops
                # remove edges
                for e in sequence_edge_ids:
                    if e is not None:
                        self.graph.remove_edge_from_index(e['edge_idx'])                
                # remove nodes?
                #self.graph.remove_nodes_from(sequence_node_ids)
                
                # update geometry of contracted stop sequence and add contracted edges
                print('zipped', sequence_zipped)
                print('new', sequence_new)
                for i in range(len(sequence_new)-1):
                    parent=sequence_new[i]
                    child=sequence_new[i+1]
                    print(parent[0], child[0])
                    #print(parent[-1])
                    total_geom=[shapely.from_wkt(parent[-1]['shape'])]
                    #start_index = next(index for index, stop in enumerate(sequence_zipped) if stop[0] == parent)
                    start_index = None
                    for index, stop in enumerate(sequence_zipped):
                        if stop[0] == parent[0]:
                            start_index = index
                            break
                    for j in range(start_index+1, len(sequence_zipped)):
                        if sequence_zipped[j][-1] is not None:
                            total_geom.append(shapely.from_wkt(sequence_zipped[j][-1]['shape']))
                            if sequence_zipped[j][0]==child[0]:
                                break
                    if len(total_geom)>1:
                        merged_line=shapely.union_all(total_geom)
                    else:
                        merged_line=total_geom[0]
                    sequence_new[i][-1]['geom']=merged_line.wkt
                    sequence_new[i][-1]['length']=merged_line.length
                    sequence_new[i][-1]['from']=parent[0]
                    sequence_new[i][-1]['to']=child[0]

                    self.graph.add_edge(self.find_idx_by_id(parent[0], self.stops), self.find_idx_by_id(child[0], self.stops), {'type': self.type,
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