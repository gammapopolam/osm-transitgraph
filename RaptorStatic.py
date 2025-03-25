from collections import defaultdict, deque
import rustworkx as rx
from functools import lru_cache
import geopandas as gpd
import shapely
from heapq import heappush, heappop
from geopy.distance import geodesic
from TransitGraphStatic import EnhTransitGraph
import heapq
from tqdm import tqdm

class RaptorRouter:
    def __init__(self, graph: EnhTransitGraph, pedestrian=False):
        """
        Инициализация маршрутизатора

        :param graph: Enhanced граф транспортной сети
        """
        self.enh = graph
        self.graph = graph.graph
        self.get_stop_ids()
        #print('stop_ids:', self.stop_ids)
        self.get_trip_ids()
        #print('trip_ids:', self.trip_ids)
        self.pedestrian = pedestrian

    def get_trip_ids(self):
        """Получение уникальных идентификаторов маршрутов"""
        self.trip_ids = set()
        for edge in self.graph.edges():
            if 'trip_id' in edge:
                self.trip_ids.add(edge['trip_id'])

    def get_stop_ids(self):
        """Получение уникальных идентификаторов остановок"""
        self.stop_ids = []
        for node in self.graph.nodes():
            if 'id' in node:
                self.stop_ids.append(node['id'])

    def get_trips_by_stop(self, stop):
        trips=[e[2]['trip_id'] for e in self.graph.out_edges(stop) if 'trip_id' in e[2].keys()]
        return trips
    
    @lru_cache(maxsize=None)  # Кэширование результатов
    def get_available_stops(self, stop, trip_id):
        """
        Находит все остановки, достижимые из текущей остановки по заданному маршруту.

        :param stop: Текущая остановка
        :param trip_id: Идентификатор маршрута
        :return: Список пар (остановка, время прибытия)
        """
        available_stops = []
        current_stop = stop
        visited_stops = set()  # Для проверки зацикленности
        total_time = 0  # Общее время с начала маршрута

        while True:
            # Проверяем, чтобы не зациклиться
            if current_stop in visited_stops:
                break  # Прерываем, если остановка уже посещена
            visited_stops.add(current_stop)

            # Ищем следующую остановку на маршруте
            out_edges = [e for e in self.graph.out_edges(current_stop) if 'trip_id' in e[2].keys()]
            next_stops = [
                (e[1], e[2]['traveltime'])  # (остановка, время в пути)
                for e in out_edges
                if e[2]['trip_id'] == trip_id
            ]

            if not next_stops:
                break  # Нет следующей остановки на маршруте

            next_stop, travel_time = next_stops[0]
            total_time += travel_time
            available_stops.append((next_stop, total_time))
            current_stop = next_stop

        return available_stops

    def RAPTOR_source_all_destination_transitlabels(self, start, pedestrian, max_transfers=1, pedestrian_cutoff=5, k_stops=4, bandwidth=0.003, wc_access='all'):
        """Нахождение достижимых вершин из остановки
        
        :param start: точка начала (lon, lat)
        :param max_transfers: Максимальное число пересадок/итераций
        :param pedestrian_cutoff: Время отсечки пешеходного графа от остановки

        :return: arrival_labels: {arrival, transfers, endpoint T/F, type}
        """
        self.max_transfers = max_transfers
            
        arrival_times = defaultdict(lambda: float('inf'))

        arrival_ntransfers = defaultdict(lambda: float('inf'))

        arrival_labels = defaultdict(lambda: float('inf'))
        
        

        nearest_stops=self.enh.get_kn_stops_node_idx(start, k_stops, bandwidth)

        if wc_access == 'all':
            pass
        elif wc_access == 'limited': 
            nearest_stops=[stop_idx for stop_idx in nearest_stops if self.enh.graph[stop_idx]['wc_access'] != 'no']
        elif wc_access == 'wheelchair':
            nearest_stops=[stop_idx for stop_idx in nearest_stops if self.enh.graph[stop_idx]['wc_access'] != 'no' and self.enh.graph[stop_idx]['wc_access'] != 'limited']

        
        if pedestrian==True:
            start_node_idx = self.enh.get_nearest_node_idx(start)
        # довести точку старта к остановкам. BFS - для изохроны, A*/Djikstra - для минимума памяти. закинуть в arrival_times
        #TODO: поиск конечных вершин - вершин, из которых нет выходящих ребер != pedestrian, connector, interchange
        # уже после нахождения меток в arrival_times
        for stop in nearest_stops:
            self.start_stop=stop
            # Начальная и конечная вершины
            if pedestrian:
                start_node = start_node_idx
                target_node = self.start_stop
                weight_fn = lambda edge: edge["traveltime"]
            # Запуск A*
                path, cost = self.a_star(start_node, target_node, weight_fn, self.heuristic)
            else:
                cost = 0
            arrival_times[self.start_stop] = cost  # Время старта
            arrival_ntransfers[self.start_stop] = 0
            #print(f'Calculating {self.start_stop} with {max_transfers} iters, {pedestrian_cutoff} min cutoff')

            queue = deque()
            queue.append((self.start_stop, cost))

            for i in range(self.max_transfers + 1):
                next_queue = set()
                while queue:
                    # Забираем остановку из очереди
                    current_stop, current_time = queue.popleft()
                    if self.graph[current_stop]['type']!='pedestrian':
                        #print(self.graph[stop])
                        # Находим маршруты этой остановки
                        stop_trips = self.get_trips_by_stop(current_stop)
                        for trip in stop_trips:
                            # Находим все остановки, достижимые по этому маршруту

                            #/!\ Добавить ребра пересадки и добавление их в next_queue
                            available_stops = self.get_available_stops(current_stop, trip)
                            # Обновляем времена прибытия
                            for available_stop, travel_time in available_stops:
                                total_time = current_time + travel_time

                                wc_label=self.enh.graph[available_stop]['wc_access']
                                if wc_access == 'all':
                                    pass
                                elif wc_access == 'limited': 
                                    if wc_label != 'no':
                                        pass
                                    else:
                                        break
                                elif wc_access == 'wheelchair':
                                    if wc_label != 'no' and wc_label != 'limited':
                                        pass
                                    else:
                                        break
                                if total_time < arrival_times[available_stop] and i < arrival_ntransfers[available_stop]:

                                    arrival_times[available_stop] = total_time
                                    arrival_ntransfers[available_stop] = i
                                    # Добавляем остановку в очередь для следующей итерации
                                    next_queue.add((available_stop, total_time))
                                    # Пересадки
                                    for edge in self.graph.out_edges(available_stop):
                                        edata = edge[2]
                                        if edata['type']=='interchange':
                                            transferred = edge[1]
                                            wc_label=self.enh.graph[transferred]['wc_access']
                                            if wc_access == 'all':
                                                pass
                                            elif wc_access == 'limited': 
                                                if wc_label != 'no':
                                                    pass
                                                else:
                                                    break
                                            elif wc_access == 'wheelchair':
                                                if wc_label != 'no' and wc_label != 'limited':
                                                    pass
                                                else:
                                                    break
                                            if total_time+edata['traveltime'] < arrival_times[transferred] and i < arrival_ntransfers[transferred]:
                                                arrival_times[transferred] = total_time+edata['traveltime']
                                                arrival_ntransfers[transferred] = i
                                                next_queue.add((transferred, total_time+edata['traveltime']))
                                

                    if pedestrian==True:
                        # Пешеходное плечо от этой остановки
                        for edge in self.graph.out_edges(stop):
                            edata = edge[2]
                            if edata['type'] == 'connector' or edata['type'] == 'pedestrian':
                                connected_pedestrian_node = edge[1]
                                # Находим все пешеходные вершины, достижимые от connected_pedestrian_node
                                available_pedestrian_nodes = self.bfs_pedestrian(connected_pedestrian_node, cutoff=pedestrian_cutoff)
                                for available_node, pedestrian_time in available_pedestrian_nodes.items():
                                    total_time = arrival_times[stop] + pedestrian_time
                                    if total_time < arrival_times[available_node]:
                                        arrival_times[available_node] = total_time
                                        arrival_ntransfers[available_node] = i
                                        # Добавляем пешеходную вершину в очередь для следующей итерации
                                        if total_time <= pedestrian_cutoff:
                                            next_queue.add((available_node, total_time))
                if pedestrian==True:
                    # После обработки всех остановок в текущей очереди, добавляем пешеходные вершины для всех достижимых остановок
                    for stop in list(arrival_times.keys()):  # Используем list для избежания изменения словаря во время итерации
                        # Проверяем, что stop является остановкой (не пешеходной вершиной)
                        if self.graph[stop]['type'] != 'pedestrian':
                            for edge in self.graph.out_edges(stop):
                                edata = edge[2]
                                if edata['type'] == 'connector' or edata['type'] == 'pedestrian':
                                    connected_pedestrian_node = edge[1]
                                    available_pedestrian_nodes = self.bfs_pedestrian(connected_pedestrian_node, cutoff=pedestrian_cutoff)
                                    for available_node, pedestrian_time in available_pedestrian_nodes.items():
                                        total_time = arrival_times[stop] + pedestrian_time
                                        if total_time < arrival_times[available_node]:
                                            arrival_times[available_node] = total_time
                                            arrival_ntransfers[available_node] = i
                                            if total_time <= pedestrian_cutoff:
                                                next_queue.add((available_node, total_time))
                
                # Обновляем очередь для следующей итерации
                queue = deque(next_queue)
        dict_sample = {'arrival': None, 'transfers': None, 'endpoint': False, 'type': None}
        
        for node in arrival_times.keys():
            label = dict_sample.copy()
            label['arrival'] = arrival_times[node]
            label['transfers'] = arrival_ntransfers[node]

            out_transit_nodes = [edge for edge in self.graph.out_edges(node) if edge[2]['type']!='pedestrian' and edge[2]['type']!='connector' and edge[2]['type']!='interchange']
            if len(out_transit_nodes) == 0:
                label['endpoint'] = True

            label['type']=self.graph[node]['type']
            arrival_labels[node] = label

        return arrival_labels
    # Эвристическая функция (например, минимальное возможное время до цели)
    def heuristic(self, node, goal):
        """
        Эвристическая функция для A* (например, расстояние между вершинами).

        :param node: Текущая вершина
        :param goal: Целевая вершина
        :return: Оценка расстояния
        """
        # Здесь можно использовать географическое расстояние или другую метрику
        node_geom=shapely.from_wkt(self.graph[node]['geom'])
        goal_geom=shapely.from_wkt(self.graph[goal]['geom'])
        distance=geodesic((node_geom.y, node_geom.x), (goal_geom.y, goal_geom.x)).kilometers
        return distance

    # Реализация алгоритма A*
    def a_star(self, start, target, weight_fn, heuristic):
        # Приоритетная очередь: (приоритет, текущая вершина, пройденный путь, стоимость пути)
        queue = [(0, start, [], 0)]
        visited = set()

        while queue:
            _, current, path, cost = heapq.heappop(queue)

            if current == target:
                return path + [current], cost

            if current in visited:
                continue
            visited.add(current)

            for edge in self.enh.graph.incident_edges(current):
                neighbor = edge[1] if edge[0] == current else edge[0]
                if neighbor not in visited:
                    edge_data = self.enh.graph.get_edge_data(edge[0], edge[1])
                    new_cost = cost + weight_fn(edge_data)
                    priority = new_cost + heuristic(neighbor, target)
                    heapq.heappush(queue, (priority, neighbor, path + [current], new_cost))

        return None, float("inf")  # Если путь не найден

    
    @lru_cache(maxsize=None)  # Кэширование результатов
    def bfs_pedestrian(self, node, cutoff=5):
        """
        Поиск в ширину (BFS) для пешеходных вершин.

        :param node: Начальная вершина
        :param cutoff: Время отсечки в секундах
        :return: Словарь {вершина: время прибытия}
        """
        arrival_times = defaultdict(lambda: float('inf'))
        arrival_times[node] = 0  # Время старта

        queue = deque()
        queue.append((node, 0))  # (вершина, время прибытия)

        while queue:
            current_node, current_time = queue.popleft()
            
            # Перебираем все исходящие пешеходные ребра
            for edge in self.graph.out_edges(current_node):
                edge_data = edge[2]
                if edge_data['type'] == 'pedestrian':
                    target_node = edge[1]
                    travel_time = edge_data['traveltime']  # Время в секундах
                    new_time = current_time + travel_time
                    
                    # Если время не превышает отсечку и новое время лучше текущего
                    if new_time <= cutoff and new_time < arrival_times[target_node]:
                        arrival_times[target_node] = new_time
                        # Добавляем вершину в очередь только если время не превышает отсечку
                        if new_time < cutoff:
                            queue.append((target_node, new_time))
        
        return arrival_times

    def get_gdf(self, arrival_labels):
        nodes=[]
        for node in arrival_labels.keys():
            nodedata=self.graph[node]
            nodedata['arrival']=arrival_labels[node]['arrival']
            nodedata['transfers']=arrival_labels[node]['transfers']
            nodedata['endpoint']=arrival_labels[node]['endpoint']
            nodedata['raptor_type']=arrival_labels[node]['type']
            nodes.append(nodedata)
        nodes_gdf=gpd.GeoDataFrame(nodes)
        nodes_gdf['geom']=nodes_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
        nodes_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
        return nodes_gdf
    
    def get_td2(self, endpoints):
        subgraph_nodes = endpoints
        subg = self.graph.subgraph(subgraph_nodes)

        sh_p = rx.digraph_floyd_warshall(subg, weight_fn=lambda edge: edge['traveltime'])
        max_diameter = 0
        for i in subgraph_nodes:
            for j in subgraph_nodes:
                if i != j and sh_p[i][j] is not None:
                    if sh_p[i][j] > max_diameter:
                        max_diameter = sh_p[i][j]
        return max_diameter

class Deprecated:
    def source_all_destinations_calculator(self, start_point_id, all_points, max_transfers=3, pedestrian_cutoff=10):
        """
        Нахождение кратчайших путей между одной вершиной и всеми остальными

        :param start_point_id: node_idx начальной точки
        :param all_points: Список точек, которые нужно достичь `{point_id: node_idx}`
        """
        arrival_points = defaultdict(lambda: float('inf'))
        for p in all_points.keys():
            arrival_points[p] = float('inf')
        arrival_points[start_point_id] = 0

        # 1. Найти ближайшие остановки к начальной точке
        nearest_stops = self.find_nearest_stops(start_point_id)

        # 2. Запустить RAPTOR для каждой ближайшей остановки
        for stop in nearest_stops:
            stop_arrival_times = self.source_all_destinations_isochrones(stop, max_transfers=max_transfers, pedestrian_cutoff=pedestrian_cutoff)
            
            # 3. Обновить временные метки для всех точек в all_points
            for point_id, point_node_idx in all_points.items():
                if point_node_idx in stop_arrival_times:
                    total_time = stop_arrival_times[point_node_idx]
                    if total_time < arrival_points[point_id]:
                        arrival_points[point_id] = total_time

        return arrival_points

    def find_nearest_stops(self, start_point_id):
        """
        Найти ближайшие остановки к начальной точке с помощью A*.

        :param start_point_id: node_idx начальной точки
        :return: Список ближайших остановок
        """
        nearest_stops = []
        visited = set()
        queue = [(0, start_point_id)]  # (время, вершина)

        while queue:
            current_time, current_node = heappop(queue)
            if current_node in visited:
                continue
            visited.add(current_node)

            # Если текущая вершина - остановка, добавляем её в список ближайших
            if self.graph[current_node]['type'] != 'pedestrian':
                nearest_stops.append(current_node)
                continue

            # Перебираем все исходящие ребра
            for edge in self.graph.out_edges(current_node):
                target_node = edge[1]
                edge_data = edge[2]
                if edge_data['type'] == 'pedestrian':
                    travel_time = edge_data['traveltime']
                    heappush(queue, (current_time + travel_time, target_node))

        return nearest_stops

    def a_star(self, start, goal):
        """
        Алгоритм A* для поиска кратчайшего пути между двумя вершинами.

        :param start: Начальная вершина
        :param goal: Целевая вершина
        :return: Кратчайший путь и его стоимость
        """
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current), g_score[current]

            for edge in self.graph.out_edges(current):
                neighbor = edge[1]
                edge_data = edge[2]
                tentative_g_score = g_score[current] + edge_data['traveltime']

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None, float('inf')

    def heuristic(self, node, goal):
        """
        Эвристическая функция для A* (например, расстояние между вершинами).

        :param node: Текущая вершина
        :param goal: Целевая вершина
        :return: Оценка расстояния
        """
        # Здесь можно использовать географическое расстояние или другую метрику
        node_geom=shapely.from_wkt(self.graph[node]['geom'])
        goal_geom=shapely.from_wkt(self.graph[node]['geom'])
        distance=geodesic((node_geom.y, node_geom.x), (goal_geom.y, goal_geom.x)).kilometers
        return distance

    def reconstruct_path(self, came_from, current):
        """
        Восстановление пути после выполнения A*.

        :param came_from: Словарь, содержащий информацию о пути
        :param current: Текущая вершина
        :return: Список вершин, представляющих путь
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    def source_to_target_calculator(self):
        """Расчет кратчайшего пути между вершинами по алгоритму RAPTOR"""
        # 1. Найти ближайшиеу к точкам пешеходную вершину
        # 2. Рассчитать путь от найденной пешеходной вершины до ближайшей остановки
        # 3. Запустить RAPTOR. В конце каждого раунда найти ближайшую остановку к финальной вершине и рассчитать пешеходный путь до нее по A*
        # 4. По мере прохождения раундов обновлять временные метки для всех вершин
        pass
    def origin_destination_calculator(self):
        """Расчет временных меток для всех заданных пар вершин"""
        # воспользоваться source_all_destinations_calculator
        # он должен отдавать путь от выбранной точки до всех других - одна строка матрицы
        pass

def test1():
        
    # Создание тестового графа
    G = rx.PyDiGraph()

    # Добавление узлов-остановок
    stops = [
        {'id': 0, 'name': "A"},
        {'id': 1, 'name': "B"},
        {'id': 2, 'name': "C"},
        {'id': 3, 'name': "D"},
    ]
    for stop in stops:
        G.add_node(stop)

    # Добавление ребер (маршрутов и пересадок)
    G.add_edge(0, 1, {'trip_id': 'R1', 'traveltime': 300})  # Маршрут R1: A->B за 5 минут
    G.add_edge(1, 2, {'trip_id': 'R1', 'traveltime': 180})  # B->C за 3 минуты
    G.add_edge(1, 3, {'trip_id': 'R2', 'traveltime': 420})  # B->D за 7 минут (пересадка)
    G.add_edge(2, 3, {'trip_id': 'R3', 'traveltime': 600})  # C->D за 10 минут

    # Инициализация маршрутизатора
    router = RaptorRouter(G, start=0, max_transfers=2, interval=600)
    result = router.main_calculator()

    print(result)  # Вывод времен достижения для каждой остановки

def test2():
    from TransitGraphStatic import TransitGraph, EnhTransitGraph

    bus=TransitGraph(trips=r"D:\osm2gtfs\kryukovo\bus_trips.json", stops=r"D:\osm2gtfs\kryukovo\bus_stops.json", s2s=r"D:\osm2gtfs\kryukovo\bus_s2s.json", speed=21, type='bus')
    print(len(bus.graph.nodes()), len(bus.graph.edges()))
    random_stop=318
    #print(random_stop)
    #print(bus.graph.nodes()[random_stop])
    print([e[2]['ref'] for e in bus.graph.out_edges(random_stop)])
    raptor=RaptorRouter(bus.graph, random_stop, max_transfers=0, interval=480)
    arrivals=raptor.main_calculator()
    
    for stop in arrivals.keys():
        print(bus.graph.nodes()[stop])
        print('traveltime', arrivals[stop])
        print([e[2]['ref'] for e in bus.graph.out_edges(stop)])

def test3():
    from TransitGraphStatic import TransitGraph, EnhTransitGraph, PedestrianGraph
    bus=TransitGraph(trips=r"D:\osm2gtfs\kja\bus_trips.json", stops=r"D:\osm2gtfs\kja\bus_stops.json", s2s=r"D:\osm2gtfs\kja\bus_s2s.json", speed=21, type='bus')
    print('bus', len(bus.graph.nodes()), len(bus.graph.edges()))

    tram=TransitGraph(trips=r"D:\osm2gtfs\kja\tram_trips.json", stops=r"D:\osm2gtfs\kja\tram_stops.json", s2s=r"D:\osm2gtfs\kja\tram_s2s.json", speed=24, type='tram')
    print('tram', len(tram.graph.nodes()), len(tram.graph.edges()))
    
    pedestrian=PedestrianGraph(pbf=r'd:\osm2gtfs\kja\krasnoyarsk.osm.pbf')
    print('pedestrian', len(pedestrian.graph.nodes()), len(pedestrian.graph.edges()))

    enh=EnhTransitGraph([bus.graph, tram.graph], pedestrian.graph)
    print('init enhanced')
    selected_stop=149

    raptor=RaptorRouter(enh.graph, selected_stop, max_transfers=0, interval=480)
    arrivals=raptor.main_calculator()

    for stop in arrivals.keys():
        print(stop, end=' ')
        print(enh.graph.nodes()[stop], end=' ')
        print('traveltime', arrivals[stop], end=' ')
        out_edges=[e for e in enh.graph.out_edges(stop) if e[2]['type']!='pedestrian' and e[2]['type']!='connector']
        print(out_edges)
        print([e[2]['ref'] for e in out_edges])
def test4():
    from TransitGraphStatic import TransitGraph, EnhTransitGraph, PedestrianGraph

    tram=TransitGraph(trips=r"D:\osm2gtfs\ru_spe_osmgrabber\tram_trips.json", stops=r"D:\osm2gtfs\ru_spe_osmgrabber\tram_stops.json", s2s=r"D:\osm2gtfs\ru_spe_osmgrabber\tram_s2s.json", speed=24, type='tram', wc_mode=False, keep_limited=True)
    print('tram', len(tram.graph.nodes()), len(tram.graph.edges()))

    #tram_limited=TransitGraph(trips=r"D:\osm2gtfs\ru_spe_osmgrabber\tram_trips.json", stops=r"D:\osm2gtfs\ru_spe_osmgrabber\tram_stops.json", s2s=r"D:\osm2gtfs\ru_spe_osmgrabber\tram_s2s.json", speed=24, type='tram', wc_mode=True, keep_limited=True)
    #print('tram_limited', len(tram_limited.graph.nodes()), len(tram_limited.graph.edges()))

    bus=TransitGraph(trips=r"D:\osm2gtfs\ru_spe_osmgrabber\bus_trips.json", stops=r"D:\osm2gtfs\ru_spe_osmgrabber\bus_stops.json", s2s=r"D:\osm2gtfs\ru_spe_osmgrabber\bus_s2s.json", speed=18, type='bus')
    print('bus', len(bus.graph.nodes()), len(bus.graph.edges()))

    subway=TransitGraph(trips=r"D:\osm2gtfs\ru_spe_osmgrabber\subway_trips.json", stops=r"D:\osm2gtfs\ru_spe_osmgrabber\subway_stops.json", s2s=r"D:\osm2gtfs\ru_spe_osmgrabber\subway_s2s.json", speed=31, type='subway')
    print('subway', len(subway.graph.nodes()), len(subway.graph.edges()))

    highway_tags = [
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "primary_link",
        "secondary_link",
        "tertiary_link",
        "living_street",
        "pedestrian",
        "footway",
        "bridleway",
        "steps",
        "path",
        "crossing",
    ]
    #pedestrian=PedestrianGraph(pbf=r'd:\osm2gtfs\ru_spe_osmgrabber\saint_petersburg-filter.osm.pbf', tags=highway_tags, walking=3)
    #print('pedestrian', len(pedestrian.graph.nodes()), len(pedestrian.graph.edges()))

    enh=EnhTransitGraph([bus.graph, tram.graph, subway.graph], walking=2)
    print('init enhanced')
    enh.init_interchanges(k=4, d=0.003)

    #enh_limited=EnhTransitGraph([bus.graph, tram_limited.graph], walking=2)
    #print('init enhanced limited')
    #enh_limited.init_interchanges(k=4, d=0.003)

    zones = gpd.read_file('spe_tests.gpkg', layer='spe_zones')
    zones['zone_id']=zones.index
    print('Calculating raptor router for each zone')
    raptor = RaptorRouter(enh)
    res_all = []
    res_wh = []
    k = 4

    for _, z in tqdm(zones.iterrows(), total=zones.shape[0]):
        import math

        # Создаем полигон (пример)
        polygon = z.geometry

        # 1. Находим вершины полигона
        vertices = list(polygon.exterior.coords)

        # 2. Находим центр полигона (центр масс)
        center = polygon.centroid

        # 3. Находим радиус описанной окружности
        radius = max(math.hypot(center.x - x, center.y - y) for x, y in vertices)
        bw = radius + 0.001
        zone_cnt = (z.geometry.centroid.y, z.geometry.centroid.x)
        arrival_labels_all = raptor.RAPTOR_source_all_destination_transitlabels(zone_cnt,
                                                                        max_transfers=k, 
                                                                        pedestrian=False, 
                                                                        pedestrian_cutoff=15, 
                                                                        k_stops=6, bandwidth=bw, wc_access='all')
        arrival_labels_wh = raptor.RAPTOR_source_all_destination_transitlabels(zone_cnt,
                                                                        max_transfers=k, 
                                                                        pedestrian=False, 
                                                                        pedestrian_cutoff=15, 
                                                                        k_stops=6, bandwidth=bw, wc_access='wheelchair')

        #self.topodiam_dict[z.zone_id] = max(arrival_times.values())
        for node in arrival_labels_all.keys():
            ndata = enh.graph[node]
            nodedata=ndata.copy()
            nodedata['arrival'] = arrival_labels_all[node]['arrival']
            nodedata['transfers']=arrival_labels_all[node]['transfers']
            nodedata['endpoint']=arrival_labels_all[node]['endpoint']
            nodedata['raptor_type']=arrival_labels_all[node]['type']
            nodedata['zone_id'] = z.zone_id
            res_all.append(nodedata)

        for node in arrival_labels_wh.keys():
            ndata = enh.graph[node]
            nodedata=ndata.copy()
            nodedata['arrival'] = arrival_labels_wh[node]['arrival']
            nodedata['transfers']=arrival_labels_wh[node]['transfers']
            nodedata['endpoint']=arrival_labels_wh[node]['endpoint']
            nodedata['raptor_type']=arrival_labels_wh[node]['type']
            nodedata['zone_id'] = z.zone_id
            res_wh.append(nodedata)
    if len(res_all)>0 and len(res_wh)>0:
        res_all_gdf = gpd.GeoDataFrame(res_all)
        res_all_gdf['geom']=res_all_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
        res_all_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
        print(f'Saving all with {k} int')
        res_all_gdf.to_file('spe_tests.gpkg', layer=f'test_int{k}_all')

        res_wh_gdf = gpd.GeoDataFrame(res_wh)
        res_wh_gdf['geom']=res_wh_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
        res_wh_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
        print(f'Saving wh with {k} int')
        res_wh_gdf.to_file('spe_tests.gpkg', layer=f'test_int{k}_wh')
#test4()