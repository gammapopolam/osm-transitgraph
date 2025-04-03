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

ACCESSIBILITY_RULES = {
    "all": lambda x: True,
    "limited": lambda x: x != "no" or (isinstance(x, dict) and x['lowFloor']==True),
    "wheelchair": lambda x: x == "yes" or x == 'cross-platform' or (isinstance(x, dict) and x['lowFloor']==True)
}
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

    def source_all_static(self, start, pedestrian, max_transfers=1, pedestrian_cutoff=5, k_stops=4, bandwidth=0.003, wc_access='all'):
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


        nearest_accessible_stops = [stop_idx for stop_idx in nearest_stops if ACCESSIBILITY_RULES[wc_access](self.enh.graph[stop_idx]['wc_access'])]


        #if pedestrian==True:
        #    start_node_idx = self.enh.get_nearest_node_idx(start)
        # довести точку старта к остановкам. BFS - для изохроны, A*/Djikstra - для минимума памяти. закинуть в arrival_times
        #TODO: поиск конечных вершин - вершин, из которых нет выходящих ребер != pedestrian, connector, interchange
        # уже после нахождения меток в arrival_times
        for stop in nearest_accessible_stops:
            self.start_stop=stop
            # Начальная и конечная вершины
            #if pedestrian:
            #    start_node = start_node_idx
            #    target_node = self.start_stop
            #    weight_fn = lambda edge: edge["traveltime"]
            # Запуск A*
            #    path, cost = self.a_star(start_node, target_node, weight_fn, self.heuristic)
            #else:
            #    cost = 0
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
                                if ACCESSIBILITY_RULES[wc_access](self.enh.graph[available_stop]['wc_access']):
                                    pass
                                else:
                                    break
                                if total_time < arrival_times[available_stop] and i < arrival_ntransfers[available_stop]:
                                #if total_time < arrival_times[available_stop]:
                                    arrival_times[available_stop] = total_time
                                    arrival_ntransfers[available_stop] = i
                                    # Добавляем остановку в очередь для следующей итерации
                                    next_queue.add((available_stop, total_time))
                                    # Пересадки
                                    if self.enh.graph[available_stop]['wc_access']=='cross-platform' and wc_access!='all':
                                        cp_flag=True
                                    else:
                                        cp_flag=False
                                    for edge in self.graph.out_edges(available_stop):
                                        edata = edge[2]
                                        if edata['type']=='interchange':
                                            transferred = edge[1]
                                            if ACCESSIBILITY_RULES[wc_access](self.enh.graph[transferred]['wc_access']):
                                                if cp_flag==True:
                                                    if self.enh.graph[available_stop]['type']==self.enh.graph[transferred]['type']:
                                                        pass
                                                    else:
                                                        break
                                                else:
                                                    pass
   
                                            if total_time+edata['traveltime'] < arrival_times[transferred] and i+1 < arrival_ntransfers[transferred]:
                                            #if total_time+edata['traveltime'] < arrival_times[transferred]:
                                                arrival_times[transferred] = total_time+edata['traveltime']
                                                arrival_ntransfers[transferred] = i+1
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

