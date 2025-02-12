from collections import defaultdict, deque
import rustworkx as rx
from functools import lru_cache
import geopandas as gpd
import shapely

class RaptorRouter:
    def __init__(self, graph: rx.PyDiGraph, start, pedestrian=True, max_transfers=2, interval=600):
        """
        Инициализация маршрутизатора

        :param graph: Граф транспортной сети (узлы - остановки, ребра - соединения с атрибутами)
        :param start: Начальная остановка (ID узла или список координат)
        :param max_transfers: Максимальное число пересадок/итераций
        :param interval: Интервал движения транспорта (в секундах)
        """
        self.graph = graph
        self.interval = interval
        self.get_stop_ids()
        #print('stop_ids:', self.stop_ids)
        self.get_trip_ids()
        #print('trip_ids:', self.trip_ids)
        
        if isinstance(start, list):
            self.start_stop = self.find_nearest_node(start)
        else:
            self.start_stop = start
        
        self.max_transfers = max_transfers

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

    def main_calculator(self, pedestrian_cutoff=5):
        """Основной метод расчета временных меток с использованием очереди"""
        arrival_times = defaultdict(lambda: float('inf'))
        arrival_times[self.start_stop] = 0  # Время старта

        queue = deque()
        queue.append(self.start_stop)

        for _ in range(self.max_transfers + 1):
            next_queue = set()
            while queue:
                # Забираем остановку из очереди
                stop = queue.popleft()
                print(self.graph[stop])
                # Находим маршруты этой остановки
                stop_trips = self.get_trips_by_stop(stop)
                for trip in stop_trips:
                    # Находим все остановки, достижимые по этому маршруту
                    available_stops = self.get_available_stops(stop, trip)
                    # Обновляем времена прибытия
                    for available_stop, arrival_time in available_stops:
                        if arrival_time < arrival_times[available_stop]:
                            arrival_times[available_stop] = arrival_time
                            # Добавляем остановку в очередь для следующей итерации
                            next_queue.add(available_stop)
                
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
                                # Добавляем пешеходную вершину в очередь для следующей итерации
                                if total_time <= pedestrian_cutoff:
                                    next_queue.add(available_node)
            
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
                                    if total_time <= pedestrian_cutoff:
                                        next_queue.add(available_node)
            
            # Обновляем очередь для следующей итерации
            queue = deque(next_queue)
        
        return dict(arrival_times)
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

    def find_nearest_node(self, start):
        """Нахождение ближайшего узла к заданным координатам"""
        # В данном примере просто возвращаем первый узел
        return self.stop_ids[0] if self.stop_ids else None
    def vizard(self, arrival_times):
        nodes=[]
        for node in arrival_times.keys():
            nodedata=self.graph[node]
            nodedata['arrival']=arrival_times['node']
            nodes.append(nodedata)
        nodes_gdf=gpd.GeoDataFrame(nodes)
        nodes_gdf['geom']=nodes_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
        nodes_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
        return nodes_gdf

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
#test3()