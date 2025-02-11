from collections import defaultdict, deque
import rustworkx as rx
from functools import lru_cache

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
        trips=[e[2]['trip_id'] for e in self.graph.out_edges(stop)]
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
            next_stops = [
                (e[1], e[2]['traveltime'] * 60)  # (остановка, время в пути)
                for e in self.graph.out_edges(current_stop)
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

        for _ in range(self.max_transfers+1):
            next_queue = set()
            while queue:
                # забрали остановку
                stop = queue.popleft()
                # найти маршруты этой остановки
                stop_trips=self.get_trips_by_stop(stop)
                for trip in stop_trips:
                    # найти все остановки, достижимые этой поездкой
                    available_stops = self.get_available_stops(stop, trip)
                    #print(trip, self.get_available_stops(stop, trip))
                    # нашли. теперь надо добавить всё в arrival_times
                    for available_stop in available_stops:
                        if available_stop[0] not in arrival_times.keys(): # если остановки нет в словаре
                            arrival_times[available_stop[0]] = available_stop[1]
                        elif available_stop[0] in arrival_times.keys() and available_stop[1]<arrival_times[available_stop[0]]: # если остановка есть и метка прибытия меньше чем была
                            arrival_times[available_stop[0]] = available_stop[1]
                        # по каждому маршруту у нас есть те остановки, которые он проходит. это - очередь обработки для следующей итерации
                        next_queue.add(available_stop[0])

                # пешеходное плечо от этой остановки
                # find connector
                for edge in self.graph.out_edges(stop):
                    edata=self.graph.get_edge_data(*edge)
                    if edata['type']=='connector':
                        connector=edge
                        connected_pedestrian_node=edge[1]
                        available_pedestrian_nodes=self.bfs_pedestrian(connected_pedestrian_node, cutoff=5)
                        for available_node in available_pedestrian_nodes.keys():
                            if available_node not in arrival_times.keys():
                                arrival_times[available_node]=available_pedestrian_nodes[available_node]
                            elif available_node in arrival_times.keys() and available_pedestrian_nodes[available_node]<arrival_times[available_node]:
                                arrival_times[available_node]=available_pedestrian_nodes[available_node]
            queue=deque(next_queue)
        return arrival_times
    def bfs_pedestrian(self, node, cutoff=5):

        arrival_times = defaultdict(lambda: float('inf'))
        arrival_times[node] = 0  # Время старта

        queue = deque()
        queue.append(node)

        while queue:
            next_queue = set()
            current_node = queue.popleft()
            current_time = arrival_times[current_node]
            out_edges=[edge for edge in self.graph.out_edges(current_node) if edge[2]['type']=='pedestrian']
            for out_edge in out_edges:
                out_time = current_time + out_edge[2]['traveltime']
                out_node = out_edge[1]
                arrival_times[out_node] = out_time
                if out_time<cutoff:
                    next_queue.add(out_node)
            queue=deque(next_queue)
        return arrival_times

    def find_nearest_node(self, start):
        """Нахождение ближайшего узла к заданным координатам"""
        # В данном примере просто возвращаем первый узел
        return self.stop_ids[0] if self.stop_ids else None
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
test2()