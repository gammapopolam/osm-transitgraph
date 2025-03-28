from collections import defaultdict, deque
import rustworkx as rx
from functools import lru_cache
import geopandas as gpd
import polars as pl
import shapely
from heapq import heappush, heappop
from geopy.distance import geodesic
from TransitGraphTD import PermTransitGraph
from datetime import datetime
import json
import numpy as np
import math

ACCESSIBILITY_RULES = {
    "all": lambda x: True,
    "limited": lambda x: x != "no" or (isinstance(x, dict) and x['lowFloor']==True),
    "wheelchair": lambda x: x == "yes" or (isinstance(x, dict) and x['lowFloor']==True)
}

class RaptorTDRouter:
    def __init__(self, ptg: PermTransitGraph, schedules: pl.DataFrame):
        self.graph = ptg.graph
        self.ptg = ptg
        self.schedules = self.add_stop_sequence(schedules)
        try:
            self.stop_id_to_idx = {self.graph[n]['stop_id']: n for n in self.graph.node_indices()}
        except KeyError:
            self.stop_id_to_idx = {self.graph[n]['id']: n for n in self.graph.node_indices()}
        try:
            self.idx_to_stop_id = {n: self.graph[n]['stop_id'] for n in self.graph.node_indices()}
        except:
            self.idx_to_stop_id = {n: self.graph[n]['id'] for n in self.graph.node_indices()}


    def add_stop_sequence(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .with_row_count("stop_sequence")
            .with_columns(
                pl.col("stop_sequence")
                .over(["route_id", "direction"])
                .alias("stop_sequence")
            )
        )

    def is_accessible(self, node):
        try:
            isa=ACCESSIBILITY_RULES[self.mode](self.graph[node]['wheelchair'])
        except:
            isa=ACCESSIBILITY_RULES[self.mode](self.graph[node]['wc_access'])
        return isa

    def source_all_td(self, start, time='09:00:00', max_transfers=1, k=4, bw=0.003, mode='all', transfer_time_limit=15):
        self.mode = mode
        arrival_data = {stop: (float('inf'), float('inf')) for stop in self.graph.node_indices()}
        arrival_labels = {}

        start_time = self.time_to_minutes(time)
        nearest_stops = self.ptg.get_kn_stops_node_idx(start, k, bw)
        nearest_stops_ext = self.extend_nearest_stops_with_interchanges(tuple(nearest_stops))

        for stop in nearest_stops_ext:
            arrival_data[stop] = (start_time, 0)
            queue = deque([(stop, start_time)])

            for i in range(max_transfers + 1):
                next_queue = set()
                while queue:
                    current_stop, enroute_time = queue.popleft()
                    earliest_available_stops = self.get_earliest_available_stops(current_stop, enroute_time, self.mode, transfer_time_limit)
                    #print(earliest_available_stops)
                    for available_stop, earliest_arrival in earliest_available_stops:
                        if self.is_accessible(available_stop):
                            if earliest_arrival < arrival_data[available_stop][0] and i < arrival_data[available_stop][1]:
                                arrival_data[available_stop] = (earliest_arrival, i)
                                next_queue.add((available_stop, earliest_arrival))
                queue = deque(next_queue)

        for node, (arrival_time, transfers) in arrival_data.items():
            if arrival_time != float('inf'):
                arrival_labels[node] = {'arrival': arrival_time, 'transfers': transfers, 'endpoint': False, 'type': None}

        return arrival_labels

    @lru_cache(maxsize=None)
    def get_earliest_available_stops(self, stop_idx, enroute_time, mode='all', transfer_time_limit=15):
        stop_id = self.get_stop_id_by_idx(stop_idx)
        route_data = self.schedules.filter(pl.col('stoppoint_id') == stop_id)
        if route_data.is_empty():
            return [(stop_idx, enroute_time)]

        route_ids, directions = route_data.select(['route_id', 'direction']).to_pandas().values.T
        result = [(stop_idx, enroute_time)]

        for route_id, direction in zip(route_ids, directions):
            #print(route_id, direction)
            stop_pos = route_data.filter(
                (pl.col('route_id') == route_id) & (pl.col('direction') == direction)
            )
            stop_sequences = stop_pos['stop_sequence'].to_list()
            if not stop_sequences:
                continue
            current_stop_sequence = stop_sequences[0]

            stops_next = self.schedules.filter(
                (pl.col('route_id') == route_id) &
                (pl.col('direction') == direction) &
                (pl.col('stop_sequence') > current_stop_sequence)
            ).select(["stoppoint_id", "schedule"])

            schedules_on_stop = stop_pos['schedule'].to_list()
            if not schedules_on_stop or not isinstance(schedules_on_stop[0], list):
                continue
            schedules_on_stop = schedules_on_stop[0]

            valid_schedules = [
                arrival for arrival in schedules_on_stop
                if self.time_to_minutes(arrival['scheduledTime']) >= enroute_time
                and ACCESSIBILITY_RULES[mode](arrival)
                and self.time_to_minutes(arrival['scheduledTime']) - enroute_time <= transfer_time_limit
            ]
            
            if not valid_schedules:
                continue

            earliest_back = min(self.time_to_minutes(arrival['scheduledTime']) for arrival in valid_schedules)
            #if self.mode=='wheelchair': print(earliest_back)
            for next_stop in stops_next.iter_rows(named=True):
                stoppoint_id = next_stop['stoppoint_id']
                next_stop_labels = next_stop['schedule']
                valid_next_arrivals = [
                    arrival for arrival in next_stop_labels
                    if self.time_to_minutes(arrival['scheduledTime']) >= earliest_back
                    and ACCESSIBILITY_RULES[mode](arrival)
                ]
                
                if len(valid_next_arrivals)>0:
                    #if self.mode=='wheelchair': print(valid_next_arrivals[0])
                    node_idx = self.get_idx_by_stop_id(stoppoint_id)
                    result.append((node_idx, self.time_to_minutes(valid_next_arrivals[0]['scheduledTime'])))
        #if self.mode=='wheelchair': print(result)
        return self.keep_earliest(result)

    def keep_earliest(self, arrivals):
        unique_arrivals = {}
        for node_idx, arrival in arrivals:
            unique_arrivals.setdefault(node_idx, arrival)
        return list(unique_arrivals.items())

    def time_to_minutes(self, time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours * 60 + minutes + seconds // 60
    @lru_cache(maxsize=None)
    def get_stop_id_by_idx(self, stop):
        return self.idx_to_stop_id.get(stop)

    @lru_cache(maxsize=None)
    def get_idx_by_stop_id(self, stop):
        return self.stop_id_to_idx.get(stop)

    def extend_nearest_stops_with_interchanges(self, nearest_stops):
        nearest_stops_ext = list(nearest_stops)
        for node in nearest_stops:
            for edge in self.graph.out_edges(node):
                edata = edge[2]
                if edata['type'] == 'interchange':
                    transferred = edge[1]
                    wc_label = self.graph[transferred]['wheelchair']
                    if ACCESSIBILITY_RULES[self.mode](wc_label) and transferred not in nearest_stops_ext:
                        nearest_stops_ext.append(transferred)
        return tuple(nearest_stops_ext)

def test1():
    stops = gpd.read_file('stops.json')
    ptg = PermTransitGraph('trip_info', stops)
    def create_dataframe(data) -> pl.DataFrame:
        normalized = []
        for route in data:
            for direction in ["fwd", "bkwd"]:
                stops = route["directions"][direction]
                for stop in stops:
                    normalized.append({
                        "route_id": route["routeId"],
                        "direction": direction,
                        "stoppoint_id": stop["stoppointId"],
                        "schedule": stop["schedule"]
                    })
        return pl.DataFrame(normalized)
    with open("normalized_schedules.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    schedules = create_dataframe(result)
    raptor = RaptorTDRouter(ptg, schedules)
    start = (58.0396, 56.168)
    labels = raptor.source_all_td(start, max_transfers=0, mode='all')
    gdf = raptor.get_gdf(labels)
    gdf.to_file('perm_tests.gpkg', layer='td_raptor_test_all')

    start = (58.0396, 56.168)
    labels = raptor.source_all_td(start, max_transfers=0, mode='wheelchair')
    gdf = raptor.get_gdf(labels)
    gdf.to_file('perm_tests.gpkg', layer='td_raptor_test_wh')
#test1()