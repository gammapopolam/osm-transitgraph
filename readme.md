# OSM Static Transit Graph
Generate static and time-dependent public transit graphs using [`rustworkx`](https://github.com/Qiskit/rustworkx) and [`osm2gtfs`](https://github.com/gammapopolam/osm2gtfs).
# `TransitGraphStatic.py`

## `class TransitGraph`

Generate a static transit graph for any type of transport with no schedule labels using `trips.json`, `stops.json`, `s2s.json` files from [`osm2gtfs`](https://github.com/gammapopolam/osm2gtfs)
 module.

 Initialize `TransitGraph` class:
```python
from TransitGraphStatic import TransitGraph, EnhTransitGraph

tram=TransitGraph(trips=r"tram_trips.json", stops=r"tram_stops.json", s2s=r"tram_s2s.json", speed=21, type='tram')
 ```


Visualize graph:
```python
import geopandas as gpd
import pandas as pd
import shapely

route=[]
stations=[]

for node in subway.graph.node_indices():
    data=subway.graph[node]
    data['node_idx']=node
    stations.append(data)

for node in tram.graph.node_indices():
    data=tram.graph[node]
    data['node_idx']=node
    stations.append(data)

for node in commuter.graph.node_indices():
    data=commuter.graph[node]
    data['node_idx']=node
    stations.append(data)

stations_gdf=gpd.GeoDataFrame(stations)
stations_gdf['geom']=stations_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
stations_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
stations_gdf.explore()
```
```python
for edge in subway.graph.edge_list():
    route.append(subway.graph.get_edge_data(*edge))
for edge in tram.graph.edge_list():
    route.append(tram.graph.get_edge_data(*edge))
for edge in commuter.graph.edge_list():
    route.append(commuter.graph.get_edge_data(*edge))

route_gdf=gpd.GeoDataFrame(route)
route_gdf['geom']=route_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
route_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)

# color edges by traveltime (minutes)
route_gdf.explore('traveltime')
# or by transport type
route_gdf.explore('type')
```
## `class PedestrianGraph`

**Deprecated**: Generate pedestrian graph from `*.osm.pbf` file using `osmium`
```python
from TransitGraphStatic import PedestrianGraph

pedestrian=PedestrianGraph('moscow-latest.osm.pbf').graph
```

Also there is `filter_pbf.py` script to filter large-sized pbf by bounding box

## `class EnhTransitGraph`

Merge transit graphs into one enhanced transit graph

#### /!\ Currently there could be issues with pedestrian graph because it is not tested properly and is being deprecated :(

```python
from TransitGraphStatic import EnhTransitGraph

enh=EnhTransitGraph([commuter.graph, tram.graph, subway.graph])
```



## `class RaptorRouter`

**Experimental**: An implementation of RAPTOR routing algorithm in `RaptorStatic.py`. It works with `EnhTransitGraph` to get all possible stops that can be reached from `start_stop`

# `TransitGraphTD.py`

## `class PermTransitGraph`

Works with parsed data from Permskiy Transport API. Generates a transit graph from jsons with route data.

## `class KjaTransitGraph`

Unlike `PermTransitGraph`, works with [`osm2gtfs`](https://github.com/gammapopolam/osm2gtfs) data.

Messy sample to get time-dependent graph. Note that there should be pre-parsed `schedules.json` to initialize RAPTOR router.

```python
stops=gpd.read_file(r'kja\kja_stops.json')
stops.rename(columns={'id': 'stop_id'}, inplace=True)

kja_bus=TransitGraph(trips=r"kja\bus_trips.json", stops=r'kja\bus_stops.json', s2s=r"kja\bus_s2s.json", speed=16, type='bus')
kja_tram=TransitGraph(trips=r"kja\tram_trips.json", stops=r'kja\tram_stops.json', s2s=r"kja\tram_s2s.json", speed=20, type='tram')

kja_pt=EnhTransitGraph([kja_bus.graph, kja_tram.graph])

kja_td=KjaTransitGraph(kja_pt, stops)
```

Public transport schedule with low-floor flags can be generated by average or min-max intervals with max number of enroute vehicles. See `model_schedule.py`

## `class RaptorTDRouter`

An implementation of RAPTOR algorithm for time-dependent transit graphs to get all possible stops that can be reached from some point. Supports wheelchair public transit routing (`all|limited|wheelchair`). Based on `polars` dataframe to optimize evaluations.

```python
raptor=RaptorTDRouter(kja_td, schedules: pl.DataFrame)

arrival_labels=raptor.source_all_td(zone_cnt, time, max_transfers=k, bw=bw, mode='all')
```