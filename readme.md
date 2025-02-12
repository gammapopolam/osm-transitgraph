# OSM Static Transit Graph
Generate static transit graphs using [`rustworkx`](https://github.com/Qiskit/rustworkx) and [`osm2gtfs`](https://github.com/gammapopolam/osm2gtfs).

## `class TransitGraph`

Generate a static transit graph for any type of transport with no schedule labels using `trips.json`, `stops.json`, `s2s.json` files from [`osm2gtfs`](https://github.com/gammapopolam/osm2gtfs)
 module.

 Initialize `TransitGraph` class:
```python
from TransitGraphStatic import TransitGraph, EnhTransitGraph

tram=TransitGraph(trips=r"tram_trips.json", stops=r"tram_stops.json", s2s=r"tram_s2s.json", speed=21, type='tram')
 ```
`TransitGraph` allow to contract edges by wheelchair accessibility of nodes. Use ```wc_mode=True``` to contract edges

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

**Experimental**: Generate pedestrian graph from `*.osm.pbf` file using `osmium`
```python
from TransitGraphStatic import PedestrianGraph

pedestrian=PedestrianGraph('moscow-latest.osm.pbf').graph
```

Also there is `filter_pbf.py` script to filter large-sized pbf by bounding box

## `class EnhTransitGraph`

Merge transit graphs into one enhanced transit graph

#### /!\ Currently there could be issues with pedestrian graph because it is not tested properly

```python
from TransitGraphStatic import EnhTransitGraph

enh=EnhTransitGraph([commuter.graph, tram.graph, subway.graph], pedestrian=pedestrian)
```



## `class RaptorRouter`

**Experimental**: An implementation of RAPTOR routing algorithm in `RaptorStatic.py`. It works with `EnhTransitGraph` to get pedestrian paths for each visited stop in any cutoff (isochrone-alike approach)

```python
from TransitGraphStatic import TransitGraph, EnhTransitGraph
from RaptorStatic import RaptorRouter

bus=TransitGraph(trips="bus_trips.json", stops="bus_stops.json", s2s="bus_s2s.json", speed=21, type='bus')
print('bus', len(bus.graph.nodes()), len(bus.graph.edges()))

tram=TransitGraph(trips="tram_trips.json", stops="tram_stops.json", s2s="tram_s2s.json", speed=24, type='tram')
print('tram', len(tram.graph.nodes()), len(tram.graph.edges()))

pedestrian=PedestrianGraph(pbf='*.osm.pbf', walking=5)
print('pedestrian', len(pedestrian.graph.nodes()), len(pedestrian.graph.edges()))

random_stop=318
print([e[2]['ref'] for e in enh.graph.out_edges(random_stop)])

raptor=RaptorRouter(bus.graph, random_stop, max_transfers=0, interval=480)
# Get all arrival labels for stop
arrivals=raptor.main_calculator(pedestrian_cutoff=5)

nodes=[]
for node in arrivals.keys():
    #print(node)
    nodedata=enh.graph[node]
    nodedata['arrival']=arrivals[node]
    nodes.append(nodedata)
nodes_gdf=gpd.GeoDataFrame(nodes)
nodes_gdf['geom']=nodes_gdf['geom'].apply(lambda x: shapely.from_wkt(x))
nodes_gdf.set_geometry('geom', crs='EPSG:4326', inplace=True)
```

`nodes_gdf` can be visualized with stops and pedestrian nodes

```python
fig, ax = plt.subplots(figsize=(15, 15))

nodes_gdf[nodes_gdf['type']!='pedestrian'].plot(ax=ax, color='red', markersize=10, zorder=3)
nodes_gdf[nodes_gdf['type']!='pedestrian'].plot(ax=ax, color='white', markersize=15, zorder=2)
nodes_gdf[nodes_gdf['type']=='pedestrian'].plot(ax=ax, column='arrival', markersize=1, zorder=1)
```