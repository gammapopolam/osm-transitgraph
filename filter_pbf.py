import osmium as osm

class BBoxFilterHandler(osm.SimpleHandler):
    def __init__(self, writer, bbox):
        super().__init__()
        self.writer = writer
        self.bbox = bbox  # Bounding box: (min_lon, min_lat, max_lon, max_lat)
        self.nodes_in_bbox = set()  # Track nodes within the bounding box

    def node(self, n):
        # Check if the node is within the bounding box
        if (self.bbox[0] <= n.location.lon <= self.bbox[2] and
            self.bbox[1] <= n.location.lat <= self.bbox[3]):
            self.nodes_in_bbox.add(n.id)
            self.writer.add_node(n)

    def way(self, w):
        # Check if at least one node of the way is within the bounding box
        for node in w.nodes:
            if node.ref in self.nodes_in_bbox:
                self.writer.add_way(w)
                break

    def relation(self, r):
        # Check if at least one member of the relation is within the bounding box
        for member in r.members:
            if member.type == 'n' and member.ref in self.nodes_in_bbox:
                self.writer.add_relation(r)
                break

# Define the bounding box for Krasnoyarsk
#bbox = (92.7222, 55.9887, 93.1123, 56.0123)
bbox = (92.6, 55.8, 93.2, 56.2)

# Input and output file paths
input_pbf = r'd:\osm2gtfs\kja\krasnoyarsk_krai-latest.osm.pbf'
output_pbf = r'd:\osm2gtfs\kja\krasnoyarsk.osm.pbf'

# Create a writer for the output file
writer = osm.SimpleWriter(output_pbf)

# Create the handler and apply it to the input file
handler = BBoxFilterHandler(writer, bbox)
handler.apply_file(input_pbf, locations=True)

# Close the writer to finalize the output file
writer.close()