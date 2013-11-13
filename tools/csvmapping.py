import csv
from shapely.geometry import Point, mapping, Polygon
from fiona import collection

def demo_points():
    schema = { 'geometry': 'Point', 'properties': { 'name': 'str' } }
    with collection(
        "some.shp", "w", "ESRI Shapefile", schema) as output:
        with open('some.csv', 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                point = Point(float(row['lon']), float(row['lat']))
                output.write({
                    'properties': {
                        'name': row['name']
                    },
                    'geometry': mapping(point)
                })

def create_contours(fname):
    schema = { 'geometry': 'Polygon', 'properties': { 'depth': 'str' } }
    with collection(
        "FMB.shp", "w", "ESRI Shapefile", schema) as output:
        with open(fname, 'rb') as f:
            reader = csv.DictReader(f)
            polygon_coord = []

            for row in reader:
                polygon_coord.append((float(row['lon']), float(row['lat'])))

            poly = Polygon(polygon_coord)
            output.write({
                'properties': {
                    'depth': row['depth']
                },
                'geometry': mapping(poly)
            })



if __name__ == "__main__":
    # demo_points()
    create_contours('test.csv')
    print "Done"
