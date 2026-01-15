import utm

class UWConvert:
    def __init__(self, offset: str = "data/offset.txt"):
        self.offset_file = offset
        self.utm_zone = None
        self.hemisphere = None
        self.x_coordinate = None
        self.y_coordinate = None

        self.get_utm(self.offset_file)

    def get_utm(self, path):
        with open(path, 'r') as file:
            content = file.read()
        values = content.split()
        self.utm_zone = int(values[2][:2])
        self.hemisphere = values[2][-1]
        self.x_coordinate = float(values[-2])
        self.y_coordinate = float(values[-1])

    def W2U(self, wgs84: list):  # [lat,lon,alt]  to [x,y,z]
        utm_point = utm.from_latlon(
            wgs84[0], wgs84[1], self.utm_zone, self.hemisphere)
        return [utm_point[0] - self.x_coordinate, utm_point[1]-self.y_coordinate, wgs84[2]]

    def U2W(self, utm_o: list):  # [x,y,z]  to [lat,lon,alt]
        utm_point = utm.to_latlon(
            utm_o[0]+self.x_coordinate, utm_o[1]+self.y_coordinate, self.utm_zone, self.hemisphere)
        return [utm_point[0], utm_point[1], utm_o[2]]


if __name__ == "__main__":
    c = UWConvert("data/map/tianjin/georeferencing_model_geo.txt")
    # 1719375002354
# "altitude":0.0,"latitude":39.055077,"longitude":117.70836}
# {"altitude":0.0,"latitude":39.055077,"longitude":117.70836}
# {"altitude":0.0,"latitude":39.054142,"longitude":117.70444}
# "altitude":0.0,"latitude":39.054516,"longitude":117.70811}
# "altitude":0.0,"latitude":39.05474347756207,"longitude":117.70861233597032
    point1 = [39.0545071048, 117.70958548232, 0]
    print(c.W2U(point1))

    # points = [point1, point2, point3, point4, point5]
    # for i in points:
    #     print([*c.W2U(i[:2]), i[2]])
