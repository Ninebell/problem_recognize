class Rectangle:
    def __init__(self, points):
        self.points = points
        self.width = points[1][0] - points[0][0]
        self.height = points[1][1] - points[0][1]
        self.area = self.width*self.height
        if self.height == 0:
            self.ratio = 0
        else:
            self.ratio = self.width/self.height

    def __str__(self):
        return "{}".format(self.points)

    @staticmethod
    def union(regions):
        points = regions[0].points
        left = points[0][0]
        top = points[0][1]
        right = points[1][0]
        bottom = points[1][1]
        for region in regions:
            points = region.points
            if left > points[0][0]:
                left = points[0][0]

            if top > points[0][1]:
                top = points[0][1]

            if right < points[1][0]:
                right = points[1][0]

            if bottom < points[1][1]:
                bottom = points[1][1]

        points = [(left, top), (right, bottom)]
        return Rectangle(points)

