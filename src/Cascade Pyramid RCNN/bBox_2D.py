import math


class bBox_2D(object):
    def __init__(self, length, width, xc, yc,
                 alpha):  # alpha is the bbox's orientation in degrees, theta is the relative angle to the sensor in rad
        self.yc = yc
        self.xc = xc
        self.center = (self.xc, self.yc)
        self.width = width
        self.length = length
        self.alpha = alpha
        self.vertex1 = 0
        self.vertex2 = 0
        self.vertex3 = 0
        self.vertex4 = 0
        #     x width
        # ------
        # |        |
        # |        |
        # |        |   y length
        # |        |
        # -------

    def bBoxCalcVertxex(self):
        self.vertex1 = (self.xc + self.width / 2, self.yc + self.length / 2)
        self.vertex2 = (self.xc + self.width / 2, self.yc - self.length / 2)
        self.vertex3 = (self.xc - self.width / 2, self.yc + self.length / 2)
        self.vertex4 = (self.xc - self.width / 2, self.yc - self.length / 2)

        self.vertex1 = self._rotate(self.vertex1, self.center, self.alpha)
        self.vertex2 = self._rotate(self.vertex2, self.center, self.alpha)
        self.vertex3 = self._rotate(self.vertex3, self.center, self.alpha)
        self.vertex4 = self._rotate(self.vertex4, self.center, self.alpha)

        self.vertex1 = (int(self.vertex1[0]), int(self.vertex1[1]))
        self.vertex2 = (int(self.vertex2[0]), int(self.vertex2[1]))
        self.vertex3 = (int(self.vertex3[0]), int(self.vertex3[1]))
        self.vertex4 = (int(self.vertex4[0]), int(self.vertex4[1]))

    def scale(self, ratio, offsx, offsy):
        self.yc = (self.yc)* ratio + offsy
        self.xc = (self.xc) * ratio + offsx
        self.center = (self.xc, self.yc)
        self.width = self.width * ratio
        self.length = self.length * ratio

    def _rotate(self, point, origin, alpha):
        return ((point[0] - origin[0]) * math.cos(alpha * math.pi / 180) - (point[1] - origin[1]) * math.sin(
            alpha * math.pi / 180) + origin[0],
                (point[0] - origin[0]) * math.sin(alpha * math.pi / 180) + (point[1] - origin[1]) * math.cos(
                    alpha * math.pi / 180) + origin[1])

    def rotate(self, delta):
        self.vertex1 = (self.xc + self.width / 2, self.yc + self.length / 2)
        self.vertex2 = (self.xc + self.width / 2, self.yc - self.length / 2)
        self.vertex3 = (self.xc - self.width / 2, self.yc + self.length / 2)
        self.vertex4 = (self.xc - self.width / 2, self.yc - self.length / 2)

        self.vertex1 = self._rotate(self.vertex1, self.center, self.alpha + delta)
        self.vertex2 = self._rotate(self.vertex2, self.center, self.alpha + delta)
        self.vertex3 = self._rotate(self.vertex3, self.center, self.alpha + delta)
        self.vertex4 = self._rotate(self.vertex4, self.center, self.alpha + delta)

        self.vertex1 = (int(self.vertex1[0]), int(self.vertex1[1]))
        self.vertex2 = (int(self.vertex2[0]), int(self.vertex2[1]))
        self.vertex3 = (int(self.vertex3[0]), int(self.vertex3[1]))
        self.vertex4 = (int(self.vertex4[0]), int(self.vertex4[1]))

    def resize(self, ratio):
        self.width = self.width * ratio
        self.length = self.length * ratio

    def translate(self, offsx, offsy):
        self.yc = self.yc + offsy
        self.xc = self.xc + offsx
        self.center = (self.xc, self.yc)

    def flipx(self, axis):
        self.xc = 2 * axis - self.xc
        self.alpha = -self.alpha
        self.center = (self.xc, self.yc)

    def xcyc2topleft(self):
        self.xtl=self.xc-self.width/2
        self.ytl=self.yc-self.length/2

    def xcyc2bottomright(self):
        self.xbr = self.xc + self.width / 2
        self.ybr = self.yc + self.length / 2

