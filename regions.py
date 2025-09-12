class Region:
    def __contains__(self, point): ...
    def size(self): ...


class Rectangle(Region):

    def __init__(self, x=0, y=0, width=1, height=1):
        x1, y1, x2, y2 = x, y, x + width, y + height
        self.bottomleft_x = min(x1, x2)
        self.bottomleft_y = min(y1, y2)
        self.topright_x = max(x1, x2)
        self.topright_y = max(y1, y2)

    def __contains__(self, point):
        return (self.bottomleft_x <= point[0] <= self.topright_x) and (
            self.bottomleft_y <= point[1] <= self.topright_y
        )

    def size(self):
        if not hasattr(self, "cache_size"):
            self.cache_size = (self.topright_x - self.bottomleft_x) * (
                self.topright_y - self.bottomleft_y
            )
        return self.cache_size
