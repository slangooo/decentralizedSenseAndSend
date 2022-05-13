from dataclasses import dataclass
import numpy as np
import timeit
from scipy.spatial import distance


@dataclass
class Coords3d:
    __slots__ = ["x", "y", "z"]
    x: float
    y: float
    z: float

    def __str__(self):
        return f'{{x: {str(self.x)}, y:{str(self.y)}, z:{str(self.z)}}}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, np.ndarray):
            if len(other) == 2:
                return Coords3d(self.x + other[0], self.y + other[1], self.z)
            elif len(other) == 3:
                return Coords3d(self.x + other[0], self.y + other[1], self.z + other[2])
        else:
            raise ValueError("Undefined operation for given operand!")

    def __sub__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, np.ndarray):
            if len(other) == 2:
                return Coords3d(self.x - other[0], self.y - other[1], self.z)
            elif len(other) == 3:
                return Coords3d(self.x - other[0], self.y - other[1], self.z - other[2])
        else:
            raise ValueError("Undefined operation for given operand!")

    def __truediv__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __rtruediv__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __mul__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __rmul__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def get_distance_to(self, other_coords, flag_2d=False):

        if isinstance(other_coords, Coords3d):
            return np.sqrt(((self.x - other_coords.x) ** 2 + (self.y - other_coords.y) ** 2
                            + ((self.z - other_coords.z) ** 2 if not flag_2d else 0)))
        elif isinstance(other_coords, tuple) or isinstance(other_coords, list) or isinstance(other_coords, np.ndarray):
            squared_sum = (other_coords[0] - self.x) ** 2 + (other_coords[1] - self.y) ** 2
            if len(other_coords) > 2 and not flag_2d:
                squared_sum += (other_coords[2] - self.z) ** 2
            return np.sqrt(squared_sum)
        else:
            raise ValueError('Unidentified input format!')

    def copy(self):
        return Coords3d(self.x, self.y, self.z)

    def np_array(self):
        return np.asarray((self.x, self.y, self.z))

    def as_2d_array(self):
        return np.asarray((self.x, self.y))

    def __array__(self, dtype=None):
        if dtype is None:
            return np.asarray((self.x, self.y, self.z))
        elif dtype == Coords3d:
            # arr = np.empty(1, dtype=Coords3d)
            # arr[0] = self
            return self

    def __len__(self):
        return 3

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise ValueError("Out of bounds!")

    @staticmethod
    def from_array(array):
        return Coords3d(array[0], array[1], array[2])


def to_coords_3d(_array):
    return Coords3d(_array[0], _array[1], _array[2] if len(_array) > 2 else 0)


if __name__ == '__main__':
    a = Coords3d(1, 2, 3)
    b = Coords3d(1.5, 2.5, 3.5)
    an = np.array([1, 2, 3])
    bn = np.array([1.5, 2.5, 3])
    print(np.linalg.norm(an - bn), a.get_distance_to(b, flag_2d=True), distance.euclidean(an, bn))
    setup = '''
from src.data_structures import Coords3d
a = Coords3d(1, 2, 3)
b = Coords3d(1.5, 2.5, 3.5)
'''
    print(min(timeit.Timer('a.get_distance_to(b)', setup=setup).repeat(7, 1000)))
    setup = '''
import numpy as np
from scipy.spatial import distance
an = np.array([1, 2, 3])
bn = np.array([1.5, 2.5, 3.5])
'''
    print(min(timeit.Timer('np.linalg.norm(an-bn)', setup=setup).repeat(7, 1000)))
    print(min(timeit.Timer('distance.euclidean(an, bn)', setup=setup).repeat(7, 1000)))
