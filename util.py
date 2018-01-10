import numpy as np
import pandas as pd
import random
import math

class Vector(object):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.dist(other) == 0

    def dist(self, other: 'Vector'):
        return (self - other).length()

    def length(self):
        import math
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __sub__(self, other: 'Vector'):
        return Vector(self.x - other.x,
                      self.y - other.y,
                      self.z - other.z)

    def translate(self, dist, axis: int):
        assert(axis >= 1 and axis <= 3)
        if axis == 1:
            return self.translate_x(dist)
        elif axis == 2:
            return self.translate_y(dist)
        elif axis == 3:
            return self.translate_z(dist)

    def translate_x(self, dist):
        return Vector(self.x + dist, self.y, self.z)

    def translate_y(self, dist):
        return Vector(self.x, self.y + dist, self.z)

    def translate_z(self, dist):
        return Vector(self.x, self.y, self.z + dist)

    def rotate(self, angle, axis: int):
        assert(axis >= 1 and axis <= 3)
        axis_dict = {
            1: self.x,
            2: self.y,
            3: self.z,
            4: self.x,
            5: self.y
        }
        """
        essentially the component along the axis is unchange while the other two are changed by the 2D rotation matrix
        we now define the rotation in 2D with c1 and c2
        """
        c1 = axis_dict[axis + 1]
        c2 = axis_dict[axis + 2]

        new_c1 = math.cos(angle) * c1 - math.sin(angle) * c2
        new_c2 = math.sin(angle) * c1 + math.cos(angle) * c2

        if axis == 1:
            return Vector(self.x, new_c1, new_c2)
        elif axis == 2:
            return Vector(new_c2, self.y, new_c1)
        else:
            return Vector(new_c1, new_c2, self.z)

    def rotate_x(self, angle):
        return self.rotate(angle, 1)

    def rotate_y(self, angle):
        return self.rotate(angle, 2)

    def rotate_z(self, angle):
        return self.rotate(angle, 3)

    def to_numpy(self):
        return np.asarray([self.x, self.y, self.z])

    def max_component(self):
        return max(self.x, self.y, self.z)

    def __str__(self):
        return str(self.to_numpy())

class UnitCell(object):
    def __init__(self, lattice1: Vector, lattice2: Vector, lattice3: Vector):
        self.lattice1 = lattice1
        self.lattice2 = lattice2
        self.lattice3 = lattice3
        self.atoms = {}

    def add_atom(self, atom: str, coord: Vector):
        if atom not in self.atoms:
            self.atoms[atom] = []
        if coord not in self.atoms[atom]:
            self.atoms[atom].append(coord)

    def to_numpy(self, atom_0=None):
        atoms = []
        all_coords = []
        for atom, coords in self.atoms.items():
            if atom_0 is None or atom == atom_0:
                atoms += [atom] * len(coords)
                all_coords += list(map(lambda v: v.to_numpy(), coords))
        return atoms, np.asarray(all_coords)

    def rotate(self, angle, axis: int):
        assert(axis >= 1 and axis <= 3)
        uc = UnitCell(self.lattice1.rotate(angle, axis),
                      self.lattice2.rotate(angle, axis),
                      self.lattice3.rotate(angle, axis))
        for atom, coords in self.atoms.items():
            uc.atoms[atom] = list(map(lambda v: v.rotate(angle, axis), coords))

        return uc

    def rotate_x(self, angle):
        return self.rotate(angle, 1)

    def rotate_y(self, angle):
        return self.rotate(angle, 2)

    def rotate_z(self, angle):
        return self.rotate(angle, 3)

    def translate(self, dist, axis):
        uc = UnitCell(self.lattice1,
                      self.lattice2,
                      self.lattice3)

        for atom, coords in self.atoms.items():
            uc.atoms[atom] = list(map(lambda v: v.translate(dist, axis), coords))
        return uc

    def __str__(self):
        lattice_vecs = pd.DataFrame(numpy.asarray([self.lattice1.to_numpy(), self.lattice2.to_numpy(), self.lattice3.to_numpy()]))
        lattice_vecs.index = ['lv1', 'lv2', 'lv3']
        lattice_vecs.columns = ['x', 'y', 'z']

        a, c = self.to_numpy()
        atoms = pd.DataFrame(c)
        atoms.index = a
        atoms.columns = ['x', 'y', 'z']

        return '{}\n\n{}'.format(str(lattice_vecs), str(atoms))

    def longest_lv(self):
        return max(self.lattice1.length(), self.lattice2.length(), self.lattice3.length())

    def shortest_lv(self):
        return min(self.lattice1.length(), self.lattice2.length(), self.lattice3.length())


class SuperCell(object):
    def __init__(self, unit_cell: UnitCell,
                 x_rot = 0.0, y_rot = 0.0, z_rot = 0.0,
                 x_tran = 0.0, y_tran = 0.0, z_tran = 0.0):
        self.unit_cell = unit_cell
        self.x_rot = x_rot
        self.y_rot = y_rot
        self.z_rot = z_rot
        self.x_tran = x_tran
        self.y_tran = y_tran
        self.z_tran = z_tran

    def rotate(self, angle, axis):
        return SuperCell(self.unit_cell.rotate(angle, axis))

    def translate(self, dist, axis):
        return SuperCell(self.unit_cell.translate(dist, axis))

    def repetitions(self):
        return (int(25.0 / self.unit_cell.lattice1.length()) + 1,
                int(25.0 / self.unit_cell.lattice2.length()) + 1,
                int(25.0 / self.unit_cell.lattice3.length()) + 1)

    def random_transform(self):
        x_rot, y_rot, z_rot = math.pi / 2 * random.random(), math.pi / 2 * random.random(), math.pi / 2 * random.random()
        x_max, y_max, z_max = max(self.unit_cell.lattice1.x, self.unit_cell.lattice2.x, self.unit_cell.lattice3.x),\
            max(self.unit_cell.lattice1.y, self.unit_cell.lattice2.y, self.unit_cell.lattice3.y),\
            max(self.unit_cell.lattice1.z, self.unit_cell.lattice2.z, self.unit_cell.lattice3.z),
        x_tran, y_tran, z_tran = x_max / 2, y_max / 2, z_max / 2
        return SuperCell(self.unit_cell.rotate_x(x_rot).rotate_y(y_rot).rotate_z(z_rot)
                         .translate(x_tran, 1).translate(y_tran, 2).translate(z_tran, 3),
                         x_rot=x_rot, y_rot=y_rot, z_rot=z_rot,
                         x_tran=x_tran, y_tran=y_tran, z_tran=z_tran)

    def make_supercell(self, atom):
        return self.unit_cell.to_numpy(atom)

    def to_tensor(self, cell_size=12.0, fineness=2.0,
                  gamma_in=1.0, gamma_ga=1.0, gamma_al=1.0, gamma_o=1.0,
                  f=lambda r, gamma: np.exp(-gamma*np.linalg.norm(r, axis=1))):
        atom_map = {
            "In": 0,
            "Ga": 1,
            "Al": 2,
            "O": 3
        }
        gamma_map = {
            "In": gamma_in,
            "Ga": gamma_ga,
            "Al": gamma_al,
            "O": gamma_o
        }
        N = 2 * int(cell_size / fineness) + 1
        discr = np.linspace(-cell_size, cell_size, N)
        X, Y, Z = np.meshgrid(discr, discr, discr)
        out_tensor = np.zeros((N, N, N, 4), dtype=np.float32)

        supercells = {
            k: self.make_supercell(k) for k in self.unit_cell.atoms.keys()
        }

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    xyz = np.asarray([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    for a, sc in supercells.items():
                        v = f(sc[1] - xyz, gamma_map[a]).sum()
                        out_tensor[i,j,k, atom_map[a]] = v
        return out_tensor



def read_file(filename):
    with open(filename) as inputfile:
        lines = list(filter(lambda s: '#' not in s, inputfile.readlines()))
    lattice_vecs = lines[:3]
    atoms = lines[3:]
    x, y, z = list(map(lambda m: Vector(float(m[1]), float(m[2]), float(m[3])),
                         list(map(lambda l: l.split(' '), lattice_vecs))))
    unit_cell = UnitCell(x, y, z)
    for atom in atoms:
        _, x, y, z, a = atom.split(' ')
        unit_cell.add_atom(a.strip(), Vector(float(x), float(y), float(z)))
    return unit_cell

if __name__ == '__main__':
    cell = read_file('train/1/geometry.xyz')
    supercell = SuperCell(cell)
    import time
    start = time.time()
    supercell.to_tensor()
    duration = time.time() - start
    print('took time: {}'.format(duration))
    start = time.time()
    supercell.to_tensor(fineness=1.5)
    duration = time.time() - start
    print('took time: {}'.format(duration))
    start = time.time()
    supercell.to_tensor(fineness=1.0)
    duration = time.time() - start
    print('took time: {}'.format(duration))

