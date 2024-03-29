import numpy as np
import pandas as pd
import random
import math

cov_rad = {
    'Ga': 1.22,
    'Al': 1.21,
    'In': 1.42,
    'O' : 0.66
}

class Vector(object):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.dist(other) == 0

    def get_unit_vec(self):
        return (self.x / self.length(), self.y / self.length(), self.z / self.length())

    def dist(self, other: 'Vector'):
        return (self - other).length()

    def dot(self, other: 'Vector'):
        return self.x * other.x  + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector'):
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def angle(self, other: 'Vector'):
        return np.arccos(self.dot(other) / (self.length() * other.length()))

    def length(self):
        import math
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __sub__(self, other: 'Vector'):
        return Vector(self.x - other.x,
                      self.y - other.y,
                      self.z - other.z)

    def free_translate(self, x, y, z): # translate the atoms, not frame of reference
        return Vector(self.x + x, self.y + y, self.z + z)

    def translate(self, dist, axis: int):
        assert(1 <= axis <= 3)
        if axis == 1:
            return self.free_translate(dist, 0, 0)
        elif axis == 2:
            return self.free_translate(0, dist, 0)
        elif axis == 3:
            return self.free_translate(0, 0, dist)

    def translate_x(self, dist):
        return self.translate(dist, 1)

    def translate_y(self, dist):
        return self.translate(dist, 2)

    def translate_z(self, dist):
        return self.translate(dist, 3)

    def free_rotate(self, theta_x, theta_y, theta_z):
        new_x, new_y, new_z = self.x, self.y, self.z
        # x-rotation
        new_y2 = math.cos(theta_x) * new_y - math.sin(theta_x) * new_z
        new_z2 = math.sin(theta_x) * new_y + math.cos(theta_x) * new_z
        new_y, new_z = new_y2, new_z2
        # y-rotation
        new_x2 = math.cos(theta_y) * new_x + math.sin(theta_y) * new_z
        new_z2 = -math.sin(theta_y) * new_x + math.cos(theta_y) * new_z
        new_x, new_z = new_x2, new_z2
        # z-rotation
        new_x2 = math.cos(theta_z) * new_x - math.sin(theta_z) * new_y
        new_y2 = math.sin(theta_z) * new_x + math.cos(theta_z) * new_y
        new_x, new_y = new_x2, new_y2
        return Vector(new_x, new_y, new_z)

    def rotate_around(self, axis, angle):
        ux, uy, uz = axis
        R = np.asarray([[
            np.cos(angle) + ux**2 * (1 - np.cos(angle)), ux*uy*(1-np.cos(angle)) - uz*np.sin(angle), ux*uz*(1-np.cos(angle)) + uy*np.sin(angle)
        ],[
            uy*ux*(1-np.cos(angle)) + uz*np.sin(angle), np.cos(angle) + uy**2*(1-np.cos(angle)), uy*uz*(1-np.cos(angle)) - ux*np.sin(angle)
        ],[
            uz*ux*(1-np.cos(angle)) - uy*np.sin(angle), uz*uy*(1-np.cos(angle)) + ux*np.sin(angle), np.cos(angle) + uz**2*(1-np.cos(angle))
        ]])
        new_x, new_y, new_z = np.dot(R, np.asarray([self.x, self.y, self.z]))
        return Vector(new_x, new_y, new_z)

    def rotate(self, angle, axis: int):
        assert(1 <= axis <= 3)
        if axis == 1:
            return self.free_rotate(angle, 0, 0)
        elif axis == 2:
            return self.free_rotate(0, angle, 0)
        else:
            return self.free_rotate(0, 0, angle)

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
    def __init__(self, lattice1: Vector, lattice2: Vector, lattice3: Vector, isHexagonal=False):
        self.lattice1 = lattice1
        self.lattice2 = lattice2
        self.lattice3 = lattice3
        self.atoms = {}
        self.isHexagonal = isHexagonal

    def realign(self):
        """
        two stage rotation to make first axis align with x axis, and second axis to be in xy plane
        1. align cross(lv1, lav2) to z-axis
        2. align lv1 to x-axis
        """
        def realign_z(uc: 'UnitCell'):
            vec3 = uc.lattice1.cross(uc.lattice2)
            angle = vec3.angle(Vector(0,0,1))
            axis = vec3.cross(Vector(0,0,1))
            return uc.rotate_from(axis.get_unit_vec(), angle)
        def realign_x(uc: 'UnitCell'):
            axis = uc.lattice1.cross(Vector(1,0,0))
            angle = uc.lattice1.angle(Vector(1,0,0))
            return uc.rotate_from(axis.get_unit_vec(), angle)
        return realign_x(realign_z(self))

    def  show_lvs(self):
        df = lattice_vecs = pd.DataFrame(np.asarray([self.lattice1.to_numpy(), 
            self.lattice2.to_numpy(), 
            self.lattice3.to_numpy()]), index=['lv1', 'lv2', 'lv3'], columns=['x', 'y', 'z'])
        print(df)


    def add_atom(self, atom: str, coord: Vector):
        if atom not in self.atoms:
            self.atoms[atom] = []
        if coord not in self.atoms[atom]:
            self.atoms[atom].append(coord)
            if self.isHexagonal:
                self.atoms[atom].append(
                    coord.rotate_around(self.lattice3.get_unit_vec(), np.pi * 2 / 3)
                )
                self.atoms[atom].append(
                    coord.rotate_around(self.lattice3.get_unit_vec(), np.pi * 4 / 3)
                )

    def to_numpy(self, atom_0=None):
        atoms = []
        all_coords = []
        for atom, coords in self.atoms.items():
            if atom_0 is None or atom == atom_0:
                atoms += [atom] * len(coords)
                all_coords += list(map(lambda v: v.to_numpy(), coords))
        return atoms, np.asarray(all_coords)

    def free_rotate(self, theta_x, theta_y, theta_z):
        uc = UnitCell(self.lattice1.free_rotate(theta_x, theta_y, theta_z),
                      self.lattice2.free_rotate(theta_x, theta_y, theta_z),
                      self.lattice3.free_rotate(theta_x, theta_y, theta_z))
        for atom, coords in self.atoms.items():
            uc.atoms[atom] = list(map(lambda v: v.free_rotate(theta_x, theta_y, theta_z), coords))
        return uc

    def rotate_from(self, axis, angle):
        uc = UnitCell(self.lattice1.rotate_around(axis, angle),
                      self.lattice2.rotate_around(axis, angle),
                      self.lattice3.rotate_around(axis, angle))
        for atom, coords in self.atoms.items():
            uc.atoms[atom] = list(map(lambda v: v.rotate_around(axis, angle), coords))
        return uc

    def rotate(self, angle, axis: int):
        assert(1 <= axis <= 3)
        if axis == 1:
            return self.free_rotate(angle, 0, 0)
        elif axis == 2:
            return self.free_rotate(0, angle, 0)
        else:
            return self.free_rotate(0, 0, angle)

    def rotate_x(self, angle):
        return self.rotate(angle, 1)

    def rotate_y(self, angle):
        return self.rotate(angle, 2)

    def rotate_z(self, angle):
        return self.rotate(angle, 3)

    def free_translate(self, x, y, z):
        uc = UnitCell(self.lattice1,
                      self.lattice2,
                      self.lattice3)
        for atom, coords in self.atoms.items():
            uc.atoms[atom] = list(map(lambda v: v.free_translate(x, y, z), coords))
        return uc

    def translate(self, dist, axis):
        assert(1 <= axis <= 3)
        if axis == 1:
            return self.free_translate(dist, 0, 0)
        elif axis == 2:
            return self.free_translate(0, dist, 0)
        else:
            return self.free_translate(0, 0, dist)



    def __str__(self):
        lattice_vecs = pd.DataFrame(np.asarray([self.lattice1.to_numpy(), self.lattice2.to_numpy(), self.lattice3.to_numpy()]), index=['lv1', 'lv2', 'lv3'], columns=['x', 'y', 'z'])
        a, c = self.to_numpy()
        atoms = pd.DataFrame(c, index=a, columns=['x', 'y', 'z'])

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

    def free_rotate(self, theta_x, theta_y, theta_z):
        return SuperCell(self.unit_cell.free_rotate(theta_x, theta_y, theta_z),
                         x_rot=theta_x+self.x_rot, y_rot=theta_y+self.y_rot, z_rot=theta_z+self.z_rot,
                         x_tran=self.x_tran, y_tran=self.y_tran, z_tran=self.z_tran)

    def free_translate(self, x, y, z):
        return SuperCell(self.unit_cell.free_translate(x, y, z),
                         x_tran=x+self.x_tran, y_tran=y+self.y_tran, z_tran=z+self.z_tran,
                         x_rot=self.x_rot, y_rot=self.y_rot, z_rot=self.z_rot)

    def rotate(self, angle, axis):
        return SuperCell(self.unit_cell.rotate(angle, axis))

    def translate(self, dist, axis):
        return SuperCell(self.unit_cell.translate(dist, axis))

    def repetitions(self):
        return (int(25.0 / self.unit_cell.lattice1.length()) + 2,
                int(25.0 / self.unit_cell.lattice2.length()) + 2,
                int(25.0 / self.unit_cell.lattice3.length()) + 2)

    def random_transform(self):
        x_max, y_max, z_max = min(self.unit_cell.lattice1.x, self.unit_cell.lattice2.x, self.unit_cell.lattice3.x),\
            min(self.unit_cell.lattice1.y, self.unit_cell.lattice2.y, self.unit_cell.lattice3.y),\
            min(self.unit_cell.lattice1.z, self.unit_cell.lattice2.z, self.unit_cell.lattice3.z),
        x_tran, y_tran, z_tran = x_max / 2 * random.random(), y_max / 2 * random.random(), z_max / 2 * random.random()
        return self.free_translate(x_tran, y_tran, z_tran)

    def make_supercell(self, atom):
        x, y, z = self.repetitions()
        _, primary_cell = self.unit_cell.to_numpy(atom)
        supercell = primary_cell
        lv1 = self.unit_cell.lattice1.to_numpy()
        lv2 = self.unit_cell.lattice2.to_numpy()
        lv3 = self.unit_cell.lattice3.to_numpy()
        if True:
            for i in range(-x, x):
                for j in range(-y, y):
                    for k in range(-z, z):
                        subcell = primary_cell + i * lv1 + j * lv2 + k * lv3
                        if supercell is None:
                            supercell = subcell
                        else:
                            supercell = np.concatenate((supercell, subcell))
        #else:
        #    supercell = np.concatenate((supercell, primary_cell +  lv1))
        return supercell

    def realign_cell(self):
        # TODO rotate cell so that first component is in x only, first and second in xy plane only
        pass

    def to_tensor2(self, cell_size=12.0, fineness=1.5):
        atom_map = {
            "In": 0,
            "Ga": 1,
            "Al": 2,
            "O": 3
        }

        n = 2 * int(cell_size/fineness) + 1

        out_tensor = np.zeros((n,n,n,4), dtype=np.float32)

        supercells = {
            k: self.make_supercell(k) for k in self.unit_cell.atoms.keys()
        }

        for k, sc in supercells.items():
            coord = (sc / fineness).astype(np.int)
            for c in coord:
                x, y, z = c
                if np.abs(x) <= int(n/2) and np.abs(y) <= int(n/2) and np.abs(z) <=int(n/2):
                    out_tensor[x, y, z, atom_map[k]] = 1.0

        return out_tensor

    def to_tensor(self, cell_size=12.0, fineness=2,
                  gamma_in=cov_rad['In'], gamma_ga=cov_rad['Ga'],
                  gamma_al=cov_rad['Al'], gamma_o=cov_rad['O'],
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
        n = 2 * int(cell_size / fineness) + 1
        discr = np.linspace(-cell_size, cell_size, n)
        X, Y, Z = np.meshgrid(discr, discr, discr)
        out_tensor = np.zeros((n, n, n, 4), dtype=np.float32)

        supercells = {
            k: self.make_supercell(k) for k in self.unit_cell.atoms.keys()
        }

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    xyz = np.asarray([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    for a, sc in supercells.items():
                        v = f(sc - xyz, gamma_map[a]).sum()
                        out_tensor[i,j,k, atom_map[a]] = v
        return out_tensor

    def __str__(self):
        df = pd.DataFrame([[self.x_rot, self.y_rot, self.z_rot],
                           [self.x_tran, self.y_tran, self.z_tran]],
                          index=['rot', 'tran'], columns=['x', 'y', 'z'])
        return '{}\n\n{}'.format(str(df), str(self.unit_cell))



def read_file(filename, isHexagonal=False):
    with open(filename) as inputfile:
        lines = list(filter(lambda s: '#' not in s, inputfile.readlines()))
    lattice_vecs = lines[:3]
    atoms = lines[3:]
    x, y, z = list(map(lambda m: Vector(float(m[1]), float(m[2]), float(m[3])),
                         list(map(lambda l: l.split(' '), lattice_vecs))))
    unit_cell = UnitCell(x, y, z, isHexagonal)
    for atom in atoms:
        _, x, y, z, a = atom.split(' ')
        unit_cell.add_atom(a.strip(), Vector(float(x), float(y), float(z)))
    return unit_cell.realign()

def super_cell(filename, isHexagonal=False):
    return SuperCell(read_file(filename, isHexagonal=isHexagonal))

if __name__ == '__main__':
    import sys
    uc = read_file('train/{}/geometry.xyz'.format(sys.argv[1]), isHexagonal=True)
    uc.show_lvs()
    uc.realign().show_lvs()
    #sc = SuperCell(uc)
    #y = sc.make_supercell('Al')
    #y = np.concatenate((y, sc.make_supercell('Ga')))
    #y = np.concatenate((y, sc.make_supercell('O')))
    #df = pd.DataFrame(y, columns=['x','y','z'])
    #df.to_csv('hexagon.csv')
