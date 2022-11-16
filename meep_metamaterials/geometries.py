import meep as mp
import numpy as np
import copy
from typing import Callable, List, Tuple, Union

def spring(R, r, P, h, material, axis='z'):
    assert(r < R/2)

    if axis == 'z' or axis == mp.Z:
        def _spring(p):
            if p.z > h/2 or p.z < -h/2:
                return mp.air
            if p.y == 0:
                if p.x < 0:
                    if (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z - P/2)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z - P/2 + P)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z - P/2 - P)**2 < r**2:
                        return material
                    else:
                        return mp.air
                else:
                    if (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z + P/2)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z + P/2 + P)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z + P/2 - P)**2 < r**2:
                        return material
                    else:
                        return mp.air

            if (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z + P*np.arctan(p.x/p.y)/np.pi)**2 < r**2:
                return material
            elif (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z + P*np.arctan(p.x/p.y)/np.pi - P)**2 < r**2:
                return material
            elif (R-np.sqrt(p.x**2+p.y**2))**2 + (p.z + P*np.arctan(p.x/p.y)/np.pi + P)**2 < r**2:
                return material
            else:
                return mp.air

    elif axis == 'x' or axis == mp.X:
        def _spring(p):
            if p.x > h/2 or p.x < -h/2:
                return mp.air
            if p.z == 0:
                if p.y < 0:
                    if (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x - P/2)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x - P/2 + P)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x - P/2 - P)**2 < r**2:
                        return material
                    else:
                        return mp.air
                else:
                    if (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x + P/2)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x + P/2 + P)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x + P/2 - P)**2 < r**2:
                        return material
                    else:
                        return mp.air

            if (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x + P*np.arctan(p.y/p.z)/np.pi)**2 < r**2:
                return material
            elif (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x + P*np.arctan(p.y/p.z)/np.pi - P)**2 < r**2:
                return material
            elif (R-np.sqrt(p.z**2+p.y**2))**2 + (p.x + P*np.arctan(p.y/p.z)/np.pi + P)**2 < r**2:
                return material
            else:
                return mp.air
    
    elif axis == 'y' or axis == mp.Y:
        def _spring(p):
            if p.y > h/2 or p.y < -h/2:
                return mp.air
            if p.x == 0:
                if p.z < 0:
                    if (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y - P/2)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y - P/2 + P)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y - P/2 - P)**2 < r**2:
                        return material
                    else:
                        return mp.air
                else:
                    if (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y + P/2)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y + P/2 + P)**2 < r**2:
                        return material
                    elif (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y + P/2 - P)**2 < r**2:
                        return material
                    else:
                        return mp.air

            if (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y + P*np.arctan(p.z/p.x)/np.pi)**2 < r**2:
                return material
            elif (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y + P*np.arctan(p.z/p.x)/np.pi - P)**2 < r**2:
                return material
            elif (R-np.sqrt(p.x**2+p.z**2))**2 + (p.y + P*np.arctan(p.z/p.x)/np.pi + P)**2 < r**2:
                return material
            else:
                return mp.air

    return _spring


def honeycomb_triangular_rounded(a, r, rc, material, background=mp.air):
    assert(r < a)
    assert(rc < r)

    r = r+rc
    L = r*np.sqrt(3)

    e1 = mp.Vector3(0, 1)
    e2 = mp.Vector3(L/2, -r/2)
    e2 = e2.unit()
    e3 = mp.Vector3(-L/2, -r/2)
    e3 = e3.unit()

    vertices = [
        mp.Vector3(0, L*np.sqrt(3)/2),
        mp.Vector3(L/2, 0),
        mp.Vector3(-L/2, 0),
    ]

    round_triangle = [
        mp.Prism(vertices, height=1, material=material, center=mp.Vector3()),
        # Corner 1
        mp.Block(center=r*e1, size=mp.Vector3(0.5, 2.5*rc), material=background),
        mp.Cylinder(radius=rc, height=1, material=material, center=(r-2*rc)*e1),
        # Corner 2
        mp.Block(center=r*e2, size=mp.Vector3(0.5, 2.5*rc), material=background, e1=e2.rotate(mp.Vector3(z=1), np.pi/2), e2=e2),
        mp.Cylinder(radius=rc, height=1, material=material, center=(r-2*rc)*e2),
        # Corner 3
        mp.Block(center=r*e3, size=mp.Vector3(0.5, 2.5*rc), material=background, e1=e3.rotate(mp.Vector3(z=1), np.pi/2), e2=e3),
        mp.Cylinder(radius=rc, height=1, material=material, center=(r-2*rc)*e3),
    ]

    inv_vertices = [v.rotate(mp.Vector3(z=1), np.pi) for v in vertices]

    inv_round_triangle = [
        mp.Prism(inv_vertices, height=1, material=material, center=mp.Vector3()),
        # Corner 1
        mp.Block(center=-r*e1, size=mp.Vector3(0.5, 2.5*rc), material=background),
        mp.Cylinder(radius=rc, height=1, material=material, center=-(r-2*rc)*e1),
        # Corner 2
        mp.Block(center=-r*e2, size=mp.Vector3(0.5, 2.5*rc), material=background, e1=e2.rotate(mp.Vector3(z=1), np.pi/2), e2=e2),
        mp.Cylinder(radius=rc, height=1, material=material, center=-(r-2*rc)*e2),
        # Corner 3
        mp.Block(center=-r*e3, size=mp.Vector3(0.5, 2.5*rc), material=background, e1=e3.rotate(mp.Vector3(z=1), np.pi/2), e2=e3),
        mp.Cylinder(radius=rc, height=1, material=material, center=-(r-2*rc)*e3),
    ]

    t1 = copy.deepcopy(round_triangle)
    t2 = copy.deepcopy(round_triangle)
    t3 = copy.deepcopy(inv_round_triangle)
    t4 = copy.deepcopy(inv_round_triangle)
    t5 = copy.deepcopy(round_triangle)
    t6 = copy.deepcopy(inv_round_triangle)

    L = a/np.sqrt(3) # TODO: change name because it is not the same as the L above
    R1 = mp.Vector3(x=a)
    R2 = mp.Vector3(x=1/2*a, y=np.sqrt(3)/2*a)

    def shift_elem(elem, vector):
        elem.center = elem.center + vector
        return elem

    t1 = [shift_elem(elem, mp.Vector3()) for elem in t1]
    t2 = [shift_elem(elem, mp.Vector3(-np.sqrt(3)*L/2, -L-L/2)) for elem in t2]
    t3 = [shift_elem(elem, mp.Vector3(y=-L)) for elem in t3]
    t4 = [shift_elem(elem, mp.Vector3(-np.sqrt(3)*L/2, L/2)) for elem in t4]
    t5 = [shift_elem(elem, mp.Vector3(np.sqrt(3)*L/2, -L-L/2)) for elem in t5]
    t6 = [shift_elem(elem, mp.Vector3(np.sqrt(3)*L/2, L/2)) for elem in t6]

    return t1+t2+t3+t4+t5+t6


def hexagonal_strained(R: float,
                       r: float,
                       nperiods: int,
                       pad: float = 10,
                       material: mp.Medium = mp.Medium(epsilon=12),
                       strain = lambda x: 0):
    """
    Returns a hexagonal lattice with a strain defined by the function `strain`.

    Args:
        R (float):
        r (float): radius of the hexagons
        L (float): length of the cell

    """
    assert R >= r

    # Regular hexagon
    vertices = [mp.Vector3(-r,0),
                mp.Vector3(-r*0.5,r*np.sqrt(3)/2),
                mp.Vector3(r*0.5,r*np.sqrt(3)/2),
                mp.Vector3(r,0),
                mp.Vector3(r*0.5,-r*np.sqrt(3)/2),
                mp.Vector3(-r*0.5,-r*np.sqrt(3)/2)]
            
    geometry = []

    for i in range(-nperiods//2 + 1, nperiods//2+1):
        v1 = [mp.Vector3(v.x+i*3*R, v.y+strain(v.x+i*3*R)) for v in vertices]
        v2 = [mp.Vector3(3*R/2 + v.x + i*3*R, v.y + R*np.sqrt(3)/2 + strain(3*R/2+v.x+i*3*R)) for v in vertices]
        v3 = [mp.Vector3(3*R/2 + v.x + i*3*R, v.y - R*np.sqrt(3)/2 + strain(3*R/2+v.x+i*3*R)) for v in vertices]

        geometry.append(mp.Prism(v1, height=1, material=material, center=mp.Vector3(i*3*R, strain(i*3*R))))

        geometry.append(mp.Prism(v2, height=1, material=material, center=mp.Vector3(3*R/2+i*3*R, R*np.sqrt(3)/2 + strain(3*R/2+i*3*R))))
        geometry.append(mp.Prism(v2, height=1, material=material, center=mp.Vector3(3*R/2+i*3*R, -R*np.sqrt(3)/2 + strain(3*R/2+i*3*R))))
        geometry.append(mp.Prism(v1, height=1, material=material, center=mp.Vector3(i*3*R, -2*R*np.sqrt(3)/2 + strain(i*3*R))))
        geometry.append(mp.Prism(v2, height=1, material=material, center=mp.Vector3(3*R/2+i*3*R, -3*R*np.sqrt(3)/2 + strain(3*R/2+i*3*R))))
        geometry.append(mp.Prism(v1, height=1, material=material, center=mp.Vector3(i*3*R, -4*R*np.sqrt(3)/2 + strain(i*3*R))))

        if i < nperiods//2:
            geometry.append(mp.Prism(v2, height=1, material=material, center=mp.Vector3(3*R/2+i*3*R, -5*R*np.sqrt(3)/2 + strain(3*R/2+i*3*R))))

        geometry.append(mp.Prism(v1, height=1, material=material, center=mp.Vector3(i*3*R, -6*R*np.sqrt(3)/2 + strain(i*3*R))))
            
        geometry.append(mp.Prism(v2, height=1, material=material, center=mp.Vector3(3*R/2+i*3*R, -7*R*np.sqrt(3)/2 + strain(3*R/2+i*3*R))))
        geometry.append(mp.Prism(v1, height=1, material=material, center=mp.Vector3(i*3*R, -8*R*np.sqrt(3)/2 + strain(i*3*R))))




    return geometry