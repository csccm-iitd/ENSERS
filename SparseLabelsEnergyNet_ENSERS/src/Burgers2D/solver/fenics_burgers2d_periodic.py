"""
Sovling 2D viscous Burgers' equation with Fenics
"""
import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import dolfin as df
import mshr

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import time
from os.path import dirname, realpath, join
import json
import pdb
import os
import scipy.io

matplotlib.use('agg')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


# Define Dirichlet boundary
class DirichletBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class PeriodicBoundary(df.SubDomain):
    # https://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition/
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((df.near(x[0], 0) or df.near(x[1], 0, )) and
                    (not ((df.near(x[0], 0) and df.near(x[1], 1)) or
                          (df.near(x[0], 1) and df.near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if df.near(x[0], 1) and df.near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
        elif df.near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:  # df.near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


def init_field_fenics(mesh, save_dir, V, order=4, seed=0):
    # https://fenicsproject.org/qa/3975/interpolating-vector-function-from-python-code-to-fenics/

    u0 = df.Function(V)
    # Extract x and y coordinates of mesh and
    # align with dof structure
    dim = V.dim()
    N = mesh.geometry().dim()
    coor = V.tabulate_dof_coordinates().reshape(dim, N)
    f0_dofs = V.sub(0).dofmap().dofs()
    f1_dofs = V.sub(1).dofmap().dofs()

    x = coor[:, 0]  # x for fx and fy
    y = coor[:, 1]  # y for fx and fy
    f0_x, f0_y = x[f0_dofs], y[f0_dofs]  # x, y of components
    # f1_x, f1_y = x[f1_dofs], y[f1_dofs]

    np.random.seed(seed)  # ################################
    lam = np.random.randn(2, 2, (2 * order + 1) ** 2)
    c = -1 + np.random.rand(2) * 2
    aa, bb = np.meshgrid(np.arange(-order, order + 1), np.arange(-order, order + 1))
    k = np.stack((aa.flatten(), bb.flatten()), 1)
    kx_plus_ly = (np.outer(k[:, 0], f0_x) + np.outer(k[:, 1], f0_y)) * 2 * np.pi

    # vector field
    f = np.dot(lam[0], np.cos(kx_plus_ly)) + np.dot(lam[1], np.sin(kx_plus_ly))
    f = f * 2 / np.amax(np.abs(f), axis=1, keepdims=True) + c[:, None]

    # Insert values of fx and fy into the function fe
    u0.vector()[f0_dofs] = f[0]
    u0.vector()[f1_dofs] = f[1]

    return u0, lam, c


def generateMesh(save_dir):
    # Define 2D geometry
    domain =   mshr.Rectangle(df.Point(0., 0.), df.Point(1., 1.))
    # domain.set_subdomain(1, Rectangle(dolfin.Point(1., 1.), dolfin.Point(4., 3.)))
    # domain.set_subdomain(2, Rectangle(dolfin.Point(2., 2.), dolfin.Point(3., 4.)))

    print("Verbose output of 2D geometry:")
    df.info(domain, True)

    # Generate and plot mesh
    mesh2d = mshr.generate_mesh(domain, 45)

    # ************* save mesh vertices
    np.save(save_dir + '/meshVertices.npy', mesh2d.coordinates())
    return mesh2d


def saveBoundaryVertices(mesh, save_dir):
    """ https://fenicsproject.org/qa/2989/vertex-on-mesh-boundary/
    """
    # Mark a CG1 Function with ones on the boundary
    V = df.FunctionSpace(mesh, 'CG', 1)
    bc = df.DirichletBC(V, 1, df.DomainBoundary())
    u = df.Function(V)
    bc.apply(u.vector())

    # Get vertices sitting on boundary
    d2v = df.dof_to_vertex_map(V)
    vertices_on_boundary = d2v[u.vector() == 1.0]
    # ************* save Boundary vertices
    np.save(save_dir + '/BoundaryVertices.npy', vertices_on_boundary)


def saveParams(sp):
    path = join(sp.save_dir, f'dataParams.json')
    with open(path, 'w') as file:  
        json.dump(vars(sp), file, indent = 4) 


def burgers2d(sp):
    """
    Vars:
        u_out_vertex (ndarray): (numNodes)
        simData (ndarray): (timeStep, 2, numNodes)
    """

    sp.save_dir = sp.save_dir + f'/run{sp.run}'
    mkdirs(sp.save_dir)
    saveParams(sp)

    mesh = df.UnitSquareMesh(sp.numHorizCell - 1, sp.numHorizCell - 1)
    mesh_out = mesh
    saveBoundaryVertices(mesh_out, sp.save_dir)

    V = df.VectorFunctionSpace(mesh, 'CG', 2, constrained_domain=PeriodicBoundary())
    Vout = df.VectorFunctionSpace(mesh_out, 'CG', 1, constrained_domain=PeriodicBoundary())

    # initial vector field
    u0, lam, c = init_field_fenics(mesh, sp.save_dir, V, order=sp.order, seed=sp.run)
    
    bc = df.DirichletBC(V, u0, DirichletBoundary())

    u = df.Function(V)
    u_old = df.Function(V)
    v = df.TestFunction(V)

    u = df.project(u0, V)
    u_old.assign(u)

    # backward Euler
    F = (df.inner((u - u_old) / sp.dt, v) \
         + df.inner(df.dot(u, df.nabla_grad(u)), v) \
         + sp.nu * df.inner(df.grad(u), df.grad(v))) * df.dx


    vtkfile = df.File(sp.save_dir + f'/soln.pvd')
    u_save = []
    
    # not much log
    df.set_log_level(30)
    tic = time.time()

    for k, t in enumerate(sp.timeGrid):

        if k % sp.saveVecInterval == 0 and sp.saveVec:
            u_out = df.project(u, Vout)
            u_out_vertex = u_out.compute_vertex_values(mesh_out).reshape(2, sp.numHorizCell, sp.numVertCell)
            u_save.append(u_out_vertex[:, :-1, :-1].reshape(2, -1))
            
        if k % sp.savePvdInterval == 0 and sp.savePvd:
            u_out = df.project(u, Vout)
            u_out.rename('u', 'u')
            vtkfile << (u_out, t)

        df.solve(F == 0, u)
        u_old.assign(u)     


        print(f'Run {sp.run}: solved {k} steps with total {time.time() - tic:.3f} seconds')

    simData = np.stack(u_save, 0)
    np.save(sp.save_dir+f'/raw{sp.run}.npy', simData)
    return time.time() - tic   
    


if __name__ == '__main__':

    class SimParams: pass
    sp = SimParams()

    sp.istart = 14
    sp.iend = 15
    sp.processes = 4

    sp.numHorizCell = 64
    sp.numVertCell = 64
    sp.numNodes = sp.numHorizCell*sp.numVertCell
    sp.nu = 0.005
    sp.dt = 0.005
    sp.numtimeStep = 200
    sp.timeGrid = [sp.dt*i for i in range(sp.numtimeStep)]
    sp.order = 2
    sp.savePvdInterval = 5
    sp.saveVecInterval = 4
    sp.savePvd = True
    sp.saveVec = True
    sp.plot = False
    sp.save_dir = './fenics_data_periodic'
    sp.save_dt = sp.dt*sp.saveVecInterval

    for sp.run in range(sp.istart, sp.iend):
        burgers2d(sp)
