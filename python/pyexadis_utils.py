"""@package docstring

ExaDiS python utilities

Implements utility functions for the ExaDiS python binding

* insert_frank_read_src()
* insert_infinite_line()
* insert_prismatic_loop()
* generate_line_config()
* generate_prismatic_config()

* get_segments_end_points()
* get_segments_length()
* dislocation_density()
* dislocation_charge()

* replicate_network()
* combine_networks()
* extract_segments()
* delete_segments()

* read_paradis()
* write_data()
* write_vtk()

Nicolas Bertin
bertin1@llnl.gov
"""

import numpy as np
import pyexadis
from pyexadis_base import NodeConstraints, ExaDisNet
try:
    # Try importing DisNetManager from OpenDiS
    from framework.disnet_manager import DisNetManager
except ImportError:
    # Use dummy DisNetManager if OpenDiS is not available
    from pyexadis_base import DisNetManager

from typing import Tuple


def insert_frank_read_src(cell, nodes, segs, burg, plane, length, center, theta=0.0, linedir=None, numnodes=10):
    """Insert a Frank-Read source into the list of nodes and segments
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the source
    plane: habit plane normal of the source
    theta: character angle of the source in degrees
    linedir: line direction of the source
    length: length of the source
    center: center position of the source
    numnodes: number of discretization nodes for the source
    """
    plane = plane / np.linalg.norm(plane)
    if np.abs(np.dot(burg, plane)) >= 1e-5:
        print('Warning: Burgers vector and plane normal are not orthogonal')
    
    if not linedir is None:
        ldir = np.array(linedir)
        ldir = ldir / np.linalg.norm(ldir)
    else:
        b = burg / np.linalg.norm(burg)
        y = np.cross(plane, b)
        y = y / np.linalg.norm(y)
        ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y
    
    istart = len(nodes)
    for i in range(numnodes):
        p = center -0.5*length*ldir + i*length/(numnodes-1)*ldir
        constraint = NodeConstraints.PINNED_NODE if (i == 0 or i == numnodes-1) else NodeConstraints.UNCONSTRAINED
        nodes.append(np.concatenate((p, [constraint])))
    
    for i in range(numnodes-1):
        segs.append(np.concatenate(([istart+i, istart+i+1], burg, plane)))
    
    return nodes, segs


def insert_infinite_line(cell, nodes, segs, burg, plane, origin, theta=0.0, linedir=None, maxseg=-1, trial=False):
    """Insert an infinite line into the list of nodes and segments
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the line
    plane: habit plane normal of the line
    origin: origin position of the line
    theta: character angle of the line in degrees
    linedir: line direction
    maxseg: maximum discretization length of the line
    trial: do a trial insertion only (to test if insertion is possible)
    """
    plane = plane / np.linalg.norm(plane)
    if np.abs(np.dot(burg, plane)) >= 1e-5:
        print('Warning: Burgers vector and plane normal are not orthogonal')
    
    if not linedir is None:
        ldir = np.array(linedir)
        ldir = ldir / np.linalg.norm(ldir)
    else:
        b = burg / np.linalg.norm(burg)
        y = np.cross(plane, b)
        y = y / np.linalg.norm(y)
        ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y

    h = np.array(cell.h)
    Lmin = np.min(np.linalg.norm(h, axis=0))
    seglength = 0.15*Lmin
    
    if maxseg > 0:
        seglength = np.min([seglength, maxseg])

    length = 0.0
    meet = 0
    maxnodes = 1000
    numnodes = 0
    origin = np.array(origin)
    p = 1.0*origin
    originpbc = 1.0*origin
    while ((~meet) & (numnodes < maxnodes)):
        p += seglength*ldir
        pp = np.asarray(cell.closest_image(Rref=origin, R=p))
        dist = np.linalg.norm(pp-origin)
        if ((numnodes > 0) & (dist < seglength)):
            originpbc = np.asarray(cell.closest_image(Rref=p, R=origin))
            meet = 1
        numnodes += 1

    if numnodes == maxnodes:
        if trial:
            return -1.0
        else:
            print('Warning: infinite line is too long, aborting')
            return nodes, segs

    if trial:
        return np.linalg.norm(originpbc-origin)
    else:
        istart = len(nodes)
        for i in range(numnodes):
            p = origin + 1.0*i/numnodes*(originpbc-origin)
            constraint = NodeConstraints.UNCONSTRAINED
            nodes.append(np.concatenate((p, [constraint])))
        for i in range(numnodes):
            segs.append(np.concatenate(([istart+i, istart+(i+1)%numnodes], burg, plane)))
        return nodes, segs


def insert_prismatic_loop(crystal, cell, nodes, segs, burg, radius, center, maxseg=-1, Rorient=None):
    """Insert a prismatic dislocation loop into the list of nodes and segments
    Input Burgers vector must be of the 1/2<111> type for bcc and 1/2<110> type for fcc.
    Arguments:
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the loop
    radius: radius of the loop
    center: center position of the loop
    maxseg: maximum discretization length
    Rorient: crystal orientation matrix
    """ 
    b = -1.0*burg
    
    if crystal in ['BCC', 'bcc']:
        b0 = 1.0/np.sqrt(3.0)*np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
        ib = bcol.argmin()
        if bcol[ib] > 1e-5:
            raise ValueError('BCC Burgers vector must be of the 1/2<111> type in insert_prismatic_loop()')
        Nsides = 6
        if 1:
            # Loop with arms on {110} planes (default)
            e = np.array([[-2.0*b[0],b[1],b[2]],[-b[0],-b[1],2.0*b[2]],
                          [b[0],-2.0*b[1],b[2]],[2.0*b[0],-b[1],-b[2]],
                          [b[0],b[1],-2.0*b[2]],[-b[0],2.0*b[1],-b[2]]])
        else:
            # Loop with arms on {112} planes
            e = np.array([[-b[0],0.0,b[2]],[0.0,-b[1],b[2]],
                          [b[0],-b[1],0.0],[b[0],0.0,-b[2]],
                          [0.0,b[1],-b[2]],[-b[0],b[1],0.0]])
        
        n = np.cross(b, e[(np.arange(6)+1)%6]-e[np.arange(6)])
        e = e / np.linalg.norm(e, axis=1)[:,None]
        
    elif crystal in ['FCC', 'fcc']:
        Nsides = 4
        b0 = 1.0/np.sqrt(2.0)*np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]])
        n01 = np.array([[-1,-1,1],[-1,1,1],[-1,1,1],[1,1,1],[-1,1,1],[1,1,1]])
        n02 = np.array([[1,-1,1],[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,-1],[1,1,-1]])
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
        ib = bcol.argmin()
        if bcol[ib] > 1e-5:
            raise ValueError('FCC Burgers vector must be of the 1/2<110> type in insert_prismatic_loop()')
        p1 = n01[ib] / np.linalg.norm(n01[ib])
        p2 = n02[ib] / np.linalg.norm(n02[ib])
        l1 = np.cross(p1, b)
        l1 = l1 / np.linalg.norm(l1)
        l2 = np.cross(p2, b)
        l2 = l2 / np.linalg.norm(l2)
        e = np.array([-0.5*l1-0.5*l2, +0.5*l1-0.5*l2, +0.5*l1+0.5*l2, -0.5*l1+0.5*l2])
        n = np.array([p1, p2, p1, p2])
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in insert_prismatic_loop()' % crystal)
    
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        e = np.matmul(e, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    istart = len(nodes)
    Nnodes = 0
    for i in range(Nsides):
        l = radius*(e[(i+1)%Nsides]-e[i])
        Nseg = int(np.ceil(np.linalg.norm(l)/maxseg)) if maxseg > 0 else 1
        for j in range(Nseg):
            p = radius*e[i]+1.0*j/Nseg*l+center
            nodes.append(np.concatenate((p, [NodeConstraints.UNCONSTRAINED])))
            n1 = istart+Nnodes
            n2 = istart if (i == Nsides-1 and j == Nseg-1) else n1+1
            segs.append(np.concatenate(([n1, n2], b, n[i])))
            Nnodes += 1
            
    return nodes, segs


def generate_line_config(crystal, Lbox, num_lines, theta=None, maxseg=-1, Rorient=None, seed=-1, verbose=True):
    """Generate a configuration made of straight, infinite dislocation lines
    * Dislocation lines are generated by cycling through the list of signed
      slip systems (+/- Burgers vectors). I.e., for a balanced configuration 
      (neutral Burgers charge), it is advised to use a number of dislocation lines 
      as a multiple of 24 (=12*2), so that dislocation dipoles are created.
    * If a list of character angles (theta) is provided, each dislocation will be
      randomly assigned one of the character angles from the list. If not provided,
      the character angles will be chosen such that the dislocation density is
      roughly equal between all slip systems.
    Arguments:
    * crystal: crystal structure
    * Lbox: box size or cell object
    * num_lines: number of dislocation lines
    * theta: list of possible character angles in degrees
    * maxseg: maximum discretization length of the lines
    * Rorient: crystal orientation matrix
    * seed: seed for random number generation
    * verbose: print information
    """    
    if verbose: print('generate_line_config()')
    
    if crystal in ['BCC', 'bcc']:
        # Define the 12 <111>{110} slip systems
        b = np.array([
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.]
        ])
        n = np.array([
            [0.,-1.,1.], [0.,-1.,1.], [0.,1.,1.], [0.,1.,1.],
            [1.,0.,1.], [-1.,0.,1.], [1.,0.,1.], [-1.,0.,1.],
            [1.,1.,0.], [-1.,1.,0.], [-1.,1.,0.], [1.,1.,0.]
        ])
        
    elif crystal in ['FCC', 'fcc']:
        # Define the 12 <110>{111} slip systems
        b = np.array([
            [0.,1.,-1.], [1.,0.,-1.], [1.,-1.,0.],
            [0.,1.,-1.], [1.,0.,1.], [1.,1.,0.],
            [0.,1.,1.], [1.,0.,-1.], [1.,1.,0.],
            [0.,1.,1.], [1.,0.,1.], [1.,-1.,0.]
        ])
        n = np.array([
            [1.,1.,1.], [1.,1.,1.], [1.,1.,1.],
            [-1.,1.,1.], [-1.,1.,1.], [-1.,1.,1.],
            [1.,-1.,1.], [1.,-1.,1.], [1.,-1.,1.],
            [1.,1.,-1.], [1.,1.,-1.], [1.,1.,-1.]
        ])
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in generate_line_config()' % crystal)
    
    nsys = b.shape[0]
    b = b / np.linalg.norm(b, axis=1)[:,None]
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    cell = pyexadis.Cell(Lbox)
    Lmax = np.max(np.linalg.norm(cell.h, axis=0))
    
    if theta is None:
        # Determine the character angles of each dipole
        # such that the densities among slip systems are close.
        # We need to do this because the line length of each dipole
        # depends on the crystal orientation and slip system,
        # with each of which likely to have a different periodicity.
        # Here we first determine the dipole with maximum length.
        ntheta = 19
        theta = 90.0/(ntheta-1)*np.arange(ntheta)
        theta_minlength = np.zeros((nsys, ntheta))
        for isys in range(nsys):
            burg, plane = b[isys], n[isys]
            c = np.array(cell.center())
            # Find character angle that minimizes the line length
            minlength = 1e20
            for t in range(ntheta):
                nodes, segs = [], []
                length = insert_infinite_line(cell, nodes, segs, burg, plane, c,
                                              theta=theta[t], maxseg=maxseg, trial=True)
                theta_minlength[isys,t] = length
        
        # Maximum dipole size among all slip systems
        theta_minlength = np.ma.masked_less(theta_minlength, 0.0)
        minlength = theta_minlength.min(axis=1).filled(-1.0)
        maxlength = np.max(minlength)
        if maxlength > 10*Lmax or np.min(minlength) < 0.0:
            raise ValueError('Error: cannot find appropriate line to insert')
        
        # Select character angle for the slip system that is
        # the closest to the maximum dipole length across
        # all the slip systems
        theta_sys = np.argmin(np.abs(theta_minlength-maxlength), axis=1)
        theta_sys = theta[theta_sys][:,None]
    else:
        theta_sys = np.tile(np.array(theta), (nsys, 1))
    
    # Insert the lines
    if seed > 0: np.random.seed(seed)
    pos = np.random.rand(num_lines, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    ithe = np.random.randint(0, theta_sys.shape[1], num_lines)
    nodes, segs = [], []
    
    for i in range(num_lines):
        isys = i % nsys
        burg, plane = b[isys], n[isys]
        
        idip = np.floor(i/nsys).astype(int) % 2 # alternate sign to create dipoles
        lsign = 1-2*idip
        
        edir = np.cross(plane, burg)
        edir = edir / np.linalg.norm(edir)
        theta = theta_sys[isys,ithe[i-idip*nsys]]
        ldir = np.cos(theta*np.pi/180.0)*burg + np.sin(theta*np.pi/180.0)*edir
        
        if verbose: print(' insert dislocation: b = %.3f %.3f %.3f, n = %.3f %.3f %.3f, theta = %.1f deg' % (*burg, *plane, theta))
        nodes, segs = insert_infinite_line(cell, nodes, segs, burg, plane, pos[i],
                                           linedir=lsign*ldir, maxseg=maxseg)
    
    G = ExaDisNet(cell, nodes, segs)
    return G


def generate_prismatic_config(crystal, Lbox, num_loops, radius, maxseg=-1, Rorient=None, seed=-1, uniform=False):
    """Generate a configuration made of prismatic dislocation loops
    * Dislocation loops are generated by cycling through the list of native
      Burgers vectors for the crystal structure: 6 1/2<110> Burgers vectors
      for fcc and 4 1/2<111> Burgers vectors for bcc.
    Arguments:
    * crystal: crystal structure
    * Lbox: box size or cell object
    * num_loops: number of dislocation loops
    * radius: radius of the loops, or [min_radius, max_radius]
    * maxseg: maximum discretization length of the lines
    * Rorient: crystal orientation matrix
    * seed: seed for random number generation
    * uniform: make the spatial loop distribution close to uniform 
    """    
    #print('generate_prismatic_config()')
    if crystal in ['BCC', 'bcc']:
        b = np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
    elif crystal in ['FCC', 'fcc']:
        b = np.array([[1.,1.,0.],[-1.,1.,0.],[1.,0.,1.],[-1.,0.,1.],[0.,1.,1.],[0.,-1.,1.]])
    else:
        raise ValueError('Error: unsupported crystal type = %s in generate_prismatic_config()' % crystal)
    
    nburg = b.shape[0]
    b = b / np.linalg.norm(b, axis=1)[:,None]
    
    # Insert the loops
    cell = pyexadis.Cell(Lbox)
    if seed > 0: np.random.seed(seed)
    if uniform:
        # random uniform positions
        ngrid = np.ceil((1.0*num_loops)**(1.0/3.0))
        H = 1.0/ngrid
        x = 0.5*H + H*np.arange(ngrid)
        x, y, z = np.meshgrid(x, x, x)
        p = np.random.permutation(len(x.flatten()))
        x, y, z = x.flatten()[p], y.flatten()[p], z.flatten()[p]
        pos = np.vstack((x, y, z)).T + 0.5*H*(np.random.rand(len(x), 3)-0.5)
    else:
        pos = np.random.rand(num_loops, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    if isinstance(radius, list):
        R = np.random.uniform(radius[0], radius[1], size=(num_loops,))
    else:
        R = radius*np.ones(num_loops)
    
    nodes, segs = [], []
    for i in range(num_loops):
        iburg = i % nburg
        burg = b[iburg]
        nodes, segs = insert_prismatic_loop(crystal, cell, nodes, segs, burg,
                                            R[i], pos[i], maxseg, Rorient)
    
    G = ExaDisNet(cell, nodes, segs)
    return G


def get_segments_end_points(N: DisNetManager) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the list of dislocation segments end points of the network
    for which the closest image convention is applied to the second end point
    """
    data = N.export_data()
    # cell
    cell = pyexadis.Cell(**data["cell"])
    # nodes
    nodes = data.get("nodes")
    rn = nodes.get("positions")
    # segments
    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    # end points
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))
    return r1, r2


def get_segments_length(N: DisNetManager) -> np.ndarray:
    """ Returns the list of dislocation segment lenghts of the network
    """
    r1, r2 = get_segments_end_points(N)
    Lseg = np.linalg.norm(r2-r1, axis=1)
    return Lseg


def dislocation_density(N: DisNetManager, burgmag: float) -> float:
    """ Returns the dislocation density of the network
    """
    len = get_segments_length(N).sum()
    vol = np.abs(np.linalg.det(N.export_data().get("cell")["h"]))
    rho = len/vol/burgmag**2
    return rho


def dislocation_charge(N: DisNetManager) -> np.ndarray:
    """ Returns the dislocation charge (net Nye's tensor) of the network
    """
    r1, r2 = get_segments_end_points(N)
    t = r2-r1
    b = N.export_data()["segs"]["burgers"]
    alpha = np.einsum('ij,ik->jk', b, t)
    return alpha


def read_paradis(datafile: str) -> DisNetManager:
    """ Read dislocation network in ParaDiS format
    """
    G = ExaDisNet().read_paradis(datafile)
    return DisNetManager(G)


def replicate_network(N: DisNetManager, Nrep) -> DisNetManager:
    """ Periodically replicate a dislocation network along the three dimensions
    """
    import copy
    
    if np.isscalar(Nrep): Nrep = Nrep*np.ones(3)
    Nrep = np.array(Nrep).astype(int)
    if np.any(Nrep < 1):
        raise ValueError('replicate_network(): periodic replica (%d,%d,%d) must be at least 1 in each direction' % tuple(Nrep))
    if np.all(Nrep == 1):
        return N
    
    # cell
    data = N.export_data()
    cell0 = pyexadis.Cell(**data["cell"])
    h0 = np.array(cell0.h)
    c1, c2, c3 = h0[:,0], h0[:,1], h0[:,2]
    data["cell"]["h"] *= Nrep
    nodes = data["nodes"]
    nodes0 = copy.deepcopy(nodes)
    num_nodes = nodes["positions"].shape[0]
    
    # periodic nodes replica
    for i3 in range(Nrep[2]):
        for i2 in range(Nrep[1]):
            for i1 in range(Nrep[0]):
                if i1 == 0 and i2 == 0 and i3 == 0: continue
                cnodes = copy.deepcopy(nodes0)
                cnodes["positions"] += i1*c1 + i2*c2 + i3*c3
                for k, v in nodes.items():
                    nodes[k] = np.vstack((nodes[k], cnodes[k]))
    
    # periodic link replica
    segs = data["segs"]
    segs0 = copy.deepcopy(segs)
    nodeids = segs["nodeids"]
    num_segs = nodeids.shape[0]
    p1 = nodes0["positions"][nodeids[:,0]]
    p2 = nodes0["positions"][nodeids[:,1]]
    p2 = np.array(cell0.closest_image(Rref=p1, R=p2))
    repseg = ~np.array(cell0.are_inside(p2), dtype=bool)
    
    nr = 1
    for i3 in range(Nrep[2]):
        for i2 in range(Nrep[1]):
            for i1 in range(Nrep[0]):
                if i1 == 0 and i2 == 0 and i3 == 0: continue
                csegs = copy.deepcopy(segs0)
                csegs["nodeids"] += nr * num_nodes
                for k, v in segs.items():
                    segs[k] = np.vstack((segs[k], csegs[k]))
                nr += 1
    
    # reconnect links across PBC boundaries
    cell = pyexadis.Cell(**data["cell"])
    for s in range(num_segs):
        if not repseg[s]: continue
        for r1 in range(np.prod(Nrep)):
            sr = s + r1 * num_segs
            n1 = segs["nodeids"][sr,0]
            n2 = segs["nodeids"][sr,1] % num_nodes
            # find closest neighbor among PBC images
            n2p = [n2 + r2 * num_nodes for r2 in range(np.prod(Nrep))]
            p1 = nodes["positions"][n1]
            p2 = nodes["positions"][n2p]
            p2 = np.array(cell.closest_image(Rref=p1, R=p2))
            dist = np.linalg.norm(p2-p1, axis=1)
            imin = np.argmin(dist).ravel()[0]
            segs["nodeids"][sr,1] = n2p[imin]
    
    # reset node tags to make sure they are unique
    num_tot_nodes = num_nodes * np.prod(Nrep)
    nodes["tags"] = np.stack((np.zeros(num_tot_nodes), np.arange(num_tot_nodes))).T
    Nnew = DisNetManager(ExaDisNet().import_data(data))
    
    return Nnew


def combine_networks(Nlist) -> DisNetManager:
    """ Combine several DisNetManager into a single network
    """
    if not isinstance(Nlist, list) or len(Nlist) == 0:
        raise ValueError('combine_networks() argument must be a list of DisNetManager')
    
    # combine networks
    for i, Ni in enumerate(Nlist):
        if i == 0:
            data = Ni.export_data()
            nodes, segs = data["nodes"], data["segs"]
            num_nodes = Ni.num_nodes()
        else:
            datai = Ni.export_data()
            if not np.all(datai["cell"]["h"] == data["cell"]["h"]) or \
               not np.all(datai["cell"]["origin"] == data["cell"]["origin"]):
                raise ValueError('combine_networks() networks must use the same cell')
            for k, v in nodes.items():
                nodes[k] = np.vstack((nodes[k], datai["nodes"][k]))
            for k, v in segs.items():
                if k == 'nodeids':
                    segs[k] = np.vstack((segs[k], datai["segs"][k]+num_nodes))
                else:
                    segs[k] = np.vstack((segs[k], datai["segs"][k]))
            num_nodes += Ni.num_nodes()
            
    # reset node tags to make sure they are unique
    nodes["tags"] = np.stack((np.zeros(num_nodes), np.arange(num_nodes))).T
    N = DisNetManager(ExaDisNet().import_data(data))
    return N


def extract_segments(N: DisNetManager, seglist) -> DisNetManager:
    """ Return a new network that contains a subset of segments
    from the input network
    """
    data = N.export_data()
    # keep segments from the list
    segs = data["segs"]
    for k, v in segs.items():
        segs[k] = v[seglist]
    # remove unconnected nodes
    nodelist, nind = np.unique(segs["nodeids"].ravel(), return_inverse=True)
    nodes = data["nodes"]
    for k, v in nodes.items():
        nodes[k] = v[nodelist]
    # update node indices
    segs["nodeids"] = nind[np.arange(segs["nodeids"].size).reshape(-1,2)]
    # create new DisNet
    G = ExaDisNet().import_data(data)
    return DisNetManager(G)


def delete_segments(N: DisNetManager, seglist) -> DisNetManager:
    """ Return a new network in which segments have been deleted
    from the input network
    """
    keeplist = np.setxor1d(seglist, np.arange(N.num_segments()))
    return extract_segments(N, keeplist)


def write_data(N: DisNetManager, datafile: str):
    """ Write dislocation network in ParaDiS format
    """
    N.get_disnet(ExaDisNet).write_data(datafile)


def write_vtk(N: DisNetManager, vtkfile: str, segprops={}, pbc_wrap=True):
    """ Write dislocation network in vtk format
    """
    data = N.export_data()
    # cell
    cell = pyexadis.Cell(**data["cell"])
    cell_origin, cell_center, h = np.array(cell.origin), np.array(cell.center()), np.array(cell.h)
    c = cell_origin + np.array([np.zeros(3), h[0], h[1], h[2], h[0]+h[1],
                                h[0]+h[2], h[1]+h[2], h[0]+h[1]+h[2]])
    # nodes
    nodes = data.get("nodes")
    rn = nodes.get("positions")
    # segments
    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))
    b = segs.get("burgers")
    p = segs.get("planes")
    
    # PBC wrapping
    if np.all(np.array(cell.is_periodic()) == 0): pbc_wrap = False
    if pbc_wrap:
        
        eps = 1e-10
        hinv = np.linalg.inv(h)
        is_periodic = np.array(cell.is_periodic())
        def outside_box(p):
            s = np.matmul(hinv, p - cell_origin)
            return np.any(((s < -eps)|(s > 1.0+eps))&(is_periodic))
        
        def facet_intersection_position(r1, r2, i):
            s1 = np.matmul(hinv, r1 - cell_origin)
            s2 = np.matmul(hinv, r2 - cell_origin)
            t = s2 - s1
            t0 = -(s1 - 0.0) / (t + eps)
            t1 = -(s1 - 1.0) / (t + eps)
            s = np.hstack((t0, t1))
            s[s < eps] = 1.0
            facet = np.argmin(s)
            if s[facet] < 1.0:
                pos = np.matmul(h, s1 + s[facet]*t) + cell_origin
                sfacet = s[facet]
            else:
                facet = -1
                pos = r2
                sfacet = 1.0
            return pos, facet, sfacet
            
        segsid = []
        rsegs = []
        for i in range(segsnid.shape[0]):
            n1, n2 = segsnid[i]
            r1 = np.array(cell.closest_image(Rref=cell_center, R=rn[n1]))
            r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))
            out = outside_box(r2)
            while out:
                pos, facet, sfacet = facet_intersection_position(r1, r2, i)
                if facet < 0: break
                segsid.append(i)
                rsegs.append([r1, pos])
                r1 = pos + (1-2*np.floor(facet/3)) * (1.0-2*eps)*h[:,facet%3]
                r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))
                out = outside_box(r2)
            segsid.append(i)
            rsegs.append([r1, r2])
        
        segsid = np.array(segsid).astype(int)
        nsegs = segsid.shape[0]
        rsegs = np.array(rsegs).reshape(-1,3)
        b = b[segsid]
        p = p[segsid]
        for k, v in segprops.items():
            segprops[k] = v[segsid]
    else:
        nsegs = segsnid.shape[0]
        rsegs = np.hstack((r1, r2)).reshape(-1,3)
    
    f = open(vtkfile, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Configuration exported from OpenDiS\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    
    f.write("POINTS %d FLOAT\n" % (c.shape[0]+2*nsegs))
    np.savetxt(f, c, fmt='%f')
    np.savetxt(f, rsegs, fmt='%f')
    
    f.write("CELLS %d %d\n" % (1+nsegs, 9+3*nsegs))
    f.write("8 0 1 4 2 3 5 7 6\n")
    nid = np.hstack((2*np.ones((nsegs,1)), np.arange(2*nsegs).reshape(-1,2)+8))
    np.savetxt(f, nid, fmt='%d')
    
    f.write("CELL_TYPES %d\n" % (1+nsegs))
    f.write("12\n")
    np.savetxt(f, 4*np.ones(nsegs), fmt='%d')
    
    f.write("CELL_DATA %d\n" % (1+nsegs))
    
    f.write("VECTORS Burgers FLOAT\n")
    f.write("%f %f %f\n" % tuple(np.zeros(3)))
    np.savetxt(f, b, fmt='%f')
    
    f.write("VECTORS Planes FLOAT\n")
    f.write("%f %f %f\n" % tuple(np.zeros(3)))
    np.savetxt(f, p, fmt='%f')
    
    for k, v in segprops.items():
        vals = np.atleast_2d(v.T).T
        if vals.shape[0] != nsegs:
            raise ValueError('segprop value must the same size as the number of segments')
        f.write("SCALARS %s FLOAT %d\n" % (str(k), vals.shape[1]))
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, np.vstack((np.zeros(vals.shape[1]), vals)), fmt='%f')
    
    f.close()

def write_vtk_character(N: DisNetManager, vtkfile: str, segprops={}, pbc_wrap=True):
    """ Write dislocation network in vtk format
        Junjie: add character angle as cell data to each segment, add also classify the
        slip plane of each segment. slip plane types: 1. {110} 2. {112} 3. {123} 4. others
    """ 
    data = N.export_data()
    # cell
    cell = pyexadis.Cell(**data["cell"])
    cell_origin, cell_center, h = np.array(cell.origin), np.array(cell.center()), np.array(cell.h)
    c = cell_origin + np.array([np.zeros(3), h[0], h[1], h[2], h[0]+h[1],
                                h[0]+h[2], h[1]+h[2], h[0]+h[1]+h[2]])
    # nodes
    nodes = data.get("nodes")
    rn = nodes.get("positions")
    # segments
    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))
    b = segs.get("burgers")
    p = segs.get("planes")
    
    # PBC wrapping
    if np.all(np.array(cell.is_periodic()) == 0): pbc_wrap = False
    if pbc_wrap:
        
        eps = 1e-10
        hinv = np.linalg.inv(h)
        is_periodic = np.array(cell.is_periodic())
        def outside_box(p):
            s = np.matmul(hinv, p - cell_origin)
            return np.any(((s < -eps)|(s > 1.0+eps))&(is_periodic))
        
        def facet_intersection_position(r1, r2, i):
            s1 = np.matmul(hinv, r1 - cell_origin)
            s2 = np.matmul(hinv, r2 - cell_origin)
            t = s2 - s1
            t0 = -(s1 - 0.0) / (t + eps)
            t1 = -(s1 - 1.0) / (t + eps)
            s = np.hstack((t0, t1))
            s[s < eps] = 1.0
            facet = np.argmin(s)
            if s[facet] < 1.0:
                pos = np.matmul(h, s1 + s[facet]*t) + cell_origin
                sfacet = s[facet]
            else:
                facet = -1
                pos = r2
                sfacet = 1.0
            return pos, facet, sfacet
            
        segsid = []
        rsegs = []
        for i in range(segsnid.shape[0]):
            n1, n2 = segsnid[i]
            r1 = np.array(cell.closest_image(Rref=cell_center, R=rn[n1]))
            r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))
            out = outside_box(r2)
            while out:
                pos, facet, sfacet = facet_intersection_position(r1, r2, i)
                if facet < 0: break
                segsid.append(i)
                rsegs.append([r1, pos])
                r1 = pos + (1-2*np.floor(facet/3)) * (1.0-2*eps)*h[:,facet%3]
                r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))
                out = outside_box(r2)
            segsid.append(i)
            rsegs.append([r1, r2])
        
        segsid = np.array(segsid).astype(int)
        nsegs = segsid.shape[0]
        rsegs = np.array(rsegs).reshape(-1,3)
        b = b[segsid]
        p = p[segsid]
        charactor_angles = []
        for i in range(nsegs):
            burg_vec = b[i]
            line_vec = rsegs[2*i+1] - rsegs[2*i]
            burg_vec_norm = np.linalg.norm(burg_vec)
            line_vec_norm = np.linalg.norm(line_vec)
            cos_theta = np.dot(burg_vec, line_vec) / (burg_vec_norm * line_vec_norm)
            # make sure the theta is acute
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta) * 180.0 / np.pi
            theta = min(theta, 180.0 - theta)
            charactor_angles.append(theta)
        charactor_angles = np.array(charactor_angles)

        slip_plane_types = []
        for i in range(nsegs):
            plane_vec = p[i]
            # normalize the normal vector
            normal = plane_vec / np.linalg.norm(plane_vec)
            # convert to approximated integer indices
            # Find the smallest integer to scale the normal vector
            abs_normal = np.abs(normal)
            non_zero = abs_normal[abs_normal > 1e-6]
            if len(non_zero) == 0:
                slip_plane_types.append(4)  # others
                continue
            scale_factor = 1.0 / np.min(non_zero)
            scaled_abs_normal = abs_normal * scale_factor
            int_abs_indices = np.round(scaled_abs_normal).astype(int)

            # Normalize the indices by their greatest common divisor
            h, k, l = int_abs_indices
            gcd_value = np.gcd(np.gcd(abs(h), abs(k)), abs(l))
            if gcd_value > 0:
                h //= gcd_value
                k //= gcd_value
                l //= gcd_value

            # Classify the slip plane type
            indices = sorted([abs(h), abs(k), abs(l)])
            if indices == [0,1,1]:
                slip_plane_types.append(1)  # {110}
            elif indices == [1,1,2]:
                slip_plane_types.append(2)  # {112}
            elif indices == [1,2,3]:    
                slip_plane_types.append(3)  # {123}
            else:
                slip_plane_types.append(4)  # others


        for k, v in segprops.items():
            segprops[k] = v[segsid]
    else:
        nsegs = segsnid.shape[0]
        rsegs = np.hstack((r1, r2)).reshape(-1,3)
    
    f = open(vtkfile, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Configuration exported from OpenDiS\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    
    f.write("POINTS %d FLOAT\n" % (c.shape[0]+2*nsegs))
    np.savetxt(f, c, fmt='%f')
    np.savetxt(f, rsegs, fmt='%f')
    
    f.write("CELLS %d %d\n" % (1+nsegs, 9+3*nsegs))
    f.write("8 0 1 4 2 3 5 7 6\n")
    nid = np.hstack((2*np.ones((nsegs,1)), np.arange(2*nsegs).reshape(-1,2)+8))
    np.savetxt(f, nid, fmt='%d')
    
    f.write("CELL_TYPES %d\n" % (1+nsegs))
    f.write("12\n")
    np.savetxt(f, 4*np.ones(nsegs), fmt='%d')
    
    f.write("CELL_DATA %d\n" % (1+nsegs))
    
    f.write("VECTORS Burgers FLOAT\n")
    f.write("%f %f %f\n" % tuple(np.zeros(3)))
    np.savetxt(f, b, fmt='%f')
    
    f.write("VECTORS Planes FLOAT\n")
    f.write("%f %f %f\n" % tuple(np.zeros(3)))
    np.savetxt(f, p, fmt='%f')

    f.write("SCALARS Character_Angle FLOAT 1\n")
    f.write("LOOKUP_TABLE default\n")
    f.write("%f\n" % 0.0)
    np.savetxt(f, charactor_angles, fmt='%f')

    f.write("SCALARS Slip_Plane_Type INT 1\n")
    f.write("LOOKUP_TABLE default\n")
    f.write("%d\n" % 0)
    np.savetxt(f, slip_plane_types, fmt='%d')
    
    for k, v in segprops.items():
        vals = np.atleast_2d(v.T).T
        if vals.shape[0] != nsegs:
            raise ValueError('segprop value must the same size as the number of segments')
        f.write("SCALARS %s FLOAT %d\n" % (str(k), vals.shape[1]))
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, np.vstack((np.zeros(vals.shape[1]), vals)), fmt='%f')
    
    f.close()


# FCC的12个滑移系定义
# 滑移面 {111} 和滑移方向 <110>
FCC_SLIP_SYSTEMS = {
    'planes': np.array([
        [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1],  # (111) 面
        [-1,  1,  1], [-1,  1,  1], [-1,  1,  1],  # (-111) 面
        [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1],  # (1-11) 面
        [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1],  # (11-1) 面
    ], dtype=float),#滑移面法向量
    'directions': np.array([
        [ 1, -1,  0], [ 1,  0, -1], [ 0,  1, -1],  # (111) 面上的滑移方向
        [ 1,  1,  0], [ 1,  0, -1], [ 0,  1,  1],  # (-111) 面上的滑移方向
        [ 1,  1,  0], [ 1,  0,  1], [ 0,  1, -1],  # (1-11) 面上的滑移方向
        [ 1, -1,  0], [ 1,  0,  1], [ 0,  1,  1],  # (11-1) 面上的滑移方向
    ], dtype=float)#滑移方向向量
}

def normalize_vector(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:#避免除以零
        return v
    return v / norm

def is_parallel(v1, v2, tol=1e-6):
    """判断两个向量是否平行（包括反平行）"""
    v1_norm = normalize_vector(v1)#归一化向量1
    v2_norm = normalize_vector(v2)
    dot = np.abs(np.dot(v1_norm, v2_norm))#计算两个归一化向量的点积的绝对值
    return dot > (1.0 - tol)#如果点积接近1，则认为两个向量平行

def identify_fcc_slip_system(burgers, plane, tol=1e-6):
    """
    识别是否为FCC标准滑移系
    
    Parameters:
    -----------
    burgers : array-like, 伯格斯矢量
    plane : array-like, 滑移面法向量
    tol : float, 容差
    
    Returns:
    --------
    int : 12 表示标准FCC滑移系，13 表示其他滑移情况
    """
    burgers = np.array(burgers, dtype=float)#将输入转换为浮点型数组
    plane = np.array(plane, dtype=float)
    
    # 检查是否匹配12个FCC滑移系中的任意一个
    for i in range(12):#遍历12个FCC滑移系
        fcc_plane = FCC_SLIP_SYSTEMS['planes'][i]#取第 i 个滑移面法向量
        fcc_dir = FCC_SLIP_SYSTEMS['directions'][i]#取第 i 个滑移方向向量
        
        # 检查滑移面和滑移方向是否都匹配
        if is_parallel(plane, fcc_plane, tol) and is_parallel(burgers, fcc_dir, tol):#判断滑移面和滑移方向是否都匹配
            return 12
    
    return 13

def compute_slip_system_labels(burgers_vectors, plane_vectors, tol=1e-6):
    """
    批量计算滑移系标签
    
    Parameters:
    -----------
    burgers_vectors : ndarray, shape (n, 3), 伯格斯矢量数组
    plane_vectors : ndarray, shape (n, 3), 滑移面法向量数组
    tol : float, 容差
    
    Returns:
    --------
    ndarray : 滑移系标签数组 (12 或 13)
    """
    n = burgers_vectors.shape[0]#位错段数量
    labels = np.zeros(n, dtype=int)#创建长度为 n 的整型数组，初始化为 0
    
    for i in range(n):
        labels[i] = identify_fcc_slip_system(burgers_vectors[i], plane_vectors[i], tol)#计算每个位错段的滑移系标签
    
    return labels


def write_vtk_fcc(N: DisNetManager, vtkfile: str, segprops={}, pbc_wrap=True, identify_slip_system=True, slip_tol=1e-6):
    """
    Write dislocation network in vtk format
    
    Parameters:
    -----------
    N : DisNetManager, 位错网络管理器
    vtkfile : str, 输出VTK文件路径
    segprops : dict, 额外的段属性
    pbc_wrap : bool, 是否进行周期性边界条件包裹
    identify_slip_system : bool, 是否识别FCC滑移系 (新增参数)
    slip_tol : float, 滑移系识别容差 (新增参数)
    """
    data = N.export_data()
    
    # cell
    cell = pyexadis.Cell(**data["cell"])#用 data["cell"] 的字段构造一个 Cell 对象（** 解包字典为关键字参数）
    cell_origin, cell_center, h = np.array(cell.origin), np.array(cell.center()), np.array(cell.h)#取出晶胞原点 origin、中心 center、晶胞矩阵 h，并转成 numpy 数组
    c = cell_origin + np.array([np.zeros(3), h[0], h[1], h[2], h[0]+h[1],
                                h[0]+h[2], h[1]+h[2], h[0]+h[1]+h[2]])#计算晶胞8个顶点的坐标  origin + (0, a, b, c, a+b, a+c, b+c, a+b+c)
    
    # nodes
    nodes = data.get("nodes")#取节点数据字典
    rn = nodes.get("positions")#取节点坐标数组 rn（每个节点一个 3D 坐标）
    
    # segments
    segs = data.get("segs")#取段数据字典
    segsnid = segs.get("nodeids")#取段的节点索引数组 segsnid（每个段由两个节点索引定义）
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))#取每段第一个端点的坐标，并映射到“离 cell.center 最近的周期像”
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))#取每段第二个端点的坐标，并映射到“离 r1 最近的周期像”
    b = segs.get("burgers")#取段的伯格斯矢量数组 b
    p = segs.get("planes")#取段的滑移面法向量数组 p
    
    # PBC wrapping
    if np.all(np.array(cell.is_periodic()) == 0): pbc_wrap = False#若晶胞三个方向都不是周期的（is_periodic 全为 0），强制关闭 pbc_wrap
    if pbc_wrap:
        
        eps = 1e-10#很小的数，用于避免除 0 与边界判断抖动
        hinv = np.linalg.inv(h)#计算晶胞矩阵 h 的逆矩阵 hinv
        is_periodic = np.array(cell.is_periodic())#取周期性标志（每个方向 0/1）
        
        def outside_box(p):#判断点 p 是否在晶胞外部
            s = np.matmul(hinv, p - cell_origin)#计算分数坐标 s，使得 p = origin + h*s（取决于 h 的定义）
            return np.any(((s < -eps)|(s > 1.0+eps))&(is_periodic))#若在任一“周期方向”上分数坐标超出 [0,1]（留 eps 容差），返回 True
        
        def facet_intersection_position(r1, r2, i):#求线段 r1->r2 首次与盒子哪个面相交，以及相交位置
            s1 = np.matmul(hinv, r1 - cell_origin)
            s2 = np.matmul(hinv, r2 - cell_origin)
            t = s2 - s1#分数坐标下的方向向量（参数化用）
            t0 = -(s1 - 0.0) / (t + eps)#分别计算与 s=0 三个面的交点参数（避免 t=0 用 eps）
            t1 = -(s1 - 1.0) / (t + eps)#分别计算与 s=1 三个面的交点参数
            s = np.hstack((t0, t1))#把 6 个候选参数拼成长度 6 的数组（3个“0面”+3个“1面”）
            s[s < eps] = 1.0#把非常小/负的参数设为 1.0（相当于忽略“不在前方”的交点）
            facet = np.argmin(s)# 取最小的正参数对应的面 → 即最先撞到的盒子面索引 facet∈[0..5]
            if s[facet] < 1.0:#如果交点出现在 r1->r2 的线段内部（参数<1）
                pos = np.matmul(h, s1 + s[facet]*t) + cell_origin#计算交点的笛卡尔坐标：origin + h*(s1 + alpha*t)
                sfacet = s[facet]#交点参数
            else:
                facet = -1#用 -1 表示“没撞到面/无需切割”
                pos = r2#交点设为 r2（线段终点）
                sfacet = 1.0
            return pos, facet, sfacet#返回交点坐标、面索引、交点参数
            
        segsid = []#新段列表中每一段对应的“原始段 id”
        rsegs = []#新段的端点坐标列表（每段存 [r_start, r_end]）
        for i in range(segsnid.shape[0]):#遍历每一原始段
            n1, n2 = segsnid[i]#取该段的两个节点 id
            r1 = np.array(cell.closest_image(Rref=cell_center, R=rn[n1]))# 取端点1坐标并选离盒子中心最近的周期像
            r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))# 取端点2坐标并选离端点1最近的周期像
            out = outside_box(r2)#判断 r2 是否在盒子外（周期方向）
            while out:#如果在盒子外，说明该段跨越了周期边界，需要切段
                pos, facet, sfacet = facet_intersection_position(r1, r2, i)#求从 r1 到 r2 首次撞到盒子边界的位置 pos
                if facet < 0: break#facet=-1 表示没有可切割的面，退出循环
                segsid.append(i)#新产生的这一小段来自原始段 i
                rsegs.append([r1, pos])#把切割出来的这一小段 [r1, pos] 存入新段列表
                r1 = pos + (1-2*np.floor(facet/3)) * (1.0-2*eps)*h[:,facet%3]#把 r1 “穿越周期边界”跳到对面继续;(1-2*floor(facet/3)) 给出 +1 或 -1 的方向因子
                r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))#重新计算 r2 的最近周期像（现在参考点换成新的 r1）
                out = outside_box(r2)#再次判断是否还在盒子外：可能一段跨多次周期，需要多次切割
            segsid.append(i)#退出 while 后，把剩余最后一段也记录下来
            rsegs.append([r1, r2])#保存最后一段端点：r1 -> r2（此时 r2 应已在盒子内或无需再切）
        
        segsid = np.array(segsid).astype(int)#把 segsid 转成整型 numpy 数组
        nsegs = segsid.shape[0]#新段总数（切割后可能大于原始段数）
        rsegs = np.array(rsegs).reshape(-1,3)#把新段端点列表转成 numpy 数组，形状 (nsegs*2, 3)
        b = b[segsid]#根据 segsid 取出新段对应的伯格斯矢量
        p = p[segsid]#根据 segsid 取出新段对应的滑移面法向量
        for k, v in segprops.items():#根据 segsid 取出新段对应的其他属性
            segprops[k] = v[segsid]#更新 segprops 中每个属性的值
    else:
        nsegs = segsnid.shape[0]#原始段总数
        rsegs = np.hstack((r1, r2)).reshape(-1,3)#把原始段端点拼接成形状 (nsegs*2, 3) 的数组
    
    # ============================================
    # 新增：计算FCC滑移系标签
    # ============================================
    if identify_slip_system:#如果需要识别滑移系
        slip_system_labels = compute_slip_system_labels(b, p, tol=slip_tol)#计算每段的滑移系标签（12 或 13）
    
    # 写入VTK文件
    f = open(vtkfile, 'w')#打开输出文件（文本写入）
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Configuration exported from OpenDiS\n")
    f.write("ASCII\n")# 指明是 ASCII（不是二进制）
    f.write("DATASET UNSTRUCTURED_GRID\n")
    
    f.write("POINTS %d FLOAT\n" % (c.shape[0]+2*nsegs))# 写点数量：8 个盒子角点 + 每段 2 个端点
    np.savetxt(f, c, fmt='%f')#写入晶胞顶点坐标
    np.savetxt(f, rsegs, fmt='%f')#再把所有段端点坐标写入（共 2*nsegs 行）
    
    f.write("CELLS %d %d\n" % (1+nsegs, 9+3*nsegs))# 写单元数量：1 个盒子 + nsegs 段
    f.write("8 0 1 4 2 3 5 7 6\n")#写盒子单元的连接关系（8个顶点索引） 点数 = 8 顶点索引 = [0,1,4,2,3,5,7,6]
    nid = np.hstack((2*np.ones((nsegs,1)), np.arange(2*nsegs).reshape(-1,2)+8))#写段单元的连接关系（每段2个端点索引） 点数 = 2 顶点索引 = [段起点索引, 段终点索引]，段起点索引从 8 开始
    np.savetxt(f, nid, fmt='%d')#写入所有段的连接关系
    
    f.write("CELL_TYPES %d\n" % (1+nsegs))# 写单元类型数量
    f.write("12\n")#盒子单元类型 12 = VTK_HEXAHEDRON
    np.savetxt(f, 4*np.ones(nsegs), fmt='%d')#段单元类型 4 = VTK_LINE
    
    f.write("CELL_DATA %d\n" % (1+nsegs))#声明接下来写的是每个 cell 的数据（总数 1+nsegs）
    
    f.write("VECTORS Burgers FLOAT\n")#写入伯格斯矢量数据
    f.write("%f %f %f\n" % tuple(np.zeros(3)))#给第一个 cell（盒子）写占位向量 0,0,0（因为盒子本身没有 burgers）
    np.savetxt(f, b, fmt='%f')#写入每段的伯格斯矢量
    
    f.write("VECTORS Planes FLOAT\n")#写入滑移面法向量数据
    f.write("%f %f %f\n" % tuple(np.zeros(3)))#给第一个 cell（盒子）写占位向量 0,0,0
    np.savetxt(f, p, fmt='%f')#写入每段的滑移面法向量
    
    # ============================================
    # 新增：写入滑移系标签
    # 12 = 标准FCC滑移系
    # 13 = 其他滑移情况
    # ============================================
    if identify_slip_system:#如果需要识别滑移系
        f.write("SCALARS SlipSystemType INT 1\n")#声明一个标量场：SlipSystemType，整型，每个 cell 1 个分量
        f.write("LOOKUP_TABLE default\n")#VTK 标量字段常规写法：指定查找表
        f.write("0\n")  # cell的占位值
        np.savetxt(f, slip_system_labels, fmt='%d')#写每条线段的标签（12 或 13）
    
    # 写入其他段属性
    for k, v in segprops.items():#遍历用户传入的额外段属性字典（key=属性名，value=数组）
        vals = np.atleast_2d(v.T).T#把 v 强制变成二维列向量/二维数组，统一后续写入逻辑
        if vals.shape[0] != nsegs:#检查：属性长度必须等于段数（切段后的 nsegs）
            raise ValueError('segprop value must the same size as the number of segments')#不一致就报错，避免写错数据
        f.write("SCALARS %s FLOAT %d\n" % (str(k), vals.shape[1]))#声明标量场，名称为 k，浮点型，每个 cell vals.shape[1] 个分量
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, np.vstack((np.zeros(vals.shape[1]), vals)), fmt='%f')#写入数据，第一行是盒子的占位值 0，后面是每段的属性值
    
    f.close()
