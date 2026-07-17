#!/usr/bin/python3
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator, MultipleLocator
from matplotlib.patches import Rectangle, Circle

# Modified Helmholtz problem on 2**n gridpoints
# Poisson for sigma=0
def problem(sigma, n, d=1, f=lambda x: 0, g=[0, ]*2):
    if d==1:
        # N is the number of interior points and size of the matrix
        N = 2**n - 1
        # h is the distance between two consecutive points
        h = 1.0/(N + 1)
        
        # Mesh includes the endpoints
        meshx = np.linspace(0, 1, N + 2)
        
        # Make problem homogeneous
        S = np.zeros(N + 2)
        S = np.linspace(g[0], g[1], N + 2)
        
        # RHS
        F = np.zeros(N)
        F = f(meshx[1:-1])
        F -= sigma*S[1:-1]
        F[0] += g[0]*h**2
        F[-1] += g[1]*h**2
        
        # Operator
        # Off-diag
        minusones = -np.ones(N-1)
        # Diag
        diag = (2 + sigma*(h**2))*np.ones(N)
        # Tri-diag
        A = (1/h**2)*sparse.diags([minusones[::-1], diag, minusones], [-1,0,1], [N,N], format='csr')
        
        return meshx, A, F, S

def relax(A, v, rhs, iterates, method='Gauss-Seidel', weight=0.6):
    if method == 'Gauss-Seidel':
        L = sparse.tril(A, format='csr')
        U = sparse.triu(A, k=1, format='csr')
        for k in range(iterates):
            vstar = la.spsolve(L, rhs - U@v)
            v = (1 - weight)*v + weight*vstar
        
    elif method == 'Jacobi':
        D = sparse.diags([A.diagonal(), ], [0, ], format='csr')
        R = A.copy()
        R.setdiag(0)
        for k in range(iterates):
            vstar = la.spsolve(D, rhs - R@v)
            v = (1 - weight)*v + weight*vstar
            
    else:
        print(method + ' not implemented')
    
    return v

def restrict(v, method='full-weight'):
    if method == 'inject':
        return v[1::2]
        
    elif method == 'full-weight':
        NN = v.size
        N = v.size//2

        # Generate NxNN matrix

        rows = np.hstack([np.tile(np.arange(N), 3)])

        colindex = np.arange(1, NN, 2)
        cols = np.hstack([colindex, colindex - 1, colindex + 1])

        ones = np.ones(N)
        data = np.hstack([2*ones, ones, ones])

        I = 0.25*sparse.coo_matrix((data, (rows, cols)))
        I.tocsr()
        return I@v
        
    else:
        print(method + ' not implemented')

def interpolate(v):
    N = v.size
    NN = v.size*2 + 1

    # Generate NNxN matrix

    rowindex = np.arange(1, NN, 2)
    rows = np.hstack([rowindex, rowindex - 1, rowindex + 1])

    cols = np.hstack([np.tile(np.arange(N), 3)])

    ones = np.ones(N)
    data = np.hstack([2*ones, ones, ones])

    I = 0.5*sparse.coo_matrix((data, (rows, cols)))
    I.tocsr()
    return I@v

def rec_mg(A, r, e=None, g=[0,0], levels=2, smooths=3, ax=None, mesh=None, true=None, k=0):
    if levels < 2:
        raise ValueError('Levels must be greater than or equal to 2')
    if e is None:
        e = np.zeros_like(r)
    e = relax(A, e, r, smooths)
    
    if ax is not None:
        S = np.linspace(g[0], g[1], e.shape[0]+2)
        if true is None:
            ax[-levels, 0].plot(mesh, [0, *e, 0])
        else:
            ax[-levels, 0].plot(mesh, true(mesh) - ([0, *e, 0] + S), zorder=1)
            
    r2h = restrict(r - A@e)
    ref = int(np.log2(r2h.shape[0] + 1))
    m, A2h, _, S2h = problem(k**2, ref, g=[0,0])
    if levels == 2:
        e2h = la.spsolve(A2h, r2h)
        ax[-1,1].plot(m, [0, *e2h, 0])
    else:
        e2h = rec_mg(A2h, r2h, g=[0,1], levels=levels-1, ax=ax, mesh=m, true=true, k=k)
    
    e += interpolate(e2h)
    e = relax(A, e, r, smooths)
    
    if ax is not None:
        if true is None:
            ax[-levels, 2].plot(mesh, [0, *e, 0])
        else:
            ax[-levels, 2].plot(mesh, true(mesh) - ([0, *e, 0] + S), zorder = 0.1)
            
    return e

def uneven_subplots(rows, cols):
    fig = plt.figure()
    
    shape = (rows, 10)
    rowlist = []
    for ii in range(rows-1):
        collist = []
        collist.append(plt.subplot2grid(shape, (ii, 0), colspan=4))
        collist.append(plt.subplot2grid(shape, (ii, 4), colspan=2))
        collist.append(plt.subplot2grid(shape, (ii, 6), colspan=4))
        rowlist.append(np.array(collist))
    
    collist = []
    collist.append(plt.subplot2grid(shape, (rows-1, 0), colspan=3))
    collist.append(plt.subplot2grid(shape, (rows-1, 3), colspan=4))
    collist.append(plt.subplot2grid(shape, (rows-1, 7), colspan=3))
    rowlist.append(np.array(collist))
    
    ax = np.array(rowlist)
    return fig, ax
    

if __name__ == '__main__':
    k = 1
    ell = 3
    levels = 3
    U_exact = lambda x: 1 - np.cos(ell*np.pi*x) - x**2
    f = lambda x: 2.0 + (k**2)*(1 - x**2) - (k**2 + (ell*np.pi)**2)*np.cos(ell*np.pi*x)
    
    base_level = levels + 1
    mesh, A, F, S = problem(k**2, base_level, f=f, g=[0,1])
    fig, ax = uneven_subplots(levels+1, 3)
    fig.set_size_inches(7, 7)
    
    ylabel_pos = (-0.1, 0.9)
    meshlabel_pos = (1.02, 0)
    for axis in ax[0,:].ravel():
        axis.plot(mesh, U_exact(mesh), 'r:')
        axis.set_xlim(0,1)
        axis.set_xticks([0,1])
        axis.text(*ylabel_pos, r'$\phi_h$', transform=axis.transAxes)
        custom_trans = axis.get_yaxis_transform()
        axis.text(*meshlabel_pos, r'$x_h$', transform=custom_trans)
        axis.spines['top'].set_color('none')
        axis.spines['right'].set_color('none')
        axis.spines['bottom'].set_position(('data',0))
        axis.xaxis.set_minor_locator(MultipleLocator(2**-base_level))
        axis.xaxis.set_tick_params(which='minor', direction='inout', length=10)
        axis.set_facecolor((1, 1, 1, 0))
    
    # Draw box around top two
    bl = ax[0,0].transAxes.transform((0,0))
    tr = ax[0,2].transAxes.transform((1,1))
    wh = tr - bl
    
    fbl = fig.transFigure.inverted().transform(bl) - np.array((0.05, 0.01))
    fwh = fig.transFigure.inverted().transform(wh) + np.array((0.08, 0.02))
    
    rec = Rectangle(fbl, *fwh, transform=fig.transFigure,
                    fill=False, lw=2, ec=(0.5,0.5,0.5), joinstyle='round')
    pat = ax[0,0].add_patch(rec)
    pat.set_clip_on(False)
    
    edgepointfwd = []
    edgepointbck = []
    off = 0.1
    for jj, axis_col in enumerate(ax[1:,:].T):
        for ii, axis in enumerate(axis_col):
            axis.plot(mesh, np.zeros_like(mesh), 'k-', lw=0.5)
            axis.set_xlim(0,1)
            axis.set_xticks([0,1])
            if ii != 0:
                axis.text(*ylabel_pos, r'$e_{{{}h}}$'.format(2**ii), transform=axis.transAxes)
                custom_trans = axis.get_yaxis_transform()
                axis.text(*meshlabel_pos, r'$x_{{{}h}}$'.format(2**ii), transform=custom_trans)
            else:
                axis.text(*ylabel_pos, r'$e_h$', transform=axis.transAxes)
                custom_trans = axis.get_yaxis_transform()
                axis.text(*meshlabel_pos, r'$x_h$'.format(2**ii), transform=custom_trans)
            axis.spines['top'].set_color('none')
            axis.spines['right'].set_color('none')
            axis.spines['bottom'].set_position(('data',0))
            axis.xaxis.set_minor_locator(MultipleLocator(2**-(base_level-ii)))
            axis.xaxis.set_tick_params(which='minor', direction='inout', length=10)
            axis.set_facecolor((1, 1, 1, 0))
            if jj==0: # First column
                datpoint = (-off, 0.5)
                axpoint = axis.transAxes.transform(datpoint)
                edgepointfwd.append(fig.transFigure.inverted().transform(axpoint))
            elif jj==2: # Last column
                datpoint = (1+off, 0.5)
                axpoint = axis.transAxes.transform(datpoint)
                edgepointbck.append(fig.transFigure.inverted().transform(axpoint))
        if jj==0: # First column bottom to middle
            edgepointfwd.pop()
            datpoint = (0.25, -off)
            axpoint = ax[-2, 0].transAxes.transform(datpoint)
            edgepointfwd.append(fig.transFigure.inverted().transform(axpoint))
            datpoint = (-off, 0.5)
            axpoint = ax[-1, 1].transAxes.transform(datpoint)
            edgepointfwd.append(fig.transFigure.inverted().transform(axpoint))
        elif jj==2: # Last column bottom to middle
            edgepointbck.pop()
            datpoint = (0.75, -off)
            axpoint = ax[-2, 2].transAxes.transform(datpoint)
            edgepointbck.append(fig.transFigure.inverted().transform(axpoint))
            datpoint = (1, 0.5)
            axpoint = ax[-1, 1].transAxes.transform(datpoint)
            edgepointbck.append(fig.transFigure.inverted().transform(axpoint))
            
            
    for axis in ax[:-1,1]:
        fig.delaxes(axis)
    for axis in ax[-1,0::2]:
        fig.delaxes(axis)
        
    # Draw arrows pointing to meshes
    arrpoints = edgepointfwd + edgepointbck
    for ii in range(levels-2):
        ax[0,0].annotate('', xy=edgepointfwd[ii+1], xycoords='figure fraction',
                        xytext=edgepointfwd[ii], textcoords='figure fraction',
                        arrowprops=dict(color='0.5', width=2, headwidth=10,
                                        shrink=0.05, connectionstyle='arc3,rad=0.5')
                        )
        txt_pos = 0.5*(edgepointfwd[ii] + edgepointfwd[ii+1]) + np.array((-0.04, -0.005))
        ax[0,0].text(*txt_pos, 'Restrict and smooth', color='0.5', transform=fig.transFigure)
        ax[0,0].annotate('', xy=edgepointbck[ii], xycoords='figure fraction',
                        xytext=edgepointbck[ii+1], textcoords='figure fraction',
                        arrowprops=dict(color='0.5', width=2, headwidth=10,
                                        shrink=0.05, connectionstyle='arc3,rad=0.5')
                        )
        txt_pos = 0.5*(edgepointbck[ii] + edgepointbck[ii+1]) + np.array((+0.04, -0.005))
        ax[0,0].text(*txt_pos, 'Prolong and smooth', ha='right', color='0.5', transform=fig.transFigure)
    # End of for loop
    ax[0,0].annotate('', xy=edgepointfwd[-1], xycoords='figure fraction',
                    xytext=edgepointfwd[-2], textcoords='figure fraction',
                    arrowprops=dict(color='0.5', width=2, headwidth=10,
                                    shrink=0.05, connectionstyle='arc3,rad=0.5')
                    )
    txt_pos = 0.5*(edgepointfwd[-2] + edgepointfwd[-1]) + np.array((-0.01, -0.08))
    ax[0,0].text(*txt_pos, 'Restrict and smooth', ha='right', color='0.5', transform=fig.transFigure)
    ax[0,0].annotate('', xy=edgepointbck[-2], xycoords='figure fraction',
                    xytext=edgepointbck[-1], textcoords='figure fraction',
                    arrowprops=dict(color='0.5', width=2, headwidth=10,
                                    shrink=0.05, connectionstyle='arc3,rad=0.5')
                    )
    txt_pos = 0.5*(edgepointbck[-2] + edgepointbck[-1]) + np.array((0.01, -0.08))
    ax[0,0].text(*txt_pos, 'Prolong and smooth', color='0.5', transform=fig.transFigure)
    
    # Label all plots in order
    label_pos = (0.09, 0.9)
    orderedax = np.array([*ax[:-1,0], ax[-1,1], *ax[-2::-1,2]])
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    circ_props = dict(boxstyle='circle', fill=False, ec='k', lw=0.5)
    for text, axis in zip(alphabet, orderedax):
        axis.text(*label_pos, text, bbox=circ_props, transform=axis.transAxes)
    
    # Actually plot the multigrid stuff
    ax[0,0].plot(mesh, S)
    x = rec_mg(A, F, g=[0,1], levels=levels, smooths=0, ax=ax, mesh=mesh, true=U_exact, k=k)
    ax[0,2].plot(mesh, [0, *x, 0] + S)
    x = rec_mg(A, F, g=[0,1], levels=levels, smooths=1, ax=ax, mesh=mesh, true=U_exact, k=k)
    ax[0,2].plot(mesh, [0, *x, 0] + S)
    x = rec_mg(A, F, g=[0,1], levels=levels, smooths=2, ax=ax, mesh=mesh, true=U_exact, k=k)
    ax[0,2].plot(mesh, [0, *x, 0] + S)
    
    # Legend
    labels = ['True solution', '0 Smooths', '1 Smooth', '2 Smooths']
    tl = ax[0,0].transAxes.transform((0,1))
    tr = ax[0,2].transAxes.transform((1,1))
    wh = tr - bl
    
    fbl = fig.transFigure.inverted().transform(tl) + np.array((-0.05, 0.03))
    fwh = fig.transFigure.inverted().transform(wh) + np.array((0.08, 0.1))
    
    ax[0,2].legend( labels,
                    bbox_to_anchor=(*fbl, *fwh),
                    bbox_transform=fig.transFigure,
                    loc='lower left',
                    ncol=4,
                    mode='expand',
                    borderaxespad=0.)
    
    fig.savefig('multigrid.png', dpi=300)
