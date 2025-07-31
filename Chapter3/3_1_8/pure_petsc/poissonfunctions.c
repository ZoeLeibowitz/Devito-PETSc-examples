
// ED BUELER - Petsc4pdes

#include <petsc.h>
#include "poissonfunctions.h"


PetscErrorCode Poisson2DFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                      PetscReal **aF, PoissonCtx *user) {
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
               ue, uw, un, us, uee, uww, uss, unn;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    darea = hx * hy;
    scx = user->cx * hy / hx;
    scy = user->cy * hx / hy;
    scdiag = 2.5 * (scx + scy);    // diagonal scaling
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            // if (i < 2 || i >= info->mx - 2 || j < 2 || j >= info->my - 2) {
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1 || i==1 || i==info->mx-2 || j==1 || j==info->my-2) {
                aF[j][i] = au[j][i] - user->g_bdry(x,y,0.0,user);
                aF[j][i] *= scdiag;
            } else {

                uee = (i+2 == info->mx-1) ? user->g_bdry(x+2*hx, y, 0.0, user) : au[j][i+2];
                uww = (i-2 == 0)         ? user->g_bdry(x-2*hx, y, 0.0, user) : au[j][i-2];
                unn = (j+2 == info->my-1) ? user->g_bdry(x, y+2*hy, 0.0, user) : au[j+2][i];
                uss = (j-2 == 0)         ? user->g_bdry(x, y-2*hy, 0.0, user) : au[j-2][i];

                // // Second neighbors (for 4th-order)
                ue = (i+1 == info->mx-2) ? user->g_bdry(x+hx, y, 0.0, user) : au[j][i+1];
                uw = (i-1 == 1)         ? user->g_bdry(x-hx, y, 0.0, user) : au[j][i-1];
                un = (j+1 == info->my-2) ? user->g_bdry(x, y+hy, 0.0, user) : au[j+1][i];
                us = (j-1 == 1)         ? user->g_bdry(x, y-hy, 0.0, user) : au[j-1][i];


                aF[j][i] = scdiag * au[j][i]
                           + scx * ((1./12.)*uww -(4./3.)*uw -(4./3.)*ue + (1./12.)*uee) + scy * ((1./12.)*uss -(4./3.)*us -(4./3.)*un + (1./12.)*unn)
                           - darea * user->f_rhs(x,y,0.0,user);

                PetscCall(PetscPrintf(PETSC_COMM_WORLD, "aF[%d][%d] = %d\n", j, i, info->mx-2));

            }
        }
    }
    PetscCall(PetscLogFlops(11.0*info->xm*info->ym));

    return 0;
}



// PetscErrorCode Poisson2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
//                                       Mat J, Mat Jpre, PoissonCtx *user) {
//     PetscReal   xymin[2], xymax[2], hx, hy, scx, scy, scdiag, v[9];
//     PetscInt    i,j,ncols;
//     MatStencil  col[9],row;

//     PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
//     hx = (xymax[0] - xymin[0]) / (info->mx - 1);
//     hy = (xymax[1] - xymin[1]) / (info->my - 1);
//     scx = user->cx * hy / hx;
//     scy = user->cy * hx / hy;
//     scdiag = 2.5 * (scx + scy);
//     for (j = info->ys; j < info->ys+info->ym; j++) {
//         row.j = j;
//         col[0].j = j;
//         for (i = info->xs; i < info->xs+info->xm; i++) {
//             row.i = i;
//             col[0].i = i;
//             ncols = 1;
//             v[0] = scdiag;

//             if (i > 1 && i < info->mx-2 && j > 1 && j < info->my-2) {
//                 // 1st neighbors
//                 col[ncols].j = j;   col[ncols].i = i-1;  v[ncols++] = -(4./3.)*scx;
//                 col[ncols].j = j;   col[ncols].i = i+1;  v[ncols++] = -(4./3.)*scx;
//                 col[ncols].j = j-1; col[ncols].i = i;    v[ncols++] = -(4./3.)*scy;
//                 col[ncols].j = j+1; col[ncols].i = i;    v[ncols++] = -(4./3.)*scy;

//                 // 2nd neighbors
//                 col[ncols].j = j;   col[ncols].i = i-2;  v[ncols++] = (1./12.)*scx;
//                 col[ncols].j = j;   col[ncols].i = i+2;  v[ncols++] = (1./12.)*scx;
//                 col[ncols].j = j-2; col[ncols].i = i;    v[ncols++] = (1./12.)*scy;
//                 col[ncols].j = j+2; col[ncols].i = i;    v[ncols++] = (1./12.)*scy;
//             }
//             PetscCall(MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES));
//         }
//     }

//     PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
//     PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
//     if (J != Jpre) {
//         PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
//         PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
//     }
//     return 0;
// }




PetscErrorCode Poisson2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                      Mat J, Mat Jpre, PoissonCtx *user) {
    PetscReal   xymin[2], xymax[2], hx, hy, scx, scy, scdiag, v[9];
    PetscInt    i,j,ncols;
    MatStencil  col[9],row;

    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    scx = user->cx * hy / hx;
    scy = user->cy * hx / hy;
    scdiag = 2.5 * (scx + scy);
    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;
        col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.i = i;
            col[0].i = i;
            ncols = 1;
            v[0] = scdiag;

            if (i> 1 && i<info->mx-2 && j>1 && j<info->my-2) {
                if (i-1 > 1) {
                    col[ncols].j = j;    col[ncols].i = i-1;  v[ncols++] = -(4./3.)*scx;  }
                if (i+1 < info->mx-2) {
                    col[ncols].j = j;    col[ncols].i = i+1;  v[ncols++] = -(4./3.)*scx;  }
                if (j-1 > 1) {
                    col[ncols].j = j-1;  col[ncols].i = i;    v[ncols++] = -(4./3.)*scy;  }
                if (j+1 < info->my-2) {
                    col[ncols].j = j+1;  col[ncols].i = i;    v[ncols++] = -(4./3.)*scy;  }


                if (i-2 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-2;  v[ncols++] = (1./12.)*scx;  }
                if (i+2 < info->mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+2;  v[ncols++] = (1./12.)*scx;  }
                if (j-2 > 1) {
                    col[ncols].j = j-2;  col[ncols].i = i;    v[ncols++] = (1./12.)*scy;  }
                if (j+2 < info->my-1) {
                    col[ncols].j = j+2;  col[ncols].i = i;    v[ncols++] = (1./12.)*scy;  }
            }
            PetscCall(MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
    if (J != Jpre) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}


PetscErrorCode InitialState(DM da, PetscBool gbdry,
                            Vec u, PoissonCtx *user) {
    DMDALocalInfo  info;
    
    // Initial zeros
    PetscCall(VecSet(u,0.0));

    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, x, y, **au;
    PetscCall(DMDAVecGetArray(da, u, &au));
    PetscCall(DMGetBoundingBox(da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info.mx - 1);
    hy = (xymax[1] - xymin[1]) / (info.my - 1);
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            if (i <= 1 || i >= info.mx - 2 || j <= 1 || j >= info.my - 2) {
                x = xymin[0] + i * hx;
                au[j][i] = user->g_bdry(x,y,0.0,user);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da, u, &au));
    return 0;
}
