#include <petscsnes.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscvec.h>


static char help[] = "Solves 2D Poisson equation,\n\
                      using mirrored boundaries to implement Neumann boundary conditions,\n\
                      ";

                      typedef struct
{
  PetscScalar h_x;
  PetscScalar h_y;
  PetscInt x_M;
  PetscInt x_ltkn0;
  PetscInt x_ltkn1;
  PetscInt x_m;
  PetscInt x_rtkn0;
  PetscInt x_rtkn2;
  PetscInt y_M;
  PetscInt y_ltkn0;
  PetscInt y_ltkn2;
  PetscInt y_m;
  PetscInt y_rtkn0;
  PetscInt y_rtkn1;
}UserCtx0 ;


extern PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
extern PetscErrorCode FormRHS0(DM dm0, Vec B);
extern PetscErrorCode PopulateUserContext0(UserCtx0 * ctx0, DM da);
extern PetscErrorCode FormExact(DM dm0, Vec B);
extern PetscErrorCode MyComputeJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx);


int main(int argc, char **argv)
{
  Mat J0;
  Vec bglobal0;
  DM da0;
  KSP ksp0;
  PC pc0;
  PetscMPIInt size;
  SNES snes0;
  Vec xglobal0;
  Vec exact;
  PetscReal      errinf;

  UserCtx0 ctx0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&(size)));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,4,4,1,1,1,2,NULL,NULL,&(da0)));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATAIJ));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&(snes0)));
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&(J0)));

  PetscCall(SNESSetType(snes0,SNESKSPONLY));
  PetscCall(DMCreateGlobalVector(da0,&(xglobal0)));

  PetscCall(DMCreateGlobalVector(da0,&(xglobal0)));
  PetscCall(DMCreateGlobalVector(da0,&(bglobal0)));

  PetscCall(DMCreateGlobalVector(da0,&(exact)));

  PetscCall(SNESGetKSP(snes0,&(ksp0)));
  PetscCall(KSPSetTolerances(ksp0,1e-10,1e-50,100000.0,10000.0));
  PetscCall(KSPSetType(ksp0,KSPGMRES));
  PetscCall(KSPGetPC(ksp0,&(pc0)));
  PetscCall(PCSetType(pc0,PCNONE));
  PetscCall(KSPSetFromOptions(ksp0));

  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  PetscCall(PopulateUserContext0(&(ctx0), da0));
  PetscCall(MatSetDM(J0,da0));
  PetscCall(DMSetApplicationContext(da0,&(ctx0)));

  PetscCall(SNESSetJacobian(snes0,J0,J0,MyComputeJacobian,&ctx0));

//   PetscCall(MatView(J0, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(FormRHS0(da0,bglobal0));
  PetscCall(FormExact(da0,exact));

//   MatNullSpace nullspace;
//   PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
//   PetscCall(MatSetNullSpace(J0, nullspace));
//   PetscCall(MatSetTransposeNullSpace(J0, nullspace));

//   PetscCall(VecSet(xglobal0,0.001));

  // PetscCall(KSPSet)

  // PetscCall(MatNullSpaceRemove(nullspace, xglobal0));
//   PetscCall(MatNullSpaceRemove(nullspace, bglobal0));
  PetscCall(SNESSolve(snes0,bglobal0,xglobal0));

  // PetscCall(VecView(xglobal0,PETSC_VIEWER_STDOUT_WORLD));

  // compute infinity norm
  PetscCall(VecAXPY(xglobal0,-1.0,exact));   // u <- u + (-1.0) uexact
  PetscCall(VecNorm(xglobal0,NORM_INFINITY,&errinf));

//   PetscCall(VecView(xglobal0,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error |u-uexact|_inf = %.22e\n", errinf));

  PetscCall(VecDestroy(&(bglobal0)));
  PetscCall(VecDestroy(&(xglobal0)));
  PetscCall(VecDestroy(&(exact)));
  PetscCall(MatDestroy(&(J0)));
  PetscCall(SNESDestroy(&(snes0)));
  PetscCall(DMDestroy(&(da0)));

  return 0;

}


PetscErrorCode MyComputeJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx)
{
  PetscInt     i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar  v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil   row, col[5];
  DM           da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

  Hx    = 1.0 / (PetscReal)(M-1);
  Hy    = 1.0 / (PetscReal)(N-1);
  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;

  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i;
      row.j = j;

      if (i == 0 || j == 0 || i == M - 1 || j == N - 1) {
        num  = 0;
        numi = 0;
        numj = 0;
        if (j != 0) {
          v[num]     = -HxdHy;
          col[num].i = i;
          col[num].j = j - 1;
          num++;
          numj++;
        }
        if (i != 0) {
          v[num]     = -HydHx;
          col[num].i = i - 1;
          col[num].j = j;
          num++;
          numi++;
        }
        if (i != M - 1) {
          v[num]     = -HydHx;
          col[num].i = i + 1;
          col[num].j = j;
          num++;
          numi++;
        }
        if (j != N - 1) {
          v[num]     = -HxdHy;
          col[num].i = i;
          col[num].j = j + 1;
          num++;
          numj++;
        }
        v[num]     = (PetscReal)numj * HxdHy + (PetscReal)numi * HydHx;
        col[num].i = i;
        col[num].j = j;
        num++;
        PetscCall(MatSetValuesStencil(J, 1, &row, num, col, v, INSERT_VALUES));
      } else {
        v[0]     = -HxdHy;
        col[0].i = i;
        col[0].j = j - 1;
        v[1]     = -HydHx;
        col[1].i = i - 1;
        col[1].j = j;
        v[2]     = 2.0 * (HxdHy + HydHx);
        col[2].i = i;
        col[2].j = j;
        v[3]     = -HydHx;
        col[3].i = i + 1;
        col[3].j = j;
        v[4]     = -HxdHy;
        col[4].i = i;
        col[4].j = j + 1;
        PetscCall(MatSetValuesStencil(J, 1, &row, 5, col, v, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(J, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}



PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  Vec floc;
  DMDALocalInfo info;
  Vec xloc;

  UserCtx0 * ctx0;
  PetscScalar * f_u_vec;
  PetscScalar * x_u_vec;

  DM dm0 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(floc)));
  PetscCall(VecGetArray(floc,&f_u_vec));
  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* f_u)[info.gxm] = (PetscScalar (*)[info.gxm]) f_u_vec;
  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;


  PetscScalar r10 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r11 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    for (int iy = ctx0->y_m + ctx0->y_ltkn0; iy <= ctx0->y_M - ctx0->y_rtkn0; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (2.0*(r10*x_u[ix + 2][iy + 2] + r11*x_u[ix + 2][iy + 2]) - (r10*x_u[ix + 1][iy + 2] + r10*x_u[ix + 3][iy + 2] + r11*x_u[ix + 2][iy + 1] + r11*x_u[ix + 2][iy + 3]))*ctx0->h_x*ctx0->h_y;
    }
    for (int iy = ctx0->y_M - ctx0->y_rtkn1 + 1; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar r12 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r12 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 2 - (PetscInt)(abs(iy - ctx0->y_M + 1))])/((ctx0->h_y*ctx0->h_y)) - (r12 + x_u[ix + 1][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      PetscScalar r13 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r13 + x_u[ix + 2][2 + (PetscInt)(abs(iy - 1))] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r13 + x_u[ix + 1][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int iy = ctx0->y_m + ctx0->y_ltkn0; iy <= ctx0->y_M - ctx0->y_rtkn0; iy += 1)
    {
      PetscScalar r14 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r14 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r14 + x_u[2 + (PetscInt)(abs(ix - 1))][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_m + ctx0->y_ltkn0; iy <= ctx0->y_M - ctx0->y_rtkn0; iy += 1)
    {
      PetscScalar r15 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r15 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r15 + x_u[ix + 1][iy + 2] + x_u[ix + 2 - (PetscInt)(abs(ix - ctx0->x_M + 1))][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      PetscScalar r16 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r16 + x_u[ix + 2][2 + (PetscInt)(abs(iy - 1))] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r16 + x_u[2 + (PetscInt)(abs(ix - 1))][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      PetscScalar r17 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r17 + x_u[ix + 2][2 + (PetscInt)(abs(iy - 1))] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r17 + x_u[ix + 1][iy + 2] + x_u[ix + 2 - (PetscInt)(abs(ix - ctx0->x_M + 1))][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn1 + 1; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar r18 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r18 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 2 - (PetscInt)(abs(iy - ctx0->y_M + 1))])/((ctx0->h_y*ctx0->h_y)) - (r18 + x_u[2 + (PetscInt)(abs(ix - 1))][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn1 + 1; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar r19 = -2.0*x_u[ix + 2][iy + 2];
      f_u[ix + 2][iy + 2] = (-(r19 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 2 - (PetscInt)(abs(iy - ctx0->y_M + 1))])/((ctx0->h_y*ctx0->h_y)) - (r19 + x_u[ix + 1][iy + 2] + x_u[ix + 2 - (PetscInt)(abs(ix - ctx0->x_M + 1))][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_u_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm0,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(floc)));

  MatNullSpace nullspace;
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, F));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscFunctionReturn(0);
}

PetscErrorCode FormExact(DM dm0, Vec B)
{
  PetscFunctionBeginUser;

  Vec blocal0;
  DMDALocalInfo info;

  PetscScalar * b_u_vec;
  UserCtx0 * ctx0;

  PetscCall(DMGetLocalVector(dm0,&(blocal0)));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_u_vec));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* b_u)[info.gxm] = (PetscScalar (*)[info.gxm]) b_u_vec;

  for (int ix = ctx0->x_m; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_m; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar x = ctx0->h_x*ix;
      PetscScalar y = ctx0->h_y*iy;
      b_u[ix + 2][iy + 2] = PetscCosReal(2.0*PETSC_PI*x)*PetscCosReal(2.0*PETSC_PI*y);
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_u_vec));
  PetscCall(DMRestoreLocalVector(dm0,&(blocal0)));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS0(DM dm0, Vec B)
{
  PetscFunctionBeginUser;

  Vec blocal0;
  DMDALocalInfo info;

  PetscScalar * b_u_vec;
  UserCtx0 * ctx0;

  PetscCall(DMGetLocalVector(dm0,&(blocal0)));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_u_vec));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* b_u)[info.gxm] = (PetscScalar (*)[info.gxm]) b_u_vec;


  for (int ix = ctx0->x_m; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_m; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar x = ctx0->h_x*ix;
      PetscScalar y = ctx0->h_y*iy;
      b_u[ix + 2][iy + 2] = ctx0->h_x*ctx0->h_y*8.0*PETSC_PI*PETSC_PI*PetscCosReal(2.0*PETSC_PI*x)*PetscCosReal(2.0*PETSC_PI*y);
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_u_vec));
  PetscCall(DMRestoreLocalVector(dm0,&(blocal0)));

  // MatNullSpace nullspace;
  // PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
  // PetscCall(MatNullSpaceRemove(nullspace, B));
  // PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscFunctionReturn(0);
}


PetscErrorCode PopulateUserContext0(UserCtx0 * ctx0, DM da)
{
  PetscFunctionBeginUser;

  // get local info
  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(da,&(info)));

  ctx0->h_x = 1.0/(PetscReal)(info.mx - 1);
  ctx0->h_y = 1.0/(PetscReal)(info.my - 1);

  ctx0->x_M = info.mx - 1;
  ctx0->x_ltkn0 = 1;
  ctx0->x_ltkn1 = 1;
  ctx0->x_m = 0;
  ctx0->x_rtkn0 = 1;
  ctx0->x_rtkn2 = 1;
  ctx0->y_M = info.my - 1;
  ctx0->y_ltkn0 = 1;
  ctx0->y_ltkn2 = 1;
  ctx0->y_m = 0;
  ctx0->y_rtkn0 = 1;
  ctx0->y_rtkn1 = 1;

  PetscFunctionReturn(0);
}