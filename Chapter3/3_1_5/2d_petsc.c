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

extern PetscErrorCode MatMult0(Mat J, Vec X, Vec Y);
extern PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
extern PetscErrorCode FormRHS0(DM dm0, Vec B);
extern PetscErrorCode PopulateUserContext0(UserCtx0 * ctx0);
extern PetscErrorCode FormExact(DM dm0, Vec B);


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

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,1026,1026,1,1,1,2,NULL,NULL,&(da0)));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&(snes0)));
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&(J0)));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(SNESSetType(snes0,SNESKSPONLY));
  PetscCall(DMCreateGlobalVector(da0,&(xglobal0)));

  PetscCall(DMCreateGlobalVector(da0,&(xglobal0)));
  PetscCall(DMCreateGlobalVector(da0,&(bglobal0)));

  PetscCall(DMCreateGlobalVector(da0,&(exact)));

  PetscCall(SNESGetKSP(snes0,&(ksp0)));
  PetscCall(KSPSetTolerances(ksp0,1e-12,1e-50,100000.0,10000.0));
  PetscCall(KSPSetType(ksp0,KSPGMRES));
  PetscCall(KSPGetPC(ksp0,&(pc0)));
  PetscCall(PCSetType(pc0,PCNONE));
  PetscCall(KSPSetFromOptions(ksp0));

  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))MatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  PetscCall(PopulateUserContext0(&(ctx0)));
  PetscCall(MatSetDM(J0,da0));
  PetscCall(DMSetApplicationContext(da0,&(ctx0)));

  PetscCall(FormRHS0(da0,bglobal0));
  PetscCall(FormExact(da0,exact));

  MatNullSpace nullspace;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
  MatSetNullSpace(J0, nullspace);

  PetscCall(VecSet(xglobal0,0.001));

//   MatNullSpaceRemove(nullspace, bglobal0);
  MatNullSpaceRemove(nullspace, xglobal0);
  PetscCall(SNESSolve(snes0,bglobal0,xglobal0));

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

PetscErrorCode MatMult0(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  DM dm0;
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  UserCtx0 * ctx0;
  PetscScalar * x_u_vec;
  PetscScalar * y_u_vec;

  PetscCall(MatGetDM(J,&(dm0)));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(yloc)));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_u_vec));
  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;
  PetscScalar (* y_u)[info.gxm] = (PetscScalar (*)[info.gxm]) y_u_vec;

  PetscScalar r0 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r1 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    for (int iy = ctx0->y_m + ctx0->y_ltkn0; iy <= ctx0->y_M - ctx0->y_rtkn0; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (2.0*(r0*x_u[ix + 2][iy + 2] + r1*x_u[ix + 2][iy + 2]) - (r0*x_u[ix + 1][iy + 2] + r0*x_u[ix + 3][iy + 2] + r1*x_u[ix + 2][iy + 1] + r1*x_u[ix + 2][iy + 3]))*ctx0->h_x*ctx0->h_y;
    }
    for (int iy = ctx0->y_M - ctx0->y_rtkn1 + 1; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar r2 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r2 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 2 - (PetscInt)(abs(iy - ctx0->y_M + 1))])/((ctx0->h_y*ctx0->h_y)) - (r2 + x_u[ix + 1][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      PetscScalar r3 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r3 + x_u[ix + 2][2 + (PetscInt)(abs(iy - 1))] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r3 + x_u[ix + 1][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int iy = ctx0->y_m + ctx0->y_ltkn0; iy <= ctx0->y_M - ctx0->y_rtkn0; iy += 1)
    {
      PetscScalar r4 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r4 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r4 + x_u[2 + (PetscInt)(abs(ix - 1))][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_m + ctx0->y_ltkn0; iy <= ctx0->y_M - ctx0->y_rtkn0; iy += 1)
    {
      PetscScalar r5 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r5 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r5 + x_u[ix + 1][iy + 2] + x_u[ix + 2 - (PetscInt)(abs(ix - ctx0->x_M + 1))][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      PetscScalar r6 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r6 + x_u[ix + 2][2 + (PetscInt)(abs(iy - 1))] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r6 + x_u[2 + (PetscInt)(abs(ix - 1))][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      PetscScalar r7 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r7 + x_u[ix + 2][2 + (PetscInt)(abs(iy - 1))] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) - (r7 + x_u[ix + 1][iy + 2] + x_u[ix + 2 - (PetscInt)(abs(ix - ctx0->x_M + 1))][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn1 + 1; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar r8 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r8 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 2 - (PetscInt)(abs(iy - ctx0->y_M + 1))])/((ctx0->h_y*ctx0->h_y)) - (r8 + x_u[2 + (PetscInt)(abs(ix - 1))][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn1 + 1; iy <= ctx0->y_M; iy += 1)
    {
      PetscScalar r9 = -2.0*x_u[ix + 2][iy + 2];
      y_u[ix + 2][iy + 2] = (-(r9 + x_u[ix + 2][iy + 1] + x_u[ix + 2][iy + 2 - (PetscInt)(abs(iy - ctx0->y_M + 1))])/((ctx0->h_y*ctx0->h_y)) - (r9 + x_u[ix + 1][iy + 2] + x_u[ix + 2 - (PetscInt)(abs(ix - ctx0->x_M + 1))][iy + 2])/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_u_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(yloc)));

  PetscFunctionReturn(0);
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
//   struct dataobj * f_vec = ctx0->f_vec;

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

  PetscFunctionReturn(0);
}


PetscErrorCode PopulateUserContext0(UserCtx0 * ctx0)
{
  PetscFunctionBeginUser;

  ctx0->h_x = 0.00097560975;
  ctx0->h_y = 0.00097560975;
  ctx0->x_M = 1025;
  ctx0->x_ltkn0 = 1;
  ctx0->x_ltkn1 = 1;
  ctx0->x_m = 0;
  ctx0->x_rtkn0 = 1;
  ctx0->x_rtkn2 = 1;
  ctx0->y_M = 1025;
  ctx0->y_ltkn0 = 1;
  ctx0->y_ltkn2 = 1;
  ctx0->y_m = 0;
  ctx0->y_rtkn0 = 1;
  ctx0->y_rtkn1 = 1;


  PetscFunctionReturn(0);
}