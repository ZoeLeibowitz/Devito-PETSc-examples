Generating stencils for Derivative(u1(time + dt, x, y), (x, 2))
Generating stencils for Derivative(u1(time + dt, x, y), (y, 2))
/* Devito generated code for Operator `Kernel` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscsnes.h"
#include "petscdmda.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct UserCtx0
{
  PetscScalar dt;
  PetscScalar h_x;
  PetscScalar h_y;
  struct dataobj * w_u1_dx2_vec;
  struct dataobj * w_u1_dy2_vec;
  PetscInt x_M;
  PetscInt x_m;
  PetscInt y_M;
  PetscInt y_m;
  PetscInt time;
  struct dataobj * u1_vec;
} ;

struct dataobj
{
  void * data;
  PetscInt * size;
  unsigned long nbytes;
  unsigned long * npsize;
  unsigned long * dsize;
  PetscInt * hsize;
  PetscInt * hofs;
  PetscInt * oofs;
  void * dmap;
} ;

struct profiler
{
  PetscScalar section0;
} ;

PetscErrorCode SetPetscOptions0();
PetscErrorCode MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode FormRHS0(DM dm0, Vec B);
PetscErrorCode ClearPetscOptions0();
PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * u1_vec, struct dataobj * w_u1_dx2_vec, struct dataobj * w_u1_dy2_vec, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_m, const PetscInt y_M, const PetscInt y_m);

int Kernel(struct dataobj * u1_vec, const PetscInt time_M, const PetscInt time_m, struct dataobj * w_u1_dx2_vec, struct dataobj * w_u1_dy2_vec, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_m, const PetscInt y_M, const PetscInt y_m, struct profiler * timers)
{
  Mat J0;
  Vec bglobal0;
  DM da0;
  KSP ksp0;
  PetscInt localsize0;
  PetscMPIInt size;
  SNES snes0;
  Vec xglobal0;
  Vec xlocal0;

  struct UserCtx0 ctx0;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,101,101,1,1,1,2,NULL,NULL,&da0));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes0));
  PetscCall(SNESSetOptionsPrefix(snes0,"devito_1_"));
  PetscCall(SetPetscOptions0());
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&J0));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da0,&xglobal0));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,11025,PETSC_DECIDE,u1_vec->data,&xlocal0));
  PetscCall(VecGetSize(xlocal0,&localsize0));
  PetscCall(DMCreateGlobalVector(da0,&bglobal0));
  PetscCall(SNESGetKSP(snes0,&ksp0));
  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))MatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  PetscCall(PopulateUserContext0(&ctx0,u1_vec,w_u1_dx2_vec,w_u1_dy2_vec,dt,h_x,h_y,x_M,x_m,y_M,y_m));
  PetscCall(MatSetDM(J0,da0));
  PetscCall(DMSetApplicationContext(da0,&ctx0));

  for (int time = time_m; time <= time_M; time += 1)
  {
    START(section0)
    ctx0.time = time;
    PetscCall(FormRHS0(da0,bglobal0));
    PetscScalar * u1_ptr0 = (time + 1)*localsize0 + (PetscScalar*)(u1_vec->data);
    PetscCall(VecPlaceArray(xlocal0,u1_ptr0));
    PetscCall(DMLocalToGlobal(da0,xlocal0,INSERT_VALUES,xglobal0));
    PetscCall(SNESSolve(snes0,bglobal0,xglobal0));
    PetscCall(DMGlobalToLocal(da0,xglobal0,INSERT_VALUES,xlocal0));
    PetscCall(VecResetArray(xlocal0));

    STOP(section0,timers)
  }
  PetscCall(ClearPetscOptions0());

  PetscCall(VecDestroy(&bglobal0));
  PetscCall(VecDestroy(&xglobal0));
  PetscCall(VecDestroy(&xlocal0));
  PetscCall(MatDestroy(&J0));
  PetscCall(SNESDestroy(&snes0));
  PetscCall(DMDestroy(&da0));

  return 0;
}

PetscErrorCode SetPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_ksp_type","gmres"));
  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_ksp_rtol","1e-05"));
  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-devito_1_ksp_max_it","10000"));

  PetscFunctionReturn(0);
}

PetscErrorCode MatMult0(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DM dm0;
  PetscCall(MatGetDM(J,&dm0));
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  PetscScalar * x_u1_vec;
  PetscScalar * y_u1_vec;

  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_u1_vec));
  PetscCall(VecGetArray(xloc,&x_u1_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * w_u1_dx2_vec = ctx0->w_u1_dx2_vec;
  struct dataobj * w_u1_dy2_vec = ctx0->w_u1_dy2_vec;

  float (* w_u1_dx2)[w_u1_dx2_vec->size[1]][w_u1_dx2_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[w_u1_dx2_vec->size[1]][w_u1_dx2_vec->size[2]]) w_u1_dx2_vec->data;
  float (* w_u1_dy2)[w_u1_dy2_vec->size[1]][w_u1_dy2_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[w_u1_dy2_vec->size[1]][w_u1_dy2_vec->size[2]]) w_u1_dy2_vec->data;
  PetscScalar (* x_u1)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u1_vec;
  PetscScalar (* y_u1)[info.gxm] = (PetscScalar (*)[info.gxm]) y_u1_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscScalar r0 = 1.0/ctx0->dt;

  for (int x = ctx0->x_m; x <= ctx0->x_M; x += 1)
  {
    #pragma omp simd aligned(w_u1_dx2,w_u1_dy2:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      y_u1[x + 2][y + 2] = (r0*x_u1[x + 2][y + 2] - (x_u1[x + 1][y + 1]*w_u1_dx2[x][y][0] + x_u1[x + 1][y + 2]*w_u1_dx2[x][y][1] + x_u1[x + 1][y + 3]*w_u1_dx2[x][y][2] + x_u1[x + 2][y + 1]*w_u1_dx2[x][y][3] + x_u1[x + 2][y + 2]*w_u1_dx2[x][y][4] + x_u1[x + 2][y + 3]*w_u1_dx2[x][y][5] + x_u1[x + 3][y + 1]*w_u1_dx2[x][y][6] + x_u1[x + 3][y + 2]*w_u1_dx2[x][y][7] + x_u1[x + 3][y + 3]*w_u1_dx2[x][y][8]) - (x_u1[x + 1][y + 1]*w_u1_dy2[x][y][0] + x_u1[x + 1][y + 2]*w_u1_dy2[x][y][1] + x_u1[x + 1][y + 3]*w_u1_dy2[x][y][2] + x_u1[x + 2][y + 1]*w_u1_dy2[x][y][3] + x_u1[x + 2][y + 2]*w_u1_dy2[x][y][4] + x_u1[x + 2][y + 3]*w_u1_dy2[x][y][5] + x_u1[x + 3][y + 1]*w_u1_dy2[x][y][6] + x_u1[x + 3][y + 2]*w_u1_dy2[x][y][7] + x_u1[x + 3][y + 3]*w_u1_dy2[x][y][8]))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_u1_vec));
  PetscCall(VecRestoreArray(xloc,&x_u1_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&yloc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DM dm0 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  Vec floc;
  DMDALocalInfo info;
  Vec xloc;

  PetscScalar * f_u1_vec;
  PetscScalar * x_u1_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&floc));
  PetscCall(VecGetArray(floc,&f_u1_vec));
  PetscCall(VecGetArray(xloc,&x_u1_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * w_u1_dx2_vec = ctx0->w_u1_dx2_vec;
  struct dataobj * w_u1_dy2_vec = ctx0->w_u1_dy2_vec;

  PetscScalar (* f_u1)[info.gxm] = (PetscScalar (*)[info.gxm]) f_u1_vec;
  float (* w_u1_dx2)[w_u1_dx2_vec->size[1]][w_u1_dx2_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[w_u1_dx2_vec->size[1]][w_u1_dx2_vec->size[2]]) w_u1_dx2_vec->data;
  float (* w_u1_dy2)[w_u1_dy2_vec->size[1]][w_u1_dy2_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[w_u1_dy2_vec->size[1]][w_u1_dy2_vec->size[2]]) w_u1_dy2_vec->data;
  PetscScalar (* x_u1)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u1_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscScalar r1 = 1.0/ctx0->dt;

  for (int x = ctx0->x_m; x <= ctx0->x_M; x += 1)
  {
    #pragma omp simd aligned(w_u1_dx2,w_u1_dy2:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      f_u1[x + 2][y + 2] = (r1*x_u1[x + 2][y + 2] - (x_u1[x + 1][y + 1]*w_u1_dx2[x][y][0] + x_u1[x + 1][y + 2]*w_u1_dx2[x][y][1] + x_u1[x + 1][y + 3]*w_u1_dx2[x][y][2] + x_u1[x + 2][y + 1]*w_u1_dx2[x][y][3] + x_u1[x + 2][y + 2]*w_u1_dx2[x][y][4] + x_u1[x + 2][y + 3]*w_u1_dx2[x][y][5] + x_u1[x + 3][y + 1]*w_u1_dx2[x][y][6] + x_u1[x + 3][y + 2]*w_u1_dx2[x][y][7] + x_u1[x + 3][y + 3]*w_u1_dx2[x][y][8]) - (x_u1[x + 1][y + 1]*w_u1_dy2[x][y][0] + x_u1[x + 1][y + 2]*w_u1_dy2[x][y][1] + x_u1[x + 1][y + 3]*w_u1_dy2[x][y][2] + x_u1[x + 2][y + 1]*w_u1_dy2[x][y][3] + x_u1[x + 2][y + 2]*w_u1_dy2[x][y][4] + x_u1[x + 2][y + 3]*w_u1_dy2[x][y][5] + x_u1[x + 3][y + 1]*w_u1_dy2[x][y][6] + x_u1[x + 3][y + 2]*w_u1_dy2[x][y][7] + x_u1[x + 3][y + 3]*w_u1_dy2[x][y][8]))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_u1_vec));
  PetscCall(VecRestoreArray(xloc,&x_u1_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm0,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&floc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS0(DM dm0, Vec B)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  Vec blocal0;
  DMDALocalInfo info;

  PetscScalar * b_u1_vec;

  PetscCall(DMGetLocalVector(dm0,&blocal0));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_u1_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * u1_vec = ctx0->u1_vec;

  PetscScalar (* b_u1)[info.gxm] = (PetscScalar (*)[info.gxm]) b_u1_vec;
  PetscScalar (* u1)[u1_vec->size[1]][u1_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[u1_vec->size[1]][u1_vec->size[2]]) u1_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscScalar r2 = 1.0/ctx0->dt;

  for (int x = ctx0->x_m; x <= ctx0->x_M; x += 1)
  {
    #pragma omp simd aligned(u1:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      b_u1[x + 2][y + 2] = r2*ctx0->h_x*ctx0->h_y*u1[ctx0->time][x + 2][y + 2];
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_u1_vec));
  PetscCall(DMRestoreLocalVector(dm0,&blocal0));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-devito_1_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * u1_vec, struct dataobj * w_u1_dx2_vec, struct dataobj * w_u1_dy2_vec, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_m, const PetscInt y_M, const PetscInt y_m)
{
  PetscFunctionBeginUser;

  ctx0->dt = dt;
  ctx0->h_x = h_x;
  ctx0->h_y = h_y;
  ctx0->w_u1_dx2_vec = w_u1_dx2_vec;
  ctx0->w_u1_dy2_vec = w_u1_dy2_vec;
  ctx0->x_M = x_M;
  ctx0->x_m = x_m;
  ctx0->y_M = y_M;
  ctx0->y_m = y_m;
  ctx0->u1_vec = u1_vec;

  PetscFunctionReturn(0);
}

