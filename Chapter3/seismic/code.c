time_range;  TimeAxis: start=0, stop=150.313, step=0.884194, num=171
dt:  0.8841941282883075  nt:  171  kt:  168
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
  PetscScalar h_x;
  PetscScalar h_y;
  PetscInt x_M;
  PetscInt x_ltkn0;
  PetscInt x_ltkn1;
  PetscInt x_m;
  PetscInt x_rtkn0;
  PetscInt x_rtkn2;
  PetscInt y_M;
  PetscInt y_ltkn1;
  PetscInt y_ltkn2;
  PetscInt y_m;
  PetscInt y_rtkn0;
  PetscInt y_rtkn2;
  struct dataobj * p_vec;
  PetscInt time;
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
  PetscScalar section1;
  PetscScalar section2;
  PetscScalar section3;
  PetscScalar section4;
} ;

PetscErrorCode SetPetscOptions0();
PetscErrorCode MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode FormRHS0(DM dm0, Vec B);
PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc);
PetscErrorCode ClearPetscOptions0();
PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * p_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2);

int Kernel(struct dataobj * damp_vec, struct dataobj * delta_vec, struct dataobj * epsilon_vec, struct dataobj * p_vec, struct dataobj * q_vec, struct dataobj * rec_vec, struct dataobj * rec_coords_vec, struct dataobj * src_vec, struct dataobj * src_coords_vec, struct dataobj * theta_vec, struct dataobj * vp_vec, const PetscInt x_M, const PetscInt x_m, const PetscInt y_M, const PetscInt y_m, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscScalar o_x, const PetscScalar o_y, const PetscInt p_rec_M, const PetscInt p_rec_m, const PetscInt p_src_M, const PetscInt p_src_m, const PetscInt time_M, const PetscInt time_m, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_rtkn0, const PetscInt y_rtkn2, const PetscInt x_size, const PetscInt y_size, struct profiler * timers)
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
  PetscScalar * r1_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r1_vec),64,sizeof(double)*(long)y_size*(long)x_size);
  PetscScalar * r18_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r18_vec),64,sizeof(double)*(long)y_size*(8 + (long)x_size));
  PetscScalar * r19_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r19_vec),64,sizeof(double)*(long)y_size*(8 + (long)x_size));
  PetscScalar * r2_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r2_vec),64,sizeof(double)*(long)y_size*(long)x_size);
  PetscScalar * r20_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r20_vec),64,sizeof(double)*(long)y_size*(8 + (long)x_size));
  PetscScalar * r3_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r3_vec),64,sizeof(double)*(long)y_size*(long)x_size);
  PetscScalar * r4_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r4_vec),64,sizeof(double)*(long)y_size*(long)x_size);
  PetscScalar * r5_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&r5_vec),64,sizeof(double)*(long)y_size*(long)x_size);

  PetscScalar (* damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[damp_vec->size[1]]) damp_vec->data;
  PetscScalar (* delta)[delta_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[delta_vec->size[1]]) delta_vec->data;
  PetscScalar (* epsilon)[epsilon_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[epsilon_vec->size[1]]) epsilon_vec->data;
  PetscScalar (* p)[p_vec->size[1]][p_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[p_vec->size[1]][p_vec->size[2]]) p_vec->data;
  PetscScalar (* q)[q_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[q_vec->size[1]]) q_vec->data;
  PetscScalar (* r1)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r1_vec;
  PetscScalar (* r18)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r18_vec;
  PetscScalar (* r19)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r19_vec;
  PetscScalar (* r2)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r2_vec;
  PetscScalar (* r20)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r20_vec;
  PetscScalar (* r3)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r3_vec;
  PetscScalar (* r4)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r4_vec;
  PetscScalar (* r5)[y_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[y_size]) r5_vec;
  PetscScalar (* rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[rec_vec->size[1]]) rec_vec->data;
  PetscScalar (* rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  PetscScalar (* src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[src_vec->size[1]]) src_vec->data;
  PetscScalar (* src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  PetscScalar (* theta)[theta_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[theta_vec->size[1]]) theta_vec->data;
  PetscScalar (* vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,141,141,1,1,1,8,NULL,NULL,&da0));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes0));
  PetscCall(SNESSetOptionsPrefix(snes0,"poisson_"));
  PetscCall(SetPetscOptions0());
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&J0));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da0,&xglobal0));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,24649,PETSC_DECIDE,q_vec->data,&xlocal0));
  PetscCall(VecGetSize(xlocal0,&localsize0));
  PetscCall(DMCreateGlobalVector(da0,&bglobal0));
  PetscCall(SNESGetKSP(snes0,&ksp0));
  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))MatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  PetscCall(PopulateUserContext0(&ctx0,p_vec,h_x,h_y,x_M,x_ltkn0,x_ltkn1,x_m,x_rtkn0,x_rtkn2,y_M,y_ltkn1,y_ltkn2,y_m,y_rtkn0,y_rtkn2));
  PetscCall(MatSetDM(J0,da0));
  PetscCall(DMSetApplicationContext(da0,&ctx0));

  START(section0)
  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(theta:32)
    for (int y = y_m; y <= y_M; y += 1)
    {
      r1[x][y] = sin(4*theta[x + 8][y + 8]);
      PetscScalar r21 = 2*theta[x + 8][y + 8];
      r2[x][y] = sin(r21);
      r3[x][y] = cos(theta[x + 8][y + 8]);
      r4[x][y] = sin(theta[x + 8][y + 8]);
      r5[x][y] = cos(r21);
    }
  }
  STOP(section0,timers)

  PetscScalar r6 = 1.0/(h_x*h_x*h_x);
  PetscScalar r22 = 1.0/h_y;
  PetscScalar r7 = r22;
  PetscScalar r23 = 1.0/h_x;
  PetscScalar r8 = r23;
  PetscScalar r9 = 1.0/(h_y*h_y*h_y);
  PetscScalar r10 = 1.0/(h_x*h_x*h_x*h_x);
  PetscScalar r11 = 1.0/(h_y*h_y*h_y*h_y);
  PetscScalar r12 = 1.0/(h_x*h_x);
  PetscScalar r13 = 1.0/(h_y*h_y);
  PetscScalar r14 = 1.0/(dt*dt);
  PetscScalar r15 = 1.0/dt;
  PetscScalar r16 = r23;
  PetscScalar r17 = r22;

  for (int time = time_m; time <= time_M; time += 1)
  {
    START(section1)
    for (int x = x_m - 4; x <= x_M + 4; x += 1)
    {
      #pragma omp simd aligned(q:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        r18[x + 4][y] = r7*(3.57142857e-3*(q[x + 8][y + 4] - q[x + 8][y + 12]) + 3.80952381e-2*(-q[x + 8][y + 5] + q[x + 8][y + 11]) + 2.0e-1*(q[x + 8][y + 6] - q[x + 8][y + 10]) + 8.0e-1*(-q[x + 8][y + 7] + q[x + 8][y + 9]));
        r19[x + 4][y] = r9*(2.91666667e-2*(-q[x + 8][y + 4] + q[x + 8][y + 12]) + 3.0e-1*(q[x + 8][y + 5] - q[x + 8][y + 11]) + 1.408333330*(-q[x + 8][y + 6] + q[x + 8][y + 10]) + 2.033333330*(q[x + 8][y + 7] - q[x + 8][y + 9]));
        r20[x + 4][y] = r13*((-1.78571429e-3)*(q[x + 8][y + 4] + q[x + 8][y + 12]) + 2.53968254e-2*(q[x + 8][y + 5] + q[x + 8][y + 11]) + (-2.0e-1)*(q[x + 8][y + 6] + q[x + 8][y + 10]) + 1.60*(q[x + 8][y + 7] + q[x + 8][y + 9]) - 2.847222220*q[x + 8][y + 8]);
      }
    }
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp,delta,epsilon,p,q,vp:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        PetscScalar r26 = r3[x][y]*r3[x][y];
        PetscScalar r27 = r4[x][y]*r4[x][y];
        PetscScalar r24 = 2*r26*r27*delta[x + 8][y + 8];
        PetscScalar r25 = 1.0/(vp[x + 8][y + 8]*vp[x + 8][y + 8]);
        PetscScalar r28 = 2*epsilon[x + 8][y + 8];
        PetscScalar r29 = 1.1375e+1*q[x + 8][y + 8];
        p[time + 1][x + 2][y + 2] = (-r25*(-2.0*r14*p[time][x + 2][y + 2] + r14*p[time - 1][x + 2][y + 2]) + r10*(r24 + r28*r3[x][y]*r3[x][y]*r3[x][y]*r3[x][y] + 1)*(r29 + 2.91666667e-2*(q[x + 4][y + 8] + q[x + 12][y + 8]) + (-4.0e-1)*(q[x + 5][y + 8] + q[x + 11][y + 8]) + 2.816666670*(q[x + 6][y + 8] + q[x + 10][y + 8]) + (-8.133333330)*(q[x + 7][y + 8] + q[x + 9][y + 8])) + r11*(r24 + r28*r4[x][y]*r4[x][y]*r4[x][y]*r4[x][y] + 1)*(r29 + 2.91666667e-2*(q[x + 8][y + 4] + q[x + 8][y + 12]) + (-4.0e-1)*(q[x + 8][y + 5] + q[x + 8][y + 11]) + 2.816666670*(q[x + 8][y + 6] + q[x + 8][y + 10]) + (-8.133333330)*(q[x + 8][y + 7] + q[x + 8][y + 9])) + r12*((r2[x][y]*r2[x][y])*(-delta[x + 8][y + 8] + 3*epsilon[x + 8][y + 8]) + (r5[x][y]*r5[x][y])*(2*delta[x + 8][y + 8]) + 2)*((-1.78571429e-3)*(r20[x][y] + r20[x + 8][y]) + 2.53968254e-2*(r20[x + 1][y] + r20[x + 7][y]) + (-2.0e-1)*(r20[x + 2][y] + r20[x + 6][y]) + 1.60*(r20[x + 3][y] + r20[x + 5][y]) - 2.847222220*r20[x + 4][y]) + r15*damp[x + 8][y + 8]*p[time][x + 2][y + 2] + r6*(-4*r26*epsilon[x + 8][y + 8]*r2[x][y] + delta[x + 8][y + 8]*r1[x][y])*(2.91666667e-2*(-r18[x][y] + r18[x + 8][y]) + 3.0e-1*(r18[x + 1][y] - r18[x + 7][y]) + 1.408333330*(-r18[x + 2][y] + r18[x + 6][y]) + 2.033333330*(r18[x + 3][y] - r18[x + 5][y])) + r8*(-4*r27*epsilon[x + 8][y + 8]*r2[x][y] - delta[x + 8][y + 8]*r1[x][y])*(3.57142857e-3*(r19[x][y] - r19[x + 8][y]) + 3.80952381e-2*(-r19[x + 1][y] + r19[x + 7][y]) + 2.0e-1*(r19[x + 2][y] - r19[x + 6][y]) + 8.0e-1*(-r19[x + 3][y] + r19[x + 5][y])))/(r25*r14 + r15*damp[x + 8][y + 8]);
      }
    }
    STOP(section1,timers)

    START(section2)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      for (int rsrcx = 0; rsrcx <= 1; rsrcx += 1)
      {
        for (int rsrcy = 0; rsrcy <= 1; rsrcy += 1)
        {
          PetscInt posx = (PetscInt)(floor((-o_x + src_coords[p_src][0])/h_x));
          PetscInt posy = (PetscInt)(floor((-o_y + src_coords[p_src][1])/h_y));
          PetscScalar px = -floor((-o_x + src_coords[p_src][0])/h_x) + (-o_x + src_coords[p_src][0])/h_x;
          PetscScalar py = -floor((-o_y + src_coords[p_src][1])/h_y) + (-o_y + src_coords[p_src][1])/h_y;
          if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1)
          {
            PetscScalar r0 = 7.8179925649952e-1*(vp[posx + 8][posy + 8]*vp[posx + 8][posy + 8])*(rsrcx*px + (1 - rsrcx)*(1 - px))*(rsrcy*py + (1 - rsrcy)*(1 - py))*src[time][p_src];
            p[time + 1][rsrcx + posx + 2][rsrcy + posy + 2] += r0;
          }
        }
      }
    }
    STOP(section2,timers)

    START(section3)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      PetscScalar r32 = r16*(-o_x + rec_coords[p_rec][0]);
      PetscScalar r30 = floor(r32);
      PetscInt posx = (PetscInt)r30;
      PetscScalar r33 = r17*(-o_y + rec_coords[p_rec][1]);
      PetscScalar r31 = floor(r33);
      PetscInt posy = (PetscInt)r31;
      PetscScalar px = -r30 + r32;
      PetscScalar py = -r31 + r33;
      PetscScalar sum = 0.0;

      for (int rrecx = 0; rrecx <= 1; rrecx += 1)
      {
        for (int rrecy = 0; rrecy <= 1; rrecy += 1)
        {
          if (rrecx + posx >= x_m - 1 && rrecy + posy >= y_m - 1 && rrecx + posx <= x_M + 1 && rrecy + posy <= y_M + 1)
          {
            sum += (rrecx*px + (1 - rrecx)*(1 - px))*(rrecy*py + (1 - rrecy)*(1 - py))*p[time + 1][rrecx + posx + 2][rrecy + posy + 2];
          }
        }
      }

      rec[time][p_rec] = sum;
    }
    STOP(section3,timers)

    START(section4)
    ctx0.time = time;
    PetscCall(FormRHS0(da0,bglobal0));
    PetscCall(FormInitialGuess0(da0,xlocal0));
    PetscCall(DMLocalToGlobal(da0,xlocal0,INSERT_VALUES,xglobal0));
    PetscCall(SNESSolve(snes0,bglobal0,xglobal0));
    PetscCall(DMGlobalToLocal(da0,xglobal0,INSERT_VALUES,xlocal0));

    STOP(section4,timers)
  }
  PetscCall(ClearPetscOptions0());

  free(r1_vec);
  free(r18_vec);
  free(r19_vec);
  free(r2_vec);
  free(r20_vec);
  free(r3_vec);
  free(r4_vec);
  free(r5_vec);
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

  PetscCall(PetscOptionsSetValue(NULL,"-poisson_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_ksp_type","cg"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_ksp_rtol","1e-05"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_ksp_max_it","10000"));

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

  PetscScalar * x_q_vec;
  PetscScalar * y_q_vec;

  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_q_vec));
  PetscCall(VecGetArray(xloc,&x_q_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));

  PetscScalar (* x_q)[info.gxm] = (PetscScalar (*)[info.gxm]) x_q_vec;
  PetscScalar (* y_q)[info.gxm] = (PetscScalar (*)[info.gxm]) y_q_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      y_q[ix + 8][iy + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][iy + 8];
      x_q[ix + 8][iy + 8] = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      y_q[ix + 8][iy + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][iy + 8];
      x_q[ix + 8][iy + 8] = 0.0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      y_q[ix + 8][y + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][y + 8];
      x_q[ix + 8][y + 8] = 0.0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      y_q[ix + 8][y + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][y + 8];
      x_q[ix + 8][y + 8] = 0.0;
    }
  }

  PetscScalar r34 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r35 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      PetscScalar r36 = -2.847222220*x_q[ix + 8][iy + 8];
      y_q[ix + 8][iy + 8] = (r34*(r36 + (-1.78571429e-3)*(x_q[ix + 4][iy + 8] + x_q[ix + 12][iy + 8]) + 2.53968254e-2*(x_q[ix + 5][iy + 8] + x_q[ix + 11][iy + 8]) + (-2.0e-1)*(x_q[ix + 6][iy + 8] + x_q[ix + 10][iy + 8]) + 1.60*(x_q[ix + 7][iy + 8] + x_q[ix + 9][iy + 8])) + r35*(r36 + (-1.78571429e-3)*(x_q[ix + 8][iy + 4] + x_q[ix + 8][iy + 12]) + 2.53968254e-2*(x_q[ix + 8][iy + 5] + x_q[ix + 8][iy + 11]) + (-2.0e-1)*(x_q[ix + 8][iy + 6] + x_q[ix + 8][iy + 10]) + 1.60*(x_q[ix + 8][iy + 7] + x_q[ix + 8][iy + 9])))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_q_vec));
  PetscCall(VecRestoreArray(xloc,&x_q_vec));
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

  PetscScalar * f_q_vec;
  PetscScalar * x_q_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&floc));
  PetscCall(VecGetArray(floc,&f_q_vec));
  PetscCall(VecGetArray(xloc,&x_q_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));

  PetscScalar (* f_q)[info.gxm] = (PetscScalar (*)[info.gxm]) f_q_vec;
  PetscScalar (* x_q)[info.gxm] = (PetscScalar (*)[info.gxm]) x_q_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      f_q[ix + 8][iy + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][iy + 8];
      x_q[ix + 8][iy + 8] = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      f_q[ix + 8][iy + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][iy + 8];
      x_q[ix + 8][iy + 8] = 0.0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      f_q[ix + 8][y + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][y + 8];
      x_q[ix + 8][y + 8] = 0.0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      f_q[ix + 8][y + 8] = (-2.847222220/pow(ctx0->h_y, 2) - 2.847222220/pow(ctx0->h_x, 2))*ctx0->h_x*ctx0->h_y*x_q[ix + 8][y + 8];
      x_q[ix + 8][y + 8] = 0.0;
    }
  }

  PetscScalar r37 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r38 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      PetscScalar r39 = -2.847222220*x_q[ix + 8][iy + 8];
      f_q[ix + 8][iy + 8] = (r37*(r39 + (-1.78571429e-3)*(x_q[ix + 4][iy + 8] + x_q[ix + 12][iy + 8]) + 2.53968254e-2*(x_q[ix + 5][iy + 8] + x_q[ix + 11][iy + 8]) + (-2.0e-1)*(x_q[ix + 6][iy + 8] + x_q[ix + 10][iy + 8]) + 1.60*(x_q[ix + 7][iy + 8] + x_q[ix + 9][iy + 8])) + r38*(r39 + (-1.78571429e-3)*(x_q[ix + 8][iy + 4] + x_q[ix + 8][iy + 12]) + 2.53968254e-2*(x_q[ix + 8][iy + 5] + x_q[ix + 8][iy + 11]) + (-2.0e-1)*(x_q[ix + 8][iy + 6] + x_q[ix + 8][iy + 10]) + 1.60*(x_q[ix + 8][iy + 7] + x_q[ix + 8][iy + 9])))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_q_vec));
  PetscCall(VecRestoreArray(xloc,&x_q_vec));
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

  PetscScalar * b_q_vec;

  PetscCall(DMGetLocalVector(dm0,&blocal0));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_q_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * p_vec = ctx0->p_vec;

  PetscScalar (* b_q)[info.gxm] = (PetscScalar (*)[info.gxm]) b_q_vec;
  PetscScalar (* p)[p_vec->size[1]][p_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[p_vec->size[1]][p_vec->size[2]]) p_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      b_q[ix + 8][iy + 8] = 0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      b_q[ix + 8][iy + 8] = 0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      b_q[ix + 8][y + 8] = 0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      b_q[ix + 8][y + 8] = 0;
    }
  }
  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd aligned(p:32)
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      b_q[ix + 8][iy + 8] = ctx0->h_x*ctx0->h_y*p[ctx0->time + 1][ix + 2][iy + 2];
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_q_vec));
  PetscCall(DMRestoreLocalVector(dm0,&blocal0));

  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  DMDALocalInfo info;

  PetscScalar * x_q_vec;

  PetscCall(VecGetArray(xloc,&x_q_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));

  PetscScalar (* x_q)[info.gxm] = (PetscScalar (*)[info.gxm]) x_q_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      x_q[ix + 8][iy + 8] = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      x_q[ix + 8][iy + 8] = 0.0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      x_q[ix + 8][y + 8] = 0.0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      x_q[ix + 8][y + 8] = 0.0;
    }
  }
  PetscCall(VecRestoreArray(xloc,&x_q_vec));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-poisson_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * p_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2)
{
  PetscFunctionBeginUser;

  ctx0->h_x = h_x;
  ctx0->h_y = h_y;
  ctx0->x_M = x_M;
  ctx0->x_ltkn0 = x_ltkn0;
  ctx0->x_ltkn1 = x_ltkn1;
  ctx0->x_m = x_m;
  ctx0->x_rtkn0 = x_rtkn0;
  ctx0->x_rtkn2 = x_rtkn2;
  ctx0->y_M = y_M;
  ctx0->y_ltkn1 = y_ltkn1;
  ctx0->y_ltkn2 = y_ltkn2;
  ctx0->y_m = y_m;
  ctx0->y_rtkn0 = y_rtkn0;
  ctx0->y_rtkn2 = y_rtkn2;
  ctx0->p_vec = p_vec;

  PetscFunctionReturn(0);
}

norm of p:  2479.376435343486
