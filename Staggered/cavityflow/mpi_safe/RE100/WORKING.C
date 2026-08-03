/* Devito generated code for Operator `Kernel` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscsnes.h"
#include "petscdmda.h"
#include "petscsection.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "mpi.h"

struct UserCtx0
{
  struct dataobj * _stagger_border_u_vec;
  PetscScalar dt;
  PetscScalar h_x;
  PetscScalar h_y;
  PetscInt n0_M;
  PetscInt n0_m;
  PetscScalar re;
  PetscInt x_M;
  PetscInt x_ltkn1;
  PetscInt x_ltkn2;
  PetscInt x_ltkn6;
  PetscInt x_m;
  PetscInt x_rtkn0;
  PetscInt x_rtkn1;
  PetscInt x_rtkn6;
  PetscInt y_M;
  PetscInt y_ltkn1;
  PetscInt y_ltkn2;
  PetscInt y_ltkn3;
  PetscInt y_ltkn4;
  PetscInt y_ltkn6;
  PetscInt y_m;
  PetscInt y_rtkn1;
  PetscInt y_rtkn4;
  PetscInt y_rtkn6;
  PetscScalar zero;
  PetscInt t0;
  PetscInt t1;
  struct dataobj * u_vec;
  struct dataobj * v_vec;
  PetscInt t2;
} ;

struct UserCtx1
{
  struct dataobj * _stagger_border_v_vec;
  PetscScalar dt;
  PetscScalar h_x;
  PetscScalar h_y;
  PetscInt n1_M;
  PetscInt n1_m;
  PetscScalar re;
  PetscInt x_M;
  PetscInt x_ltkn2;
  PetscInt x_ltkn3;
  PetscInt x_ltkn4;
  PetscInt x_ltkn5;
  PetscInt x_ltkn7;
  PetscInt x_m;
  PetscInt x_rtkn3;
  PetscInt x_rtkn5;
  PetscInt x_rtkn7;
  PetscInt y_M;
  PetscInt y_ltkn3;
  PetscInt y_ltkn5;
  PetscInt y_ltkn7;
  PetscInt y_m;
  PetscInt y_rtkn0;
  PetscInt y_rtkn5;
  PetscInt y_rtkn7;
  PetscScalar zero;
  PetscInt t0;
  PetscInt t1;
  struct dataobj * u_vec;
  struct dataobj * v_vec;
  PetscInt t2;
} ;

struct UserCtx2
{
  struct dataobj * _stagger_border_p_vec;
  PetscScalar h_x;
  PetscScalar h_y;
  PetscInt n2_M;
  PetscInt n2_m;
  PetscInt x_M;
  PetscInt x_ltkn2;
  PetscInt x_ltkn3;
  PetscInt x_ltkn5;
  PetscInt x_ltkn8;
  PetscInt x_m;
  PetscInt x_rtkn3;
  PetscInt x_rtkn5;
  PetscInt x_rtkn8;
  PetscInt y_M;
  PetscInt y_ltkn1;
  PetscInt y_ltkn3;
  PetscInt y_ltkn4;
  PetscInt y_ltkn8;
  PetscInt y_m;
  PetscInt y_rtkn1;
  PetscInt y_rtkn4;
  PetscInt y_rtkn8;
  struct dataobj * bc_tmp_p_vec;
  PetscScalar zero;
  PetscScalar dt_c;
  PetscInt t2;
  struct dataobj * u_vec;
  struct dataobj * v_vec;
  struct dataobj * p_vec;
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

struct neighborhood
{
  int ll, lc, lr;
  int cl, cc, cr;
  int rl, rc, rr;
} ;

struct petscprofiler0
{
  KSPConvergedReason reason0;
  PetscInt kspits0;
  KSPNormType kspnormtype0;
  PetscScalar rtol0;
  PetscScalar atol0;
  PetscScalar divtol0;
  PetscInt max_it0;
  char ksptype0[64];
  PetscInt snesits0;
} ;

struct petscprofiler1
{
  KSPConvergedReason reason1;
  PetscInt kspits1;
  KSPNormType kspnormtype1;
  PetscScalar rtol1;
  PetscScalar atol1;
  PetscScalar divtol1;
  PetscInt max_it1;
  char ksptype1[64];
  PetscInt snesits1;
} ;

struct petscprofiler2
{
  KSPConvergedReason reason2;
  PetscInt kspits2;
  KSPNormType kspnormtype2;
  PetscScalar rtol2;
  PetscScalar atol2;
  PetscScalar divtol2;
  PetscInt max_it2;
  char ksptype2[64];
  PetscInt snesits2;
} ;

struct profiler
{
  PetscScalar section0;
  PetscScalar section1;
  PetscScalar section2;
  PetscScalar section3;
  PetscScalar section4;
  PetscScalar section5;
  PetscScalar section6;
  PetscScalar section7;
  PetscScalar section8;
  PetscScalar section9;
  PetscScalar section10;
} ;

static void sendrecv0(struct dataobj * f0_vec, const PetscInt x_size, const PetscInt y_size, PetscInt ogtime, PetscInt ogx, PetscInt ogy, PetscInt ostime, PetscInt osx, PetscInt osy, PetscInt fromrank, PetscInt torank, MPI_Comm comm);
static void sendrecv2(struct dataobj * f0_vec, const PetscInt x_size, const PetscInt y_size, PetscInt ogx, PetscInt ogy, PetscInt osx, PetscInt osy, PetscInt fromrank, PetscInt torank, MPI_Comm comm);
static void haloupdate0(struct dataobj * f0_vec, MPI_Comm comm, struct neighborhood * nb, PetscInt otime);
static void haloupdate2(struct dataobj * p_vec, MPI_Comm comm, struct neighborhood * nb);
static void haloupdate3(struct dataobj * p_vec, MPI_Comm comm, struct neighborhood * nb);
static void gather0(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt otime, const PetscInt ox, const PetscInt oy);
static void scatter0(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt otime, const PetscInt ox, const PetscInt oy);
static void gather2(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt ox, const PetscInt oy);
static void scatter2(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt ox, const PetscInt oy);


PetscErrorCode CountBCs0(DM dm0, PetscInt * numBCPtr0);
PetscErrorCode SetPointBCs0(DM dm0, PetscInt numBC0);
PetscErrorCode SetPetscOptions0();
PetscErrorCode MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode CountBCs1(DM dm1, PetscInt * numBCPtr1);
PetscErrorCode SetPointBCs1(DM dm1, PetscInt numBC1);
PetscErrorCode SetPetscOptions1();
PetscErrorCode MatMult1(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction1(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode CountBCs2(DM dm2, PetscInt * numBCPtr2);
PetscErrorCode SetPointBCs2(DM dm2, PetscInt numBC2);
PetscErrorCode SetPetscOptions2();
PetscErrorCode MatMult2(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction2(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode FormRHS0(DM dm0, Vec B);
PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc);
PetscErrorCode FormRHS1(DM dm1, Vec B);
PetscErrorCode FormInitialGuess1(DM dm1, Vec xloc);
PetscErrorCode FormRHS2(DM dm2, Vec B);
PetscErrorCode FormInitialGuess2(DM dm2, Vec xloc);
PetscErrorCode ClearPetscOptions0();
PetscErrorCode ClearPetscOptions1();
PetscErrorCode ClearPetscOptions2();
PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * _stagger_border_u_vec, const PetscScalar re, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar zero, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt n0_M, const PetscInt n0_m, const PetscInt x_M, const PetscInt x_ltkn1, const PetscInt x_ltkn2, const PetscInt x_ltkn6, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn1, const PetscInt x_rtkn6, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_ltkn3, const PetscInt y_ltkn4, const PetscInt y_ltkn6, const PetscInt y_m, const PetscInt y_rtkn1, const PetscInt y_rtkn4, const PetscInt y_rtkn6);
PetscErrorCode PopulateUserContext1(struct UserCtx1 * ctx1, struct dataobj * _stagger_border_v_vec, const PetscScalar re, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar zero, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt n1_M, const PetscInt n1_m, const PetscInt x_M, const PetscInt x_ltkn2, const PetscInt x_ltkn3, const PetscInt x_ltkn4, const PetscInt x_ltkn5, const PetscInt x_ltkn7, const PetscInt x_m, const PetscInt x_rtkn3, const PetscInt x_rtkn5, const PetscInt x_rtkn7, const PetscInt y_M, const PetscInt y_ltkn3, const PetscInt y_ltkn5, const PetscInt y_ltkn7, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn5, const PetscInt y_rtkn7);
PetscErrorCode PopulateUserContext2(struct UserCtx2 * ctx2, struct dataobj * _stagger_border_p_vec, struct dataobj * bc_tmp_p_vec, const PetscScalar dt_c, struct dataobj * p_vec, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar zero, const PetscScalar h_x, const PetscScalar h_y, const PetscInt n2_M, const PetscInt n2_m, const PetscInt x_M, const PetscInt x_ltkn2, const PetscInt x_ltkn3, const PetscInt x_ltkn5, const PetscInt x_ltkn8, const PetscInt x_m, const PetscInt x_rtkn3, const PetscInt x_rtkn5, const PetscInt x_rtkn8, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn3, const PetscInt y_ltkn4, const PetscInt y_ltkn8, const PetscInt y_m, const PetscInt y_rtkn1, const PetscInt y_rtkn4, const PetscInt y_rtkn8);

int Kernel(const PetscScalar dt_c, struct dataobj * p_vec, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt time_M, const PetscInt time_m, const PetscInt x_M, const PetscInt x_ltkn1, const PetscInt x_ltkn2, const PetscInt x_ltkn3, const PetscInt x_ltkn4, const PetscInt x_ltkn5, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn1, const PetscInt x_rtkn3, const PetscInt x_rtkn5, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_ltkn3, const PetscInt y_ltkn4, const PetscInt y_ltkn5, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn1, const PetscInt y_rtkn4, const PetscInt y_rtkn5, MPI_Comm comm, struct petscprofiler0 * petscinfo0, struct petscprofiler1 * petscinfo1, struct petscprofiler2 * petscinfo2, struct dataobj * _stagger_border_p_vec, struct dataobj * _stagger_border_u_vec, struct dataobj * _stagger_border_v_vec, struct dataobj * bc_tmp_p_vec, const PetscScalar dt, const PetscInt n0_M, const PetscInt n0_m, const PetscInt n1_M, const PetscInt n1_m, const PetscInt n2_M, const PetscInt n2_m, const PetscScalar re, const PetscInt x_ltkn6, const PetscInt x_ltkn7, const PetscInt x_ltkn8, const PetscInt x_rtkn6, const PetscInt x_rtkn7, const PetscInt x_rtkn8, const PetscInt y_ltkn6, const PetscInt y_ltkn7, const PetscInt y_ltkn8, const PetscInt y_rtkn6, const PetscInt y_rtkn7, const PetscInt y_rtkn8, const PetscScalar zero, struct neighborhood * nb, struct profiler * timers)
{
  Mat J0;
  Mat J1;
  Mat J2;
  PetscScalar atol0;
  PetscScalar atol1;
  PetscScalar atol2;
  Vec bglobal0;
  Vec bglobal1;
  Vec bglobal2;
  DM da0;
  DM da1;
  DM da2;
  PetscScalar divtol0;
  PetscScalar divtol1;
  PetscScalar divtol2;
  PetscSection gsection0;
  PetscSection gsection1;
  PetscSection gsection2;
  KSP ksp0;
  KSP ksp1;
  KSP ksp2;
  PetscInt kspits0;
  PetscInt kspits1;
  PetscInt kspits2;
  KSPNormType kspnormtype0;
  KSPNormType kspnormtype1;
  KSPNormType kspnormtype2;
  KSPType ksptype0;
  KSPType ksptype1;
  KSPType ksptype2;
  PetscInt localsize0;
  PetscInt localsize1;
  PetscInt localsize2;
  PetscSection lsection0;
  PetscSection lsection1;
  PetscSection lsection2;
  PetscInt max_it0;
  PetscInt max_it1;
  PetscInt max_it2;
  PetscInt numBC0 = 0;
  PetscInt numBC1 = 0;
  PetscInt numBC2 = 0;
  KSPConvergedReason reason0;
  KSPConvergedReason reason1;
  KSPConvergedReason reason2;
  PetscScalar rtol0;
  PetscScalar rtol1;
  PetscScalar rtol2;
  PetscSF sf0;
  PetscSF sf1;
  PetscSF sf2;
  PetscMPIInt size;
  SNES snes0;
  SNES snes1;
  SNES snes2;
  PetscInt snesits0;
  PetscInt snesits1;
  PetscInt snesits2;
  Vec xglobal0;
  Vec xglobal1;
  Vec xglobal2;
  Vec xlocal0;
  Vec xlocal1;
  Vec xlocal2;

  struct UserCtx0 ctx0;
  struct UserCtx1 ctx1;
  struct UserCtx2 ctx2;

  PetscScalar (* p)[p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[p_vec->size[1]]) p_vec->data;
  PetscScalar (* u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  PetscScalar (* v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCallMPI(MPI_Comm_size(comm,&size));

  PetscCall(DMDACreate2d(comm,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,65,65,1,2,1,2,NULL,NULL,&da0));
  PetscCall(PetscOptionsSetValue(NULL,"-da_use_section",NULL));
  PetscCall(DMSetFromOptions(da0));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(PopulateUserContext0(&ctx0,_stagger_border_u_vec,re,u_vec,v_vec,zero,dt,h_x,h_y,n0_M,n0_m,x_M,x_ltkn1,x_ltkn2,x_ltkn6,x_m,x_rtkn0,x_rtkn1,x_rtkn6,y_M,y_ltkn1,y_ltkn2,y_ltkn3,y_ltkn4,y_ltkn6,y_m,y_rtkn1,y_rtkn4,y_rtkn6));
  PetscCall(DMSetApplicationContext(da0,&ctx0));
  PetscCall(CountBCs0(da0,&numBC0));
  PetscCall(SetPointBCs0(da0,numBC0));
  PetscCall(DMGetLocalSection(da0,&lsection0));
  PetscCall(DMGetPointSF(da0,&sf0));
  PetscCall(PetscSectionCreateGlobalSection(lsection0,sf0,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,&gsection0));
  PetscCall(DMSetGlobalSection(da0,gsection0));
  PetscCall(DMCreateSectionSF(da0,lsection0,gsection0));
  PetscCall(SNESCreate(comm,&snes0));
  PetscCall(SNESSetOptionsPrefix(snes0,"utent_solve_"));
  PetscCall(SetPetscOptions0());
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&J0));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da0,&xglobal0));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2553,u_vec->data,&xlocal0));
  PetscCall(VecGetLocalSize(xlocal0,&localsize0));
  PetscCall(DMCreateGlobalVector(da0,&bglobal0));
  PetscCall(SNESGetKSP(snes0,&ksp0));
  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))MatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  PetscCall(MatSetDM(J0,da0));

  PetscCall(DMDACreate2d(comm,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,65,65,1,2,1,2,NULL,NULL,&da1));
  PetscCall(PetscOptionsSetValue(NULL,"-da_use_section",NULL));
  PetscCall(DMSetFromOptions(da1));
  PetscCall(DMSetUp(da1));
  PetscCall(DMSetMatType(da1,MATSHELL));
  PetscCall(PopulateUserContext1(&ctx1,_stagger_border_v_vec,re,u_vec,v_vec,zero,dt,h_x,h_y,n1_M,n1_m,x_M,x_ltkn2,x_ltkn3,x_ltkn4,x_ltkn5,x_ltkn7,x_m,x_rtkn3,x_rtkn5,x_rtkn7,y_M,y_ltkn3,y_ltkn5,y_ltkn7,y_m,y_rtkn0,y_rtkn5,y_rtkn7));
  PetscCall(DMSetApplicationContext(da1,&ctx1));
  PetscCall(CountBCs1(da1,&numBC1));
  PetscCall(SetPointBCs1(da1,numBC1));
  PetscCall(DMGetLocalSection(da1,&lsection1));
  PetscCall(DMGetPointSF(da1,&sf1));
  PetscCall(PetscSectionCreateGlobalSection(lsection1,sf1,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,&gsection1));
  PetscCall(DMSetGlobalSection(da1,gsection1));
  PetscCall(DMCreateSectionSF(da1,lsection1,gsection1));
  PetscCall(SNESCreate(comm,&snes1));
  PetscCall(SNESSetOptionsPrefix(snes1,"vtent_solve_"));
  PetscCall(SetPetscOptions1());
  PetscCall(SNESSetDM(snes1,da1));
  PetscCall(DMCreateMatrix(da1,&J1));
  PetscCall(SNESSetJacobian(snes1,J1,J1,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da1,&xglobal1));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2553,v_vec->data,&xlocal1));
  PetscCall(VecGetLocalSize(xlocal1,&localsize1));
  PetscCall(DMCreateGlobalVector(da1,&bglobal1));
  PetscCall(SNESGetKSP(snes1,&ksp1));
  PetscCall(MatShellSetOperation(J1,MATOP_MULT,(void (*)(void))MatMult1));
  PetscCall(SNESSetFunction(snes1,NULL,FormFunction1,(void*)(da1)));
  PetscCall(SNESSetFromOptions(snes1));
  PetscCall(MatSetDM(J1,da1));

  PetscCall(DMDACreate2d(comm,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,65,65,1,2,1,2,NULL,NULL,&da2));
  PetscCall(PetscOptionsSetValue(NULL,"-da_use_section",NULL));
  PetscCall(DMSetFromOptions(da2));
  PetscCall(DMSetUp(da2));
  PetscCall(DMSetMatType(da2,MATSHELL));
  PetscCall(PopulateUserContext2(&ctx2,_stagger_border_p_vec,bc_tmp_p_vec,dt_c,p_vec,u_vec,v_vec,zero,h_x,h_y,n2_M,n2_m,x_M,x_ltkn2,x_ltkn3,x_ltkn5,x_ltkn8,x_m,x_rtkn3,x_rtkn5,x_rtkn8,y_M,y_ltkn1,y_ltkn3,y_ltkn4,y_ltkn8,y_m,y_rtkn1,y_rtkn4,y_rtkn8));
  PetscCall(DMSetApplicationContext(da2,&ctx2));
  PetscCall(CountBCs2(da2,&numBC2));
  PetscCall(SetPointBCs2(da2,numBC2));
  PetscCall(DMGetLocalSection(da2,&lsection2));
  PetscCall(DMGetPointSF(da2,&sf2));
  PetscCall(PetscSectionCreateGlobalSection(lsection2,sf2,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,&gsection2));
  PetscCall(DMSetGlobalSection(da2,gsection2));
  PetscCall(DMCreateSectionSF(da2,lsection2,gsection2));
  PetscCall(SNESCreate(comm,&snes2));
  PetscCall(SNESSetOptionsPrefix(snes2,"pressure_solve_"));
  PetscCall(SetPetscOptions2());
  PetscCall(SNESSetDM(snes2,da2));
  PetscCall(DMCreateMatrix(da2,&J2));
  PetscCall(SNESSetJacobian(snes2,J2,J2,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da2,&xglobal2));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,2553,p_vec->data,&xlocal2));
  PetscCall(VecGetLocalSize(xlocal2,&localsize2));
  PetscCall(DMCreateGlobalVector(da2,&bglobal2));
  PetscCall(SNESGetKSP(snes2,&ksp2));
  PetscCall(MatShellSetOperation(J2,MATOP_MULT,(void (*)(void))MatMult2));
  PetscCall(SNESSetFunction(snes2,NULL,FormFunction2,(void*)(da2)));
  PetscCall(SNESSetFromOptions(snes2));
  PetscCall(MatSetDM(J2,da2));

  PetscScalar r6 = 1.0/h_x;
  PetscScalar r0 = r6;
  PetscScalar r1 = r6;
  PetscScalar r2 = r6;
  PetscScalar r7 = 1.0/h_y;
  PetscScalar r3 = r7;
  PetscScalar r4 = r7;
  PetscScalar r5 = r7;

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    START(section0)
    haloupdate0(u_vec,comm,nb,t0);
    haloupdate0(v_vec,comm,nb,t0);
    ctx0.t0 = t0;
    ctx0.t1 = t1;
    ctx0.t2 = t2;
    PetscCall(FormRHS0(da0,bglobal0));
    PetscScalar * u_ptr0 = t2*localsize0 + (PetscScalar*)(u_vec->data);
    PetscCall(VecPlaceArray(xlocal0,u_ptr0));
    PetscCall(FormInitialGuess0(da0,xlocal0));
    PetscCall(DMLocalToGlobal(da0,xlocal0,INSERT_VALUES,xglobal0));
    PetscCall(SNESSolve(snes0,bglobal0,xglobal0));
    PetscCall(DMGlobalToLocal(da0,xglobal0,INSERT_VALUES,xlocal0));
    PetscCall(VecResetArray(xlocal0));

    PetscCall(KSPGetConvergedReason(ksp0,&reason0));
    petscinfo0->reason0 = reason0;
    PetscCall(KSPGetIterationNumber(ksp0,&kspits0));
    petscinfo0->kspits0 = kspits0;
    PetscCall(KSPGetNormType(ksp0,&kspnormtype0));
    petscinfo0->kspnormtype0 = kspnormtype0;
    PetscCall(KSPGetTolerances(ksp0,&rtol0,&atol0,&divtol0,&max_it0));
    petscinfo0->rtol0 = rtol0;
    petscinfo0->atol0 = atol0;
    petscinfo0->divtol0 = divtol0;
    petscinfo0->max_it0 = max_it0;
    PetscCall(KSPGetType(ksp0,&ksptype0));
    PetscCall(PetscStrncpy(petscinfo0->ksptype0,ksptype0,64));
    PetscCall(SNESGetIterationNumber(snes0,&snesits0));
    petscinfo0->snesits0 = snesits0;
    STOP(section0,timers)

    START(section1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_M - y_rtkn0 + 1; y <= y_M; y += 1)
      {
        u[t2][x + 2][y + 2] = 2 - u[t2][x + 2][y + 1];
      }
    }
    STOP(section1,timers)

    START(section2)
    ctx1.t0 = t0;
    ctx1.t1 = t1;
    ctx1.t2 = t2;
    PetscCall(FormRHS1(da1,bglobal1));
    PetscScalar * v_ptr0 = t2*localsize1 + (PetscScalar*)(v_vec->data);
    PetscCall(VecPlaceArray(xlocal1,v_ptr0));
    PetscCall(FormInitialGuess1(da1,xlocal1));
    PetscCall(DMLocalToGlobal(da1,xlocal1,INSERT_VALUES,xglobal1));
    PetscCall(SNESSolve(snes1,bglobal1,xglobal1));
    PetscCall(DMGlobalToLocal(da1,xglobal1,INSERT_VALUES,xlocal1));
    PetscCall(VecResetArray(xlocal1));

    PetscCall(KSPGetConvergedReason(ksp1,&reason1));
    petscinfo1->reason1 = reason1;
    PetscCall(KSPGetIterationNumber(ksp1,&kspits1));
    petscinfo1->kspits1 = kspits1;
    PetscCall(KSPGetNormType(ksp1,&kspnormtype1));
    petscinfo1->kspnormtype1 = kspnormtype1;
    PetscCall(KSPGetTolerances(ksp1,&rtol1,&atol1,&divtol1,&max_it1));
    petscinfo1->rtol1 = rtol1;
    petscinfo1->atol1 = atol1;
    petscinfo1->divtol1 = divtol1;
    petscinfo1->max_it1 = max_it1;
    PetscCall(KSPGetType(ksp1,&ksptype1));
    PetscCall(PetscStrncpy(petscinfo1->ksptype1,ksptype1,64));
    PetscCall(SNESGetIterationNumber(snes1,&snesits1));
    petscinfo1->snesits1 = snesits1;
    STOP(section2,timers)

    START(section3)
    for (int x = x_M - x_rtkn0 + 1; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        v[t2][x + 2][y + 2] = -v[t2][x + 1][y + 2];
      }
    }
    STOP(section3,timers)

    START(section4)
    ctx2.t2 = t2;
    PetscCall(FormRHS2(da2,bglobal2));
    PetscCall(FormInitialGuess2(da2,xlocal2));
    PetscCall(DMLocalToGlobal(da2,xlocal2,INSERT_VALUES,xglobal2));
    PetscCall(SNESSolve(snes2,bglobal2,xglobal2));
    PetscCall(DMGlobalToLocal(da2,xglobal2,INSERT_VALUES,xlocal2));

    PetscCall(KSPGetConvergedReason(ksp2,&reason2));
    petscinfo2->reason2 = reason2;
    PetscCall(KSPGetIterationNumber(ksp2,&kspits2));
    petscinfo2->kspits2 = kspits2;
    PetscCall(KSPGetNormType(ksp2,&kspnormtype2));
    petscinfo2->kspnormtype2 = kspnormtype2;
    PetscCall(KSPGetTolerances(ksp2,&rtol2,&atol2,&divtol2,&max_it2));
    petscinfo2->rtol2 = rtol2;
    petscinfo2->atol2 = atol2;
    petscinfo2->divtol2 = divtol2;
    petscinfo2->max_it2 = max_it2;
    PetscCall(KSPGetType(ksp2,&ksptype2));
    PetscCall(PetscStrncpy(petscinfo2->ksptype2,ksptype2,64));
    PetscCall(SNESGetIterationNumber(snes2,&snesits2));
    petscinfo2->snesits2 = snesits2;
    STOP(section4,timers)

    START(section5)
    haloupdate2(p_vec,comm,nb);
    for (int x = x_m + x_ltkn1; x <= x_M - x_rtkn1; x += 1)
    {
      #pragma omp simd aligned(p,u:32)
      for (int y = y_m + y_ltkn1; y <= y_M - y_rtkn1; y += 1)
      {
        u[t2][x + 2][y + 2] = -dt_c*(-r0*p[x + 1][y + 2] + r0*p[x + 2][y + 2]) + u[t2][x + 2][y + 2];
      }
    }
    STOP(section5,timers)

    START(section6)
    for (int x = x_m; x <= x_m + x_ltkn2 - 1; x += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int y = y_m; y <= y_m + y_ltkn2 - 1; y += 1)
      {
        u[t2][x + 2][y + 2] = 0;
      }
    }
    for (int x = x_M - x_rtkn0 + 1; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int y = y_m; y <= y_m + y_ltkn2 - 1; y += 1)
      {
        u[t2][x + 2][y + 2] = 0;
      }
    }
    for (int x = x_m + x_ltkn1; x <= x_M - x_rtkn1; x += 1)
    {
      #pragma omp simd aligned(p,u:32)
      for (int y = y_m; y <= y_m + y_ltkn3 - 1; y += 1)
      {
        u[t2][x + 2][y + 2] = -dt_c*(-r1*p[x + 1][y + 2] + r1*p[x + 2][y + 2]) + u[t2][x + 2][y + 2];
      }
      #pragma omp simd aligned(p,u:32)
      for (int y = y_m + y_ltkn4; y <= y_M - y_rtkn4; y += 1)
      {
        u[t2][x + 2][y + 2] = -dt_c*(-r2*p[x + 1][y + 2] + r2*p[x + 2][y + 2]) + u[t2][x + 2][y + 2];
      }
    }
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_M - y_rtkn0 + 1; y <= y_M; y += 1)
      {
        u[t2][x + 2][y + 2] = 2 - u[t2][x + 2][y + 1];
      }
    }
    STOP(section6,timers)

    START(section7)
    haloupdate3(p_vec,comm,nb);
    for (int x = x_m + x_ltkn3; x <= x_M - x_rtkn3; x += 1)
    {
      #pragma omp simd aligned(p,v:32)
      for (int y = y_m + y_ltkn5; y <= y_M - y_rtkn5; y += 1)
      {
        v[t2][x + 2][y + 2] = -dt_c*(-r3*p[x + 2][y + 1] + r3*p[x + 2][y + 2]) + v[t2][x + 2][y + 2];
      }
    }
    STOP(section7,timers)

    START(section8)
    for (int x = x_m; x <= x_m + x_ltkn4 - 1; x += 1)
    {
      #pragma omp simd aligned(v:32)
      for (int y = y_M - y_rtkn0 + 1; y <= y_M; y += 1)
      {
        v[t2][x + 2][y + 2] = 0;
      }
      #pragma omp simd aligned(v:32)
      for (int y = y_m; y <= y_m + y_ltkn3 - 1; y += 1)
      {
        v[t2][x + 2][y + 2] = 0;
      }
    }
    for (int x = x_m; x <= x_m + x_ltkn2 - 1; x += 1)
    {
      #pragma omp simd aligned(p,v:32)
      for (int y = y_m + y_ltkn5; y <= y_M - y_rtkn5; y += 1)
      {
        v[t2][x + 2][y + 2] = -dt_c*(-r4*p[x + 2][y + 1] + r4*p[x + 2][y + 2]) + v[t2][x + 2][y + 2];
      }
    }
    STOP(section8,timers)

    START(section9)
    for (int x = x_m + x_ltkn5; x <= x_M - x_rtkn5; x += 1)
    {
      #pragma omp simd aligned(p,v:32)
      for (int y = y_m + y_ltkn5; y <= y_M - y_rtkn5; y += 1)
      {
        v[t2][x + 2][y + 2] = -dt_c*(-r5*p[x + 2][y + 1] + r5*p[x + 2][y + 2]) + v[t2][x + 2][y + 2];
      }
    }
    STOP(section9,timers)

    START(section10)
    for (int x = x_M - x_rtkn0 + 1; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        v[t2][x + 2][y + 2] = -v[t2][x + 1][y + 2];
      }
    }
    STOP(section10,timers)
  }
  PetscCall(ClearPetscOptions0());
  PetscCall(ClearPetscOptions1());
  PetscCall(ClearPetscOptions2());

  PetscCall(VecDestroy(&bglobal0));
  PetscCall(VecDestroy(&bglobal1));
  PetscCall(VecDestroy(&bglobal2));
  PetscCall(VecDestroy(&xglobal0));
  PetscCall(VecDestroy(&xglobal1));
  PetscCall(VecDestroy(&xglobal2));
  PetscCall(VecDestroy(&xlocal0));
  PetscCall(VecDestroy(&xlocal1));
  PetscCall(VecDestroy(&xlocal2));
  PetscCall(MatDestroy(&J0));
  PetscCall(MatDestroy(&J1));
  PetscCall(MatDestroy(&J2));
  PetscCall(SNESDestroy(&snes0));
  PetscCall(SNESDestroy(&snes1));
  PetscCall(SNESDestroy(&snes2));
  PetscCall(PetscSectionDestroy(&gsection0));
  PetscCall(PetscSectionDestroy(&gsection1));
  PetscCall(PetscSectionDestroy(&gsection2));
  PetscCall(DMDestroy(&da0));
  PetscCall(DMDestroy(&da1));
  PetscCall(DMDestroy(&da2));

  return 0;
}


static void sendrecv0(struct dataobj * f0_vec, const PetscInt x_size, const PetscInt y_size, PetscInt ogtime, PetscInt ogx, PetscInt ogy, PetscInt ostime, PetscInt osx, PetscInt osy, PetscInt fromrank, PetscInt torank, MPI_Comm comm)
{
  MPI_Request rrecv;
  MPI_Request rsend;

  PetscScalar * a0_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&a0_vec),64,sizeof(double)*(long)y_size*(long)x_size);
  PetscScalar * a1_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&a1_vec),64,sizeof(double)*(long)y_size*(long)x_size);

  MPI_Irecv(a1_vec,x_size*y_size,MPI_DOUBLE,fromrank,13,comm,&rrecv);
  if (torank != MPI_PROC_NULL)
  {
    gather0(a0_vec,x_size,y_size,f0_vec,ogtime,ogx,ogy);
  }
  MPI_Isend(a0_vec,x_size*y_size,MPI_DOUBLE,torank,13,comm,&rsend);
  MPI_Wait(&rsend,MPI_STATUS_IGNORE);
  MPI_Wait(&rrecv,MPI_STATUS_IGNORE);
  if (fromrank != MPI_PROC_NULL)
  {
    scatter0(a1_vec,x_size,y_size,f0_vec,ostime,osx,osy);
  }

  free(a0_vec);
  free(a1_vec);
}

static void sendrecv2(struct dataobj * f0_vec, const PetscInt x_size, const PetscInt y_size, PetscInt ogx, PetscInt ogy, PetscInt osx, PetscInt osy, PetscInt fromrank, PetscInt torank, MPI_Comm comm)
{
  MPI_Request rrecv;
  MPI_Request rsend;

  PetscScalar * a0_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&a0_vec),64,sizeof(double)*(long)y_size*(long)x_size);
  PetscScalar * a1_vec __attribute__ ((aligned (64)));
  posix_memalign((void**)(&a1_vec),64,sizeof(double)*(long)y_size*(long)x_size);

  MPI_Irecv(a1_vec,x_size*y_size,MPI_DOUBLE,fromrank,13,comm,&rrecv);
  if (torank != MPI_PROC_NULL)
  {
    gather2(a0_vec,x_size,y_size,f0_vec,ogx,ogy);
  }
  MPI_Isend(a0_vec,x_size*y_size,MPI_DOUBLE,torank,13,comm,&rsend);
  MPI_Wait(&rsend,MPI_STATUS_IGNORE);
  MPI_Wait(&rrecv,MPI_STATUS_IGNORE);
  if (fromrank != MPI_PROC_NULL)
  {
    scatter2(a1_vec,x_size,y_size,f0_vec,osx,osy);
  }

  free(a0_vec);
  free(a1_vec);
}

static void haloupdate0(struct dataobj * f0_vec, MPI_Comm comm, struct neighborhood * nb, PetscInt otime)
{
  sendrecv0(f0_vec,f0_vec->hsize[3],f0_vec->npsize[2],otime,f0_vec->oofs[2],f0_vec->hofs[4],otime,f0_vec->hofs[3],f0_vec->hofs[4],nb->rc,nb->lc,comm);
  sendrecv0(f0_vec,f0_vec->hsize[2],f0_vec->npsize[2],otime,f0_vec->oofs[3],f0_vec->hofs[4],otime,f0_vec->hofs[2],f0_vec->hofs[4],nb->lc,nb->rc,comm);
  sendrecv0(f0_vec,f0_vec->npsize[1],f0_vec->hsize[5],otime,f0_vec->hofs[2],f0_vec->oofs[4],otime,f0_vec->hofs[2],f0_vec->hofs[5],nb->cr,nb->cl,comm);
  sendrecv0(f0_vec,f0_vec->npsize[1],f0_vec->hsize[4],otime,f0_vec->hofs[2],f0_vec->oofs[5],otime,f0_vec->hofs[2],f0_vec->hofs[4],nb->cl,nb->cr,comm);
}

static void haloupdate2(struct dataobj * p_vec, MPI_Comm comm, struct neighborhood * nb)
{
  sendrecv2(p_vec,p_vec->hsize[0],p_vec->npsize[1],p_vec->oofs[1],p_vec->hofs[2],p_vec->hofs[0],p_vec->hofs[2],nb->lc,nb->rc,comm);
}

static void haloupdate3(struct dataobj * p_vec, MPI_Comm comm, struct neighborhood * nb)
{
  sendrecv2(p_vec,p_vec->npsize[0],p_vec->hsize[2],p_vec->hofs[0],p_vec->oofs[3],p_vec->hofs[0],p_vec->hofs[2],nb->cl,nb->cr,comm);
}

static void gather0(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt otime, const PetscInt ox, const PetscInt oy)
{
  PetscScalar (* a0)[bx_size][by_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[bx_size][by_size]) a0_vec;
  PetscScalar (* f0)[f0_vec->size[1]][f0_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[f0_vec->size[1]][f0_vec->size[2]]) f0_vec->data;

  const PetscInt x_m = 0;
  const PetscInt y_m = 0;
  const PetscInt x_M = bx_size - 1;
  const PetscInt y_M = by_size - 1;

  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(f0:32)
    for (int y = y_m; y <= y_M; y += 1)
    {
      a0[0][x][y] = f0[otime][x + ox][y + oy];
    }
  }
}

static void scatter0(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt otime, const PetscInt ox, const PetscInt oy)
{
  PetscScalar (* a0)[bx_size][by_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[bx_size][by_size]) a0_vec;
  PetscScalar (* f0)[f0_vec->size[1]][f0_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[f0_vec->size[1]][f0_vec->size[2]]) f0_vec->data;

  const PetscInt x_m = 0;
  const PetscInt y_m = 0;
  const PetscInt x_M = bx_size - 1;
  const PetscInt y_M = by_size - 1;

  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(f0:32)
    for (int y = y_m; y <= y_M; y += 1)
    {
      f0[otime][x + ox][y + oy] = a0[0][x][y];
    }
  }
}

static void gather2(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt ox, const PetscInt oy)
{
  PetscScalar (* a0)[bx_size][by_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[bx_size][by_size]) a0_vec;
  PetscScalar (* f0)[f0_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[f0_vec->size[1]]) f0_vec->data;

  const PetscInt x_m = 0;
  const PetscInt y_m = 0;
  const PetscInt x_M = bx_size - 1;
  const PetscInt y_M = by_size - 1;

  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(f0:32)
    for (int y = y_m; y <= y_M; y += 1)
    {
      a0[0][x][y] = f0[x + ox][y + oy];
    }
  }
}

static void scatter2(PetscScalar * a0_vec, PetscInt bx_size, PetscInt by_size, struct dataobj * f0_vec, const PetscInt ox, const PetscInt oy)
{
  PetscScalar (* a0)[bx_size][by_size] __attribute__ ((aligned (64))) = (PetscScalar (*)[bx_size][by_size]) a0_vec;
  PetscScalar (* f0)[f0_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[f0_vec->size[1]]) f0_vec->data;

  const PetscInt x_m = 0;
  const PetscInt y_m = 0;
  const PetscInt x_M = bx_size - 1;
  const PetscInt y_M = by_size - 1;

  for (int x = x_m; x <= x_M; x += 1)
  {
    #pragma omp simd aligned(f0:32)
    for (int y = y_m; y <= y_M; y += 1)
    {
      f0[x + ox][y + oy] = a0[0][x][y];
    }
  }
}

PetscErrorCode CountBCs0(DM dm0, PetscInt * numBCPtr0)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));

  PetscInt count_u = *numBCPtr0;
  struct dataobj * _stagger_border_u_vec = ctx0->_stagger_border_u_vec;

  PetscInt (* _stagger_border_u)[_stagger_border_u_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_u_vec->size[1]]) _stagger_border_u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int n0 = ctx0->n0_m; n0 <= ctx0->n0_M; n0 += 1)
  {
    ctx0->x_ltkn6 = _stagger_border_u[n0][0];
    ctx0->x_rtkn6 = _stagger_border_u[n0][1];
    ctx0->y_ltkn6 = _stagger_border_u[n0][2];
    ctx0->y_rtkn6 = _stagger_border_u[n0][3];

    for (int ix = ctx0->x_m + ctx0->x_ltkn6; ix <= ctx0->x_M - ctx0->x_rtkn6; ix += 1)
    {
      for (int iy = ctx0->y_m + ctx0->y_ltkn6; iy <= ctx0->y_M - ctx0->y_rtkn6; iy += 1)
      {
        count_u += 1;
      }
    }
  }

  *numBCPtr0 = count_u;

  PetscFunctionReturn(0);
}

PetscErrorCode SetPointBCs0(DM dm0, PetscInt numBC0)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  PetscInt k_iter = 0;
  IS bcPointsIS;
  DMDALocalInfo info;

  IS * bcPoints;
  PetscInt * bcPointsArr0;

  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * _stagger_border_u_vec = ctx0->_stagger_border_u_vec;
  struct dataobj * u_vec = ctx0->u_vec;

  PetscInt (* _stagger_border_u)[_stagger_border_u_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_u_vec->size[1]]) _stagger_border_u_vec->data;
  PetscScalar (* u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;

  const PetscInt x_fsz0 = u_vec->size[1];
  const PetscInt y_fsz0 = u_vec->size[2];

  const PetscInt x_stride0 = x_fsz0*y_fsz0;
  const PetscInt y_stride0 = y_fsz0;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscMalloc1(numBC0,&bcPointsArr0));
  for (int n0 = ctx0->n0_m; n0 <= ctx0->n0_M; n0 += 1)
  {
    ctx0->x_ltkn6 = _stagger_border_u[n0][0];
    ctx0->x_rtkn6 = _stagger_border_u[n0][1];
    ctx0->y_ltkn6 = _stagger_border_u[n0][2];
    ctx0->y_rtkn6 = _stagger_border_u[n0][3];

    for (int ix = ctx0->x_m + ctx0->x_ltkn6; ix <= ctx0->x_M - ctx0->x_rtkn6; ix += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int iy = ctx0->y_m + ctx0->y_ltkn6; iy <= ctx0->y_M - ctx0->y_rtkn6; iy += 1)
      {
        bcPointsArr0[k_iter++] = y_stride0*(ix + 2) + iy + 2;
      }
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)(dm0)),numBC0,bcPointsArr0,PETSC_OWN_POINTER,&bcPointsIS));
  PetscCall(PetscMalloc1(1,&bcPoints));
  bcPoints[0] = bcPointsIS;
  PetscCall(DMDASetPointBC(dm0,1,bcPoints,NULL));

  PetscCall(ISDestroy(&bcPoints[0]));
  PetscCall(PetscFree(bcPoints));

  PetscFunctionReturn(0);
}

PetscErrorCode SetPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_ksp_type","cg"));
  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_ksp_rtol","1e-07"));
  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-utent_solve_ksp_max_it","10000"));

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

  PetscScalar * x_u_vec;
  PetscScalar * y_u_vec;

  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_u_vec));
  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * _stagger_border_u_vec = ctx0->_stagger_border_u_vec;

  PetscInt (* _stagger_border_u)[_stagger_border_u_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_u_vec->size[1]]) _stagger_border_u_vec->data;
  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;
  PetscScalar (* y_u)[info.gxm] = (PetscScalar (*)[info.gxm]) y_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      y_u[ix + 2][iy + 2] = x_u[ix + 2][iy + 2];
      x_u[ix + 2][iy + 2] = 0.0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn0 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      y_u[ix + 2][iy + 2] = x_u[ix + 2][iy + 2];
      x_u[ix + 2][iy + 2] = 0.0;
    }
  }
  for (int n0 = ctx0->n0_m; n0 <= ctx0->n0_M; n0 += 1)
  {
    ctx0->x_ltkn6 = _stagger_border_u[n0][0];
    ctx0->x_rtkn6 = _stagger_border_u[n0][1];
    ctx0->y_ltkn6 = _stagger_border_u[n0][2];
    ctx0->y_rtkn6 = _stagger_border_u[n0][3];

    for (int ix = ctx0->x_m + ctx0->x_ltkn6; ix <= ctx0->x_M - ctx0->x_rtkn6; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx0->y_m + ctx0->y_ltkn6; iy <= ctx0->y_M - ctx0->y_rtkn6; iy += 1)
      {
        y_u[ix + 2][iy + 2] = x_u[ix + 2][iy + 2];
        x_u[ix + 2][iy + 2] = 0.0;
      }
    }
  }

  PetscScalar r16 = 1.0/ctx0->dt;
  PetscScalar r8 = r16;
  PetscScalar r17 = 1.0/ctx0->re;
  PetscScalar r9 = r17;
  PetscScalar r18 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r10 = r18;
  PetscScalar r19 = 1.0/(ctx0->h_y*ctx0->h_y);
  PetscScalar r11 = r19;
  PetscScalar r12 = r16;
  PetscScalar r13 = r17;
  PetscScalar r14 = r18;
  PetscScalar r15 = r19;

  for (int ix = ctx0->x_m + ctx0->x_ltkn1; ix <= ctx0->x_M - ctx0->x_rtkn1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn1; iy <= ctx0->y_M - ctx0->y_rtkn1; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (r8*x_u[ix + 2][iy + 2] - 5.0e-1*r9*(r10*x_u[ix + 1][iy + 2] + r10*x_u[ix + 3][iy + 2] + r11*x_u[ix + 2][iy + 1] + r11*x_u[ix + 2][iy + 3] + (-2.0)*(r10*x_u[ix + 2][iy + 2] + r11*x_u[ix + 2][iy + 2])))*ctx0->h_x*ctx0->h_y;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn3 - 1; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (-5.0e-1*((-3.0*x_u[ix + 2][iy + 2] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) + (x_u[ix + 1][iy + 2] - 2.0*x_u[ix + 2][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))/ctx0->re + x_u[ix + 2][iy + 2]/ctx0->dt)*ctx0->h_x*ctx0->h_y;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn4; iy <= ctx0->y_M - ctx0->y_rtkn4; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (r12*x_u[ix + 2][iy + 2] - 5.0e-1*r13*(r14*x_u[ix + 1][iy + 2] - 2.0*r14*x_u[ix + 2][iy + 2] + r14*x_u[ix + 3][iy + 2] + r15*x_u[ix + 2][iy + 1] + r15*(-3.0*x_u[ix + 2][iy + 2])))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_u_vec));
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

  PetscScalar * f_u_vec;
  PetscScalar * x_u_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&floc));
  PetscCall(VecGetArray(floc,&f_u_vec));
  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * _stagger_border_u_vec = ctx0->_stagger_border_u_vec;

  PetscInt (* _stagger_border_u)[_stagger_border_u_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_u_vec->size[1]]) _stagger_border_u_vec->data;
  PetscScalar (* f_u)[info.gxm] = (PetscScalar (*)[info.gxm]) f_u_vec;
  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      f_u[ix + 2][iy + 2] = x_u[ix + 2][iy + 2];
      x_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn0 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      f_u[ix + 2][iy + 2] = x_u[ix + 2][iy + 2];
      x_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int n0 = ctx0->n0_m; n0 <= ctx0->n0_M; n0 += 1)
  {
    ctx0->x_ltkn6 = _stagger_border_u[n0][0];
    ctx0->x_rtkn6 = _stagger_border_u[n0][1];
    ctx0->y_ltkn6 = _stagger_border_u[n0][2];
    ctx0->y_rtkn6 = _stagger_border_u[n0][3];

    for (int ix = ctx0->x_m + ctx0->x_ltkn6; ix <= ctx0->x_M - ctx0->x_rtkn6; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx0->y_m + ctx0->y_ltkn6; iy <= ctx0->y_M - ctx0->y_rtkn6; iy += 1)
      {
        f_u[ix + 2][iy + 2] = -ctx0->zero + x_u[ix + 2][iy + 2];
        x_u[ix + 2][iy + 2] = ctx0->zero;
      }
    }
  }

  PetscScalar r28 = 1.0/ctx0->dt;
  PetscScalar r20 = r28;
  PetscScalar r29 = 1.0/ctx0->re;
  PetscScalar r21 = r29;
  PetscScalar r30 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r22 = r30;
  PetscScalar r31 = 1.0/(ctx0->h_y*ctx0->h_y);
  PetscScalar r23 = r31;
  PetscScalar r24 = r28;
  PetscScalar r25 = r29;
  PetscScalar r26 = r30;
  PetscScalar r27 = r31;

  for (int ix = ctx0->x_m + ctx0->x_ltkn1; ix <= ctx0->x_M - ctx0->x_rtkn1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn1; iy <= ctx0->y_M - ctx0->y_rtkn1; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (r20*x_u[ix + 2][iy + 2] - 5.0e-1*r21*(r22*x_u[ix + 1][iy + 2] + r22*x_u[ix + 3][iy + 2] + r23*x_u[ix + 2][iy + 1] + r23*x_u[ix + 2][iy + 3] + (-2.0)*(r22*x_u[ix + 2][iy + 2] + r23*x_u[ix + 2][iy + 2])))*ctx0->h_x*ctx0->h_y;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn3 - 1; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (-5.0e-1*((-3.0*x_u[ix + 2][iy + 2] + x_u[ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) + (x_u[ix + 1][iy + 2] - 2.0*x_u[ix + 2][iy + 2] + x_u[ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))/ctx0->re + x_u[ix + 2][iy + 2]/ctx0->dt)*ctx0->h_x*ctx0->h_y;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn4; iy <= ctx0->y_M - ctx0->y_rtkn4; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (r24*x_u[ix + 2][iy + 2] - 5.0e-1*r25*(r26*x_u[ix + 1][iy + 2] - 2.0*r26*x_u[ix + 2][iy + 2] + r26*x_u[ix + 3][iy + 2] + r27*x_u[ix + 2][iy + 1] + r27*(-3.0*x_u[ix + 2][iy + 2])))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_u_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm0,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&floc));

  PetscFunctionReturn(0);
}

PetscErrorCode CountBCs1(DM dm1, PetscInt * numBCPtr1)
{
  PetscFunctionBeginUser;

  struct UserCtx1 * ctx1;
  PetscCall(DMGetApplicationContext(dm1,&ctx1));

  PetscInt count_v = *numBCPtr1;
  struct dataobj * _stagger_border_v_vec = ctx1->_stagger_border_v_vec;

  PetscInt (* _stagger_border_v)[_stagger_border_v_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_v_vec->size[1]]) _stagger_border_v_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int n1 = ctx1->n1_m; n1 <= ctx1->n1_M; n1 += 1)
  {
    ctx1->x_ltkn7 = _stagger_border_v[n1][0];
    ctx1->x_rtkn7 = _stagger_border_v[n1][1];
    ctx1->y_ltkn7 = _stagger_border_v[n1][2];
    ctx1->y_rtkn7 = _stagger_border_v[n1][3];

    for (int ix = ctx1->x_m + ctx1->x_ltkn7; ix <= ctx1->x_M - ctx1->x_rtkn7; ix += 1)
    {
      for (int iy = ctx1->y_m + ctx1->y_ltkn7; iy <= ctx1->y_M - ctx1->y_rtkn7; iy += 1)
      {
        count_v += 1;
      }
    }
  }

  *numBCPtr1 = count_v;

  PetscFunctionReturn(0);
}

PetscErrorCode SetPointBCs1(DM dm1, PetscInt numBC1)
{
  PetscFunctionBeginUser;

  struct UserCtx1 * ctx1;
  PetscCall(DMGetApplicationContext(dm1,&ctx1));
  PetscInt k_iter = 0;
  IS bcPointsIS;
  DMDALocalInfo info;

  IS * bcPoints;
  PetscInt * bcPointsArr1;

  PetscCall(DMDAGetLocalInfo(dm1,&info));
  struct dataobj * _stagger_border_v_vec = ctx1->_stagger_border_v_vec;
  struct dataobj * v_vec = ctx1->v_vec;

  PetscInt (* _stagger_border_v)[_stagger_border_v_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_v_vec->size[1]]) _stagger_border_v_vec->data;
  PetscScalar (* v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  const PetscInt x_fsz1 = v_vec->size[1];
  const PetscInt y_fsz1 = v_vec->size[2];

  const PetscInt x_stride1 = x_fsz1*y_fsz1;
  const PetscInt y_stride1 = y_fsz1;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscMalloc1(numBC1,&bcPointsArr1));
  for (int n1 = ctx1->n1_m; n1 <= ctx1->n1_M; n1 += 1)
  {
    ctx1->x_ltkn7 = _stagger_border_v[n1][0];
    ctx1->x_rtkn7 = _stagger_border_v[n1][1];
    ctx1->y_ltkn7 = _stagger_border_v[n1][2];
    ctx1->y_rtkn7 = _stagger_border_v[n1][3];

    for (int ix = ctx1->x_m + ctx1->x_ltkn7; ix <= ctx1->x_M - ctx1->x_rtkn7; ix += 1)
    {
      #pragma omp simd aligned(v:32)
      for (int iy = ctx1->y_m + ctx1->y_ltkn7; iy <= ctx1->y_M - ctx1->y_rtkn7; iy += 1)
      {
        bcPointsArr1[k_iter++] = y_stride1*(ix + 2) + iy + 2;
      }
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)(dm1)),numBC1,bcPointsArr1,PETSC_OWN_POINTER,&bcPointsIS));
  PetscCall(PetscMalloc1(1,&bcPoints));
  bcPoints[0] = bcPointsIS;
  PetscCall(DMDASetPointBC(dm1,1,bcPoints,NULL));

  PetscCall(ISDestroy(&bcPoints[0]));
  PetscCall(PetscFree(bcPoints));

  PetscFunctionReturn(0);
}

PetscErrorCode SetPetscOptions1()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_ksp_type","cg"));
  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_ksp_rtol","1e-07"));
  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-vtent_solve_ksp_max_it","10000"));

  PetscFunctionReturn(0);
}

PetscErrorCode MatMult1(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  struct UserCtx1 * ctx1;
  DM dm1;
  PetscCall(MatGetDM(J,&dm1));
  PetscCall(DMGetApplicationContext(dm1,&ctx1));
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  PetscScalar * x_v_vec;
  PetscScalar * y_v_vec;

  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm1,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm1,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm1,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm1,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_v_vec));
  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm1,&info));
  struct dataobj * _stagger_border_v_vec = ctx1->_stagger_border_v_vec;

  PetscInt (* _stagger_border_v)[_stagger_border_v_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_v_vec->size[1]]) _stagger_border_v_vec->data;
  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;
  PetscScalar (* y_v)[info.gxm] = (PetscScalar (*)[info.gxm]) y_v_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn4 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_M - ctx1->y_rtkn0 + 1; iy <= ctx1->y_M; iy += 1)
    {
      y_v[ix + 2][iy + 2] = x_v[ix + 2][iy + 2];
      x_v[ix + 2][iy + 2] = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx1->y_m; iy <= ctx1->y_m + ctx1->y_ltkn3 - 1; iy += 1)
    {
      y_v[ix + 2][iy + 2] = x_v[ix + 2][iy + 2];
      x_v[ix + 2][iy + 2] = 0.0;
    }
  }
  for (int n1 = ctx1->n1_m; n1 <= ctx1->n1_M; n1 += 1)
  {
    ctx1->x_ltkn7 = _stagger_border_v[n1][0];
    ctx1->x_rtkn7 = _stagger_border_v[n1][1];
    ctx1->y_ltkn7 = _stagger_border_v[n1][2];
    ctx1->y_rtkn7 = _stagger_border_v[n1][3];

    for (int ix = ctx1->x_m + ctx1->x_ltkn7; ix <= ctx1->x_M - ctx1->x_rtkn7; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx1->y_m + ctx1->y_ltkn7; iy <= ctx1->y_M - ctx1->y_rtkn7; iy += 1)
      {
        y_v[ix + 2][iy + 2] = x_v[ix + 2][iy + 2];
        x_v[ix + 2][iy + 2] = 0.0;
      }
    }
  }

  PetscScalar r52 = 1.0/ctx1->dt;
  PetscScalar r53 = 1.0/ctx1->re;
  PetscScalar r54 = 1.0/(ctx1->h_x*ctx1->h_x);
  PetscScalar r55 = 1.0/(ctx1->h_y*ctx1->h_y);

  for (int ix = ctx1->x_m + ctx1->x_ltkn3; ix <= ctx1->x_M - ctx1->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      y_v[ix + 2][iy + 2] = (r52*x_v[ix + 2][iy + 2] - 5.0e-1*r53*(r54*x_v[ix + 1][iy + 2] + r54*x_v[ix + 3][iy + 2] + r55*x_v[ix + 2][iy + 1] + r55*x_v[ix + 2][iy + 3] + (-2.0)*(r54*x_v[ix + 2][iy + 2] + r55*x_v[ix + 2][iy + 2])))*ctx1->h_x*ctx1->h_y;
    }
  }
  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      y_v[ix + 2][iy + 2] = (-5.0e-1*((x_v[ix + 2][iy + 1] - 2.0*x_v[ix + 2][iy + 2] + x_v[ix + 2][iy + 3])/((ctx1->h_y*ctx1->h_y)) + (-3.0*x_v[ix + 2][iy + 2] + x_v[ix + 3][iy + 2])/((ctx1->h_x*ctx1->h_x)))/ctx1->re + x_v[ix + 2][iy + 2]/ctx1->dt)*ctx1->h_x*ctx1->h_y;
    }
  }

  PetscScalar r56 = 1.0/ctx1->dt;
  PetscScalar r57 = 1.0/ctx1->re;
  PetscScalar r58 = 1.0/(ctx1->h_x*ctx1->h_x);
  PetscScalar r59 = 1.0/(ctx1->h_y*ctx1->h_y);

  for (int ix = ctx1->x_m + ctx1->x_ltkn5; ix <= ctx1->x_M - ctx1->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      y_v[ix + 2][iy + 2] = (r56*x_v[ix + 2][iy + 2] - 5.0e-1*r57*(r58*x_v[ix + 1][iy + 2] + r58*(-3.0*x_v[ix + 2][iy + 2]) + r59*x_v[ix + 2][iy + 1] - 2.0*r59*x_v[ix + 2][iy + 2] + r59*x_v[ix + 2][iy + 3]))*ctx1->h_x*ctx1->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_v_vec));
  PetscCall(VecRestoreArray(xloc,&x_v_vec));
  PetscCall(DMLocalToGlobalBegin(dm1,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm1,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm1,&xloc));
  PetscCall(DMRestoreLocalVector(dm1,&yloc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction1(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  struct UserCtx1 * ctx1;
  DM dm1 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm1,&ctx1));
  Vec floc;
  DMDALocalInfo info;
  Vec xloc;

  PetscScalar * f_v_vec;
  PetscScalar * x_v_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm1,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm1,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm1,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm1,&floc));
  PetscCall(VecGetArray(floc,&f_v_vec));
  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm1,&info));
  struct dataobj * _stagger_border_v_vec = ctx1->_stagger_border_v_vec;

  PetscInt (* _stagger_border_v)[_stagger_border_v_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_v_vec->size[1]]) _stagger_border_v_vec->data;
  PetscScalar (* f_v)[info.gxm] = (PetscScalar (*)[info.gxm]) f_v_vec;
  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn4 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_M - ctx1->y_rtkn0 + 1; iy <= ctx1->y_M; iy += 1)
    {
      f_v[ix + 2][iy + 2] = x_v[ix + 2][iy + 2];
      x_v[ix + 2][iy + 2] = 0;
    }
    #pragma omp simd
    for (int iy = ctx1->y_m; iy <= ctx1->y_m + ctx1->y_ltkn3 - 1; iy += 1)
    {
      f_v[ix + 2][iy + 2] = x_v[ix + 2][iy + 2];
      x_v[ix + 2][iy + 2] = 0;
    }
  }
  for (int n1 = ctx1->n1_m; n1 <= ctx1->n1_M; n1 += 1)
  {
    ctx1->x_ltkn7 = _stagger_border_v[n1][0];
    ctx1->x_rtkn7 = _stagger_border_v[n1][1];
    ctx1->y_ltkn7 = _stagger_border_v[n1][2];
    ctx1->y_rtkn7 = _stagger_border_v[n1][3];

    for (int ix = ctx1->x_m + ctx1->x_ltkn7; ix <= ctx1->x_M - ctx1->x_rtkn7; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx1->y_m + ctx1->y_ltkn7; iy <= ctx1->y_M - ctx1->y_rtkn7; iy += 1)
      {
        f_v[ix + 2][iy + 2] = -ctx1->zero + x_v[ix + 2][iy + 2];
        x_v[ix + 2][iy + 2] = ctx1->zero;
      }
    }
  }

  PetscScalar r60 = 1.0/ctx1->dt;
  PetscScalar r61 = 1.0/ctx1->re;
  PetscScalar r62 = 1.0/(ctx1->h_x*ctx1->h_x);
  PetscScalar r63 = 1.0/(ctx1->h_y*ctx1->h_y);

  for (int ix = ctx1->x_m + ctx1->x_ltkn3; ix <= ctx1->x_M - ctx1->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      f_v[ix + 2][iy + 2] = (r60*x_v[ix + 2][iy + 2] - 5.0e-1*r61*(r62*x_v[ix + 1][iy + 2] + r62*x_v[ix + 3][iy + 2] + r63*x_v[ix + 2][iy + 1] + r63*x_v[ix + 2][iy + 3] + (-2.0)*(r62*x_v[ix + 2][iy + 2] + r63*x_v[ix + 2][iy + 2])))*ctx1->h_x*ctx1->h_y;
    }
  }
  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      f_v[ix + 2][iy + 2] = (-5.0e-1*((x_v[ix + 2][iy + 1] - 2.0*x_v[ix + 2][iy + 2] + x_v[ix + 2][iy + 3])/((ctx1->h_y*ctx1->h_y)) + (-3.0*x_v[ix + 2][iy + 2] + x_v[ix + 3][iy + 2])/((ctx1->h_x*ctx1->h_x)))/ctx1->re + x_v[ix + 2][iy + 2]/ctx1->dt)*ctx1->h_x*ctx1->h_y;
    }
  }

  PetscScalar r64 = 1.0/ctx1->dt;
  PetscScalar r65 = 1.0/ctx1->re;
  PetscScalar r66 = 1.0/(ctx1->h_x*ctx1->h_x);
  PetscScalar r67 = 1.0/(ctx1->h_y*ctx1->h_y);

  for (int ix = ctx1->x_m + ctx1->x_ltkn5; ix <= ctx1->x_M - ctx1->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      f_v[ix + 2][iy + 2] = (r64*x_v[ix + 2][iy + 2] - 5.0e-1*r65*(r66*x_v[ix + 1][iy + 2] + r66*(-3.0*x_v[ix + 2][iy + 2]) + r67*x_v[ix + 2][iy + 1] - 2.0*r67*x_v[ix + 2][iy + 2] + r67*x_v[ix + 2][iy + 3]))*ctx1->h_x*ctx1->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_v_vec));
  PetscCall(VecRestoreArray(xloc,&x_v_vec));
  PetscCall(DMLocalToGlobalBegin(dm1,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm1,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm1,&xloc));
  PetscCall(DMRestoreLocalVector(dm1,&floc));

  PetscFunctionReturn(0);
}

PetscErrorCode CountBCs2(DM dm2, PetscInt * numBCPtr2)
{
  PetscFunctionBeginUser;

  struct UserCtx2 * ctx2;
  PetscCall(DMGetApplicationContext(dm2,&ctx2));

  PetscInt count_p = *numBCPtr2;
  struct dataobj * _stagger_border_p_vec = ctx2->_stagger_border_p_vec;

  PetscInt (* _stagger_border_p)[_stagger_border_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_p_vec->size[1]]) _stagger_border_p_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int n2 = ctx2->n2_m; n2 <= ctx2->n2_M; n2 += 1)
  {
    ctx2->x_ltkn8 = _stagger_border_p[n2][0];
    ctx2->x_rtkn8 = _stagger_border_p[n2][1];
    ctx2->y_ltkn8 = _stagger_border_p[n2][2];
    ctx2->y_rtkn8 = _stagger_border_p[n2][3];

    for (int ix = ctx2->x_m + ctx2->x_ltkn8; ix <= ctx2->x_M - ctx2->x_rtkn8; ix += 1)
    {
      for (int iy = ctx2->y_m + ctx2->y_ltkn8; iy <= ctx2->y_M - ctx2->y_rtkn8; iy += 1)
      {
        count_p += 1;
      }
    }
  }

  *numBCPtr2 = count_p;

  PetscFunctionReturn(0);
}

PetscErrorCode SetPointBCs2(DM dm2, PetscInt numBC2)
{
  PetscFunctionBeginUser;

  struct UserCtx2 * ctx2;
  PetscCall(DMGetApplicationContext(dm2,&ctx2));
  PetscInt k_iter = 0;
  IS bcPointsIS;
  DMDALocalInfo info;

  IS * bcPoints;
  PetscInt * bcPointsArr2;

  PetscCall(DMDAGetLocalInfo(dm2,&info));
  struct dataobj * _stagger_border_p_vec = ctx2->_stagger_border_p_vec;
  struct dataobj * p_vec = ctx2->p_vec;

  PetscInt (* _stagger_border_p)[_stagger_border_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_p_vec->size[1]]) _stagger_border_p_vec->data;
  PetscScalar (* p)[p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[p_vec->size[1]]) p_vec->data;

  const PetscInt y_fsz2 = p_vec->size[1];

  const PetscInt y_stride2 = y_fsz2;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscMalloc1(numBC2,&bcPointsArr2));
  for (int n2 = ctx2->n2_m; n2 <= ctx2->n2_M; n2 += 1)
  {
    ctx2->x_ltkn8 = _stagger_border_p[n2][0];
    ctx2->x_rtkn8 = _stagger_border_p[n2][1];
    ctx2->y_ltkn8 = _stagger_border_p[n2][2];
    ctx2->y_rtkn8 = _stagger_border_p[n2][3];

    for (int ix = ctx2->x_m + ctx2->x_ltkn8; ix <= ctx2->x_M - ctx2->x_rtkn8; ix += 1)
    {
      #pragma omp simd aligned(p:32)
      for (int iy = ctx2->y_m + ctx2->y_ltkn8; iy <= ctx2->y_M - ctx2->y_rtkn8; iy += 1)
      {
        bcPointsArr2[k_iter++] = y_stride2*(ix + 2) + iy + 2;
      }
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)(dm2)),numBC2,bcPointsArr2,PETSC_OWN_POINTER,&bcPointsIS));
  PetscCall(PetscMalloc1(1,&bcPoints));
  bcPoints[0] = bcPointsIS;
  PetscCall(DMDASetPointBC(dm2,1,bcPoints,NULL));

  PetscCall(ISDestroy(&bcPoints[0]));
  PetscCall(PetscFree(bcPoints));

  PetscFunctionReturn(0);
}

PetscErrorCode SetPetscOptions2()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_ksp_type","cg"));
  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_ksp_rtol","1e-07"));
  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-pressure_solve_ksp_max_it","10000"));

  PetscFunctionReturn(0);
}

PetscErrorCode MatMult2(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  struct UserCtx2 * ctx2;
  DM dm2;
  PetscCall(MatGetDM(J,&dm2));
  PetscCall(DMGetApplicationContext(dm2,&ctx2));
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  PetscScalar * x_p_vec;
  PetscScalar * y_p_vec;

  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm2,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm2,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm2,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm2,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_p_vec));
  PetscCall(VecGetArray(xloc,&x_p_vec));
  PetscCall(DMDAGetLocalInfo(dm2,&info));
  struct dataobj * _stagger_border_p_vec = ctx2->_stagger_border_p_vec;

  PetscInt (* _stagger_border_p)[_stagger_border_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_p_vec->size[1]]) _stagger_border_p_vec->data;
  PetscScalar (* x_p)[info.gxm] = (PetscScalar (*)[info.gxm]) x_p_vec;
  PetscScalar (* y_p)[info.gxm] = (PetscScalar (*)[info.gxm]) y_p_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      y_p[ix + 2][iy + 2] = x_p[ix + 2][iy + 2];
      x_p[ix + 2][iy + 2] = 0.0;
    }
  }
  for (int n2 = ctx2->n2_m; n2 <= ctx2->n2_M; n2 += 1)
  {
    ctx2->x_ltkn8 = _stagger_border_p[n2][0];
    ctx2->x_rtkn8 = _stagger_border_p[n2][1];
    ctx2->y_ltkn8 = _stagger_border_p[n2][2];
    ctx2->y_rtkn8 = _stagger_border_p[n2][3];

    for (int ix = ctx2->x_m + ctx2->x_ltkn8; ix <= ctx2->x_M - ctx2->x_rtkn8; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx2->y_m + ctx2->y_ltkn8; iy <= ctx2->y_M - ctx2->y_rtkn8; iy += 1)
      {
        y_p[ix + 2][iy + 2] = x_p[ix + 2][iy + 2];
        x_p[ix + 2][iy + 2] = 0.0;
      }
    }
  }

  PetscScalar r82 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r83 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      y_p[ix + 2][iy + 2] = (r82*x_p[ix + 1][iy + 2] + r82*x_p[ix + 3][iy + 2] + r83*x_p[ix + 2][iy + 1] + r83*x_p[ix + 2][iy + 3] + (-2.0)*(r82*x_p[ix + 2][iy + 2] + r83*x_p[ix + 2][iy + 2]))*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      PetscScalar r90 = -1.0*x_p[ix + 2][iy + 2];
      y_p[ix + 2][iy + 2] = ((r90 + x_p[ix + 2][iy + 1])/((ctx2->h_y*ctx2->h_y)) + (r90 + x_p[ix + 3][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r84 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r85 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      y_p[ix + 2][iy + 2] = (r84*x_p[ix + 1][iy + 2] - 2.0*r84*x_p[ix + 2][iy + 2] + r84*x_p[ix + 3][iy + 2] + r85*x_p[ix + 2][iy + 1] - 1.0*r85*x_p[ix + 2][iy + 2])*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r86 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r87 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      y_p[ix + 2][iy + 2] = (r86*x_p[ix + 1][iy + 2] + r87*x_p[ix + 2][iy + 1] + (-1.0)*(r86*x_p[ix + 2][iy + 2] + r87*x_p[ix + 2][iy + 2]))*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      y_p[ix + 2][iy + 2] = ((x_p[ix + 2][iy + 1] - 2.0*x_p[ix + 2][iy + 2] + x_p[ix + 2][iy + 3])/((ctx2->h_y*ctx2->h_y)) + (-1.0*x_p[ix + 2][iy + 2] + x_p[ix + 3][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r88 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r89 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      y_p[ix + 2][iy + 2] = (r88*x_p[ix + 1][iy + 2] - 1.0*r88*x_p[ix + 2][iy + 2] + r89*x_p[ix + 2][iy + 1] - 2.0*r89*x_p[ix + 2][iy + 2] + r89*x_p[ix + 2][iy + 3])*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      y_p[ix + 2][iy + 2] = ((-1.0*x_p[ix + 2][iy + 2] + x_p[ix + 2][iy + 3])/((ctx2->h_y*ctx2->h_y)) + (x_p[ix + 1][iy + 2] - 2.0*x_p[ix + 2][iy + 2] + x_p[ix + 3][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      PetscScalar r91 = -1.0*x_p[ix + 2][iy + 2];
      y_p[ix + 2][iy + 2] = ((r91 + x_p[ix + 2][iy + 3])/((ctx2->h_y*ctx2->h_y)) + (r91 + x_p[ix + 1][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_p_vec));
  PetscCall(VecRestoreArray(xloc,&x_p_vec));
  PetscCall(DMLocalToGlobalBegin(dm2,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm2,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm2,&xloc));
  PetscCall(DMRestoreLocalVector(dm2,&yloc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction2(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  struct UserCtx2 * ctx2;
  DM dm2 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm2,&ctx2));
  Vec floc;
  DMDALocalInfo info;
  Vec xloc;

  PetscScalar * f_p_vec;
  PetscScalar * x_p_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm2,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm2,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm2,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm2,&floc));
  PetscCall(VecGetArray(floc,&f_p_vec));
  PetscCall(VecGetArray(xloc,&x_p_vec));
  PetscCall(DMDAGetLocalInfo(dm2,&info));
  struct dataobj * _stagger_border_p_vec = ctx2->_stagger_border_p_vec;
  struct dataobj * bc_tmp_p_vec = ctx2->bc_tmp_p_vec;

  PetscInt (* _stagger_border_p)[_stagger_border_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_p_vec->size[1]]) _stagger_border_p_vec->data;
  PetscScalar (* bc_tmp_p)[bc_tmp_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[bc_tmp_p_vec->size[1]]) bc_tmp_p_vec->data;
  PetscScalar (* f_p)[info.gxm] = (PetscScalar (*)[info.gxm]) f_p_vec;
  PetscScalar (* x_p)[info.gxm] = (PetscScalar (*)[info.gxm]) x_p_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd aligned(bc_tmp_p:32)
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      f_p[ix + 2][iy + 2] = -bc_tmp_p[ix + 2][iy + 2] + x_p[ix + 2][iy + 2];
      x_p[ix + 2][iy + 2] = bc_tmp_p[ix + 2][iy + 2];
    }
  }
  for (int n2 = ctx2->n2_m; n2 <= ctx2->n2_M; n2 += 1)
  {
    ctx2->x_ltkn8 = _stagger_border_p[n2][0];
    ctx2->x_rtkn8 = _stagger_border_p[n2][1];
    ctx2->y_ltkn8 = _stagger_border_p[n2][2];
    ctx2->y_rtkn8 = _stagger_border_p[n2][3];

    for (int ix = ctx2->x_m + ctx2->x_ltkn8; ix <= ctx2->x_M - ctx2->x_rtkn8; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx2->y_m + ctx2->y_ltkn8; iy <= ctx2->y_M - ctx2->y_rtkn8; iy += 1)
      {
        f_p[ix + 2][iy + 2] = -ctx2->zero + x_p[ix + 2][iy + 2];
        x_p[ix + 2][iy + 2] = ctx2->zero;
      }
    }
  }

  PetscScalar r92 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r93 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      f_p[ix + 2][iy + 2] = (r92*x_p[ix + 1][iy + 2] + r92*x_p[ix + 3][iy + 2] + r93*x_p[ix + 2][iy + 1] + r93*x_p[ix + 2][iy + 3] + (-2.0)*(r92*x_p[ix + 2][iy + 2] + r93*x_p[ix + 2][iy + 2]))*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      PetscScalar r100 = -1.0*x_p[ix + 2][iy + 2];
      f_p[ix + 2][iy + 2] = ((r100 + x_p[ix + 2][iy + 1])/((ctx2->h_y*ctx2->h_y)) + (r100 + x_p[ix + 3][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r94 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r95 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      f_p[ix + 2][iy + 2] = (r94*x_p[ix + 1][iy + 2] - 2.0*r94*x_p[ix + 2][iy + 2] + r94*x_p[ix + 3][iy + 2] + r95*x_p[ix + 2][iy + 1] - 1.0*r95*x_p[ix + 2][iy + 2])*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r96 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r97 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      f_p[ix + 2][iy + 2] = (r96*x_p[ix + 1][iy + 2] + r97*x_p[ix + 2][iy + 1] + (-1.0)*(r96*x_p[ix + 2][iy + 2] + r97*x_p[ix + 2][iy + 2]))*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      f_p[ix + 2][iy + 2] = ((x_p[ix + 2][iy + 1] - 2.0*x_p[ix + 2][iy + 2] + x_p[ix + 2][iy + 3])/((ctx2->h_y*ctx2->h_y)) + (-1.0*x_p[ix + 2][iy + 2] + x_p[ix + 3][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r98 = 1.0/(ctx2->h_x*ctx2->h_x);
  PetscScalar r99 = 1.0/(ctx2->h_y*ctx2->h_y);

  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      f_p[ix + 2][iy + 2] = (r98*x_p[ix + 1][iy + 2] - 1.0*r98*x_p[ix + 2][iy + 2] + r99*x_p[ix + 2][iy + 1] - 2.0*r99*x_p[ix + 2][iy + 2] + r99*x_p[ix + 2][iy + 3])*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      f_p[ix + 2][iy + 2] = ((-1.0*x_p[ix + 2][iy + 2] + x_p[ix + 2][iy + 3])/((ctx2->h_y*ctx2->h_y)) + (x_p[ix + 1][iy + 2] - 2.0*x_p[ix + 2][iy + 2] + x_p[ix + 3][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      PetscScalar r101 = -1.0*x_p[ix + 2][iy + 2];
      f_p[ix + 2][iy + 2] = ((r101 + x_p[ix + 2][iy + 3])/((ctx2->h_y*ctx2->h_y)) + (r101 + x_p[ix + 1][iy + 2])/((ctx2->h_x*ctx2->h_x)))*ctx2->h_x*ctx2->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_p_vec));
  PetscCall(VecRestoreArray(xloc,&x_p_vec));
  PetscCall(DMLocalToGlobalBegin(dm2,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm2,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm2,&xloc));
  PetscCall(DMRestoreLocalVector(dm2,&floc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS0(DM dm0, Vec B)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  Vec blocal0;
  DMDALocalInfo info;

  PetscScalar * b_u_vec;

  PetscCall(DMGetLocalVector(dm0,&blocal0));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * _stagger_border_u_vec = ctx0->_stagger_border_u_vec;
  struct dataobj * u_vec = ctx0->u_vec;
  struct dataobj * v_vec = ctx0->v_vec;

  PetscInt (* _stagger_border_u)[_stagger_border_u_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_u_vec->size[1]]) _stagger_border_u_vec->data;
  PetscScalar (* b_u)[info.gxm] = (PetscScalar (*)[info.gxm]) b_u_vec;
  PetscScalar (* u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  PetscScalar (* v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      b_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn0 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      b_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int n0 = ctx0->n0_m; n0 <= ctx0->n0_M; n0 += 1)
  {
    ctx0->x_ltkn6 = _stagger_border_u[n0][0];
    ctx0->x_rtkn6 = _stagger_border_u[n0][1];
    ctx0->y_ltkn6 = _stagger_border_u[n0][2];
    ctx0->y_rtkn6 = _stagger_border_u[n0][3];

    for (int ix = ctx0->x_m + ctx0->x_ltkn6; ix <= ctx0->x_M - ctx0->x_rtkn6; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx0->y_m + ctx0->y_ltkn6; iy <= ctx0->y_M - ctx0->y_rtkn6; iy += 1)
      {
        b_u[ix + 2][iy + 2] = 0;
      }
    }
  }

  PetscScalar r44 = 1.0/ctx0->dt;
  PetscScalar r32 = r44;
  PetscScalar r45 = 1.0/ctx0->re;
  PetscScalar r33 = r45;
  PetscScalar r46 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r34 = r46;
  PetscScalar r47 = 1.0/(ctx0->h_y*ctx0->h_y);
  PetscScalar r35 = r47;
  PetscScalar r48 = 1.0/ctx0->h_y;
  PetscScalar r36 = r48;
  PetscScalar r49 = 1.0/ctx0->h_x;
  PetscScalar r37 = r49;
  PetscScalar r38 = r45;
  PetscScalar r39 = r47;
  PetscScalar r40 = r44;
  PetscScalar r41 = r46;
  PetscScalar r42 = r48;
  PetscScalar r43 = r49;

  for (int ix = ctx0->x_m + ctx0->x_ltkn1; ix <= ctx0->x_M - ctx0->x_rtkn1; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx0->y_m + ctx0->y_ltkn1; iy <= ctx0->y_M - ctx0->y_rtkn1; iy += 1)
    {
      b_u[ix + 2][iy + 2] = (r32*u[ctx0->t0][ix + 2][iy + 2] + 5.0e-1*r33*(r34*u[ctx0->t0][ix + 1][iy + 2] + r34*u[ctx0->t0][ix + 3][iy + 2] + r35*u[ctx0->t0][ix + 2][iy + 1] + r35*u[ctx0->t0][ix + 2][iy + 3] - 2.0*(r34*u[ctx0->t0][ix + 2][iy + 2] + r35*u[ctx0->t0][ix + 2][iy + 2])) - 7.5e-1*(r36*(-u[ctx0->t0][ix + 2][iy + 1] + u[ctx0->t0][ix + 2][iy + 3])*(2.5e-1*v[ctx0->t0][ix + 1][iy + 2] + 2.5e-1*v[ctx0->t0][ix + 1][iy + 3] + 2.5e-1*v[ctx0->t0][ix + 2][iy + 2] + 2.5e-1*v[ctx0->t0][ix + 2][iy + 3]) + r37*(-u[ctx0->t0][ix + 1][iy + 2] + u[ctx0->t0][ix + 3][iy + 2])*u[ctx0->t0][ix + 2][iy + 2]) + 2.5e-1*(r36*(-u[ctx0->t1][ix + 2][iy + 1] + u[ctx0->t1][ix + 2][iy + 3])*(2.5e-1*v[ctx0->t1][ix + 1][iy + 2] + 2.5e-1*v[ctx0->t1][ix + 1][iy + 3] + 2.5e-1*v[ctx0->t1][ix + 2][iy + 2] + 2.5e-1*v[ctx0->t1][ix + 2][iy + 3]) + r37*(-u[ctx0->t1][ix + 1][iy + 2] + u[ctx0->t1][ix + 3][iy + 2])*u[ctx0->t1][ix + 2][iy + 2]))*ctx0->h_x*ctx0->h_y;
    }
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn3 - 1; iy += 1)
    {
      PetscScalar r50 = 1.0/ctx0->h_x;
      PetscScalar r51 = 1.0/ctx0->h_y;
      b_u[ix + 2][iy + 2] = (5.0e-1*((-3.0*u[ctx0->t0][ix + 2][iy + 2] + u[ctx0->t0][ix + 2][iy + 3])/((ctx0->h_y*ctx0->h_y)) + (u[ctx0->t0][ix + 1][iy + 2] - 2.0*u[ctx0->t0][ix + 2][iy + 2] + u[ctx0->t0][ix + 3][iy + 2])/((ctx0->h_x*ctx0->h_x)))/ctx0->re - 7.5e-1*(r50*(-u[ctx0->t0][ix + 1][iy + 2] + u[ctx0->t0][ix + 3][iy + 2])*u[ctx0->t0][ix + 2][iy + 2] + r51*(u[ctx0->t0][ix + 2][iy + 2] + u[ctx0->t0][ix + 2][iy + 3])*(2.5e-1*v[ctx0->t0][ix + 1][iy + 2] + 2.5e-1*v[ctx0->t0][ix + 1][iy + 3] + 2.5e-1*v[ctx0->t0][ix + 2][iy + 2] + 2.5e-1*v[ctx0->t0][ix + 2][iy + 3])) + 2.5e-1*(r50*(-u[ctx0->t1][ix + 1][iy + 2] + u[ctx0->t1][ix + 3][iy + 2])*u[ctx0->t1][ix + 2][iy + 2] + r51*(u[ctx0->t1][ix + 2][iy + 2] + u[ctx0->t1][ix + 2][iy + 3])*(2.5e-1*v[ctx0->t1][ix + 1][iy + 2] + 2.5e-1*v[ctx0->t1][ix + 1][iy + 3] + 2.5e-1*v[ctx0->t1][ix + 2][iy + 2] + 2.5e-1*v[ctx0->t1][ix + 2][iy + 3])) + u[ctx0->t0][ix + 2][iy + 2]/ctx0->dt)*ctx0->h_x*ctx0->h_y;
    }
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx0->y_m + ctx0->y_ltkn4; iy <= ctx0->y_M - ctx0->y_rtkn4; iy += 1)
    {
      b_u[ix + 2][iy + 2] = (r38*r39 + r38*(-1.0*(r39*u[ctx0->t0][ix + 2][iy + 2] + r41*u[ctx0->t0][ix + 2][iy + 2]) + 5.0e-1*(r39*(2 - u[ctx0->t0][ix + 2][iy + 2]) + r39*u[ctx0->t0][ix + 2][iy + 1] + r41*u[ctx0->t0][ix + 1][iy + 2] + r41*u[ctx0->t0][ix + 3][iy + 2])) + r40*u[ctx0->t0][ix + 2][iy + 2] + r42*(-1.50*(1.0 - 5.0e-1*(u[ctx0->t0][ix + 2][iy + 1] + u[ctx0->t0][ix + 2][iy + 2]))*(2.5e-1*v[ctx0->t0][ix + 1][iy + 2] + 2.5e-1*v[ctx0->t0][ix + 1][iy + 3] + 2.5e-1*v[ctx0->t0][ix + 2][iy + 2] + 2.5e-1*v[ctx0->t0][ix + 2][iy + 3]) + 5.0e-1*(1.0 - 5.0e-1*(u[ctx0->t1][ix + 2][iy + 1] + u[ctx0->t1][ix + 2][iy + 2]))*(2.5e-1*v[ctx0->t1][ix + 1][iy + 2] + 2.5e-1*v[ctx0->t1][ix + 1][iy + 3] + 2.5e-1*v[ctx0->t1][ix + 2][iy + 2] + 2.5e-1*v[ctx0->t1][ix + 2][iy + 3])) - 7.5e-1*r43*(-u[ctx0->t0][ix + 1][iy + 2] + u[ctx0->t0][ix + 3][iy + 2])*u[ctx0->t0][ix + 2][iy + 2] + 2.5e-1*r43*(-u[ctx0->t1][ix + 1][iy + 2] + u[ctx0->t1][ix + 3][iy + 2])*u[ctx0->t1][ix + 2][iy + 2])*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_u_vec));
  PetscCall(DMRestoreLocalVector(dm0,&blocal0));

  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  DMDALocalInfo info;

  PetscScalar * x_u_vec;

  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * _stagger_border_u_vec = ctx0->_stagger_border_u_vec;

  PetscInt (* _stagger_border_u)[_stagger_border_u_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_u_vec->size[1]]) _stagger_border_u_vec->data;
  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      x_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn0 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn2 - 1; iy += 1)
    {
      x_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int n0 = ctx0->n0_m; n0 <= ctx0->n0_M; n0 += 1)
  {
    ctx0->x_ltkn6 = _stagger_border_u[n0][0];
    ctx0->x_rtkn6 = _stagger_border_u[n0][1];
    ctx0->y_ltkn6 = _stagger_border_u[n0][2];
    ctx0->y_rtkn6 = _stagger_border_u[n0][3];

    for (int ix = ctx0->x_m + ctx0->x_ltkn6; ix <= ctx0->x_M - ctx0->x_rtkn6; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx0->y_m + ctx0->y_ltkn6; iy <= ctx0->y_M - ctx0->y_rtkn6; iy += 1)
      {
        x_u[ix + 2][iy + 2] = ctx0->zero;
      }
    }
  }
  PetscCall(VecRestoreArray(xloc,&x_u_vec));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS1(DM dm1, Vec B)
{
  PetscFunctionBeginUser;

  struct UserCtx1 * ctx1;
  PetscCall(DMGetApplicationContext(dm1,&ctx1));
  Vec blocal1;
  DMDALocalInfo info;

  PetscScalar * b_v_vec;

  PetscCall(DMGetLocalVector(dm1,&blocal1));
  PetscCall(DMGlobalToLocalBegin(dm1,B,INSERT_VALUES,blocal1));
  PetscCall(DMGlobalToLocalEnd(dm1,B,INSERT_VALUES,blocal1));
  PetscCall(VecGetArray(blocal1,&b_v_vec));
  PetscCall(DMDAGetLocalInfo(dm1,&info));
  struct dataobj * _stagger_border_v_vec = ctx1->_stagger_border_v_vec;
  struct dataobj * u_vec = ctx1->u_vec;
  struct dataobj * v_vec = ctx1->v_vec;

  PetscInt (* _stagger_border_v)[_stagger_border_v_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_v_vec->size[1]]) _stagger_border_v_vec->data;
  PetscScalar (* b_v)[info.gxm] = (PetscScalar (*)[info.gxm]) b_v_vec;
  PetscScalar (* u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  PetscScalar (* v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn4 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_M - ctx1->y_rtkn0 + 1; iy <= ctx1->y_M; iy += 1)
    {
      b_v[ix + 2][iy + 2] = 0;
    }
    #pragma omp simd
    for (int iy = ctx1->y_m; iy <= ctx1->y_m + ctx1->y_ltkn3 - 1; iy += 1)
    {
      b_v[ix + 2][iy + 2] = 0;
    }
  }
  for (int n1 = ctx1->n1_m; n1 <= ctx1->n1_M; n1 += 1)
  {
    ctx1->x_ltkn7 = _stagger_border_v[n1][0];
    ctx1->x_rtkn7 = _stagger_border_v[n1][1];
    ctx1->y_ltkn7 = _stagger_border_v[n1][2];
    ctx1->y_rtkn7 = _stagger_border_v[n1][3];

    for (int ix = ctx1->x_m + ctx1->x_ltkn7; ix <= ctx1->x_M - ctx1->x_rtkn7; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx1->y_m + ctx1->y_ltkn7; iy <= ctx1->y_M - ctx1->y_rtkn7; iy += 1)
      {
        b_v[ix + 2][iy + 2] = 0;
      }
    }
  }

  PetscScalar r68 = 1.0/ctx1->dt;
  PetscScalar r69 = 1.0/ctx1->re;
  PetscScalar r70 = 1.0/(ctx1->h_x*ctx1->h_x);
  PetscScalar r71 = 1.0/(ctx1->h_y*ctx1->h_y);
  PetscScalar r72 = 1.0/ctx1->h_x;
  PetscScalar r73 = 1.0/ctx1->h_y;

  for (int ix = ctx1->x_m + ctx1->x_ltkn3; ix <= ctx1->x_M - ctx1->x_rtkn3; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      b_v[ix + 2][iy + 2] = (r68*v[ctx1->t0][ix + 2][iy + 2] + 5.0e-1*r69*(r70*v[ctx1->t0][ix + 1][iy + 2] + r70*v[ctx1->t0][ix + 3][iy + 2] + r71*v[ctx1->t0][ix + 2][iy + 1] + r71*v[ctx1->t0][ix + 2][iy + 3] - 2.0*(r70*v[ctx1->t0][ix + 2][iy + 2] + r71*v[ctx1->t0][ix + 2][iy + 2])) - 7.5e-1*(r72*(-v[ctx1->t0][ix + 1][iy + 2] + v[ctx1->t0][ix + 3][iy + 2])*(2.5e-1*u[ctx1->t0][ix + 2][iy + 1] + 2.5e-1*u[ctx1->t0][ix + 2][iy + 2] + 2.5e-1*u[ctx1->t0][ix + 3][iy + 1] + 2.5e-1*u[ctx1->t0][ix + 3][iy + 2]) + r73*(-v[ctx1->t0][ix + 2][iy + 1] + v[ctx1->t0][ix + 2][iy + 3])*v[ctx1->t0][ix + 2][iy + 2]) + 2.5e-1*(r72*(-v[ctx1->t1][ix + 1][iy + 2] + v[ctx1->t1][ix + 3][iy + 2])*(2.5e-1*u[ctx1->t1][ix + 2][iy + 1] + 2.5e-1*u[ctx1->t1][ix + 2][iy + 2] + 2.5e-1*u[ctx1->t1][ix + 3][iy + 1] + 2.5e-1*u[ctx1->t1][ix + 3][iy + 2]) + r73*(-v[ctx1->t1][ix + 2][iy + 1] + v[ctx1->t1][ix + 2][iy + 3])*v[ctx1->t1][ix + 2][iy + 2]))*ctx1->h_x*ctx1->h_y;
    }
  }
  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      PetscScalar r80 = 1.0/ctx1->h_x;
      PetscScalar r81 = 1.0/ctx1->h_y;
      b_v[ix + 2][iy + 2] = (5.0e-1*((v[ctx1->t0][ix + 2][iy + 1] - 2.0*v[ctx1->t0][ix + 2][iy + 2] + v[ctx1->t0][ix + 2][iy + 3])/((ctx1->h_y*ctx1->h_y)) + (-3.0*v[ctx1->t0][ix + 2][iy + 2] + v[ctx1->t0][ix + 3][iy + 2])/((ctx1->h_x*ctx1->h_x)))/ctx1->re - 7.5e-1*(r80*(v[ctx1->t0][ix + 2][iy + 2] + v[ctx1->t0][ix + 3][iy + 2])*(2.5e-1*u[ctx1->t0][ix + 2][iy + 1] + 2.5e-1*u[ctx1->t0][ix + 2][iy + 2] + 2.5e-1*u[ctx1->t0][ix + 3][iy + 1] + 2.5e-1*u[ctx1->t0][ix + 3][iy + 2]) + r81*(-v[ctx1->t0][ix + 2][iy + 1] + v[ctx1->t0][ix + 2][iy + 3])*v[ctx1->t0][ix + 2][iy + 2]) + 2.5e-1*(r80*(v[ctx1->t1][ix + 2][iy + 2] + v[ctx1->t1][ix + 3][iy + 2])*(2.5e-1*u[ctx1->t1][ix + 2][iy + 1] + 2.5e-1*u[ctx1->t1][ix + 2][iy + 2] + 2.5e-1*u[ctx1->t1][ix + 3][iy + 1] + 2.5e-1*u[ctx1->t1][ix + 3][iy + 2]) + r81*(-v[ctx1->t1][ix + 2][iy + 1] + v[ctx1->t1][ix + 2][iy + 3])*v[ctx1->t1][ix + 2][iy + 2]) + v[ctx1->t0][ix + 2][iy + 2]/ctx1->dt)*ctx1->h_x*ctx1->h_y;
    }
  }

  PetscScalar r74 = 1.0/ctx1->dt;
  PetscScalar r75 = 1.0/ctx1->re;
  PetscScalar r76 = 1.0/(ctx1->h_x*ctx1->h_x);
  PetscScalar r77 = 1.0/(ctx1->h_y*ctx1->h_y);
  PetscScalar r78 = 1.0/ctx1->h_x;
  PetscScalar r79 = 1.0/ctx1->h_y;

  for (int ix = ctx1->x_m + ctx1->x_ltkn5; ix <= ctx1->x_M - ctx1->x_rtkn5; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx1->y_m + ctx1->y_ltkn5; iy <= ctx1->y_M - ctx1->y_rtkn5; iy += 1)
    {
      b_v[ix + 2][iy + 2] = (r74*v[ctx1->t0][ix + 2][iy + 2] + 5.0e-1*r75*(r76*v[ctx1->t0][ix + 1][iy + 2] - 3.0*r76*v[ctx1->t0][ix + 2][iy + 2] + r77*v[ctx1->t0][ix + 2][iy + 1] - 2.0*r77*v[ctx1->t0][ix + 2][iy + 2] + r77*v[ctx1->t0][ix + 2][iy + 3]) + 7.5e-1*(r78*(v[ctx1->t0][ix + 1][iy + 2] + v[ctx1->t0][ix + 2][iy + 2])*(2.5e-1*u[ctx1->t0][ix + 2][iy + 1] + 2.5e-1*u[ctx1->t0][ix + 2][iy + 2] + 2.5e-1*u[ctx1->t0][ix + 3][iy + 1] + 2.5e-1*u[ctx1->t0][ix + 3][iy + 2]) - r79*(-v[ctx1->t0][ix + 2][iy + 1] + v[ctx1->t0][ix + 2][iy + 3])*v[ctx1->t0][ix + 2][iy + 2]) + 2.5e-1*(-r78*(v[ctx1->t1][ix + 1][iy + 2] + v[ctx1->t1][ix + 2][iy + 2])*(2.5e-1*u[ctx1->t1][ix + 2][iy + 1] + 2.5e-1*u[ctx1->t1][ix + 2][iy + 2] + 2.5e-1*u[ctx1->t1][ix + 3][iy + 1] + 2.5e-1*u[ctx1->t1][ix + 3][iy + 2]) + r79*(-v[ctx1->t1][ix + 2][iy + 1] + v[ctx1->t1][ix + 2][iy + 3])*v[ctx1->t1][ix + 2][iy + 2]))*ctx1->h_x*ctx1->h_y;
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm1,blocal1,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm1,blocal1,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal1,&b_v_vec));
  PetscCall(DMRestoreLocalVector(dm1,&blocal1));

  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialGuess1(DM dm1, Vec xloc)
{
  PetscFunctionBeginUser;

  struct UserCtx1 * ctx1;
  PetscCall(DMGetApplicationContext(dm1,&ctx1));
  DMDALocalInfo info;

  PetscScalar * x_v_vec;

  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm1,&info));
  struct dataobj * _stagger_border_v_vec = ctx1->_stagger_border_v_vec;

  PetscInt (* _stagger_border_v)[_stagger_border_v_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_v_vec->size[1]]) _stagger_border_v_vec->data;
  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx1->x_m; ix <= ctx1->x_m + ctx1->x_ltkn4 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx1->y_M - ctx1->y_rtkn0 + 1; iy <= ctx1->y_M; iy += 1)
    {
      x_v[ix + 2][iy + 2] = 0;
    }
    #pragma omp simd
    for (int iy = ctx1->y_m; iy <= ctx1->y_m + ctx1->y_ltkn3 - 1; iy += 1)
    {
      x_v[ix + 2][iy + 2] = 0;
    }
  }
  for (int n1 = ctx1->n1_m; n1 <= ctx1->n1_M; n1 += 1)
  {
    ctx1->x_ltkn7 = _stagger_border_v[n1][0];
    ctx1->x_rtkn7 = _stagger_border_v[n1][1];
    ctx1->y_ltkn7 = _stagger_border_v[n1][2];
    ctx1->y_rtkn7 = _stagger_border_v[n1][3];

    for (int ix = ctx1->x_m + ctx1->x_ltkn7; ix <= ctx1->x_M - ctx1->x_rtkn7; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx1->y_m + ctx1->y_ltkn7; iy <= ctx1->y_M - ctx1->y_rtkn7; iy += 1)
      {
        x_v[ix + 2][iy + 2] = ctx1->zero;
      }
    }
  }
  PetscCall(VecRestoreArray(xloc,&x_v_vec));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS2(DM dm2, Vec B)
{
  PetscFunctionBeginUser;

  struct UserCtx2 * ctx2;
  PetscCall(DMGetApplicationContext(dm2,&ctx2));
  Vec blocal2;
  DMDALocalInfo info;

  PetscScalar * b_p_vec;

  PetscCall(DMGetLocalVector(dm2,&blocal2));
  PetscCall(DMGlobalToLocalBegin(dm2,B,INSERT_VALUES,blocal2));
  PetscCall(DMGlobalToLocalEnd(dm2,B,INSERT_VALUES,blocal2));
  PetscCall(VecGetArray(blocal2,&b_p_vec));
  PetscCall(DMDAGetLocalInfo(dm2,&info));
  struct dataobj * _stagger_border_p_vec = ctx2->_stagger_border_p_vec;
  struct dataobj * u_vec = ctx2->u_vec;
  struct dataobj * v_vec = ctx2->v_vec;

  PetscInt (* _stagger_border_p)[_stagger_border_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_p_vec->size[1]]) _stagger_border_p_vec->data;
  PetscScalar (* b_p)[info.gxm] = (PetscScalar (*)[info.gxm]) b_p_vec;
  PetscScalar (* u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  PetscScalar (* v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (PetscScalar (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      b_p[ix + 2][iy + 2] = 0;
    }
  }
  for (int n2 = ctx2->n2_m; n2 <= ctx2->n2_M; n2 += 1)
  {
    ctx2->x_ltkn8 = _stagger_border_p[n2][0];
    ctx2->x_rtkn8 = _stagger_border_p[n2][1];
    ctx2->y_ltkn8 = _stagger_border_p[n2][2];
    ctx2->y_rtkn8 = _stagger_border_p[n2][3];

    for (int ix = ctx2->x_m + ctx2->x_ltkn8; ix <= ctx2->x_M - ctx2->x_rtkn8; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx2->y_m + ctx2->y_ltkn8; iy <= ctx2->y_M - ctx2->y_rtkn8; iy += 1)
      {
        b_p[ix + 2][iy + 2] = 0;
      }
    }
  }

  PetscScalar r102 = 1.0/ctx2->dt_c;
  PetscScalar r103 = 1.0/ctx2->h_x;
  PetscScalar r104 = 1.0/ctx2->h_y;

  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      b_p[ix + 2][iy + 2] = r102*(-r103*u[ctx2->t2][ix + 2][iy + 2] + r103*u[ctx2->t2][ix + 3][iy + 2] - r104*v[ctx2->t2][ix + 2][iy + 2] + r104*v[ctx2->t2][ix + 2][iy + 3])*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      b_p[ix + 2][iy + 2] = ((-u[ctx2->t2][ix + 2][iy + 2] + u[ctx2->t2][ix + 3][iy + 2])/ctx2->h_x + (-v[ctx2->t2][ix + 2][iy + 2] + v[ctx2->t2][ix + 2][iy + 3])/ctx2->h_y)*ctx2->h_x*ctx2->h_y/ctx2->dt_c;
    }
  }

  PetscScalar r105 = 1.0/ctx2->dt_c;
  PetscScalar r106 = 1.0/ctx2->h_x;
  PetscScalar r107 = 1.0/ctx2->h_y;

  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      b_p[ix + 2][iy + 2] = r105*(-r106*u[ctx2->t2][ix + 2][iy + 2] + r106*u[ctx2->t2][ix + 3][iy + 2] - r107*v[ctx2->t2][ix + 2][iy + 2] + r107*v[ctx2->t2][ix + 2][iy + 3])*ctx2->h_x*ctx2->h_y;
    }
  }

  PetscScalar r108 = 1.0/ctx2->dt_c;
  PetscScalar r109 = 1.0/ctx2->h_x;
  PetscScalar r110 = 1.0/ctx2->h_y;

  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m + ctx2->y_ltkn4; iy <= ctx2->y_M - ctx2->y_rtkn4; iy += 1)
    {
      b_p[ix + 2][iy + 2] = r108*(-r109*u[ctx2->t2][ix + 2][iy + 2] + r109*u[ctx2->t2][ix + 3][iy + 2] - r110*v[ctx2->t2][ix + 2][iy + 2] + r110*v[ctx2->t2][ix + 2][iy + 3])*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      b_p[ix + 2][iy + 2] = ((-u[ctx2->t2][ix + 2][iy + 2] + u[ctx2->t2][ix + 3][iy + 2])/ctx2->h_x + (-v[ctx2->t2][ix + 2][iy + 2] + v[ctx2->t2][ix + 2][iy + 3])/ctx2->h_y)*ctx2->h_x*ctx2->h_y/ctx2->dt_c;
    }
  }

  PetscScalar r111 = 1.0/ctx2->dt_c;
  PetscScalar r112 = 1.0/ctx2->h_x;
  PetscScalar r113 = 1.0/ctx2->h_y;

  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m + ctx2->y_ltkn1; iy <= ctx2->y_M - ctx2->y_rtkn1; iy += 1)
    {
      b_p[ix + 2][iy + 2] = r111*(-r112*u[ctx2->t2][ix + 2][iy + 2] + r112*u[ctx2->t2][ix + 3][iy + 2] - r113*v[ctx2->t2][ix + 2][iy + 2] + r113*v[ctx2->t2][ix + 2][iy + 3])*ctx2->h_x*ctx2->h_y;
    }
  }
  for (int ix = ctx2->x_m + ctx2->x_ltkn3; ix <= ctx2->x_M - ctx2->x_rtkn3; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      b_p[ix + 2][iy + 2] = ((-u[ctx2->t2][ix + 2][iy + 2] + u[ctx2->t2][ix + 3][iy + 2])/ctx2->h_x + (-v[ctx2->t2][ix + 2][iy + 2] + v[ctx2->t2][ix + 2][iy + 3])/ctx2->h_y)*ctx2->h_x*ctx2->h_y/ctx2->dt_c;
    }
  }
  for (int ix = ctx2->x_m + ctx2->x_ltkn5; ix <= ctx2->x_M - ctx2->x_rtkn5; ix += 1)
  {
    #pragma omp simd aligned(u,v:32)
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      b_p[ix + 2][iy + 2] = ((-u[ctx2->t2][ix + 2][iy + 2] + u[ctx2->t2][ix + 3][iy + 2])/ctx2->h_x + (-v[ctx2->t2][ix + 2][iy + 2] + v[ctx2->t2][ix + 2][iy + 3])/ctx2->h_y)*ctx2->h_x*ctx2->h_y/ctx2->dt_c;
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm2,blocal2,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm2,blocal2,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal2,&b_p_vec));
  PetscCall(DMRestoreLocalVector(dm2,&blocal2));

  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialGuess2(DM dm2, Vec xloc)
{
  PetscFunctionBeginUser;

  struct UserCtx2 * ctx2;
  PetscCall(DMGetApplicationContext(dm2,&ctx2));
  DMDALocalInfo info;

  PetscScalar * x_p_vec;

  PetscCall(VecGetArray(xloc,&x_p_vec));
  PetscCall(DMDAGetLocalInfo(dm2,&info));
  struct dataobj * _stagger_border_p_vec = ctx2->_stagger_border_p_vec;
  struct dataobj * bc_tmp_p_vec = ctx2->bc_tmp_p_vec;

  PetscInt (* _stagger_border_p)[_stagger_border_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscInt (*)[_stagger_border_p_vec->size[1]]) _stagger_border_p_vec->data;
  PetscScalar (* bc_tmp_p)[bc_tmp_p_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[bc_tmp_p_vec->size[1]]) bc_tmp_p_vec->data;
  PetscScalar (* x_p)[info.gxm] = (PetscScalar (*)[info.gxm]) x_p_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx2->x_m; ix <= ctx2->x_m + ctx2->x_ltkn2 - 1; ix += 1)
  {
    #pragma omp simd aligned(bc_tmp_p:32)
    for (int iy = ctx2->y_m; iy <= ctx2->y_m + ctx2->y_ltkn3 - 1; iy += 1)
    {
      x_p[ix + 2][iy + 2] = bc_tmp_p[ix + 2][iy + 2];
    }
  }
  for (int n2 = ctx2->n2_m; n2 <= ctx2->n2_M; n2 += 1)
  {
    ctx2->x_ltkn8 = _stagger_border_p[n2][0];
    ctx2->x_rtkn8 = _stagger_border_p[n2][1];
    ctx2->y_ltkn8 = _stagger_border_p[n2][2];
    ctx2->y_rtkn8 = _stagger_border_p[n2][3];

    for (int ix = ctx2->x_m + ctx2->x_ltkn8; ix <= ctx2->x_M - ctx2->x_rtkn8; ix += 1)
    {
      #pragma omp simd
      for (int iy = ctx2->y_m + ctx2->y_ltkn8; iy <= ctx2->y_M - ctx2->y_rtkn8; iy += 1)
      {
        x_p[ix + 2][iy + 2] = ctx2->zero;
      }
    }
  }
  PetscCall(VecRestoreArray(xloc,&x_p_vec));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-utent_solve_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions1()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-vtent_solve_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions2()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-pressure_solve_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * _stagger_border_u_vec, const PetscScalar re, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar zero, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt n0_M, const PetscInt n0_m, const PetscInt x_M, const PetscInt x_ltkn1, const PetscInt x_ltkn2, const PetscInt x_ltkn6, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn1, const PetscInt x_rtkn6, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_ltkn3, const PetscInt y_ltkn4, const PetscInt y_ltkn6, const PetscInt y_m, const PetscInt y_rtkn1, const PetscInt y_rtkn4, const PetscInt y_rtkn6)
{
  PetscFunctionBeginUser;

  ctx0->_stagger_border_u_vec = _stagger_border_u_vec;
  ctx0->dt = dt;
  ctx0->h_x = h_x;
  ctx0->h_y = h_y;
  ctx0->n0_M = n0_M;
  ctx0->n0_m = n0_m;
  ctx0->re = re;
  ctx0->x_M = x_M;
  ctx0->x_ltkn1 = x_ltkn1;
  ctx0->x_ltkn2 = x_ltkn2;
  ctx0->x_ltkn6 = x_ltkn6;
  ctx0->x_m = x_m;
  ctx0->x_rtkn0 = x_rtkn0;
  ctx0->x_rtkn1 = x_rtkn1;
  ctx0->x_rtkn6 = x_rtkn6;
  ctx0->y_M = y_M;
  ctx0->y_ltkn1 = y_ltkn1;
  ctx0->y_ltkn2 = y_ltkn2;
  ctx0->y_ltkn3 = y_ltkn3;
  ctx0->y_ltkn4 = y_ltkn4;
  ctx0->y_ltkn6 = y_ltkn6;
  ctx0->y_m = y_m;
  ctx0->y_rtkn1 = y_rtkn1;
  ctx0->y_rtkn4 = y_rtkn4;
  ctx0->y_rtkn6 = y_rtkn6;
  ctx0->zero = zero;
  ctx0->u_vec = u_vec;
  ctx0->v_vec = v_vec;

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext1(struct UserCtx1 * ctx1, struct dataobj * _stagger_border_v_vec, const PetscScalar re, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar zero, const PetscScalar dt, const PetscScalar h_x, const PetscScalar h_y, const PetscInt n1_M, const PetscInt n1_m, const PetscInt x_M, const PetscInt x_ltkn2, const PetscInt x_ltkn3, const PetscInt x_ltkn4, const PetscInt x_ltkn5, const PetscInt x_ltkn7, const PetscInt x_m, const PetscInt x_rtkn3, const PetscInt x_rtkn5, const PetscInt x_rtkn7, const PetscInt y_M, const PetscInt y_ltkn3, const PetscInt y_ltkn5, const PetscInt y_ltkn7, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn5, const PetscInt y_rtkn7)
{
  PetscFunctionBeginUser;

  ctx1->_stagger_border_v_vec = _stagger_border_v_vec;
  ctx1->dt = dt;
  ctx1->h_x = h_x;
  ctx1->h_y = h_y;
  ctx1->n1_M = n1_M;
  ctx1->n1_m = n1_m;
  ctx1->re = re;
  ctx1->x_M = x_M;
  ctx1->x_ltkn2 = x_ltkn2;
  ctx1->x_ltkn3 = x_ltkn3;
  ctx1->x_ltkn4 = x_ltkn4;
  ctx1->x_ltkn5 = x_ltkn5;
  ctx1->x_ltkn7 = x_ltkn7;
  ctx1->x_m = x_m;
  ctx1->x_rtkn3 = x_rtkn3;
  ctx1->x_rtkn5 = x_rtkn5;
  ctx1->x_rtkn7 = x_rtkn7;
  ctx1->y_M = y_M;
  ctx1->y_ltkn3 = y_ltkn3;
  ctx1->y_ltkn5 = y_ltkn5;
  ctx1->y_ltkn7 = y_ltkn7;
  ctx1->y_m = y_m;
  ctx1->y_rtkn0 = y_rtkn0;
  ctx1->y_rtkn5 = y_rtkn5;
  ctx1->y_rtkn7 = y_rtkn7;
  ctx1->zero = zero;
  ctx1->u_vec = u_vec;
  ctx1->v_vec = v_vec;

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext2(struct UserCtx2 * ctx2, struct dataobj * _stagger_border_p_vec, struct dataobj * bc_tmp_p_vec, const PetscScalar dt_c, struct dataobj * p_vec, struct dataobj * u_vec, struct dataobj * v_vec, const PetscScalar zero, const PetscScalar h_x, const PetscScalar h_y, const PetscInt n2_M, const PetscInt n2_m, const PetscInt x_M, const PetscInt x_ltkn2, const PetscInt x_ltkn3, const PetscInt x_ltkn5, const PetscInt x_ltkn8, const PetscInt x_m, const PetscInt x_rtkn3, const PetscInt x_rtkn5, const PetscInt x_rtkn8, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn3, const PetscInt y_ltkn4, const PetscInt y_ltkn8, const PetscInt y_m, const PetscInt y_rtkn1, const PetscInt y_rtkn4, const PetscInt y_rtkn8)
{
  PetscFunctionBeginUser;

  ctx2->_stagger_border_p_vec = _stagger_border_p_vec;
  ctx2->h_x = h_x;
  ctx2->h_y = h_y;
  ctx2->n2_M = n2_M;
  ctx2->n2_m = n2_m;
  ctx2->x_M = x_M;
  ctx2->x_ltkn2 = x_ltkn2;
  ctx2->x_ltkn3 = x_ltkn3;
  ctx2->x_ltkn5 = x_ltkn5;
  ctx2->x_ltkn8 = x_ltkn8;
  ctx2->x_m = x_m;
  ctx2->x_rtkn3 = x_rtkn3;
  ctx2->x_rtkn5 = x_rtkn5;
  ctx2->x_rtkn8 = x_rtkn8;
  ctx2->y_M = y_M;
  ctx2->y_ltkn1 = y_ltkn1;
  ctx2->y_ltkn3 = y_ltkn3;
  ctx2->y_ltkn4 = y_ltkn4;
  ctx2->y_ltkn8 = y_ltkn8;
  ctx2->y_m = y_m;
  ctx2->y_rtkn1 = y_rtkn1;
  ctx2->y_rtkn4 = y_rtkn4;
  ctx2->y_rtkn8 = y_rtkn8;
  ctx2->bc_tmp_p_vec = bc_tmp_p_vec;
  ctx2->zero = zero;
  ctx2->dt_c = dt_c;
  ctx2->u_vec = u_vec;
  ctx2->v_vec = v_vec;
  ctx2->p_vec = p_vec;

  PetscFunctionReturn(0);
}
/* Backdoor edit at Mon Apr 20 22:32:44 2026*/ 
/* Backdoor edit at Mon Apr 20 22:37:01 2026*/ 



