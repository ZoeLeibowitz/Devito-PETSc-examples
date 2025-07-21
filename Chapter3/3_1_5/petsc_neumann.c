/*
Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -u.laplace = f,  0 < x,y < 1,

with forcing function

   f = 8pi**2 * cos(2pix) * cos(2piy)

with pure Neumman boundary conditions

*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrixBackwardFirstOrder(KSP, Mat, Mat, void *);
extern PetscErrorCode ComputeRHS(KSP, Vec, void *);
extern PetscErrorCode FormExact(KSP, Vec, void*);

typedef enum {
  DIRICHLET,
  NEUMANN
} BCType;

typedef struct {
    PetscScalar nu;
  BCType    bcType;
} UserContext;

int main(int argc, char **argv)
{
  KSP         ksp;
  DM          da;
  UserContext user;
  const char *bcTypes[2] = {"dirichlet", "neumann"};
  PetscInt    bc;
  Vec         b, x, exact;
  PetscBool   testsolver = PETSC_FALSE;
  PetscScalar      errinf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 33, 33, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0));
  PetscCall(DMDASetFieldName(da, 0, "Pressure"));

  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMqq");
  user.nu = 0.1;
  PetscCall(PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user.nu, &user.nu, NULL));
  bc = (PetscInt)DIRICHLET;
  PetscCall(PetscOptionsEList("-bc_type", "Type of boundary condition", "ex29.c", bcTypes, 2, bcTypes[0], &bc, NULL));
  user.bcType = (BCType)bc;
  PetscCall(PetscOptionsBool("-testsolver", "Run solver multiple times, useful for performance studies of solver", "ex29.c", testsolver, &testsolver, NULL));
  PetscOptionsEnd();

  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &user));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrixBackwardFirstOrder, &user));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(KSPSolve(ksp, NULL, x));

  // Compute exact solution
  PetscCall(DMCreateGlobalVector(da, &exact));

  PetscCall(FormExact(ksp, exact, &user));

  // compute infinity norm
//   PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(VecAXPY(x,-1.0,exact));   // u <- u + (-1.0) uexact
  PetscCall(VecNorm(x,NORM_INFINITY,&errinf));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error |u-uexact|_inf = %.22e\n", errinf));

  // view matrix
  Mat Amat, Pmat;
  PetscCall(KSPGetOperators(ksp, &Amat, &Pmat));

  //view exact
  // PetscCall(VecView(exact, PETSC_VIEWER_STDOUT_WORLD));
  // PetscCall(MatView(Amat, PETSC_VIEWER_STDOUT_WORLD));


// PetscScalar tol = 1e-10;  // or another small tolerance
//   PetscBool isSymmetric;
//   PetscCall(MatIsSymmetric(Amat, tol, &isSymmetric));
//   if (isSymmetric) {
//       PetscPrintf(PETSC_COMM_WORLD, "Matrix is symmetric within tolerance %.1e\n", tol);
//   } else {
//       PetscPrintf(PETSC_COMM_WORLD, "Matrix is NOT symmetric within tolerance %.1e\n", tol);
//   }

  PetscCall(DMDestroy(&da));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  UserContext  *user = (UserContext *)ctx;
  PetscInt      i, j, mx, my, xm, ym, xs, ys;
  PetscScalar   Hx, Hy;
  PetscScalar **array;
  DM            da;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  Hx = 1.0 / (PetscScalar)(mx - 1);
  Hy = 1.0 / (PetscScalar)(my - 1);
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
  PetscCall(DMDAVecGetArray(da, b, &array));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++){
        PetscScalar x = Hx*j;
        PetscScalar y = Hy*i;
        array[j][i] = Hx*Hy*8.0*PETSC_PI*PETSC_PI*PetscCosReal(2.0*PETSC_PI*x)*PetscCosReal(2.0*PETSC_PI*y);
    }
  }
  PetscCall(DMDAVecRestoreArray(da, b, &array));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  // PetscCall(VecShift(b, 0.01));

  /* force right-hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
    PetscCall(MatNullSpaceRemove(nullspace, b));
    PetscCall(MatNullSpaceDestroy(&nullspace));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// first order accurate, backward differencing
PetscErrorCode ComputeMatrixBackwardFirstOrder(KSP ksp, Mat J, Mat jac, void *ctx)
{
  UserContext *user = (UserContext *)ctx;
  PetscInt     i, j, mx, my, xm, ym, xs, ys;
  PetscScalar  v[5];
  PetscScalar    Hx, Hy, HydHx, HxdHy;
  MatStencil   row, col[5];
  DM           da;
  PetscBool    check_matis = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  Hx    = 1.0 / (PetscScalar)(mx - 1);
  Hy    = 1.0 / (PetscScalar)(my - 1);
  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i;
      row.j = j;
      if (i == 0 || j == 0 || i == mx - 1 || j == my - 1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0 * (HxdHy + HydHx);
          PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
        } else if (user->bcType == NEUMANN) {
          PetscInt numx = 0, numy = 0, num = 0;
          if (j != 0) {
            v[num]     = -HxdHy;
            col[num].i = i;
            col[num].j = j - 1;
            numy++;
            num++;
          }
          if (i != 0) {
            v[num]     = -HydHx;
            col[num].i = i - 1;
            col[num].j = j;
            numx++;
            num++;
          }
          if (i != mx - 1) {
            v[num]     = -HydHx;
            col[num].i = i + 1;
            col[num].j = j;
            numx++;
            num++;
          }
          if (j != my - 1) {
            v[num]     = -HxdHy;
            col[num].i = i;
            col[num].j = j + 1;
            numy++;
            num++;
          }
          v[num]     = numx * HydHx + numy * HxdHy;
          col[num].i = i;
          col[num].j = j;
          num++;
          PetscCall(MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES));
        }
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
        PetscCall(MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(jac, NULL, "-view_mat"));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_matis", &check_matis, NULL));
  if (check_matis) {
    void (*f)(void);
    Mat       J2;
    MatType   jtype;
    PetscScalar nrm;

    PetscCall(MatGetType(jac, &jtype));
    PetscCall(MatConvert(jac, MATIS, MAT_INITIAL_MATRIX, &J2));
    PetscCall(MatViewFromOptions(J2, NULL, "-view_conv"));
    PetscCall(MatConvert(J2, jtype, MAT_INPLACE_MATRIX, &J2));
    PetscCall(MatGetOperation(jac, MATOP_VIEW, &f));
    PetscCall(MatSetOperation(J2, MATOP_VIEW, f));
    PetscCall(MatSetDM(J2, da));
    PetscCall(MatViewFromOptions(J2, NULL, "-view_conv_assembled"));
    PetscCall(MatAXPY(J2, -1., jac, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(J2, NORM_FROBENIUS, &nrm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error MATIS %g\n", (double)nrm));
    PetscCall(MatViewFromOptions(J2, NULL, "-view_conv_err"));
    PetscCall(MatDestroy(&J2));
  }
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
    PetscCall(MatSetNullSpace(J, nullspace));
    PetscCall(MatNullSpaceDestroy(&nullspace));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode FormExact(KSP ksp, Vec b, void *ctx)
{
  UserContext  *user = (UserContext *)ctx;
  PetscInt      i, j, mx, my, xm, ym, xs, ys;
  PetscScalar   Hx, Hy;
  PetscScalar **array;
  DM            da;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  Hx = 1.0 / (PetscScalar)(mx - 1);
  Hy = 1.0 / (PetscScalar)(my - 1);
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
  PetscCall(DMDAVecGetArray(da, b, &array));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++){
        PetscScalar x = Hx*i;
        PetscScalar y = Hy*j;
        // array[j][i] = PetscCosReal(2.0*PETSC_PI*(PetscReal)i * Hx)*PetscCosReal(2.0*PETSC_PI*(PetscReal)i * Hy);
        array[j][i] = PetscCosReal(2.0*PETSC_PI*x)*PetscCosReal(2.0*PETSC_PI*y);
    }
  }
  PetscCall(DMDAVecRestoreArray(da, b, &array));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  PetscFunctionReturn(PETSC_SUCCESS);
}



/*TEST

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -da_refine 8 -ksp_rtol 1.e-3

   test:
      suffix: 2
      args: -bc_type neumann -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -da_refine 8 -mg_coarse_pc_factor_shift_type nonzero
      requires: !single

   test:
      suffix: telescope
      nsize: 4
      args: -ksp_monitor_short -da_grid_x 257 -da_grid_y 257 -pc_type mg -pc_mg_galerkin pmat -pc_mg_levels 4 -ksp_type richardson -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_ignore_kspcomputeoperators -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type chebyshev -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_coarse_pc_telescope_reduction_factor 4

   test:
      suffix: 3
      args: -ksp_view -da_refine 2 -pc_type mg -pc_mg_distinct_smoothup -mg_levels_up_pc_type jacobi

   test:
      suffix: 4
      args: -ksp_view -da_refine 2 -pc_type mg -pc_mg_distinct_smoothup -mg_levels_up_ksp_max_it 3 -mg_levels_ksp_max_it 4

   testset:
     suffix: aniso
     args: -da_grid_x 10 -da_grid_y 2 -da_refine 2 -pc_type mg -ksp_monitor_short -mg_levels_ksp_max_it 6 -mg_levels_pc_type jacobi
     test:
       suffix: first
       args: -mg_levels_ksp_chebyshev_kind first
     test:
       suffix: fourth
       args: -mg_levels_ksp_chebyshev_kind fourth
     test:
       suffix: opt_fourth
       args: -mg_levels_ksp_chebyshev_kind opt_fourth

   test:
      suffix: 5
      nsize: 2
      requires: hypre !complex
      args: -pc_type mg -da_refine 2 -ksp_monitor -matptap_via hypre -pc_mg_galerkin both

   test:
      suffix: 6
      args: -pc_type svd -pc_svd_monitor ::all

TEST*/
