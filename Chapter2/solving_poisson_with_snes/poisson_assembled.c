static char help[] = "Example adapted from Ed Bueler - petsc4pdes.\n"
                     "Solves structured-grid Poisson problem in 2D - assembled version.\n"
                     "Equation is\n"
                     "    - p_xx - p_yy = f,\n"
                     "subject to Dirichlet boundary conditions.\n"
                     "Defaults to a SNESType of KSPONLY and a KSPType of CG.\n\n";


#include <petscsnes.h>
#include <petscdmda.h>


typedef struct {
  // domain dimensions
  PetscScalar Lx, Ly;
  // right-hand-side f(x,y)
  PetscScalar (*f_rhs)(PetscScalar x, PetscScalar y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscScalar (*g_bdry)(PetscScalar x, PetscScalar y, void *ctx);
} PoissonCtx;


static PetscScalar p_exact_2D(PetscScalar x, PetscScalar y, void *ctx)
{
  return -x * PetscExpReal(y);
}

// right-hand-side function  f(x,y) = - laplacian p
static PetscScalar f_rhs_2D(PetscScalar x, PetscScalar y, void *ctx)
{
  return x * PetscExpReal(y); // note  f = - (p_xx + p_yy) = - p
}

PetscErrorCode FormJacobian(SNES snes, Vec p, Mat J, Mat Jpre, void *ctx);
PetscErrorCode FormFunction(SNES snes, Vec p, Vec F, void *ctx);
PetscErrorCode FormExact(DM da, Vec p, PoissonCtx *user);
PetscErrorCode InitialState(DM da, Vec p, PoissonCtx *user);

int main(int argc, char **argv)
{
  DM            da;
  KSP           ksp;
  SNES          snes;
  Vec           pglobal, p_exact;
  DMDALocalInfo info;
  PetscReal     errmax;
  PoissonCtx    user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // Problem data, collected in the user context
  user.Lx     = 1.0;
  user.Ly     = 1.0;
  user.f_rhs  = &f_rhs_2D;
  user.g_bdry = &p_exact_2D;

  // 17 x 17 grid, one dof per point, parallel layout chosen by PETSc, star stencil of width one
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 17, 17, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  // Coordinates, from which the callbacks compute the spacing
  PetscCall(DMDASetUniformCoordinates(da, 0.0, user.Lx, 0.0, user.Ly, 0.0, 1.0));

  // Create the nonlinear solver on the grid
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, da));
  // Set the residual callback
  PetscCall(SNESSetFunction(snes, NULL, FormFunction, NULL));
  // Jacobian routine assembles J into the MATAIJ matrix that SNES creates from the DMDA
  PetscCall(SNESSetJacobian(snes, NULL, NULL, FormJacobian, NULL));

  // Linear problem, so a single Newton step (KSPONLY), solved with CG since the Jacobian is SPD
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(SNESSetFromOptions(snes));

  // Initial guess, g on the boundary and zero elsewhere
  PetscCall(DMCreateGlobalVector(da, &pglobal));
  PetscCall(InitialState(da, pglobal, &user));
  // Solve F(p) = 0, overwriting pglobal with the solution
  PetscCall(SNESSolve(snes, NULL, pglobal));

  // Maximum error against the exact solution
  PetscCall(VecDuplicate(pglobal, &p_exact));
  PetscCall(FormExact(da, p_exact, &user));
  PetscCall(VecAXPY(pglobal, -1.0, p_exact)); // pglobal <- pglobal - p_exact
  PetscCall(VecNorm(pglobal, NORM_INFINITY, &errmax));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d x %d grid, maximum error %.3e\n", info.mx, info.my, (double)errmax));
  PetscCall(VecDestroy(&p_exact));
  PetscCall(VecDestroy(&pglobal));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());

  return 0;
}

PetscErrorCode FormFunction(SNES snes, Vec p, Vec F, void *ctx)
{
  DM            dm;
  PetscInt      i, j;
  DMDALocalInfo info;
  PetscScalar   sigma, hx, hy, x, y, pe, pw, pn, ps;
  PetscScalar **aF, **ap, xymin[2], xymax[2];
  PoissonCtx   *user;
  Vec           p_local, F_local;

  PetscFunctionBeginUser;
  // Get the DMDA from the solver
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(VecSet(F, 0.0));

  PetscCall(DMGetLocalVector(dm, &p_local));
  PetscCall(DMGetLocalVector(dm, &F_local));
  PetscCall(VecSet(F_local, 0.0));
  PetscCall(DMGlobalToLocalBegin(dm, p, INSERT_VALUES, p_local));
  PetscCall(DMGlobalToLocalEnd(dm, p, INSERT_VALUES, p_local));

  PetscCall(DMDAVecGetArray(dm, p_local, &ap));
  PetscCall(DMDAVecGetArray(dm, F_local, &aF));

  PetscCall(DMGetBoundingBox(dm, xymin, xymax));
  hx     = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy     = (xymax[1] - xymin[1]) / (info.my - 1);
  sigma  = 2.0 * (hy / hx + hx / hy); // diagonal scaling

  for (j = info.ys; j < info.ys + info.ym; j++) {
    y = xymin[1] + j * hy;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      x = xymin[0] + i * hx;
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        aF[j][i] = ap[j][i] - user->g_bdry(x, y, user);
        aF[j][i] *= sigma;
      } else {
        pe       = (i + 1 == info.mx - 1) ? user->g_bdry(x + hx, y, user) : ap[j][i + 1];
        pw       = (i - 1 == 0) ? user->g_bdry(x - hx, y, user) : ap[j][i - 1];
        pn       = (j + 1 == info.my - 1) ? user->g_bdry(x, y + hy, user) : ap[j + 1][i];
        ps       = (j - 1 == 0) ? user->g_bdry(x, y - hy, user) : ap[j - 1][i];
        aF[j][i] = sigma * ap[j][i] - (hy / hx) * (pw + pe) - (hx / hy) * (ps + pn) - hx * hy * user->f_rhs(x, y, user);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(dm, p_local, &ap));
  PetscCall(DMDAVecRestoreArray(dm, F_local, &aF));

  PetscCall(DMLocalToGlobalBegin(dm, F_local, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(dm, F_local, ADD_VALUES, F));

  PetscCall(DMRestoreLocalVector(dm, &p_local));
  PetscCall(DMRestoreLocalVector(dm, &F_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec p, Mat J, Mat Jpre, void *ctx)
{
  DM            dm;
  DMDALocalInfo info;
  PetscReal     xymin[2], xymax[2], hx, hy, sigma, v[5];
  PetscInt      i, j, ncols;
  MatStencil    col[5], row;

  PetscFunctionBeginUser;
  // Get the DMDA from the solver
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMDAGetLocalInfo(dm, &info));

  PetscCall(DMGetBoundingBox(dm, xymin, xymax));
  hx     = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy     = (xymax[1] - xymin[1]) / (info.my - 1);
  sigma  = 2.0 * (hy / hx + hx / hy); // diagonal scaling

  for (j = info.ys; j < info.ys + info.ym; j++) {
    row.j    = j;
    col[0].j = j;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      row.i    = i;
      col[0].i = i;
      ncols    = 1;
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        v[0] = sigma; // boundary rows: F = sigma * (p - g)
      } else {
        v[0] = sigma;
        if (i - 1 > 0) {
          col[ncols].j = j;     col[ncols].i = i - 1; v[ncols++] = -hy / hx;
        }
        if (i + 1 < info.mx - 1) {
          col[ncols].j = j;     col[ncols].i = i + 1; v[ncols++] = -hy / hx;
        }
        if (j - 1 > 0) {
          col[ncols].j = j - 1; col[ncols].i = i;     v[ncols++] = -hx / hy;
        }
        if (j + 1 < info.my - 1) {
          col[ncols].j = j + 1; col[ncols].i = i;     v[ncols++] = -hx / hy;
        }
      }
      PetscCall(MatSetValuesStencil(Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
    }
  }

  // Assemble the matrix once all entries are inserted
  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InitialState(DM da, Vec p, PoissonCtx *user)
{
  DMDALocalInfo info;
  PetscInt      i, j;
  PetscScalar   xymin[2], xymax[2], hx, hy, x, y, **ap;

  PetscFunctionBeginUser;
  PetscCall(VecSet(p, 0.0));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMGetBoundingBox(da, xymin, xymax));
  hx = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy = (xymax[1] - xymin[1]) / (info.my - 1);
  PetscCall(DMDAVecGetArray(da, p, &ap));
  for (j = info.ys; j < info.ys + info.ym; j++) {
    y = xymin[1] + j * hy;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        x        = xymin[0] + i * hx;
        ap[j][i] = user->g_bdry(x, y, user);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, p, &ap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormExact(DM da, Vec p, PoissonCtx *user)
{
  DMDALocalInfo info;
  PetscInt      i, j;
  PetscScalar   xymin[2], xymax[2], hx, hy, x, y, **ap;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMGetBoundingBox(da, xymin, xymax));
  hx = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy = (xymax[1] - xymin[1]) / (info.my - 1);
  // The boundary function g is the exact solution, evaluated at every locally owned point
  PetscCall(DMDAVecGetArray(da, p, &ap));
  for (j = info.ys; j < info.ys + info.ym; j++) {
    y = xymin[1] + j * hy;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      x        = xymin[0] + i * hx;
      ap[j][i] = user->g_bdry(x, y, user);
    }
  }
  PetscCall(DMDAVecRestoreArray(da, p, &ap));
  PetscFunctionReturn(PETSC_SUCCESS);
}
