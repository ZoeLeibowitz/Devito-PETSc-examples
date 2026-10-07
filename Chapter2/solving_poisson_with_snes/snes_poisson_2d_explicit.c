static char help[] = "Example adapted from Ed Bueler - petsc4pdes.\n"
                     "Solves structured-grid Poisson problem in 2D.\n"
                     "Equation is\n"
                     "    - p_xx - p_yy = b,\n"
                     "subject to Dirichlet boundary conditions.\n"
                     "Defaults to a SNESType of KSPONLY and a KSPType of CG.\n\n";


#include <petscsnes.h>
#include <petscdmda.h>


typedef struct {
  // domain dimensions
  PetscScalar Lx, Ly;
  // right-hand-side b(x,y)
  PetscScalar (*b_rhs)(PetscScalar x, PetscScalar y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscScalar (*g_bdry)(PetscScalar x, PetscScalar y, void *ctx);
} PoissonCtx;


static PetscScalar p_exact_2D(PetscScalar x, PetscScalar y, void *ctx)
{
  return -x * PetscExpReal(y);
}

// right-hand-side function  b(x,y) = - laplacian p
static PetscScalar b_rhs_2D(PetscScalar x, PetscScalar y, void *ctx)
{
  return x * PetscExpReal(y); // note  b = - (p_xx + p_yy) = - p
}

PetscErrorCode FormJacobian(SNES snes, Vec p, Mat J, Mat Jpre, void *dummy);
PetscErrorCode FormFunctionGlobal(SNES snes, Vec p, Vec F, void *dummy);
PetscErrorCode FormExact(DMDALocalInfo *info, Vec p, PoissonCtx *user);
PetscErrorCode InitialState(DM da, Vec p, PoissonCtx *user);

int main(int argc, char **argv)
{
  DM            da;
  KSP           ksp;
  SNES          snes;
  Vec           pglobal, p_exact, p_exact_local;
  DMDALocalInfo info;
  PetscScalar   errinf, normconst2h, err2h;
  char          gridstr[99];
  PoissonCtx    user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  user.Lx     = 1.0;
  user.Ly     = 1.0;
  user.g_bdry = &p_exact_2D;

  user.b_rhs  = &b_rhs_2D;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 17, 17, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, user.Lx, 0.0, user.Ly, 0.0, 1.0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(SNESSetDM(snes, da));
  PetscCall(SNESSetFunction(snes, NULL, FormFunctionGlobal, (void *)(da)));
  // Assemble the Jacobian explicitly (AIJ matrix created from the DMDA)
  PetscCall(SNESSetJacobian(snes, NULL, NULL, FormJacobian, (void *)(da)));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(DMCreateGlobalVector(da, &pglobal));
  PetscCall(InitialState(da, pglobal, &user)); // zero interior, p = g on boundary
  PetscCall(SNESSolve(snes, NULL, pglobal));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateLocalVector(da, &p_exact_local));
  PetscCall(DMCreateGlobalVector(da, &p_exact));
  PetscCall(FormExact(&info, p_exact_local, &user));
  PetscCall(DMLocalToGlobal(da, p_exact_local, INSERT_VALUES, p_exact));
  PetscCall(VecAXPY(pglobal, -1.0, p_exact)); // p <- p + (-1.0) pexact
  PetscCall(VecDestroy(&p_exact));            // no longer needed
  PetscCall(VecDestroy(&p_exact_local));
  PetscCall(VecNorm(pglobal, NORM_INFINITY, &errinf));
  PetscCall(VecNorm(pglobal, NORM_2, &err2h));
  normconst2h = PetscSqrtReal((PetscScalar)(info.mx - 1) * (info.my - 1));
  snprintf(gridstr, 99, "%d x %d point 2D", info.mx, info.my);
  err2h /= normconst2h; // like continuous L2
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "problem on %s grid:\n"
                        "  error |p-pexact|_inf = %.3e, |p-pexact|_h = %.3e\n",
                        gridstr, errinf, err2h));
  PetscCall(VecDestroy(&pglobal));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());

  return 0;
}

PetscErrorCode FormFunctionGlobal(SNES snes, Vec p, Vec F, void *dummy)
{
  DM            dm = (DM)(dummy);
  PetscInt      i, j;
  DMDALocalInfo info;
  PetscScalar   scdiag, hx, hy, darea, x, y, pe, pw, pn, ps;
  PetscScalar **aF, **ap, xymin[2], xymax[2];
  PoissonCtx   *user;
  Vec           p_local, F_local;

  PetscFunctionBeginUser;
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
  darea  = hx * hy;
  scdiag = 2.0 * (hy / hx + hx / hy); // diagonal scaling

  for (j = info.ys; j < info.ys + info.ym; j++) {
    y = j * hy;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      x = i * hx;
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        aF[j][i] = ap[j][i] - user->g_bdry(x, y, user);
        aF[j][i] *= scdiag;
      } else {
        pe       = (i + 1 == info.mx - 1) ? user->g_bdry(x + hx, y, user) : ap[j][i + 1];
        pw       = (i - 1 == 0) ? user->g_bdry(x - hx, y, user) : ap[j][i - 1];
        pn       = (j + 1 == info.my - 1) ? user->g_bdry(x, y + hy, user) : ap[j + 1][i];
        ps       = (j - 1 == 0) ? user->g_bdry(x, y - hy, user) : ap[j - 1][i];
        aF[j][i] = scdiag * ap[j][i] - (hy / hx) * (pw + pe) - (hx / hy) * (ps + pn) - darea * user->b_rhs(x, y, user);
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

PetscErrorCode FormJacobian(SNES snes, Vec p, Mat J, Mat Jpre, void *dummy)
{
  DM            dm = (DM)(dummy);
  DMDALocalInfo info;
  Vec           p_local;
  PetscScalar **ap;
  PetscReal     xymin[2], xymax[2], hx, hy, scdiag, v[5];
  PetscInt      i, j, ncols;
  MatStencil    col[5], row;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(dm, &info));

  PetscCall(DMGetLocalVector(dm, &p_local));
  PetscCall(DMGlobalToLocalBegin(dm, p, INSERT_VALUES, p_local));
  PetscCall(DMGlobalToLocalEnd(dm, p, INSERT_VALUES, p_local));
  PetscCall(DMDAVecGetArray(dm, p_local, &ap));

  PetscCall(DMGetBoundingBox(dm, xymin, xymax));
  hx     = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy     = (xymax[1] - xymin[1]) / (info.my - 1);
  scdiag = 2.0 * (hy / hx + hx / hy); // diagonal scaling

  for (j = info.ys; j < info.ys + info.ym; j++) {
    row.j    = j;
    col[0].j = j;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      row.i    = i;
      col[0].i = i;
      ncols    = 1;
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        v[0] = scdiag; // boundary rows: F = scdiag * (p - g)
      } else {
        v[0] = scdiag;
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

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(DMDAVecRestoreArray(dm, p_local, &ap));
  PetscCall(DMRestoreLocalVector(dm, &p_local));
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

PetscErrorCode FormExact(DMDALocalInfo *info, Vec p, PoissonCtx *user)
{
  PetscInt    i, j;
  PetscScalar xymin[2], xymax[2], hx, hy, x, y, **ap;

  PetscFunctionBeginUser;
  PetscCall(DMGetBoundingBox(info->da, xymin, xymax));
  hx = (xymax[0] - xymin[0]) / (info->mx - 1);
  hy = (xymax[1] - xymin[1]) / (info->my - 1);
  PetscCall(DMDAVecGetArray(info->da, p, &ap));
  for (j = info->gys; j < info->gys + info->gym; j++) {
    y = xymin[1] + j * hy;
    for (i = info->gxs; i < info->gxs + info->gxm; i++) {
      x        = xymin[0] + i * hx;
      ap[j][i] = user->g_bdry(x, y, user);
    }
  }
  PetscCall(DMDAVecRestoreArray(info->da, p, &ap));
  PetscFunctionReturn(PETSC_SUCCESS);
}
