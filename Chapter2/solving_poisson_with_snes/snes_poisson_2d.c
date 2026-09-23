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

PetscErrorCode JacMult(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunctionGlobal(SNES snes, Vec p, Vec F, void *dummy);
PetscErrorCode FormExact(DMDALocalInfo *info, Vec p, PoissonCtx *user);

int main(int argc, char **argv)
{
  Mat           J;
  DM            da;
  KSP           ksp;
  PC            pc;
  SNES          snes;
  Vec           pglobal, p_exact, p_exact_local;
  DMDALocalInfo info;
  PetscScalar   errinf, normconst2h, err2h;
  char          gridstr[99];
  PetscInt      N = 17;
  PoissonCtx    user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  user.Lx     = 1.0;
  user.Ly     = 1.0;
  user.g_bdry = &p_exact_2D;

  user.b_rhs  = &b_rhs_2D;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, N, N, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, user.Lx, 0.0, user.Ly, 0.0, 1.0));
  PetscCall(DMSetMatType(da, MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(SNESSetDM(snes, da));
  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, NULL));
  // Set the matrix-free matmult action for J
  PetscCall(MatShellSetOperation(J, MATOP_MULT, (void (*)(void))JacMult));
  PetscCall(SNESSetFunction(snes, NULL, FormFunctionGlobal, (void *)(da)));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(MatSetDM(J, da));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(DMCreateGlobalVector(da, &pglobal));
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
  PetscCall(MatDestroy(&J));
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
      } else {
        pe       = (i + 1 == info.mx - 1) ? user->g_bdry(x + hx, y, user) : ap[j][i + 1];
        pw       = (i - 1 == 0) ? user->g_bdry(x - hx, y, user) : ap[j][i - 1];
        pn       = (j + 1 == info.my - 1) ? user->g_bdry(x, y + hy, user) : ap[j + 1][i];
        ps       = (j - 1 == 0) ? user->g_bdry(x, y - hy, user) : ap[j - 1][i];
        aF[j][i] = scdiag * ap[j][i] - (pw + pe) - (ps + pn) - darea * user->b_rhs(x, y, user);
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

PetscErrorCode JacMult(Mat J, Vec X, Vec Y)
{
  DM            dm;
  DMDALocalInfo info;
  Vec           xloc, yloc;
  PetscScalar   xymin[2], xymax[2], hx, hy, scdiag;
  PetscInt      i, j, xs, ys, xm, ym;
  PetscScalar   pe, pw, pn, ps;
  PetscScalar **x_p;
  PetscScalar **y_p;
  PetscFunctionBeginUser;
  PetscCall(VecSet(Y, 0.0));
  PetscCall(MatGetDM(J, &dm));
  PetscCall(DMDAGetLocalInfo(dm, &info));

  PetscCall(DMGetLocalVector(dm, &xloc));
  PetscCall(DMGetLocalVector(dm, &yloc));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, xloc));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, xloc));
  PetscCall(VecSet(yloc, 0.0));
  PetscCall(DMDAVecGetArray(dm, yloc, &y_p));
  PetscCall(DMDAVecGetArray(dm, xloc, &x_p));
  PetscCall(DMGetBoundingBox(dm, xymin, xymax));
  hx     = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy     = (xymax[1] - xymin[1]) / (info.my - 1);
  scdiag = 2.0 * (hy / hx + hx / hy); // diagonal scaling

  DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);

  for (j = info.ys; j < info.ys + info.ym; j++) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        y_p[j][i] = x_p[j][i];
      } else {
        pe        = (i + 1 == info.mx - 1) ? 0.0 : x_p[j][i + 1];
        pw        = (i - 1 == 0) ? 0.0 : x_p[j][i - 1];
        pn        = (j + 1 == info.my - 1) ? 0.0 : x_p[j + 1][i];
        ps        = (j - 1 == 0) ? 0.0 : x_p[j - 1][i];
        y_p[j][i] = scdiag * x_p[j][i] - (pw + pe) - (ps + pn);
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(dm, yloc, &y_p));
  PetscCall(DMDAVecRestoreArray(dm, xloc, &x_p));

  PetscCall(DMLocalToGlobalBegin(dm, yloc, ADD_VALUES, Y));
  PetscCall(DMLocalToGlobalEnd(dm, yloc, ADD_VALUES, Y));

  PetscCall(DMRestoreLocalVector(dm, &xloc));
  PetscCall(DMRestoreLocalVector(dm, &yloc));
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
