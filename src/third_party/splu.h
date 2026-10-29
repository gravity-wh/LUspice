/* splu.h - faithful templated port of CSparse cs_lu/cs_spsolve/cs_dfs/
 * cs_reach (Tim Davis, SuiteSparse 5.10.1), for float (LU_FLOAT) or double.
 * P * A(:,q) = L * U.
 *   L: CSC, unit diagonal stored FIRST in each column, rows pre-permuted.
 *   U: CSC, diagonal stored LAST in each column.
 *   pinv: pinv[orig_row] = permuted_position (of the LU rows); the row
 *         permutation P maps permuted position k <- original row i with
 *         pinv[i] = k.
 */
#ifndef SPLU_H
#define SPLU_H
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef LU_FLOAT
typedef float  lu_real;
#define LU_ABS  fabsf
#else
typedef double lu_real;
#define LU_ABS  fabs
#endif

#define LU_FLIP(i) (-(i)-2)
#define LU_UNFLIP(i) (((i) < 0) ? LU_FLIP(i) : (i))
#define LU_MARKED(w,j) ((w)[j] < 0)
#define LU_MARK(w,j) { (w)[j] = LU_FLIP((w)[j]); }

typedef struct {
    int n;
    int *Lp, *Li; lu_real *Lx;
    int *Up, *Ui; lu_real *Ux;
    long Lcap, Ucap, lnnz, unnz;
} lu_csc;   /* generic CSC matrix (used for L, U, A) */

typedef struct {
    int n;
    int *pinv;   /* pinv[orig_row] = position (row perm) */
    lu_csc L, U;
    int singular_col;
} lu_factors;

static lu_csc *lu_csc_alloc(int n, long cap)
{
    lu_csc *M = calloc(1, sizeof(lu_csc));
    M->n = n; M->Lcap = M->Ucap = 0;
    M->Lp = malloc((n+1)*sizeof(int));
    M->Li = malloc(cap*sizeof(int));
    M->Lx = malloc(cap*sizeof(lu_real));
    M->Up = malloc((n+1)*sizeof(int));
    M->Ui = malloc(cap*sizeof(int));
    M->Ux = malloc(cap*sizeof(lu_real));
    M->Lcap = M->Ucap = cap;
    M->lnnz = M->unnz = 0;
    return M;
}
static void lu_csc_growL(lu_csc *M, long need)
{
    if (need <= M->Lcap) return;
    long nc = M->Lcap*2 + 16;
    if (nc < need) nc = need;
    M->Li = realloc(M->Li, nc*sizeof(int));
    M->Lx = realloc(M->Lx, nc*sizeof(lu_real));
    M->Lcap = nc;
}
static void lu_csc_growU(lu_csc *M, long need)
{
    if (need <= M->Ucap) return;
    long nc = M->Ucap*2 + 16;
    if (nc < need) nc = need;
    M->Ui = realloc(M->Ui, nc*sizeof(int));
    M->Ux = realloc(M->Ux, nc*sizeof(lu_real));
    M->Ucap = nc;
}

/* depth-first search of the graph of G, starting at node j (cs_dfs port) */
static int lu_dfs(int j, lu_csc *G, int top, int *xi, int *pstack,
                  const int *pinv)
{
    int i, p, p2, done, jnew, head = 0;
    int *Gp = G->Lp; const int *Gi = G->Li;
    xi[0] = j;
    while (head >= 0) {
        j = xi[head];
        jnew = pinv ? pinv[j] : j;
        if (!LU_MARKED(Gp, j)) {
            LU_MARK(Gp, j);
            pstack[head] = (jnew < 0) ? 0 : LU_UNFLIP(Gp[jnew]);
        }
        done = 1;
        p2 = (jnew < 0) ? 0 : LU_UNFLIP(Gp[jnew+1]);
        for (p = pstack[head]; p < p2; p++) {
            i = Gi[p];
            if (LU_MARKED(Gp, i)) continue;
            pstack[head] = p;
            xi[++head] = i;
            done = 0;
            break;
        }
        if (done) {
            head--;
            xi[--top] = j;
        }
    }
    return top;
}

/* xi[top..n-1] = reachable nodes (cs_reach port) */
static int lu_reach(lu_csc *G, const int *Bp, const int *Bi, int k,
                    int *xi, const int *pinv)
{
    int p, n, top;
    int *Gp = G->Lp;
    n = G->n;
    top = n;
    for (p = Bp[k]; p < Bp[k+1]; p++) {
        if (!LU_MARKED(Gp, Bi[p])) {
            top = lu_dfs(Bi[p], G, top, xi, xi+n, pinv);
        }
    }
    for (p = top; p < n; p++) LU_MARK(Gp, xi[p]);   /* restore G */
    return top;
}

/* solve G x = b(:,k) (G lower triangular, lo=1) (cs_spsolve port) */
static int lu_spsolve(lu_csc *G, const int *Bp, const int *Bi,
                      const lu_real *Bx, int k, int *xi, lu_real *x,
                      const int *pinv)
{
    int j, J, p, q, px, top, n;
    const int *Gp = G->Lp, *Gi = G->Li;
    const lu_real *Gx = G->Lx;
    n = G->n;
    top = lu_reach(G, Bp, Bi, k, xi, pinv);
    for (p = top; p < n; p++) x[xi[p]] = 0;
    for (p = Bp[k]; p < Bp[k+1]; p++) x[Bi[p]] = Bx[p];
    for (px = top; px < n; px++) {
        j = xi[px];
        J = pinv ? pinv[j] : j;
        if (J < 0) continue;
        x[j] /= Gx[Gp[J]];                  /* L(j,j) is first entry */
        for (q = Gp[J]+1; q < Gp[J+1]; q++)
            x[Gi[q]] -= Gx[q] * x[j];
    }
    return top;
}

/* factor A (CSC) with column order q (NULL = natural): P*A(:,q) = L*U.
 * Returns -1 ok, else column index of exact zero pivot. tol: partial
 * pivoting threshold (tol=1: pure partial pivoting). */
static int lu_factor(int n, const int *Ap, const int *Ai, const lu_real *Ax,
                     const int *q, lu_factors *F, lu_real tol, int *xi,
                     lu_real *x)
{
    int k, col, p, i, ipiv, top;
    long lnz = 0, unz = 0;
    lu_real a, t, pivot;
    int *pinv = F->pinv;
    lu_csc *L = &F->L, *U = &F->U;
    int *Lp = L->Lp, *Up = U->Up;

    for (i = 0; i < n; i++) x[i] = 0;
    for (i = 0; i < n; i++) pinv[i] = -1;
    for (k = 0; k <= n; k++) { Lp[k] = 0; Up[k] = 0; }
    lnz = unz = 0;
    for (k = 0; k < n; k++) {
        Lp[k] = (int)lnz;
        Up[k] = (int)unz;
        lu_csc_growL(L, lnz + n + 16);
        lu_csc_growU(U, unz + n + 16);
        col = q ? q[k] : k;
        top = lu_spsolve(L, Ap, Ai, Ax, col, xi, x, pinv);
        /* find pivot */
        ipiv = -1; a = -1;
        for (p = top; p < n; p++) {
            i = xi[p];
            if (pinv[i] < 0) {
                if ((t = LU_ABS(x[i])) > a) { a = t; ipiv = i; }
            } else {
                U->Ui[unz] = pinv[i];
                U->Ux[unz++] = x[i];
            }
        }
        if (ipiv == -1 || a <= 0) { F->singular_col = k; return k; }
        if (pinv[col] < 0 && LU_ABS(x[col]) >= a * tol) ipiv = col;
        pivot = x[ipiv];
        U->Ui[unz] = k; U->Ux[unz++] = pivot;
        pinv[ipiv] = k;
        L->Li[lnz] = ipiv; L->Lx[lnz++] = 1;
        for (p = top; p < n; p++) {
            i = xi[p];
            if (pinv[i] < 0) {
                L->Li[lnz] = i;
                L->Lx[lnz++] = x[i] / pivot;
            }
            x[i] = 0;
        }
    }
    Lp[n] = (int)lnz;
    Up[n] = (int)unz;
    /* remap L row indices: Li[p] = pinv[Li[p]] */
    for (p = 0; p < lnz; p++) L->Li[p] = pinv[L->Li[p]];
    L->lnnz = lnz; U->unnz = unz;
    F->singular_col = -1;
    return -1;
}

/* solve A x = b (A factored as P A(:,q) = L U, with Ruiz-free usage:
 * caller handles scaling). q is the column order used at factor time. */
static void lu_solve_perm(int n, const lu_factors *F, const int *q,
                          const lu_real *b, lu_real *x)
{
    /* P b: (P b)[k] = b[orig_row_k] where orig_row_k = {i: pinv[i]=k} */
    int k, i, p;
    lu_real acc;
    const lu_csc *L = &F->L, *U = &F->U;
    lu_real *pb = malloc(n * sizeof(lu_real));
    lu_real *y = malloc(n * sizeof(lu_real));
    /* build inverse of pinv */
    int *rowat = malloc(n * sizeof(int));
    for (k = 0; k < n; k++) rowat[k] = -1;
    for (i = 0; i < n; i++) if (F->pinv[i] >= 0) rowat[F->pinv[i]] = i;
    for (k = 0; k < n; k++) pb[k] = rowat[k] >= 0 ? b[rowat[k]] : 0.0;
    /* forward: L y = pb (column sweep; L unit diag first in each column) */
    for (k = 0; k < n; k++) {
        y[k] = pb[k] / L->Lx[L->Lp[k]];
        for (p = L->Lp[k]+1; p < L->Lp[k+1]; p++)
            pb[L->Li[p]] -= L->Lx[p] * y[k];
    }
    /* backward: U z = y (column sweep; U diag LAST in each column) */
    for (k = n-1; k >= 0; k--) {
        y[k] /= U->Ux[U->Up[k+1]-1];
        for (p = U->Up[k]; p < U->Up[k+1]-1; p++)
            y[U->Ui[p]] -= U->Ux[p] * y[k];
    }
    /* x = Q z: x[q[j]] = z[j] */
    for (k = 0; k < n; k++) x[q ? q[k] : k] = y[k];
    free(pb); free(y); free(rowat);
}
#endif
