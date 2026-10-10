"""Hessiano nuclear y frecuencias armonicas NOF por diferencias finitas centrales de gradientes analiticos.

    H[i,:] = [g(x + h e_i) - g(x - h e_i)] / (2h)        (Eh/bohr^2)

Cada geometria desplazada se converge en dos etapas:
  1) compute_energy (orb_method/occ_method de p) arrancando de la solucion de referencia, sin
     multi-arranque ni chequeo de estabilidad de ocupaciones, para no cambiar de solucion NOF;
  2) pulido (p.hess_polish): rotaciones orbitales alternadas con la optimizacion de ocupaciones.
     "lbfgs" (defecto): L-BFGS con p.hess_polish_maxiter iteraciones por ciclo, costo acotado;
     "trust-krylov": Hessiano orbital completo por diferencias (bueno con bases pequenas, inviable
     con ~100 funciones); "hessp": trust-krylov con productos Hessiano-vector; "none": sin pulido.
El gradiente nuclear hereda un error de primer orden de los orbitales no convergidos (~1e-4 en moleculas
organicas pulidas, ~1e-3 Eh/bohr en cianuros metalicos de capa abierta), mientras que la energia lo
hereda de segundo orden. Por eso, ademas del Hessiano por gradientes, las frecuencias se refinan con
energias a lo largo de cada modo normal (p.hess_refine): 6 puntos por modo con continuacion y ajuste
E(s) = c0 + g s + omega_k^2 s^2 / 2 (s en coordenada normal ponderada por masa).

Uso:  compute_energy(mol, p, hessian=True)   o   compute_hessian(mol, p, C, n)
Parametros (en p): hess_step, hess_polish, hess_polish_maxiter, hess_max_polish, hess_refine,
hess_refine_step, hess_linear, hess_verbose.
Salidas: <title>_hessian.npy, <title>_freq.json, <title>_modes.molden
"""
import contextlib
import copy
import io
import json
import os
from time import time

import numpy as np
import psi4
from scipy.optimize import minimize
from psi4.driver.qcdb import vib

import pynof

CM2KCAL = 2.859144e-3  # cm-1 -> kcal/mol (por cuanto: ZPE = 0.5 * sum(w) * CM2KCAL)


def _displaced(mol, p, X, title):
    """Molecula y parametros en la geometria X (bohr, natom x 3), con la misma base y opciones que p."""
    mol_d = mol.clone()
    mol_d.fix_com(True)
    mol_d.fix_orientation(True)
    mol_d.set_geometry(psi4.core.Matrix.from_array(np.asarray(X, dtype=float)))
    mol_d.update_geometry()
    p_d = copy.copy(p)
    p_d.wfn = psi4.core.Wavefunction.build(mol_d, psi4.core.get_global_option("basis"))
    p_d.title = title
    return mol_d, p_d


def _orbopt_trust_krylov(gamma, C, H, I, b_mnl, p, maxiter=100):
    """Rotaciones orbitales con trust-krylov y productos Hessiano-vector por diferencias centrales del
    gradiente orbital (2 gradientes por producto). orbopt_rotations usa hess="2-point", que construye
    el Hessiano completo (p.nvar gradientes por iteracion) y es inviable con bases de ~100 funciones."""
    n, _ = pynof.ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = pynof.PNOFi_selector(n, p)

    def fg(y):
        return pynof.calcorbeg(y, n, cj12, ck12, C, H, I, b_mnl, p)

    def hessp(y, v):
        nv = np.linalg.norm(v)
        if nv == 0.0:
            return np.zeros_like(v)
        eps = 1e-4 / nv
        return (fg(y + eps * v)[1] - fg(y - eps * v)[1]) / (2 * eps)

    res = minimize(fg, np.zeros(p.nvar), jac=True, hessp=hessp, method="trust-krylov",
                   options={"maxiter": maxiter})
    return pynof.rotate_orbital(res.x, C, p)


def _orbopt_lbfgs(gamma, C, H, I, b_mnl, p, maxiter=200):
    """Rotaciones orbitales con L-BFGS (un gradiente orbital por iteracion)."""
    n, _ = pynof.ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = pynof.PNOFi_selector(n, p)
    res = minimize(pynof.calcorbeg, np.zeros(p.nvar), args=(n, cj12, ck12, C, H, I, b_mnl, p), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 30, "gtol": 1e-7, "ftol": 1e-14})
    return pynof.rotate_orbital(res.x, C, p)


def polish_solution(mol, p, C, n, max_polish=None):
    """Rotaciones orbitales (segun p.hess_polish) + ocupaciones hasta que la energia se estanca.

    Devuelve E total, gradiente nuclear (3N), C, n y un dict con gorb, lag y ciclos."""
    S, T, V, H, I, b_mnl, Dipole = pynof.compute_integrals(p.wfn, mol, p)
    E_nuc = mol.nuclear_repulsion_energy()
    method = getattr(p, "hess_polish", "lbfgs")
    if max_polish is None:
        max_polish = getattr(p, "hess_max_polish", None)
    if max_polish is None:
        max_polish = 20 if method in ("trust-krylov", "hessp") else 3
    q = copy.copy(p)
    q.orb_method = "Rotations"
    q.orbital_optimizer = "trust-krylov"
    q.maxloop = 100
    gamma = pynof.n_to_gammas_softmax(n, q.no1, q.ndoc, q.ndns, q.ncwo)
    E_old, stall, k = None, 0, -1
    _, _, _, gamma, n, cj12, ck12 = pynof.occoptr(gamma, C, H, I, b_mnl, q)
    E, elag, _, lag = pynof.ENERGY1r(C, n, H, I, b_mnl, cj12, ck12, q)
    for k in range(max_polish if method != "none" else 0):
        if method == "trust-krylov":
            _, C, _, _ = pynof.orbopt_rotations(gamma, C, H, I, b_mnl, q)
        elif method == "hessp":
            C = _orbopt_trust_krylov(gamma, C, H, I, b_mnl, q, maxiter=q.maxloop)
        else:
            C = _orbopt_lbfgs(gamma, C, H, I, b_mnl, q, maxiter=getattr(p, "hess_polish_maxiter", 200))
        _, _, _, gamma, n, cj12, ck12 = pynof.occoptr(gamma, C, H, I, b_mnl, q)
        E, elag, _, lag = pynof.ENERGY1r(C, n, H, I, b_mnl, cj12, ck12, q)
        if E_old is not None and abs(E - E_old) < 1e-9:
            stall += 1
            if stall >= 2:
                break
        else:
            stall = 0
        E_old = E
    grad = pynof.compute_geom_gradients(q.wfn, mol, n, C, cj12, ck12, elag, q).flatten()
    gorb = np.linalg.norm(pynof.calcorbg(np.zeros(q.nvar), n, cj12, ck12, C, H, I, b_mnl, q))
    return E + E_nuc, grad, C, n, dict(gorb=float(gorb), lag=float(lag), cycles=k + 1)


def _solve_point(mol, p, X, C0, n0, title, max_polish, verbose):
    mol_d, p_d = _displaced(mol, p, X, title)
    p_d.occ_multistart = False
    p_d.occ_stability = False
    out = io.StringIO()
    with contextlib.redirect_stdout(out) if not verbose else contextlib.nullcontext():
        r = pynof.compute_energy(mol_d, p_d, C=C0, n=n0, printmode=False)
        E, g, C, n, info = polish_solution(mol_d, p_d, r[1], r[2], max_polish)
    info["E_first"] = float(r[0])
    return E, g, C, n, info


def _linear_axis(X, tol=1e-4):
    """Eje unitario si los nucleos son colineales (desviacion relativa < tol), None si no."""
    if len(X) < 2:
        return None
    d = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(d)
    return vt[0] if len(s) < 2 or s[1] < tol * max(s[0], 1.0) else None


def straighten(X, axis):
    """Proyecta los nucleos sobre la recta (centroide, axis): geometria exactamente lineal."""
    c = X.mean(axis=0)
    return c + np.outer((X - c) @ axis, axis)


def symmetrize_linear(Hc, axis):
    """Impone simetria cilindrica a un Hessiano cartesiano de una molecula lineal: en el marco con z
    sobre el eje, promedia los bloques xx/yy y anula los acoplamientos entre ejes."""
    z = axis / np.linalg.norm(axis)
    x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    R = np.array([x, np.cross(z, x), z])                      # filas: ejes del marco local
    nat = Hc.shape[0] // 3
    Rb = np.kron(np.eye(nat), R)
    Hl = (Rb @ Hc @ Rb.T).reshape(nat, 3, nat, 3)
    xy = 0.5 * (Hl[:, 0, :, 0] + Hl[:, 1, :, 1])
    Hs = np.zeros_like(Hl)
    Hs[:, 0, :, 0] = xy
    Hs[:, 1, :, 1] = xy
    Hs[:, 2, :, 2] = Hl[:, 2, :, 2]
    Hs = Hs.reshape(3 * nat, 3 * nat)
    Hs = Rb.T @ Hs @ Rb
    return 0.5 * (Hs + Hs.T)


def harmonic_frequencies(mol, p, Hc, title=None, printmode=True):
    """Analisis armonico de Psi4 (proyecta traslaciones y rotaciones). Devuelve vibinfo y frecuencias."""
    geom = np.asarray(mol.geometry())
    mass = np.array([mol.mass(i) for i in range(mol.natom())])
    symbols = [mol.symbol(i) for i in range(mol.natom())]
    vibinfo, text = vib.harmonic_analysis(Hc, geom, mass, p.wfn.basisset(), mol.irrep_labels())
    freqs = vibinfo["omega"].data
    if printmode:
        print(text)
        print(vib.print_vibs(vibinfo, symbols, shortlong=True))
    if title:
        with open(title + "_modes.molden", "w") as fh:
            fh.write(vib.print_molden_vibs(vibinfo, symbols, geom))
    return vibinfo, freqs


def compute_hessian(mol, p, C, n, printmode=True):
    """Hessiano NOF por diferencias finitas de gradientes, en la geometria de mol y desde la solucion (C, n).

    Devuelve un dict con E0, grad, hessian (Eh/bohr^2, 3N x 3N), frequencies (cm-1, complejas para
    modos imaginarios), zpe (kcal/mol) y controles de calidad; ademas escribe
    <title>_hessian.npy, <title>_freq.json y <title>_modes.molden."""
    t0 = time()
    h = getattr(p, "hess_step", 0.02)
    max_polish = getattr(p, "hess_max_polish", None)
    verbose = getattr(p, "hess_verbose", False)
    X0 = np.asarray(mol.geometry())
    nat = X0.shape[0]
    nc = 3 * nat

    wdir = p.title + "_hess"                       # archivos de los puntos desplazados (_C.npy, .fchk, ...)
    os.makedirs(wdir, exist_ok=True)

    # Molecula casi lineal (p. ej. 179.8 grados tras una optimizacion): se alinea exactamente; si no, el
    # analisis armonico la trata como no lineal y una de las dos flexiones se proyecta como rotacion.
    linear = getattr(p, "hess_linear", None)
    axis = _linear_axis(X0, getattr(p, "hess_linear_tol", 5e-3)) if linear is not False else None
    if axis is not None:
        Xs = straighten(X0, axis)
        if np.abs(Xs - X0).max() > 1e-8:
            print("Near-linear geometry straightened (max shift {:.2e} bohr).".format(np.abs(Xs - X0).max()))
            X0 = Xs
            mol, p = _displaced(mol, p, X0, p.title)
    E0, g0, C0, n0, info0 = _solve_point(mol, p, X0, C, n, os.path.join(wdir, "ref"), max_polish, verbose)
    print("\nNOF Hessian (central differences of analytic gradients, h = {:.3f} bohr)".format(h))
    print("==================")
    print("Reference: E = {:.10f}  |grad| = {:.2e}  |grad orb| = {:.1e}".format(E0, np.linalg.norm(g0), info0["gorb"]))
    if np.abs(g0).max() > 1e-3:
        print("WARNING: max|grad| = {:.1e} Eh/bohr; the geometry is not a stationary point,".format(np.abs(g0).max()))
        print("         harmonic frequencies are only meaningful at an optimized geometry.")

    Hc = np.zeros((nc, nc))
    log = []
    for i in range(nc):
        gpm = []
        for s in (+1, -1):
            Xd = X0.copy().flatten()
            Xd[i] += s * h
            tag = os.path.join(wdir, "d{}{}".format(i, "p" if s > 0 else "m"))
            E, g, _, _, info = _solve_point(mol, p, Xd.reshape(-1, 3), C0, n0, tag, max_polish, verbose)
            gpm.append(g)
            log.append(dict(coord=i, sign=s, E=float(E), dE=float(E - E0), **info))
            if printmode:
                print("  coord {:3d} {}: E-E0 = {:+.3e}  cycles = {:2d}  |grad orb| = {:.1e}  t = {:.0f} s".format(
                    i, "+" if s > 0 else "-", E - E0, info["cycles"], info["gorb"], time() - t0), flush=True)
        Hc[i] = (gpm[0] - gpm[1]) / (2 * h)

    # Consistencia de estado: un desplazamiento claramente por debajo de la referencia (mas de lo que
    # permite el gradiente) indica que la optimizacion salto a otra solucion NOF.
    jumps = [d for d in log if d["dE"] < -(abs(g0[d["coord"]]) * h + 2e-5)]
    if jumps:
        print("WARNING: {} displacements lower the energy beyond the gradient: possible change of NOF solution.".format(len(jumps)))
        for d in jumps:
            print("   coord {} sign {:+d}: dE = {:+.2e}".format(d["coord"], d["sign"], d["dE"]))
    Hc_raw = Hc.copy()
    asym = float(np.abs(Hc - Hc.T).max())
    Hc = 0.5 * (Hc + Hc.T)
    trans = float(np.abs(Hc.reshape(nc, nat, 3).sum(axis=1)).max())

    if linear is None:
        linear = axis is not None
    if linear and axis is not None:
        Hc = symmetrize_linear(Hc, axis)
        print("Linear molecule: cylindrical symmetry imposed on the Hessian.")
    np.save(p.title + "_hessian.npy", Hc)

    vibinfo, freqs = harmonic_frequencies(mol, p, Hc, title=p.title, printmode=printmode)
    real = [float(np.real(w)) for w in freqs if abs(w) > 1e-6 and abs(np.imag(w)) < 1e-6]
    zpe = 0.5 * sum(w for w in real if w > 0) * CM2KCAL
    print("Quality: max|H-H^T| = {:.2e} Eh/bohr^2 ; translational sum rule = {:.2e} ; state jumps = {}".format(asym, trans, len(jumps)))
    print("ZPE = {:.3f} kcal/mol    Hessian time: {:.0f} s".format(zpe, time() - t0))

    # Diagonal del Hessiano desde energias (error de 2o orden en los orbitales) vs desde gradientes
    Ed = {(d["coord"], d["sign"]): d["E"] for d in log}
    diagE = np.array([(Ed[(i, 1)] + Ed[(i, -1)] - 2 * E0) / h**2 for i in range(nc)])
    diagG = np.diag(Hc_raw)
    print("Diagonal check (Eh/bohr^2):  coord   gradients    energies")
    for i in range(nc):
        print("                             {:3d}   {:10.5f}  {:10.5f}".format(i, diagG[i], diagE[i]))

    # Refinamiento energetico a lo largo de cada modo normal (2 energias por modo)
    refined = None
    if getattr(p, "hess_refine", True):
        sref = getattr(p, "hess_refine_step", None) or 0.04
        mass = np.array([mol.mass(i) for i in range(nat)])
        minv = np.repeat(1.0 / np.sqrt(mass), 3)
        qL = vibinfo["q"].data
        TRV = vibinfo["TRV"].data
        uconv = np.sqrt(psi4.constants.na * psi4.constants.hartree2J * 1.0e19) / (2 * np.pi * psi4.constants.c * psi4.constants.bohr2angstroms)
        refined = []
        print("\nEnergy refinement along normal modes (adaptive step, max Cartesian displacement <= {:.3f} bohr)".format(sref))
        print("   mode   omega(grad)   omega(energy)    E(+)-E0      E(-)-E0     max dx (bohr)")
        for k in range(qL.shape[1]):
            if TRV[k] != "V":
                continue
            if refined and linear and abs(freqs[k] - freqs[refined[-1]["mode"]]) < 1e-3:
                refined.append(dict(refined[-1], mode=k))   # segunda componente de la flexion degenerada
                print("   {:4d}   {:10.1f}    {:10.1f}     (degenerate with mode {})".format(k, refined[-1]["omega_grad"], refined[-1]["omega_energy"], k - 1))
                continue
            dx = minv * qL[:, k]                       # bohr por (a0 u^1/2) de coordenada normal
            # paso tal que E(+-s) - E0 ~ hess_refine_de (2e-4 Eh): grande en las flexiones (ruido relativo
            # pequeno) y corto en las tensiones rigidas (anarmonicidad pequena), acotado a [0.01, sref] bohr
            lam0 = max(abs(np.real(freqs[k]) ** 2 - np.imag(freqs[k]) ** 2) / uconv**2, 1e-12)
            step = np.sqrt(2 * getattr(p, "hess_refine_de", 2e-4) / lam0)
            step = min(max(step * np.abs(dx).max(), 0.01), sref) / np.abs(dx).max()
            # 6 puntos (+-1/3, +-2/3, +-1 del paso) con continuacion: cada punto arranca de la solucion del
            # punto anterior del mismo lado, para no caer en soluciones atascadas; ajuste E = c0 + g s + k s^2/2
            ss, ee = [0.0], [E0]
            for sgn in (+1, -1):
                Cc, nc_ = C0, n0
                for frac in (1 / 3, 2 / 3, 1.0):
                    si = sgn * frac * step
                    Xd = X0.flatten() + si * dx
                    tag = os.path.join(wdir, "q{}_{:+.3f}".format(k, si))
                    Ei, _, Cc, nc_, _ = _solve_point(mol, p, Xd.reshape(-1, 3), Cc, nc_, tag, max_polish, verbose)
                    ss.append(si); ee.append(Ei)
            ss, ee = np.array(ss), np.array(ee) - E0
            A = np.vstack([np.ones_like(ss), ss, 0.5 * ss**2]).T
            cfit, *_ = np.linalg.lstsq(A, ee, rcond=None)
            rms = float(np.sqrt(np.mean((A @ cfit - ee) ** 2)))
            lam = cfit[2]
            w = float(np.sign(lam) * np.sqrt(abs(lam)) * uconv)   # negativo = imaginaria
            Epm = [ee[3], ee[6]]
            refined.append(dict(mode=k, omega_grad=float(np.real(freqs[k]) if abs(np.imag(freqs[k])) < 1e-6 else -np.imag(freqs[k])),
                                omega_energy=w, dEp=float(Epm[0]), dEm=float(Epm[1]), step=float(step),
                                fit_rms=rms, fit_grad=float(cfit[1]), s=ss.tolist(), dE=ee.tolist()))
            print("   {:4d}   {:10.1f}    {:10.1f}     {:+.3e}   {:+.3e}   {:.3f}   rms {:.1e}".format(
                k, refined[-1]["omega_grad"], w, Epm[0], Epm[1], step * np.abs(dx).max(), rms), flush=True)
        zpe_ref = 0.5 * sum(r["omega_energy"] for r in refined if r["omega_energy"] > 0) * CM2KCAL
        print("ZPE (energy-refined) = {:.3f} kcal/mol".format(zpe_ref))

    res = dict(E0=float(E0), grad=g0.tolist(), step_bohr=h, ipnof=p.ipnof,
               polish=getattr(p, "hess_polish", "lbfgs"), diag_grad=diagG.tolist(), diag_energy=diagE.tolist(),
               refined=refined, zpe_kcal_refined=(float(zpe_ref) if refined else None),
               basis=psi4.core.get_global_option("basis"),
               frequencies=[(float(np.real(w)), float(np.imag(w))) for w in freqs],
               zpe_kcal=float(zpe), max_asym=asym, trans_sum=trans, linear=bool(linear and axis is not None),
               state_jumps=len(jumps), displacements=log, wall_s=time() - t0)
    with open(p.title + "_freq.json", "w") as fh:
        json.dump(res, fh, indent=1)
    res["hessian"] = Hc
    res["vibinfo"] = vibinfo
    return res
