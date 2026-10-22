import numpy as np
from scipy.linalg import eigh
from time import time
import pynof
import psi4

HARTREE2KCAL = 627.5094740631


# companero debil "casi apagado": exp(-5) ~ 7e-3  ->  n_weak ~ 0.01 (capa casi
# cerrada) PERO con gradiente no nulo, para que ADAM pueda relajar hasta la
# ocupacion correlacionada real.  Un valor muy negativo (p.ej. -25) da
# gradiente ~0 y deja el optimizador atrapado en n exactamente = 2/0.
_GAMMA_NEG = -5.0


def _gamma_all_closed_softmax(p):
    """Vector gamma (Softmax) de capa CERRADA: todos los companeros debiles
    apagados  ->  n[i] = 1  (ocupacion real 2) para cada orbital fuerte."""
    return np.full(p.ndoc * p.ncwo, _GAMMA_NEG, dtype=float)


def _gamma_frontier_softmax(gamma_base, p, npairs=1, mode="open"):
    """Copia de 'gamma_base' (Softmax) con los 'npairs' pares fuertes de mas
    alta energia (HOMO, HOMO-1, ...) forzados a capa ABIERTA o CERRADA; el
    resto de orbitales conserva sus valores de 'gamma_base'.

    ocupacion_softmax lee los companeros del orbital fuerte  i = ndoc-1-k
    en los indices  ig = k + j*ndoc  (j = 0..ncwo-1).
        mode="open"  -> companero dominante (j=0) a 0  ->  n[i] ~ 0.5
                        (par pi/pi* dirradical);  resto apagado.
        mode="close" -> todos apagados            ->  n[i] ~ 1  (enlace pi).
    """
    g = np.array(gamma_base, dtype=float).copy()
    ndoc, ncwo = p.ndoc, p.ncwo
    for k in range(min(int(npairs), ndoc)):
        for j in range(ncwo):
            ig = k + j * ndoc
            if ig < g.size:
                g[ig] = 0.0 if (mode == "open" and j == 0) else _GAMMA_NEG
    return g


def _occ_stability_probe(run_scf, E_ref, C_ref, gamma_ref, fmiug0_ref, p, E_nuc,
                         npairs=2, tol=1e-5):
    """Solucion 2: chequeo de estabilidad de la ocupacion.

    Toma la solucion convergida y la re-optimiza partiendo de dos
    perturbaciones del par frontera (mas ABIERTO y mas CERRADO).  Si alguna
    da una energia mas baja que 'E_ref' - tol, avisa y devuelve esa (la
    solucion convergida no era el minimo global de ocupaciones: habia otra
    rama NOF por debajo).

    Devuelve (E, C, gamma, fmiug0, n, cj12, ck12, cambiada_bool).
    """
    E_w, C_w, g_w, f_w = E_ref, C_ref, gamma_ref, fmiug0_ref
    n_w = cj_w = ck_w = None
    changed = False
    probes = (("occ->abierto", _gamma_frontier_softmax(gamma_ref, p, npairs, "open")),
              ("occ->cerrado", _gamma_frontier_softmax(gamma_ref, p, npairs, "close")))
    for tag, g0 in probes:
        if np.allclose(g0, g_w):
            continue
        r = run_scf(g0, C_ref, fmiug0_ref, p.maxit, tag=tag)
        if r[0] < E_w - tol:
            print("  [estabilidad occ] '{}' encuentra una rama NOF mas baja: "
                  "{:.8f} -> {:.8f}  ({:+.2f} kcal/mol). Se adopta.".format(
                      tag, E_w + E_nuc, r[0] + E_nuc, (r[0] - E_w) * HARTREE2KCAL))
            E_w, C_w, g_w, f_w, n_w, cj_w, ck_w = r
            changed = True
    if not changed:
        print("  [estabilidad occ] solucion estable frente a apertura/cierre "
              "del par frontera.")
        n_w, _ = pynof.ocupacion(g_w, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5,
                                 p.ndns, p.ncwo, p.HighSpin, p.occ_method)
        cj_w, ck_w = pynof.PNOFi_selector(n_w, p)
    return E_w, C_w, g_w, f_w, n_w, cj_w, ck_w, changed


def compute_energy(
    mol,
    p=None,
    C=None,
    n=None,
    fmiug0=None,
    guess="HF",
    nofmp2=False,
    mbpt=False,
    gradients=False,
    printmode=True,
    ekt=False,
    mulliken_pop=False,
    lowdin_pop=False,
    dipole=False,
    m_diagnostic=False,
    perturb=False,
    erpa=False,
    iter_erpa=0,
    hessian=False,
):
    """Compute Natural Orbital Functional single point energy"""

    t1 = time()

    wfn = p.wfn
    S, T, V, H, I, b_mnl, Dipole = pynof.compute_integrals(wfn, mol, p)

    if printmode:
        print("Number of basis functions                   (NBF)    =", p.nbf)
        if p.RI:
            print("Number of auxiliary basis functions         (NBFAUX) =", p.nbfaux)
        print("Inactive Doubly occupied orbitals up to     (NO1)    =", p.no1)
        print("No. considered Strongly Doubly occupied MOs (NDOC)   =", p.ndoc)
        print("No. considered Strongly Singly occupied MOs (NSOC)   =", p.nsoc)
        print("No. of Weakly occ. per St. Doubly occ.  MOs (NCWO)   =", p.ncwo)
        print("Dimension of the Nat. Orb. subspace         (NBF5)   =", p.nbf5)
        print("No. of electrons                                     =", p.ne)
        print("Multiplicity                                         =", p.mul)
        print("")

    # Nuclear Energy
    E_nuc = mol.nuclear_repulsion_energy()

    # Selection of reference
    if p.mul > 1:
        psi4.set_options({"reference": "rohf"})

    # Guess de MO (C)
    _c_was_given = C is not None
    if C is None:
        if guess == "Core" or guess == None:
            Eguess, C = eigh(H, S)  # (HC = SCe)
        elif guess == "HFIDr":
            Eguess, Cguess = eigh(H, S)
            EHF, C, fmiug0guess = pynof.hfidr(Cguess, H, I, b_mnl, E_nuc, p, printmode)
        else:
            EHF, wfn_HF = psi4.energy(guess, return_wfn=True)
            EHF = EHF - E_nuc
            C = wfn_HF.Ca().np
    else:
        guess = None
    C = pynof.check_ortho(C, S, p)

    # Guess Occupation Numbers (n)
    if n is None:
        if p.occ_method == "Trigonometric":
            gamma = pynof.compute_gammas_trigonometric(p.ndoc, p.ncwo)
        if p.occ_method == "Softmax":
             #p.nv = p.nbf5 - p.no1 - p.nsoc
            gamma = pynof.compute_gammas_softmax(p.ndoc, p.ncwo)
        if p.occ_method == "EBI":
            p.nbf5 = p.nbf
            p.nvar = int(p.nbf * (p.nbf - 1) / 2)
            p.nv = p.nbf
            gamma = pynof.compute_gammas_ebi(p.ndoc, p.nbf)
    else:
        if p.occ_method == "Trigonometric":
            gamma = pynof.n_to_gammas_trigonometric(n, p.no1, p.ndoc, p.ndns, p.ncwo)
        if p.occ_method == "Softmax":
            gamma = pynof.n_to_gammas_softmax(n, p.no1, p.ndoc, p.ndns, p.ncwo)
        if p.occ_method == "EBI":
            p.nbf5 = p.nbf
            p.nvar = int(p.nbf * (p.nbf - 1) / 2)
            p.nv = p.nbf
            gamma = pynof.n_to_gammas_ebi(n)

    elag = np.zeros((p.nbf, p.nbf))

    # ============================================================
    #  Nucleo orbital+ocupacion, extraido para poder lanzarlo desde
    #  varios guesses de ocupacion (Solucion 1: multi-arranque;
    #  Solucion 2: chequeo de estabilidad de la ocupacion).  El cuerpo
    #  es identico al bucle original de compute_energy.
    # ============================================================
    def _run_scf(gamma0, C0, fmiug0_0, maxit_loc, tag=""):
        C_ = np.array(C0, dtype=float).copy()   # C0 ya viene ortonormalizada
        gamma_ = np.array(gamma0, dtype=float).copy()
        fmiug0_ = fmiug0_0
        E_occ, nit_occ, success_occ, gamma_, n_, cj12_, ck12_ = pynof.occoptr(gamma_, C_, H, I, b_mnl, p)

        itlim = 0
        E_, E_old, E_diff = 9999, 9999, 9999
        Estored, Cstored, gammastored = 0, 0, 0

        if printmode:
            print("")
            hdr = "PNOF{} Calculation ({}/{} Optimization)".format(p.ipnof, p.orb_method, p.occ_method)
            if tag:
                hdr += "   [guess ocup.: {}]".format(tag)
            print(hdr)
            print("==================")
            print("")
            print(
                "{:^7} {:^7}  {:^7}  {:^14} {:^14} {:^14}   {:^6}   {:^6} {:^6} {:^6}".format(
                    "Nitext", "Nit_orb", "Nit_occ", "Eelec", "Etot", "Ediff", "Grad_orb", "Grad_occ", "Conv Orb", "Conv Occ"
                )
            )
        for i_ext in range(maxit_loc):
            if p.orb_method == "ID":
                E_orb, C_, nit_orb, success_orb, itlim, fmiug0_ = pynof.orboptr(
                    C_, n_, H, I, b_mnl, cj12_, ck12_, i_ext, itlim, fmiug0_, p, printmode
                )
            if p.orb_method == "Rotations":
                E_orb, C_, nit_orb, success_orb = pynof.orbopt_rotations(gamma_, C_, H, I, b_mnl, p)
            if p.orb_method == "ADAM":
                E_orb, C_, nit_orb, success_orb = pynof.orbopt_adam(gamma_, C_, H, I, b_mnl, p)

            E_occ, nit_occ, success_occ, gamma_, n_, cj12_, ck12_ = pynof.occoptr(gamma_, C_, H, I, b_mnl, p)
            if p.occ_method == "Softmax":
                C_, gamma_ = pynof.order_occupations_softmax(C_, gamma_, H, I, b_mnl, p)

            E_ = E_orb
            E_diff = E_ - E_old
            E_old = E_

            y = np.zeros((p.nvar))
            grad_orb = pynof.calcorbg(y, n_, cj12_, ck12_, C_, H, I, b_mnl, p)
            J_MO, K_MO, H_core = pynof.computeJKH_MO(C_, H, I, b_mnl, p)
            grad_occ = pynof.calcoccg(gamma_, J_MO, K_MO, H_core, p)
            print(
                "{:6d} {:6d} {:6d}   {:14.8f} {:14.8f} {:15.8f}      {:3.1e}    {:3.1e}   {}   {}".format(
                    i_ext, nit_orb, nit_occ, E_, E_ + E_nuc, E_diff,
                    np.linalg.norm(grad_orb), np.linalg.norm(grad_occ), success_orb, success_occ,
                )
            )

            if (success_orb or np.linalg.norm(grad_orb) < 1e-3) and (success_occ or np.linalg.norm(grad_occ) < 1e-3):
                if perturb and E_ - Estored < -1e-4:
                    y = np.zeros((p.nvar))
                    grad_orb = pynof.calcorbg(y, n_, cj12_, ck12_, C_, H, I, b_mnl, p)
                    J_MO, K_MO, H_core = pynof.computeJKH_MO(C_, H, I, b_mnl, p)
                    grad_occ = pynof.calcoccg(gamma_, J_MO, K_MO, H_core, p)
                    print("Increasing Gradient")
                    Estored, Cstored, gammastored = E_, C_.copy(), gamma_.copy()
                    C_, gamma_ = pynof.perturb_solution(C_, gamma_, grad_orb, grad_occ, p)
                else:
                    print("Solution does not improve anymore")
                    if Estored < E_:
                        E_, C_, gamma_ = Estored, Cstored, gammastored
                    break

        return E_, C_, gamma_, fmiug0_, n_, cj12_, ck12_

    # ------- guesses de ocupacion (Solucion 1 / Solucion 2) --------
    _softmax = (p.occ_method == "Softmax")
    _ms = getattr(p, "occ_multistart", None)      # None => automatico (GNOF si)
    _st = getattr(p, "occ_stability", None)
    do_multistart = (p.ipnof == 8 if _ms is None else bool(_ms)) and _softmax
    do_occ_stab = (p.ipnof == 8 if _st is None else bool(_st)) and _softmax
    n_pairs = int(getattr(p, "occ_multistart_npairs", 1))
    ms_tol = float(getattr(p, "occ_multistart_tol", 1e-5))

    gamma_start = np.array(gamma, dtype=float).copy()
    C_start = np.array(C, dtype=float).copy()
    fmiug0_start = fmiug0

    # arranque primario (guess que venia: n dado, o el por defecto)
    E, C, gamma, fmiug0, n, cj12, ck12 = _run_scf(
        gamma_start, C_start, fmiug0_start, p.maxit, tag=("primario" if do_multistart else "")
    )

    if do_multistart:
        # Solucion 1: repetir la optimizacion desde ocupaciones de capa
        # CERRADA y ABIERTA (par pi/pi* dirradical) y quedarse con la de
        # menor energia.  GNOF tiene varias soluciones NOF al romper un
        # enlace; el guess por defecto es de capa cerrada y ADAM se queda
        # atrapado ahi (se observo en la torsion del etileno con GNOFm).
        #
        # La 'cuenca' la fijan tanto la ocupacion (gamma) COMO los orbitales
        # (C).  Un restart sesga ambos, asi que los guesses alternativos se
        # lanzan desde orbitales HF (neutros, de capa cerrada), no desde el
        # C del restart.  occ_multistart_level: 1 = solo C_hf (~3x coste),
        # 2 = C_hf y C_start (~5x).
        level = int(getattr(p, "occ_multistart_level", 1))
        C_hf = C_start
        if _c_was_given:            # habia restart -> no se calculo HF; hazlo ahora
            try:
                _ehf, _wfnhf = psi4.energy("HF", return_wfn=True)
                C_hf = pynof.check_ortho(_wfnhf.Ca().np, S, p)
            except Exception as _e:
                print("  [multi-arranque] no se pudo obtener guess HF ({}); "
                      "se usan los orbitales del restart".format(_e))
                C_hf = C_start

        g_closed = _gamma_all_closed_softmax(p)
        base_gammas = [("cerrado", g_closed),
                       ("defecto", pynof.compute_gammas_softmax(p.ndoc, p.ncwo))]
        for k in range(1, n_pairs + 1):
            base_gammas.append(("abierto-{}par".format(k),
                                _gamma_frontier_softmax(g_closed, p, k, "open")))

        alt_runs = [(tg + " /C_HF", gg, C_hf) for tg, gg in base_gammas]
        if level >= 2:
            alt_runs += [(tg + " /C_prim", gg, C_start) for tg, gg in base_gammas]

        win = "primario"
        for tag, g0, C0 in alt_runs:
            r = _run_scf(g0, C0, fmiug0_start, p.maxit, tag=tag)
            if r[0] < E - ms_tol:
                print("  [multi-arranque] '{}' mejora: {:.8f} -> {:.8f}  "
                      "({:+.2f} kcal/mol)".format(tag, E + E_nuc, r[0] + E_nuc,
                                                  (r[0] - E) * HARTREE2KCAL))
                E, C, gamma, fmiug0, n, cj12, ck12 = r
                win = tag
        print("  [multi-arranque] solucion de ocupacion adoptada: '{}'  "
              "(E = {:.8f})".format(win, E + E_nuc))

    if do_occ_stab:
        E, C, gamma, fmiug0, n, cj12, ck12, _moved = _occ_stability_probe(
            _run_scf, E, C, gamma, fmiug0, p, E_nuc,
            npairs=max(1, n_pairs), tol=ms_tol,
        )

    if p.orb_method == "ID":
        np.save(p.title + "_fmiug0.npy", fmiug0)
    if p.orb_method == "ADAM":
        np.save(p.title + "_fmiug0.npy", fmiug0)

    n, dR = pynof.ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = pynof.PNOFi_selector(n, p)
    E, elag, sumdiff, maxdiff = pynof.ENERGY1r(C, n, H, I, b_mnl, cj12, ck12, p)
    print("\nLagrage sumdiff {:3.1e} maxfdiff {:3.1e}".format(sumdiff, maxdiff))

    if p.ipnof > 4:
        C, n, elag = pynof.order_subspaces(C, n, elag, H, I, b_mnl, p)

    np.save(p.title + "_C.npy", C)
    np.save(p.title + "_n.npy", n)

    if printmode:
        print("")
        print("RESULTS OF THE OCCUPATION OPTIMIZATION")
        print("========================================")

        e_val = elag[np.diag_indices(p.nbf5)]
        print(" {:^3}    {:^9}   {:>12}".format("Idx", "n", "E (Hartree)"))
        for i in range(p.nbeta):
            print(" {:3d}    {:9.7f}  {:12.8f}".format(i + 1, 2 * n[i], e_val[i]))
        for i in range(p.nbeta, p.nalpha):
            if not p.HighSpin:
                print(" {:3d}    {:9.7f}  {:12.8f}".format(i + 1, 2 * n[i], e_val[i]))
            else:
                print(" {:3d}    {:9.7f}  {:12.8f}".format(i + 1, n[i], e_val[i]))
        for i in range(p.nalpha, p.nbf5):
            print(" {:3d}    {:9.7f}  {:12.8f}".format(i + 1, 2 * n[i], e_val[i]))

        print("")

        print("----------------")
        print(" Final Energies ")
        print("----------------")

        if guess == "HF" or guess == "HFIDr":
            print("       HF Total Energy = {:15.7f}".format(E_nuc + EHF))
        print("Final NOF Total Energy = {:15.7f}".format(E_nuc + E))
        if guess == "HF" or guess == "HFIDr":
            print("    Correlation Energy = {:15.7f}".format(E - EHF))
        print("")
        print("")

    E_t = E_nuc + E

    pynof.fchk(p.title, wfn, mol, "Energy", E_t, elag, n, C, p)

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2 - t1))

    if nofmp2:
        pynof.nofmp2(n, C, H, I, b_mnl, E_nuc, p)

    if mbpt:
        pynof.mbpt(n, C, H, I, b_mnl, Dipole, E_nuc, E, p)

    if ekt:
        pynof.ext_koopmans(p, elag, n)

    if mulliken_pop:
        pynof.mulliken_pop(p, wfn, n, C, S)

    if lowdin_pop:
        pynof.lowdin_pop(p, wfn, n, C, S)

    if dipole:
        pynof.dipole_moment(p, wfn, n, C, Dipole)

    if m_diagnostic:
        pynof.M_diagnostic(p, n)

    if erpa:
        mode = getattr(p, "erpa_iterative", False)
        if mode:
            n_st = getattr(p, "n_erpa_states", 10)
            # CAMBIO F: se reenvian los parametros del solver desde p.
            erpa_kw = dict(
                tol_rms=getattr(p, "erpa_tol_rms", 1e-6),
                tol_max=getattr(p, "erpa_tol_max", 1e-5),
                k_max=getattr(p, "erpa_kmax", 100),
                omega_min_eV=getattr(p, "erpa_omega_min_eV", 1.0),
                B0=getattr(p, "erpa_B0", None),
            )
            if str(mode).upper() in ("ERPA1", "1"):
                pynof.iterative_ERPA1(
                    wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, n_st, p,
                    **erpa_kw,
                )
            else:  # True / "ERPA0"
                pynof.iterative_ERPA0_v2(
                    wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, n_st, p,
                    **erpa_kw,
                )
        else:
            pynof.ERPA(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, p)

    if iter_erpa > 0:
        pynof.iterative_ERPA0(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, iter_erpa, p)

    out = (E_t, C, n, fmiug0)
    if gradients:
        grad = pynof.compute_geom_gradients(wfn, mol, n, C, cj12, ck12, elag, p)
        out += (grad.flatten(),)
    if hessian:
        # Hessiano por diferencias finitas de gradientes (pynof/hessian.py); el ultimo elemento devuelto
        # es un dict con hessian, frequencies, zpe_kcal y controles de calidad
        out += (pynof.compute_hessian(mol, p, C, n, printmode=printmode),)
    return out
