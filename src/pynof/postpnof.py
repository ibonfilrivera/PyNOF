import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.optimize import root
from time import time
import pynof
from numba import prange, njit, jit
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, minres
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.special import roots_legendre


def build_XmY_XpY(L, tempm, bigomega, nab):

    XmY = np.dot(L, tempm)

    XpY = np.linalg.solve(np.transpose(L), tempm)

    XmY = np.einsum("mi,i->mi", XmY, 1 / np.sqrt(bigomega), optimize=True)
    XpY = np.einsum("mi,i->mi", XpY, np.sqrt(bigomega), optimize=True)

    return XmY, XpY


@njit(parallel=True, cache=True)
def get_AmB(eig, nalpha, nbf):
    nvir = nbf - nalpha
    AmB = np.zeros((nalpha, nvir, nalpha, nvir))
    for i in prange(nalpha):
        for aidx, a in enumerate(range(nalpha, nbf)):
            AmB[i, aidx, i, aidx] = eig[a] - eig[i]

    return AmB


def build_AmB_ApB(eig, pqrt_at, nalpha, nbf, nab):

    nvir = nbf - nalpha

    ApB = 4 * pqrt_at[0:nalpha, nalpha:nbf, 0:nalpha, nalpha:nbf]
    AmB = get_AmB(eig, nalpha, nbf)

    ApB = ApB + AmB

    EcRPA = -0.25 * (np.einsum("iaia->", ApB, optimize=True) + np.einsum("iaia->", AmB, optimize=True))

    ApB = ApB.reshape(nab, nab)
    AmB = AmB.reshape(nab, nab)

    return EcRPA, AmB, ApB


@njit(parallel=True, cache=True)
def ccsd_eq(eig, pqrt, pqrt_at, nbeta, nalpha, nbf):
    # CCSD
    Eccsd = 0
    FockM_in = np.zeros((nbf, nbf))
    for pp in range(nbf):
        FockM_in[pp, pp] = eig[pp]
    ERItmp = np.zeros((nbf, nbf, nbf, nbf))

    nocc = max(nbeta, nalpha)
    nocc2 = 2 * nocc
    nbf2 = 2 * nbf
    nvir2 = nbf2 - nocc2

    FockM = np.zeros((nbf2, nbf2))
    for i in range(nbf2):
        for j in range(nbf2):
            if i % 2 == j % 2:
                FockM[j, i] = FockM_in[slbasis(j), slbasis(i)]

    td = np.zeros((nocc2, nocc2, nvir2, nvir2))
    for i in range(nocc2):
        for j in range(nocc2):
            for aidx, a in enumerate(range(nocc2, nbf2)):
                for bidx, b in enumerate(range(nocc2, nbf2)):
                    td[i, j, aidx, bidx] = spin_int(i, j, a, b, pqrt_at) / (
                        FockM[i, i] + FockM[j, j] - FockM[a, a] - FockM[b, b] + 1e-8
                    )

    deltaE = 1
    itera = 0
    ts = np.zeros((nocc2, nvir2))
    print("")
    while np.abs(deltaE) > 1e-6 and itera < 1000000:

        Fmi, Wmnij, Fae, Wabef, Fme, Wmbej = ccsd_update_interm(ts, td, FockM, pqrt_at, nocc2, nvir2, nbf2)

        tsnew, tdnew = ccsd_update_t1_t2(ts, td, Fmi, Wmnij, Fae, Wabef, Fme, Wmbej, FockM, pqrt_at, nocc2, nvir2, nbf2)

        for i in range(nocc2):
            for j in range(nocc2):
                for aidx, a in enumerate(range(nocc2, nbf2)):
                    if j == 0:
                        if np.abs(tsnew[i, aidx]) < 1e-8:
                            tsnew[i, aidx] = 0
                    for bidx, b in enumerate(range(nocc2, nbf2)):
                        if np.abs(tdnew[i, j, aidx, bidx]) < 1e-8:
                            tdnew[i, j, aidx, bidx] = 0.0
        ts = tsnew.copy()
        td = tdnew.copy()

        ccsd_en_nof = 0
        nsocc = 2 * nalpha
        ndocc = 2 * nbeta
        for aidx, a in enumerate(range(nsocc, nbf2)):
            for bidx, b in enumerate(range(nsocc, nbf2)):
                for i in range(ndocc):
                    if b == nsocc:
                        ccsd_en_nof += FockM[i, a] * ts[i, aidx]
                    for j in range(ndocc):
                        ccsd_en_nof += (
                            0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                            + 0.5 * spin_int(i, j, a, b, pqrt) * ts[i, aidx] * ts[j, bidx]
                        )
                    for j in range(ndocc, nsocc):
                        ccsd_en_nof += 0.5 * (
                            0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                            + 0.5 * spin_int(i, j, a, b, pqrt) * ts[i, aidx] * ts[j, bidx]
                        )
                for i in range(ndocc, nsocc):
                    for j in range(ndocc):
                        ccsd_en_nof += 0.5 * (
                            0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                            + 0.5 * spin_int(i, j, aidx, bidx, pqrt) * ts[i, aidx] * ts[j, bidx]
                        )
                    for j in range(ndocc, nsocc):
                        if i % 2 == 0:
                            k = i + 1
                        else:
                            k = i - 1
                        if j != i and j != k:
                            ccsd_en_nof += 0.25 * (
                                0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                                + 0.5 * spin_int(i, j, a, b, pqrt) * ts[i, aidx] * ts[j, bidx]
                            )
        Eccsd_new = ccsd_en_nof
        deltaE = Eccsd - Eccsd_new
        Eccsd = Eccsd_new
        itera += 1
        if itera % 2 == 0:
            # print("CCSD corr. energy {:5.3f} iter {:d} deltaE {:5.3f}".format(Eccsd,itera,deltaE))
            print(" ....CCSD corr. energy", Eccsd, " iter", itera, "deltaE", deltaE)
        # break
    # print("CCSD procedure finished after {:5.3f} iter. with deltaE {:5.3f}".format(itera,deltaE))
    print(" ....CCSD procedure finished after", itera, "iter. with deltaE", deltaE)
    print("")
    EcCCSD = Eccsd_new

    return EcCCSD


@njit(parallel=True, cache=True)
def ccsd_update_interm(ts, td, FockM, pqrt_at, nocc2, nvir2, nbf2):
    Fmi = np.zeros((nocc2, nocc2))
    Wmnij = np.zeros((nocc2, nocc2, nocc2, nocc2))
    for m in range(nocc2):
        for i in range(nocc2):
            if i != m:
                Fmi[m, i] = FockM[m, i]
            for n in range(nocc2):
                for j in range(nocc2):
                    Wmnij[m, n, i, j] = spin_int(m, n, i, j, pqrt_at)
                    for eidx, e in enumerate(range(nocc2, nbf2)):
                        if j == 0:
                            Fmi[m, i] += ts[n, eidx] * spin_int(m, n, i, e, pqrt_at)
                            if n == 0:
                                Fmi[m, i] += 0.5 * ts[i, eidx] * FockM[m, e]
                        Wmnij[m, n, i, j] += ts[j, eidx] * spin_int(m, n, i, e, pqrt_at) - ts[i, eidx] * spin_int(
                            m, n, j, e, pqrt_at
                        )
                        for fidx, f in enumerate(range(nocc2, nbf2)):
                            tau = td[i, j, eidx, fidx] + ts[i, eidx] * ts[j, fidx] - ts[i, fidx] * ts[j, eidx]
                            Wmnij[m, n, i, j] += 0.25 * tau * spin_int(m, n, e, f, pqrt_at)
                            if j == 0:
                                taus = td[i, n, eidx, fidx] + 0.5 * (
                                    ts[i, eidx] * ts[j, fidx] - ts[i, fidx] * ts[j, eidx]
                                )
                                Fmi[m, i] += 0.5 * taus * spin_int(m, n, e, f, pqrt_at)

    Fae = np.zeros((nvir2, nvir2))
    Wabef = np.zeros((nvir2, nvir2, nvir2, nvir2))
    for aidx, a in enumerate(range(nocc2, nbf2)):
        for eidx, e in enumerate(range(nocc2, nbf2)):
            if a != e:
                Fae[aidx, eidx] = FockM[a, e]
            for bidx, b in enumerate(range(nocc2, nbf2)):
                for fidx, f in enumerate(range(nocc2, nbf2)):
                    Wabef[aidx, bidx, eidx, fidx] = spin_int(a, b, e, f, pqrt_at)
                    for m in range(nocc2):
                        if b == nocc2:
                            if f == nocc2:
                                Fae[aidx, eidx] += -0.5 * ts[m, aidx] * FockM[m, e]
                            Fae[aidx, eidx] += ts[m, fidx] * spin_int(m, a, f, e, pqrt_at)
                        Wabef[aidx, bidx, eidx, fidx] += -ts[m, bidx] * spin_int(a, m, e, f, pqrt_at) + ts[
                            m, aidx
                        ] * spin_int(b, m, e, f, pqrt_at)
                        for n in range(nocc2):
                            tau = td[m, n, aidx, bidx] + ts[m, aidx] * ts[n, bidx] - ts[m, bidx] * ts[n, aidx]
                            Wabef[aidx, bidx, eidx, fidx] += 0.25 * tau * spin_int(m, n, e, f, pqrt_at)
                            if b == nocc2:
                                taus = td[m, n, aidx, fidx] + 0.5 * (
                                    ts[m, aidx] * ts[n, fidx] - ts[m, fidx] * ts[n, aidx]
                                )
                                Fae[aidx, eidx] += -0.5 * taus * spin_int(m, n, e, f, pqrt_at)

    Fme = np.zeros((nocc2, nvir2))
    Wmbej = np.zeros((nocc2, nvir2, nvir2, nocc2))
    for m in range(nocc2):
        for eidx, e in enumerate(range(nocc2, nbf2)):
            Fme[m, eidx] = FockM[m, e]
            for bidx, b in enumerate(range(nocc2, nbf2)):
                for j in range(nocc2):
                    Fme[m, eidx] += ts[j, bidx] * spin_int(m, j, e, b, pqrt_at)
                    Wmbej[m, bidx, eidx, j] = spin_int(m, b, e, j, pqrt_at)
                    for fidx, f in enumerate(range(nocc2, nbf2)):
                        Wmbej[m, bidx, eidx, j] += ts[j, fidx] * spin_int(m, b, e, f, pqrt_at)
                    for n in range(nocc2):
                        Wmbej[m, bidx, eidx, j] += -ts[n, bidx] * spin_int(m, n, e, j, pqrt_at)
                        for fidx, f in enumerate(range(nocc2, nbf2)):
                            Wmbej[m, bidx, eidx, j] += -(
                                0.5 * td[j, n, fidx, bidx] + ts[j, fidx] * ts[n, bidx]
                            ) * spin_int(m, n, e, f, pqrt_at)

    return Fmi, Wmnij, Fae, Wabef, Fme, Wmbej


@njit(parallel=True, cache=True)
def ccsd_update_t1_t2(ts, td, Fmi, Wmnij, Fae, Wabef, Fme, Wmbej, FockM, pqrt_at, nocc2, nvir2, nbf2):
    tsnew = np.zeros((nocc2, nvir2))
    tdnew = np.zeros((nocc2, nocc2, nvir2, nvir2))

    for aidx, a in enumerate(range(nocc2, nbf2)):
        for bidx, b in enumerate(range(nocc2, nbf2)):
            for i in range(nocc2):
                # T1
                if b == nocc2:
                    tsnew[i, aidx] = FockM[i, a]
                    for m in range(nocc2):
                        tsnew[i, aidx] -= ts[m, aidx] * Fmi[m, i]
                        for eidx, e in enumerate(range(nocc2, nbf2)):
                            tsnew[i, aidx] += td[i, m, aidx, eidx] * Fme[m, eidx]
                            tsnew[i, aidx] += -ts[m, eidx] * spin_int(m, a, i, e, pqrt_at)
                            if m == 0:
                                tsnew[i, aidx] += ts[i, eidx] * Fae[aidx, eidx]
                            for fidx, f in enumerate(range(nocc2, nbf2)):
                                tsnew[i, aidx] += -0.5 * td[i, m, eidx, fidx] * spin_int(m, a, e, f, pqrt_at)
                            for n in range(nocc2):
                                tsnew[i, aidx] += -0.5 * td[m, n, aidx, eidx] * spin_int(n, m, e, i, pqrt_at)
                    #                    if(qnewton):
                    #                        ria=tsnew[i,a]-ts[i,a]*(FockM[i,i]-FockM[a,a])
                    #                        if(abs(ria)>tol8):
                    #                            tsnew[i,a] += ria/(FockM[i,i]-FockM[a,a]+1e-8)
                    #                    else:
                    tsnew[i, aidx] = tsnew[i, aidx] / (FockM[i, i] - FockM[a, a] + 1e-8)

                # T2
                for j in range(nocc2):
                    tdnew[i, j, aidx, bidx] = spin_int(i, j, a, b, pqrt_at)
                    for eidx, e in enumerate(range(nocc2, nbf2)):
                        tdnew[i, j, aidx, bidx] += (
                            td[i, j, aidx, eidx] * Fae[bidx, eidx] - td[i, j, bidx, eidx] * Fae[aidx, eidx]
                        )
                        for m in range(nocc2):
                            tdnew[i, j, aidx, bidx] += (
                                -0.5 * td[i, j, aidx, eidx] * ts[m, bidx] * Fme[m, eidx]
                                + 0.5 * td[i, j, bidx, eidx] * ts[m, aidx] * Fme[m, eidx]
                            )
                            if e == nocc2:
                                tdnew[i, j, aidx, bidx] += (
                                    -td[i, m, aidx, bidx] * Fmi[m, j] + td[j, m, aidx, bidx] * Fmi[m, i]
                                )
                                tdnew[i, j, aidx, bidx] += -ts[m, aidx] * spin_int(m, b, i, j, pqrt_at) + ts[
                                    m, bidx
                                ] * spin_int(m, a, i, j, pqrt_at)
                                for n in range(nocc2):
                                    tau = td[m, n, aidx, bidx] + ts[m, aidx] * ts[n, bidx] - ts[m, bidx] * ts[n, aidx]
                                    tdnew[i, j, aidx, bidx] += 0.5 * tau * Wmnij[m, n, i, j]
                            tdnew[i, j, aidx, bidx] += (
                                -0.5 * td[i, m, aidx, bidx] * ts[j, eidx] * Fme[m, eidx]
                                + 0.5 * td[j, m, aidx, bidx] * ts[i, eidx] * Fme[m, eidx]
                            )
                            tdnew[i, j, aidx, bidx] += td[i, m, aidx, eidx] * Wmbej[m, bidx, eidx, j] - ts[
                                i, eidx
                            ] * ts[m, aidx] * spin_int(m, b, e, j, pqrt_at)
                            tdnew[i, j, aidx, bidx] += -td[j, m, aidx, eidx] * Wmbej[m, bidx, eidx, i] + ts[
                                j, eidx
                            ] * ts[m, aidx] * spin_int(m, b, e, i, pqrt_at)
                            tdnew[i, j, aidx, bidx] += -td[i, m, bidx, eidx] * Wmbej[m, aidx, eidx, j] + ts[
                                i, eidx
                            ] * ts[m, bidx] * spin_int(m, a, e, j, pqrt_at)
                            tdnew[i, j, aidx, bidx] += td[j, m, bidx, eidx] * Wmbej[m, aidx, eidx, i] - ts[
                                j, eidx
                            ] * ts[m, bidx] * spin_int(m, a, e, i, pqrt_at)
                        tdnew[i, j, aidx, bidx] += ts[i, eidx] * spin_int(a, b, e, j, pqrt_at) - ts[j, eidx] * spin_int(
                            a, b, e, i, pqrt_at
                        )
                        for fidx, f in enumerate(range(nocc2, nbf2)):
                            tau = td[i, j, eidx, fidx] + ts[i, eidx] * ts[j, fidx] - ts[i, fidx] * ts[j, eidx]
                            tdnew[i, j, aidx, bidx] += 0.5 * tau * Wabef[aidx, bidx, eidx, fidx]
                    tdnew[i, j, aidx, bidx] = tdnew[i, j, aidx, bidx] / (
                        FockM[i, i] + FockM[j, j] - FockM[a, a] - FockM[b, b] + 1e-8
                    )

    return tsnew, tdnew


@njit(parallel=True, cache=True)
def slbasis(i):
    if i % 2 == 0:
        return int(i / 2)
    else:
        return int((i + 1) / 2) - 1


@njit(parallel=True, cache=True)
def spin_int(p, q, r, s, ERImol):
    value1, value2 = 0, 0
    if p % 2 == r % 2 and q % 2 == s % 2:
        value1 = ERImol[slbasis(s), slbasis(p), slbasis(r), slbasis(q)]
    if p % 2 == s % 2 and q % 2 == r % 2:
        value2 = ERImol[slbasis(r), slbasis(p), slbasis(s), slbasis(q)]
    return value1 - value2


@njit(parallel=True, cache=True)
def W(i, a, j, b, eri_at, wmn_at, eig, nab, bigomega, cfreq):

    w_iajb = eri_at
    for s in range(nab):
        w_iajb += wmn_at[i, a, s] * wmn_at[j, b, s] * (1 / (cfreq - bigomega[s]) - 1 / (cfreq + bigomega[s]))

    return w_iajb


@njit(parallel=True, cache=True)
def integrated_omega(i, a, j, b, eri_at, wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order):

    integral = 0
    for ii in prange(order):
        integral += (
            W(i, a, j, b, eri_at, wmn_at, eig, nab, bigomega, cfreqs[ii])
            * weights[ii]
            * 2
            * (eig[i] - eig[a])
            / ((eig[i] - eig[a]) ** 2 + freqs[ii] ** 2)
            * 2
            * (eig[j] - eig[b])
            / ((eig[j] - eig[b]) ** 2 + freqs[ii] ** 2)
        )

    if np.abs(integral.imag) > 1e-4:
        print("Warning, large imaginary component", integral)

    integral = integral.real

    return integral


@njit(parallel=True, cache=True)
def rpa_sosex(freqs, weights, sumw, order, wmn_at, eig, pqrt, pqrt_at, bigomega, nab, nbeta, nalpha, nbf):

    iEcRPA = 0
    iEcSOSEX = 0

    weights = weights / sumw
    freqs = (freqs + 1) / 2

    weights = weights / (1 - freqs) ** 2
    freqs = freqs / (1 - freqs)
    cfreqs = freqs * 1j

    for a in prange(nalpha, nbf):
        for b in prange(nalpha, nbf):
            for i in prange(nbeta):
                for j in prange(nbeta):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -pqrt[i, a, j, b] * integral
                    iEcSOSEX += pqrt[i, b, j, a] * integral
                for j in prange(nbeta, nalpha):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -0.5 * pqrt[i, a, j, b] * integral
                    iEcSOSEX += 0.5 * pqrt[i, b, j, a] * integral
            for i in prange(nbeta, nalpha):
                for j in prange(nbeta):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -0.5 * pqrt[i, a, j, b] * integral
                    iEcSOSEX += 0.5 * pqrt[i, b, j, a] * integral
                for j in prange(nbeta, nalpha):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -0.25 * pqrt[i, a, j, b] * integral
                    iEcSOSEX += 0.25 * pqrt[i, b, j, a] * integral

    iEcRPA = iEcRPA / np.pi
    iEcSOSEX = 0.5 * iEcSOSEX / np.pi

    return iEcRPA, iEcSOSEX


@njit(parallel=True, cache=True)
def gw_gm_eq(wmn_at, pqrt, eig, bigomega, XpY, nab, nbeta, nalpha, nbf):
    EcGoWo = 0
    EcGMSOS = 0

    for a in prange(nalpha, nbf):
        for b in range(nalpha, nbf):
            for i in range(nbeta):
                for j in range(nbeta):
                    l = j * (nbf - nalpha) + (b - nalpha)
                    for s in range(nab):
                        EcGoWo += (
                            wmn_at[i, a, s]
                            * pqrt[i, a, j, b]
                            * XpY[l, s]
                            * np.sqrt(2)
                            / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                        )
                        EcGMSOS -= (
                            wmn_at[i, a, s]
                            * pqrt[i, b, j, a]
                            * XpY[l, s]
                            * np.sqrt(2)
                            / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                        )
    if nalpha != nbeta:
        for a in range(nalpha, nbf):
            for b in range(nalpha, nbf):
                for i in range(nbeta):
                    for j in range(nbeta, nalpha):
                        l = j * (nbf - nalpha) + (b - nalpha)
                        for s in range(nab):
                            EcGoWo += (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, a, j, b]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                            EcGMSOS -= (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, b, j, a]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                for i in range(nbeta, nalpha):
                    for j in range(nbeta, nalpha):
                        l = j * (nbf - nalpha) + (b - nalpha)
                        for s in range(nab):
                            EcGoWo += (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, a, j, b]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                            EcGMSOS -= (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, b, j, a]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                    for j in range(nbeta, nalpha):
                        if i != j:
                            l = j * (nbf - nalpha) + (b - nalpha)
                            for s in range(nab):
                                EcGoWo += (
                                    0.25
                                    * wmn_at[i, a, s]
                                    * pqrt[i, a, j, b]
                                    * XpY[l, s]
                                    * np.sqrt(2)
                                    / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                                )
                                EcGMSOS -= (
                                    0.25
                                    * wmn_at[i, a, s]
                                    * pqrt[i, b, j, a]
                                    * XpY[l, s]
                                    * np.sqrt(2)
                                    / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                                )

    return EcGoWo, EcGMSOS


def build_wmn(pqrt, pqrt_at, XpY, nab, nalpha, nbf):

    nvir = nbf - nalpha
    XpY = XpY.reshape(nalpha, nvir, nab)
    wmn = np.einsum("pqia,iak->pqk", pqrt[:, :, :nalpha, nalpha:], XpY, optimize=True)
    wmn_at = np.einsum("pqia,iak->pqk", pqrt_at[:, :, :nalpha, nalpha:], XpY, optimize=True)

    return wmn, wmn_at


def td_polarizability(EcRPA, C, Dipole, bigomega, XpY, nab, nalpha, nbf):

    Dipole_MO = np.einsum("mi,qmn,nj->qij", C, Dipole, C, optimize=True)

    nvir = nbf - nalpha
    XpY = XpY.reshape(nalpha, nvir, nab)

    tempm = np.sqrt(2) * np.einsum("qia,iak->qk", Dipole_MO[:, :nalpha, nalpha:nbf], XpY, optimize=True)
    oscstr = 2 / 3 * np.einsum("qk,qk,k->k", tempm, tempm, bigomega, optimize=True)
    EcRPA += 0.5 * np.sum(np.abs(bigomega))

    print("TD-H (RPA) CASIDA eq. solved")
    print("N. excitation   a.u.         eV            nm      osc. strenght")
    for i in range(nab):
        if oscstr[i] > 10**-6:
            print(
                "{:^ 10d} {: 11.5f} {: 11.5f} {: 11.4f} {: 11.6f} ".format(
                    i, bigomega[i], bigomega[i] * 27.211399, 1239.84193 / (bigomega[i] * 27.211399), oscstr[i]
                )
            )
    print("")

    print("Static Polarizability:")
    print("")
    staticpol = 2 * np.einsum("ji,ki,i->jk", tempm, tempm, 1 / (bigomega + 1e-10), optimize=True)
    for xyz in range(3):
        print("{:15.5f} {:15.5f} {:15.5f}".format(staticpol[xyz, 0], staticpol[xyz, 1], staticpol[xyz, 2]))
    print("")
    print(
        "Trace of the static polarizability {:9.5f}".format((staticpol[0, 0] + staticpol[1, 1] + staticpol[2, 2]) / 3)
    )
    print("")

    return EcRPA


def build_F_MO(C, H, I, b_mnl, p):
    D = pynof.computeD_HF(C, p)
    if p.MSpin == 0:
        if p.nsoc > 0:
            Dalpha = pynof.computeDalpha_HF(C, p)
            D = D + 0.5 * Dalpha
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = H + 2 * J - K
        EHFL = np.trace(np.matmul(D, H) + np.matmul(D, F))
    elif not p.MSpin == 0:
        Dalpha = pynof.computeDalpha_HF(C, p)
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = 2 * J - K
        EHFL = 2 * np.trace(np.matmul(D + 0.5 * Dalpha, H)) + np.trace(np.matmul(D + Dalpha, F))
        F = H + F
        if p.nsoc > 1:
            J, K = pynof.computeJK_HF(0.5 * Dalpha, I, b_mnl, p)
            Falpha = J - K
            EHFL = EHFL + 2 * np.trace(np.matmul(0.5 * Dalpha, Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(C), F), C)

    # eig = np.einsum("ii->i",F_MO) #JFHLY: Print info

    return EHFL, F_MO


@njit(parallel=True, cache=True)
def ERIS_attenuated(pqrt, Cintra, Cinter, no1, ndoc, nsoc, ndns, ncwo, nbf5, nbf):
    subspaces = np.zeros((nbf))
    for i in range(no1):
        subspaces[i] = i
    for i in range(ndoc):
        subspaces[no1 + i] = no1 + i
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ncwo * ndoc
        subspaces[ll:ul:ndoc] = no1 + i
    for i in range(nsoc):
        subspaces[no1 + ndoc + i] = no1 + ndoc + i
    subspaces[nbf5:] = -1

    pqrt_at = np.zeros((nbf, nbf, nbf, nbf))
    for p in prange(nbf):
        for q in prange(nbf):
            for r in prange(nbf):
                for t in prange(nbf):
                    if (
                        subspaces[p] == subspaces[q]
                        and subspaces[p] == subspaces[r]
                        and subspaces[p] == subspaces[t]
                        and subspaces[p] != -1
                    ):
                        pqrt_at[p, q, r, t] = pqrt[p, q, r, t] * Cintra[p] * Cintra[q] * Cintra[r] * Cintra[t]
                    else:
                        pqrt_at[p, q, r, t] = pqrt[p, q, r, t] * Cinter[p] * Cinter[q] * Cinter[r] * Cinter[t]

    return pqrt_at


@njit(parallel=True, cache=True)
def F_MO_attenuated(F_MO, Cintra, Cinter, no1, nalpha, ndoc, nsoc, ndns, ncwo, nbf5, nbf):

    F_MO_at = np.zeros((nbf, nbf))

    subspaces = np.zeros((nbf))
    for i in range(no1):
        subspaces[i] = i
    for i in range(ndoc):
        subspaces[no1 + i] = no1 + i
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ncwo * ndoc
        subspaces[ll:ul:ndoc] = no1 + i
    for i in range(nsoc):
        subspaces[no1 + ndoc + i] = no1 + ndoc + i
    subspaces[nbf5:] = -1

    for p in prange(nbf):
        for q in prange(nbf):
            if p != q:
                if subspaces[p] == subspaces[q]:
                    F_MO_at[p, q] = F_MO[p, q] * Cintra[p] * Cintra[q]
                else:
                    F_MO_at[p, q] = F_MO[p, q] * Cinter[p] * Cinter[q]
            else:
                F_MO_at[p, q] = F_MO[p, q]
    F_MO_at[:nalpha, nalpha:nbf] = 0.0
    F_MO_at[nalpha:nbf, :nalpha] = 0.0

    return F_MO_at


@njit(parallel=True, cache=True)
def mp2_eq(eig, pqrt, pqrt_at, nbeta, nalpha, nbf):

    EcMP2 = 0
    for a in prange(nalpha, nbf):
        for b in prange(nalpha, nbf):
            for i in prange(nbeta):
                for j in prange(nbeta):
                    EcMP2 += (
                        pqrt[i, a, j, b]
                        * (2 * pqrt_at[i, a, j, b] - pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )
                for j in prange(nbeta, nalpha):
                    EcMP2 += (
                        pqrt[i, a, j, b]
                        * (pqrt_at[i, a, j, b] - 0.5 * pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )
            for i in prange(nbeta, nalpha):
                for j in prange(nbeta):
                    EcMP2 += (
                        pqrt[i, a, j, b]
                        * (pqrt_at[i, a, j, b] - 0.5 * pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )
                for j in prange(nbeta, nalpha):
                    EcMP2 += (
                        0.5
                        * pqrt[i, a, j, b]
                        * (pqrt_at[i, a, j, b] - 0.5 * pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )

    return EcMP2


def ECorrNonDyn(n, C, H, I, b_mnl, p):
    fi = 2 * n * (1 - n)

    CK12nd = np.outer(fi, fi)

    beta = np.sqrt((1 - abs(1 - 2 * n)) * n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + (p.ndoc - l - 1)
        ul = ll + p.ncwo * p.ndoc
        CK12nd[p.no1 + l, ll:ul:p.ndoc] = beta[p.no1 + l] * beta[ll:ul:p.ndoc]
        CK12nd[ll:ul:p.ndoc, p.no1 + l] = beta[ll:ul:p.ndoc] * beta[p.no1 + l]
        CK12nd[ll:ul:p.ndoc, ll:ul:p.ndoc] = -np.outer(beta[ll:ul:p.ndoc], beta[ll:ul:p.ndoc])

    # C^K KMO
    J_MO, K_MO, H_core = pynof.computeJKH_MO(C, H, I, b_mnl, p)

    ECndHF = 0
    ECndl = 0
    if p.MSpin == 0:
        ECndHF = -np.einsum(
            "ii,ii->", CK12nd[p.nbeta : p.nalpha, p.nbeta : p.nalpha], K_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha]
        )  # sum_ij
        ECndl -= np.einsum("ij,ji->", CK12nd, K_MO)  # sum_ij
        ECndl += np.einsum("ii,ii->", CK12nd, K_MO)  # Quita i=j
    elif not p.MSpin == 0:
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # Quita i=j
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # Quita i=j

    return ECndHF, ECndl


def compare(original, new, byelement=False):

    if byelement:
        original_lin = original.flatten()
        new_lin = new.flatten()

        for ol, nl in zip(original_lin, new_lin):
            if np.abs(ol - nl) > 1e-5:
                print(ol, nl, ol - nl)

    print("Original pqrt", np.sum(original))
    print("New pqrt", np.sum(new))
    print("Diff pqrt", np.sum(new - original))
    print("Abs Diff pqrt", np.sum(np.abs(new - original)))


def mbpt(n, C, H, I, b_mnl, Dipole, E_nuc, E_elec, p):

    #####################################################
    #
    # Reference Article
    # Coupling Natural Orbital Functional Theory
    # and Many-Bodyi Perturbation Theory by Using
    # Nondynamically Correlated Canonical Orbitals
    # https://doi.org/10.1021/acs.jctc.1c00858
    #
    #####################################################

    t1 = time()

    print(" MBPT")
    print("=========")

    nab = p.nvir * p.nalpha
    last_coup = p.nalpha + p.ncwo * (p.ndoc)

    print("Number of orbitals        (NBASIS) = {}".format(p.nbf))
    print("Number of frozen pairs       (NFR) = {}".format(p.no1))
    print("Number of occupied orbs.     (NOC) = {}".format(p.nalpha))
    print("Number of virtual orbs.     (NVIR) = {}".format(p.nvir))
    print("Size of A+B and A-B (NAB=NOCxNVIR) = {}".format(nab))
    print("")

    occ = np.zeros((p.nbf))
    occ[: p.nbf5] = n

    print(" ....Building F_MO")
    EHFL, F_MO = build_F_MO(C, H, I, b_mnl, p)

    print(" ....Transforming ERIs mnsl->pqrt")
    pqrt = pynof.compute_pqrt(C, I, b_mnl, p)

    # Eq. (17) and (18)
    Cintra = 1 - (1 - abs(1 - 2 * occ)) ** 2
    Cinter = abs(1 - 2 * occ) ** 2
    Cinter[: p.nalpha] = 1.0

    # Eq. (19) Canonicalization of NOs Step 1
    print(" ....Attenuating F_MO")
    F_MO_at = F_MO_attenuated(F_MO, Cintra, Cinter, p.no1, p.nalpha, p.ndoc, p.nsoc, p.ndns, p.ncwo, p.nbf5, p.nbf)

    # Eq. (20)
    print(" ....Attenuating pqrt")
    pqrt_at = ERIS_attenuated(pqrt, Cintra, Cinter, p.no1, p.ndoc, p.nsoc, p.ndns, p.ncwo, p.nbf5, p.nbf)

    # Canonicalization of NOs Step 2 and 3
    eig, C_can = eigh(F_MO_at)

    # Canonicalization of NOs step 4
    print(" ....Canonicalizing pqrt")
    pqrt = np.einsum("pqrt,pm,qn,rs,tl->mnsl", pqrt, C_can, C_can, C_can, C_can, optimize=True)
    print(" ....Canonicalizing pqrt_at")
    pqrt_at = np.einsum("pqrt,pm,qn,rs,tl->mnsl", pqrt_at, C_can, C_can, C_can, C_can, optimize=True)
    print("")

    print("List of qp-orbital energies (a.u.) and occ numbers used")
    print()
    mu = (eig[p.nalpha] + eig[p.nalpha - 1]) / 2
    eig = eig - mu
    for i in range(p.nbf):
        print(" {: 8.6f} {:5.3f}".format(eig[i], occ[i]))
    print("Chemical potential used for qp-orbital energies: {}".format(mu))
    print("")

    # Supp. Info. Eq. (4) and (5)
    EcRPA, AmB, ApB = build_AmB_ApB(eig, pqrt_at, p.nalpha, p.nbf, nab)

    L = sp.linalg.cholesky(ApB, lower=True)

    bigomega2, tempm = np.linalg.eigh(np.matmul(np.matmul(np.transpose(L), AmB), L))
    #    bigomega2,tempm = np.linalg.eigh(np.matmul(ApB,AmB))

    bigomega = np.sqrt(bigomega2)

    XmY, XpY = build_XmY_XpY(L, tempm, bigomega, nab)

    # Eq (8) second term
    print(" ....Computing Polarizabilities")
    EcRPA = td_polarizability(EcRPA, C, Dipole, bigomega, XpY, nab, p.nalpha, p.nbf)

    # Supp. Info. Eq. (6)
    print(" ....Computing wmn")
    wmn, wmn_at = build_wmn(pqrt, pqrt_at, XpY, nab, p.nalpha, p.nbf)

    # Eq. (27) and (29)
    print(" ....Computing gw_gm")
    EcGoWo, EcGMSOS = gw_gm_eq(wmn_at, pqrt, eig, bigomega, XpY, nab, p.nbeta, p.nalpha, p.nbf)

    EcGoWo *= 2
    EcGoWoSOS = EcGoWo + EcGMSOS

    # Eq. (22)
    print(" ....Computing mp2")
    EcMP2 = mp2_eq(eig, pqrt, pqrt_at, p.nbeta, p.nalpha, p.nbf)

    # Eq. (24) and (28)
    order = 40
    freqs, weights, sumw = roots_legendre(order, mu=True)
    print(" ....Computing rpa_sosex")
    iEcRPA, iEcSOSEX = (
        0,
        0,
    )  # rpa_sosex(freqs,weights,sumw,order,wmn_at,eig,pqrt,pqrt_at,bigomega,nab,p.nbeta,p.nalpha,p.nbf)

    iEcRPASOS = iEcRPA + iEcSOSEX

    # Eq. (26)
    print(" ....Computing ccsd")
    pqrt_at = np.einsum("pqsr->sqpr", pqrt_at, optimize=True)
    pqrt = np.einsum("pqsr->sqpr", pqrt, optimize=True)
    EcCCSD = 0  # ccsd_eq(eig,pqrt,pqrt_at,p.nbeta,p.nalpha,p.nbf)

    ECndHF, ECndl = ECorrNonDyn(n, C, H, I, b_mnl, p)

    ESD = EHFL + E_nuc + ECndHF
    ESDc = ESD + ECndl
    EPNOF = E_elec + E_nuc

    print(" E(SD)                = {: f}".format(ESD))
    print(" E(SD+ND)             = {: f}".format(ESDc))
    print(" E(PNOFi)             = {: f}".format(EPNOF))
    print("")
    print(" Ec(ND)               = {: f}".format(ECndl))
    print(" Ec(RPA-FURCHE)       = {: f}".format(EcRPA))
    print(" Ec(RPA)              = {: f}".format(iEcRPA))
    print(" Ec(AC-SOSEX)         = {: f}".format(iEcSOSEX))
    print(" Ec(RPA+AC-SOSEX)     = {: f}".format(iEcRPASOS))
    print(" Ec(GW@GM)            = {: f}".format(EcGoWo))
    print(" Ec(SOSEX@GM)         = {: f}".format(EcGMSOS))
    print(" Ec(GW@GM+SOSEX@GM)   = {: f}".format(EcGoWoSOS))
    print(" Ec(MP2)              = {: f}".format(EcMP2))
    print(" Ec(CCSD)             = {: f}".format(EcCCSD))
    print("")
    print(" E(RPA-FURCHE)       = {: f}".format(ESD + EcRPA))
    print(" E(RPA)              = {: f}".format(ESD + iEcRPA))
    print(" E(RPA+AC-SOSEX)     = {: f}".format(ESD + iEcRPASOS))
    print(" E(GW@GM)            = {: f}".format(ESD + EcGoWo))
    print(" E(SOSEX@GM)         = {: f}".format(ESD + EcGMSOS))
    print(" E(GW@GM+SOSEX@GM)   = {: f}".format(ESD + EcGoWoSOS))
    print(" E(MP2)              = {: f}".format(ESD + EcMP2))
    print(" E(CCSD)             = {: f}".format(ESD + EcCCSD))
    print("")
    print(" E(NOF-c-RPA-FURCHE)       = {: f}".format(ESDc + EcRPA))
    print(" E(NOF-c-RPA)              = {: f}".format(ESDc + iEcRPA))
    print(" E(NOF-c-RPA+AC+SOSEX)     = {: f}".format(ESDc + iEcRPASOS))
    print(" E(NOF-c-GW@GM)            = {: f}".format(ESDc + EcGoWo))
    print(" E(NOF-c-SOSEX@GM)         = {: f}".format(ESDc + EcGMSOS))
    print(" E(NOF-c-GW@GM+SOSEX@GM)   = {: f}".format(ESDc + EcGoWoSOS))
    print(" E(NOF-c-MP2)              = {: f}".format(ESDc + EcMP2))
    print(" E(NOF-c-CCSD)             = {: f}".format(ESDc + EcCCSD))

    print("")

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2 - t1))


def nofmp2(n, C, H, I, b_mnl, E_nuc, p):


    print("================= Warning ====================")
    print("    The NOF-MP2 requires a revision due to    ")
    print("    new orbital sorting in the subspaces      ")
    print("==============================================")

    t1 = time()

    print(" NOF-MP2")
    print("=========")

    occ = n[p.no1 : p.nbf5]
    vec = C[:, p.no1 : p.nbf]

    D = pynof.computeD_HF(C, p)
    if p.MSpin == 0:
        if p.nsoc > 0:
            Dalpha = pynof.computeDalpha_HF(C, p)
            D = D + 0.5 * Dalpha
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = H + 2 * J - K
        EHFL = np.trace(np.matmul(D, H) + np.matmul(D, F))
    elif not p.MSpin == 0:
        Dalpha = pynof.computeDalpha_HF(C, p)
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = 2 * J - K
        EHFL = 2 * np.trace(np.matmul(D + 0.5 * Dalpha, H)) + np.trace(np.matmul(D + Dalpha, F))
        F = H + F
        if p.nsoc > 1:
            J, K = pynof.computeJK_HF(0.5 * Dalpha, I, b_mnl, p)
            Falpha = J - K
            EHFL = EHFL + 2 * np.trace(np.matmul(0.5 * Dalpha, Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(vec), F), vec)

    eig = np.einsum("ii->i", F_MO[: p.nbf - p.no1, : p.nbf - p.no1])

    iajb = pynof.compute_iajb(C, I, b_mnl, p)
    FI1 = np.ones(p.nbf - p.no1)
    FI2 = np.ones(p.nbf - p.no1)

    FI1[: p.nbf5 - p.no1] = 1 - (1 - abs(1 - 2 * occ[: p.nbf5 - p.no1])) ** 2

    FI2[p.nalpha - p.no1 : p.nbf5 - p.no1] = abs(1 - 2 * occ[p.nalpha - p.no1 : p.nbf5 - p.no1]) ** 2

    Tijab = CalTijab(iajb, F_MO, eig, FI1, FI2, p)
    ECd = 0
    for k in range(p.nvir):
        for l in range(p.nvir):
            for i in range(p.ndoc):
                for j in range(p.ndoc):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    ECd = ECd + Xijkl * (2 * Tijab[ijkl] - Tijab[ijlk])
                for j in range(p.ndoc, p.ndns):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    ECd = ECd + Xijkl * (Tijab[ijkl] - 0.5 * Tijab[ijlk])
            for i in range(p.ndoc, p.ndns):
                for j in range(p.ndoc):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    ECd = ECd + Xijkl * (Tijab[ijkl] - 0.5 * Tijab[ijlk])
                for j in range(p.ndoc, p.ndns):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    if j != i:
                        ECd = ECd + Xijkl * (Tijab[ijkl] - 0.5 * Tijab[ijlk]) / 2

    fi = 2 * n * (1 - n)

    CK12nd = np.outer(fi, fi)

    beta = np.sqrt((1 - abs(1 - 2 * n)) * n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + (p.ndoc - l - 1)
        ul = ll + p.ncwo * p.ndoc
        CK12nd[p.no1 + l, ll:ul:p.ndoc] = beta[p.no1 + l] * beta[ll:ul:p.ndoc]
        CK12nd[ll:ul:p.ndoc, p.no1 + l] = beta[ll:ul:p.ndoc] * beta[p.no1 + l]
        CK12nd[ll:ul:p.ndoc, ll:ul:p.ndoc] = -np.outer(beta[ll:ul:p.ndoc], beta[ll:ul:p.ndoc])

    # C^K KMO
    J_MO, K_MO, H_core = pynof.computeJKH_MO(C, H, I, b_mnl, p)

    ECndHF = 0
    ECndl = 0
    if p.MSpin == 0:
        ECndHF = -np.einsum(
            "ii,ii->", CK12nd[p.nbeta : p.nalpha, p.nbeta : p.nalpha], K_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha]
        )  # sum_ij
        ECndl -= np.einsum("ij,ji->", CK12nd, K_MO)  # sum_ij
        ECndl += np.einsum("ii,ii->", CK12nd, K_MO)  # Quita i=j
    elif not p.MSpin == 0:
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # Quita i=j
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # Quita i=j

    print("      Ehfc      = {:f}".format(EHFL + E_nuc + ECndHF))
    print("")
    print("      ECd       = {:f}".format(ECd))
    print("      ECnd      = {:f}".format(ECndl))
    print("      Ecorre    = {:f}".format(ECd + ECndl))
    print("      E(NOFMP2) = {:f}".format(EHFL + ECd + ECndl + E_nuc + ECndHF))
    print("")

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2 - t1))


def CalTijab(iajb, F_MO, eig, FI1, FI2, p):

    print("Starting CalTijab")

    B = build_B(iajb, FI1, FI2, p.ndoc, p.ndns, p.nvir, p.ncwo)
    print("....B vector Computed")

    Tijab = Tijab_guess(iajb, eig, p.ndoc, p.ndns, p.nvir)
    print("....Tijab Guess Computed")

    # A_CSR = csr_matrix(build_A(F_MO,FI1,FI2,p.no1,p.ndoc,p.ndns,p.nvir,p.ncwo,p.nbf))
    # print("A matrix has {}/{} elements with Tol = {}".format(len(A),p.nvir**4*p.ndoc**4,1e-10))
    # Tijab = solve_Tijab(A_CSR,B,Tijab,p)

    res = root(build_R, Tijab, args=(B, F_MO, FI1, FI2, p.no1, p.ndoc, p.ndns, p.nvir, p.ncwo, p.nbf), method="krylov")
    if res.success:
        print("....Tijab found as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
    else:
        print("....WARNING! Tijab NOT FOUND as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
        print(res)
    Tijab = res.x
    print("")

    return Tijab


@njit(parallel=True, cache=True)
def build_A(F_MO, FI1, FI2, no1, ndoc, ndns, nvir, ncwo, nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = (ndoc - i - 1)
        ul = ll + ncwo * ndoc 
        npair[ll:ul:ndoc] = i + 1

    A = np.empty((2 * ndns**2 * nvir**2 * (nbf - no1)))
    IROW = np.empty((2 * ndns**2 * nvir**2 * (nbf - no1)), dtype=np.int32)
    ICOL = np.empty((2 * ndns**2 * nvir**2 * (nbf - no1)), dtype=np.int32)

    nnz = -1
    for ib in range(nvir):
        for ia in range(nvir):
            for j in range(ndns):
                for i in range(ndns):
                    # print(nnz)
                    jab = (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    iab = i + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    ijb = i + (j) * ndns + (ib) * ndns * ndns * nvir
                    ija = i + (j) * ndns + (ia) * ndns * ndns
                    ijab = i + (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir

                    nnz = nnz + 1
                    A[nnz] = F_MO[ia + ndns, ia + ndns] + F_MO[ib + ndns, ib + ndns] - F_MO[i, i] - F_MO[j, j]
                    IROW[nnz] = ijab
                    ICOL[nnz] = i + jab

                    for k in range(i):
                        if abs(F_MO[i, k]) > 1e-10:
                            Cki = FI2[k] * FI2[i]
                            nnz += 1
                            A[nnz] = -Cki * F_MO[i, k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k + jab
                            nnz += 1
                            A[nnz] = -Cki * F_MO[i, k]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k + jab

                    for k in range(j):
                        if abs(F_MO[j, k]) > 1e-10:
                            Ckj = FI2[k] * FI2[j]
                            nnz += 1
                            A[nnz] = -Ckj * F_MO[j, k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k * ndns + iab
                            nnz += 1
                            A[nnz] = -Ckj * F_MO[j, k]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k * ndns + iab

                    for k in range(ia):
                        if abs(F_MO[ia + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ia]:
                                Ckia = FI1[k + ndns] * FI1[ia + ndns]
                            else:
                                Ckia = FI2[k + ndns] * FI2[ia + ndns]
                            nnz += 1
                            A[nnz] = Ckia * F_MO[ia + ndns, k + ndns]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k * ndns * ndns + ijb
                            nnz += 1
                            A[nnz] = Ckia * F_MO[ia + ndns, k + ndns]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k * ndns * ndns + ijb

                    for k in range(ib):
                        if abs(F_MO[ib + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ib]:
                                Ckib = FI1[k + ndns] * FI1[ib + ndns]
                            else:
                                Ckib = FI2[k + ndns] * FI2[ib + ndns]
                            nnz += 1
                            A[nnz] = Ckib * F_MO[ib + ndns, k + ndns]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k * ndns * ndns * nvir + ija
                            nnz += 1
                            A[nnz] = Ckib * F_MO[ib + ndns, k + ndns]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k * ndns * ndns * nvir + ija

    A = A[: nnz + 1]
    IROW = IROW[: nnz + 1]
    ICOL = ICOL[: nnz + 1]

    return A, (IROW, ICOL)


@njit(parallel=True, cache=True)
def build_R(T, B, F_MO, FI1, FI2, no1, ndoc, ndns, nvir, ncwo, nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = (ndoc - i - 1)
        ul = ll + ncwo * ndoc
        npair[ll:ul:ndoc] = i + 1

    Bp = np.zeros((ndns**2 * nvir**2))

    for ib in prange(nvir):
        for ia in prange(nvir):
            for j in prange(ndns):
                for i in prange(ndns):
                    jab = (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    iab = i + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    ijb = i + (j) * ndns + (ib) * ndns * ndns * nvir
                    ija = i + (j) * ndns + (ia) * ndns * ndns
                    ijab = i + (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir

                    Bp[ijab] += (F_MO[ia + ndns, ia + ndns] + F_MO[ib + ndns, ib + ndns] - F_MO[i, i] - F_MO[j, j]) * T[
                        i + jab
                    ]

                    for k in range(i):
                        if abs(F_MO[i, k]) > 1e-10:
                            Cki = FI2[k] * FI2[i]
                            Bp[ijab] += (-Cki * F_MO[i, k]) * T[k + jab]
                    for k in range(i + 1, ndns):
                        if abs(F_MO[i, k]) > 1e-10:
                            Cki = FI2[k] * FI2[i]
                            Bp[ijab] += (-Cki * F_MO[i, k]) * T[k + jab]

                    for k in range(j):
                        if abs(F_MO[j, k]) > 1e-10:
                            Ckj = FI2[k] * FI2[j]
                            Bp[ijab] += (-Ckj * F_MO[j, k]) * T[k * ndns + iab]
                    for k in range(j + 1, ndns):
                        if abs(F_MO[j, k]) > 1e-10:
                            Ckj = FI2[k] * FI2[j]
                            Bp[ijab] += (-Ckj * F_MO[j, k]) * T[k * ndns + iab]

                    for k in range(ia):
                        if abs(F_MO[ia + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ia]:
                                Ckia = FI1[k + ndns] * FI1[ia + ndns]
                            else:
                                Ckia = FI2[k + ndns] * FI2[ia + ndns]
                            Bp[ijab] += (Ckia * F_MO[ia + ndns, k + ndns]) * T[k * ndns * ndns + ijb]
                    for k in range(ia + 1, nvir):
                        if abs(F_MO[ia + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ia]:
                                Ckia = FI1[k + ndns] * FI1[ia + ndns]
                            else:
                                Ckia = FI2[k + ndns] * FI2[ia + ndns]
                            Bp[ijab] += (Ckia * F_MO[ia + ndns, k + ndns]) * T[k * ndns * ndns + ijb]

                    for k in range(ib):
                        if abs(F_MO[ib + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ib]:
                                Ckib = FI1[k + ndns] * FI1[ib + ndns]
                            else:
                                Ckib = FI2[k + ndns] * FI2[ib + ndns]
                            Bp[ijab] += (Ckib * F_MO[ib + ndns, k + ndns]) * T[k * ndns * ndns * nvir + ija]
                    for k in range(ib + 1, nvir):
                        if abs(F_MO[ib + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ib]:
                                Ckib = FI1[k + ndns] * FI1[ib + ndns]
                            else:
                                Ckib = FI2[k + ndns] * FI2[ib + ndns]
                            Bp[ijab] += (Ckib * F_MO[ib + ndns, k + ndns]) * T[k * ndns * ndns * nvir + ija]

    R = B - Bp
    return R


@njit(parallel=True, cache=True)
def build_B(iajb, FI1, FI2, ndoc, ndns, nvir, ncwo):
    B = np.zeros((ndns**2 * nvir**2))
    for i in prange(ndns):
        lmin_i = ndns + (ndns - i - 1)
        lmax_i = lmin_i + ncwo * ndoc
        for j in range(ndns):
            if i == j:
                for k in range(nvir):
                    ik = i + k * ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        if lmin_i <= kn and kn < lmax_i and lmin_i <= ln and ln < lmax_i:
                            Ciikl = FI1[kn] * FI1[ln] * FI1[i] * FI1[i]
                        else:
                            Ciikl = FI2[kn] * FI2[ln] * FI2[i] * FI2[i]
                        iikl = i + i * ndns + k * ndns * ndns + l * ndns * ndns * nvir
                        B[iikl] = -Ciikl * iajb[i, k, i, l]
            else:
                for k in range(nvir):
                    ik = i + k * ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        ijkl = i + j * ndns + k * ndns * ndns + l * ndns * ndns * nvir
                        Cijkl = FI2[kn] * FI2[ln] * FI2[i] * FI2[j]
                        B[ijkl] = -Cijkl * iajb[j, k, i, l]
    return B


@njit(parallel=True, cache=True)
def Tijab_guess(iajb, eig, ndoc, ndns, nvir):
    Tijab = np.zeros(nvir**2 * ndns**2)
    for ia in prange(nvir):
        for i in range(ndns):
            for ib in range(nvir):
                for j in range(ndns):
                    ijab = i + (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    Eijab = eig[ib + ndns] + eig[ia + ndns] - eig[j] - eig[i]
                    Tijab[ijab] = -iajb[j, ia, i, ib] / Eijab
    return Tijab


def solve_Tijab(A_CSR, B, Tijab, p):
    Tijab, info = cg(A_CSR, B, x0=Tijab)
    return Tijab


def ext_koopmans(p, elag, n):
    elag_small = elag[: p.nbf5, : p.nbf5]
    nu = -np.einsum("qp,q,p->qp", elag_small, 1 / np.sqrt(n), 1 / np.sqrt(n))

    print("")
    print("---------------------------")
    print(" Extended Koopmans Theorem ")
    print("   Ionization Potentials   ")
    print("---------------------------")

    eigval, eigvec = np.linalg.eigh(nu)
    print(" OM        (eV)")
    print("---------------------------")
    for i, val in enumerate(eigval[::-1]):
        print("{: 3d}      {: 7.3f}".format(i, val * 27.2114))

    print("")
    print("EKT IP: {: 7.3f} eV".format(eigval[0] * 27.2114))
    print("")


def mulliken_pop(p, wfn, n, C, S):

    nPS = 2 * np.einsum("i,mi,ni,nm->m", n, C[:, : p.nbf5], C[:, : p.nbf5], S, optimize=True)

    pop = np.zeros((p.natoms))

    for mu in range(C.shape[0]):
        iatom = wfn.basisset().function_to_center(mu)
        pop[iatom] += nPS[mu]

    print("")
    print("---------------------------------")
    print("  Mulliken Population Analysis   ")
    print("---------------------------------")
    print(" Idx  Atom   Population   Charge ")
    print("---------------------------------")
    for iatom in range(p.natoms):
        symbol = wfn.molecule().flabel(iatom)
        print(
            "{: 3d}    {:2s}    {: 5.2f}      {: 5.2f}".format(
                iatom, symbol, pop[iatom], wfn.molecule().Z(iatom) - pop[iatom]
            )
        )
    print("")


def lowdin_pop(p, wfn, n, C, S):

    S_12 = fractional_matrix_power(S, 0.5)

    S_12nPS_12 = 2 * np.einsum("sm,i,mi,ni,ns->s", S_12, n, C[:, : p.nbf5], C[:, : p.nbf5], S_12, optimize=True)

    pop = np.zeros((p.natoms))

    for mu in range(C.shape[0]):
        iatom = wfn.basisset().function_to_center(mu)
        pop[iatom] += S_12nPS_12[mu]

    print("")
    print("---------------------------------")
    print("   Lowdin Population Analysis    ")
    print("---------------------------------")
    print(" Idx  Atom   Population   Charge ")
    print("---------------------------------")
    for iatom in range(p.natoms):
        symbol = wfn.molecule().flabel(iatom)
        print(
            "{: 3d}    {:2s}    {: 5.2f}      {: 5.2f}".format(
                iatom, symbol, pop[iatom], wfn.molecule().Z(iatom) - pop[iatom]
            )
        )
    print("")


def M_diagnostic(p, n):

    m_vals = 2 * n
    if p.HighSpin:
        m_vals[p.nbeta : p.nalpha] = n[p.nbeta : p.nalpha]

    m_diagnostic = 0

    m_vals[p.no1 : p.nbeta] = 2.0 - m_vals[p.no1 : p.nbeta]

    if p.ndoc > 0:
        m_diagnostic += max(m_vals[p.no1 : p.nbeta])

    m_vals[p.nalpha : p.nbf5] = m_vals[p.nalpha : p.nbf5] - 0.0
    if p.ndoc > 0:
        m_diagnostic += max(m_vals[p.nalpha : p.nbf5])

    m_diagnostic = 0.5 * m_diagnostic

    print("")
    print("---------------------------------")
    print("   M Diagnostic: {:4.2f} ".format(m_diagnostic))
    print("---------------------------------")
    print("")


def matrix_product(x, M, V):

    w = x[0]
    y = x[1:]

    MM = M - w * V
    vec = np.matmul(MM, y)

    return np.linalg.norm(vec)


def ERPA(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, pp):

    time1 = time()

    print("\n---------------")
    print(" ERPA Analysis")
    print("---------------\n")

    tol_dn = 10**-3

    h = np.einsum("mi,mn,nj->ij", C[:, 0 : pp.nbf5], H, C[:, 0 : pp.nbf5], optimize=True)

    if pp.RI:
        b_pql = np.einsum("mnl,mp,nq->pql", b_mnl, C[:, 0 : pp.nbf5], C[:, 0 : pp.nbf5], optimize=True)
        I = np.einsum("pqR,slR->pqsl", b_pql, b_pql, optimize=True)
    else:
        I = np.einsum(
            "mp,nq,mnab,as,bl->pqsl",
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            I,
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            optimize=True,
        )
    if pp.gpu:
        I = I.get()

    c = np.sqrt(n)
    c[pp.no1 + pp.ndns :] *= -1

    I = np.einsum("rpsq->rspq", I, optimize=True)

    A = np.zeros((pp.nbf5, pp.nbf5, pp.nbf5, pp.nbf5))
    Id = np.identity(pp.nbf5)

    A += np.einsum("sq,pr,p->rspq", h, Id, n, optimize=True)
    A -= np.einsum("sq,pr,s->rspq", h, Id, n, optimize=True)
    A += np.einsum("pr,sq,q->rspq", h, Id, n, optimize=True)
    A -= np.einsum("pr,sq,r->rspq", h, Id, n, optimize=True)

    Daa, Dab = pynof.compute_2RDM(pp, n)

    time2 = time()

    ######################################
    A += np.einsum("stqu,purt->rspq", I, Daa, optimize=True)
    A -= np.einsum("stuq,purt->rspq", I, Daa, optimize=True)
    A += np.einsum("stqu,purt->rspq", I, Dab, optimize=True)
    A += np.einsum("stuq,putr->rspq", I, Dab, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Daa, optimize=True)
    A -= np.einsum("uprt,stqu->rspq", I, Daa, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Dab, optimize=True)
    A += np.einsum("uprt,stuq->rspq", I, Dab, optimize=True)

    time3 = time()

    ####
    A += np.einsum("pstu,turq->rspq", I, Daa, optimize=True)
    A -= np.einsum("pstu,utrq->rspq", I, Dab, optimize=True)
    A += np.einsum("tuqr,sptu->rspq", I, Daa, optimize=True)
    A -= np.einsum("tuqr,pstu->rspq", I, Dab, optimize=True)
    ####
    time4 = time()

    A += np.einsum("sq,tpwu,wurt->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("sq,tpwu,uwrt->rspq", Id, I, Dab, optimize=True)
    A += np.einsum("pr,tuwq,swtu->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("pr,tuwq,wstu->rspq", Id, I, Dab, optimize=True)

    M = np.zeros((pp.nbf5**2, pp.nbf5**2))

    # i is rows
    # j is columns

    time5 = time()

    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time6 = time()

    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time7 = time()

    for r in range(pp.nbf5):
        i += 1
        j = -1
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, p, q]
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, q, p]
        for p in range(pp.nbf5):
            j += 1
            M[i, j] = A[r, r, p, p]

    v = np.zeros((pp.nbf5 * (pp.nbf5 - 1)))
    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = +(n[s] - n[r])
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = -(n[s] - n[r])

    idx = []
    for i, vi in enumerate(v):
        if np.abs(vi) < tol_dn:
            idx.append(i)

    dim = (pp.nbf5) * (pp.nbf5 - 1) - len(idx)

    for i, j in enumerate(idx):
        tmp = v[j - i]
        v[j - i : -1] = v[j - i + 1 :]
        v[-1] = tmp

        tmp = M[j - i, :].copy()
        M[j - i : -1, :] = M[j - i + 1 :, :]
        M[-1, :] = tmp
        tmp = M[:, j - i].copy()
        M[:, j - i : -1] = M[:, j - i + 1 :]
        M[:, -1] = tmp

    print("M_ERPA Orig dim: {} New dim: {} Elements below tol_dN: {}".format((pp.nbf5) * (pp.nbf5 - 1), dim, len(idx)))

    ######## ERPA0 ########

    dd = int(dim / 2)
    AA = M[:dd, :dd]
    BB = M[:dd, dd : int(2 * dd)]

    ApB = AA + BB
    AmB = AA - BB

    dN = np.zeros((dd, dd))
    np.fill_diagonal(dN, v[:dd])
    dNm1 = np.linalg.pinv(dN)

    maxApBsym = np.max(np.abs(ApB - ApB.T))
    maxAmBsym = np.max(np.abs(AmB - AmB.T))
    print("Max diff ApB {:3.1e} and Max AmB {:3.1e}".format(maxApBsym, maxAmBsym))

    MM = np.einsum("ij,jk,kl,lm->im", dNm1, ApB, dNm1, AmB, optimize=True)
    vals = np.linalg.eigvals(MM)
    vals = np.sqrt(vals)

    vals = vals * 27.2114
    vals_complex = np.array([val for val in vals if (np.abs(np.imag(val)) > 0.00000001)])

    print("  Excitation energies ERPA0/PNOF{}(eV)".format(pp.ipnof))
    print("  ===================================")
    vals_real = np.array([np.real(val) for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 1.0)])

    sort_idx = np.argsort(vals_real)
    vals_real = vals_real[sort_idx]
    for i in range(min(10, len(vals_real))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals_real[i]))
    print("    Vals Complex: {}\n".format(np.size(vals_complex)))

    time8 = time()

    ######## ERPA  ########

    dd = int(dim / 2)
    AA = M[:dd, :dd]
    BB = M[:dd, dd : int(2 * dd)]

    ApB = AA + BB
    AmB = AA - BB

    dN = np.zeros((dd, dd))
    np.fill_diagonal(dN, v[:dd])
    dNm1 = np.linalg.pinv(dN)

    CC = M[:dd, int(2 * dd) : int(2 * dd) + pp.nbf5]
    EE = M[int(2 * dd) : int(2 * dd) + pp.nbf5, :dd]
    FF = M[int(2 * dd) : int(2 * dd) + pp.nbf5, int(2 * dd) : int(2 * dd) + pp.nbf5]

    FFm1 = np.linalg.inv(FF)

    tmpMat = 2 * np.einsum("ij,jk,kl->il", CC, FFm1, EE, optimize=True)

    MM = np.einsum("ij,jk,kl,lm->im", dNm1, ApB - tmpMat, dNm1, AmB, optimize=True)
    vals = np.linalg.eigvals(MM)
    vals = np.sqrt(vals)

    vals = vals * 27.2114
    vals_complex = np.array([val for val in vals if (np.abs(np.imag(val)) > 0.00000001)])

    print("  Excitation energies ERPA/PNOF{}(eV)".format(pp.ipnof))
    print("  ===================================")
    vals_real = np.sort(
        np.array([np.real(val) for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 1.0)])
    )
    for i in range(min(10, len(vals_real))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals_real[i]))
    print("    Vals Complex: {}\n".format(np.size(vals_complex)))

    time9 = time()

    ######## ERPA2 ########

    # CAMBIO ERPA2-2: corte de deflacion propio para ERPA2, independiente
    # del tol_dn=1e-3 compartido por ERPA0/ERPA1 (que se deja intacto para
    # no alterar esos resultados).  Un tol_dn2 algo mayor descarta mas
    # pares casi degenerados del bloque fuera-de-diagonal antes de
    # contraer con V^-1, evitando amplificar ruido numerico en esa
    # contraccion.  Configurable via pp.erpa2_tol_dn.
    tol_dn2 = getattr(pp, "erpa2_tol_dn", 3e-3)
    # CAMBIO ERPA2-1: epsilon de regularizacion (Tikhonov) para las
    # divisiones por c[r], c[p] y c[r]+c[s] -- ver _safe_inv_reg.  Solo
    # afectan a las columnas "diagonales" (p=q) de ERPA2, que no existen en
    # ERPA0/ERPA1.  Configurable via pp.erpa2_reg_eps.
    eps2 = getattr(pp, "erpa2_reg_eps", 1e-4)

    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p] * 0.5 * c[s] * _safe_inv_reg(c[p], eps2) * _safe_inv_reg(c[r] + c[s], eps2)

    time10 = time()

    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p] * 0.5 * c[r] * _safe_inv_reg(c[p], eps2) * _safe_inv_reg(c[r] + c[s], eps2)

    time11 = time()

    for r in range(pp.nbf5):
        i += 1
        j = -1
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, p, q] * _safe_inv_reg(c[r], eps2)
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, q, p] * _safe_inv_reg(c[r], eps2)
        for p in range(pp.nbf5):
            j += 1
            M[i, j] = A[r, r, p, p] * 0.25 * _safe_inv_reg(c[p], eps2) * _safe_inv_reg(c[r], eps2)

    v = np.zeros((pp.nbf5**2))
    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = -(n[r] - n[s])
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = n[r] - n[s]
    for r in range(pp.nbf5):
        i += 1
        v[i] = 1

    lidx = []
    for i, vi in enumerate(v):
        if np.abs(vi) < tol_dn2:
            lidx.append(i)

    dim = pp.nbf5**2 - len(lidx)

    for i, j in enumerate(lidx):
        tmp = v[j - i]
        v[j - i : -1] = v[j - i + 1 :]
        v[-1] = tmp

        tmp = M[j - i, :].copy()
        M[j - i : -1, :] = M[j - i + 1 :, :]
        M[-1, :] = tmp
        tmp = M[:, j - i].copy()
        M[:, j - i : -1] = M[:, j - i + 1 :]
        M[:, -1] = tmp

    M_ERPA2 = M[:dim, :dim]

    time12 = time()

    M_ERPA2 = np.real(M_ERPA2)

    # CAMBIO ERPA2-3: transformacion de congruencia simetrica en vez de la
    # division por filas M[i,:]/v[i] original, para mejorar el
    # condicionamiento SIN alterar el problema de autovalores.
    #
    # El problema original es el generalizado V^-1 M x = lambda x, con
    # V = diag(v) indefinida (v cambia de signo) y M_ERPA2 NO simetrica por
    # construccion (a diferencia de ApB/AmB en ERPA0/ERPA1): se comprobo
    # empiricamente que max|M-M.T| es ~1-3% de max|M|, es decir, la parte
    # antisimetrica codifica informacion real (no es ruido de redondeo), y
    # forzar M <- (M+M.T)/2 antes de resolver cambia el espectro de forma
    # apreciable (aparecen autovalores bajos espureos que no estaban antes).
    # Por eso NO se simetriza el operador usado para los autovalores.
    #
    # Lo que si se hace es escribir V = D S D con D = diag(sqrt(|v|)) y
    # S = diag(sign(v)), y = D x; entonces D (V^-1 M) D^-1 = S (D^-1 M D^-1)
    # es una transformacion de SEMEJANZA (conserva EXACTAMENTE los
    # autovalores de V^-1 M) pero reparte la escala de forma simetrica
    # entre filas y columnas en vez de la division unilateral por filas
    # M[i,:]/v[i] original, que amplifica el ruido de las direcciones con
    # |v[i]| pequeno solo por el lado de las filas.
    v_dim = v[:dim]
    dinv = 1.0 / np.sqrt(np.abs(v_dim))
    Mtilde = (dinv[:, None] * M_ERPA2) * dinv[None, :]
    sign_v = np.sign(v_dim)

    # CAMBIO ERPA2-4: diagnostico de (in)estabilidad -- ver
    # _erpa2_report_instability.  Usa solo la parte simetrica de Mtilde como
    # heuristica barata (no es un teorema exacto para ERPA2, ya que Mtilde
    # no es simetrica; sirve como proxy de en que geometrias el problema
    # esta peor condicionado / la referencia NOF es mas sospechosa).
    Msym_diag = 0.5 * (Mtilde + Mtilde.T)
    _erpa2_report_instability(Msym_diag, sign_v, pp)

    M_ERPA2_eff = sign_v[:, None] * Mtilde

    time13 = time()

    vals = np.linalg.eigvals(M_ERPA2_eff)

    vals = vals * 27.2114
    vals_complex = np.array([val for val in vals if (np.abs(np.imag(val)) > 0.00000001)])

    time14 = time()

    print("  Excitation energies ERPA2/PNOF{}(eV)".format(pp.ipnof))
    print("  ==================================")
    vals_real = np.sort(
        np.array([np.real(val) for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 1.0)])
    )
    for i in range(min(11, len(vals_real))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals_real[i]))
    print("\n    Number of small diagonal elements of V moved to the end:", len(lidx))
    print("    Vals Complex: {}\n".format(np.size(vals_complex)))

    time15 = time()

    # Opt

    times = [
        time1,
        time2,
        time3,
        time4,
        time5,
        time6,
        time7,
        time8,
        time9,
        time10,
        time11,
        time12,
        time13,
        time14,
        time15,
    ]

    time_total = times[-1] - times[0]

    print("Δt            s              %")
    print("=================================")

    for i in range(len(times) - 1):
        d = times[i + 1] - times[i]
        print("t{j:>} - t{i:>}:   {d:3.1e}   ...   {p:4.1f}".format(i=i + 1, j=i + 2, d=d, p=d * 100 / time_total))


def iterative_ERPA0(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, number_of_excitations, pp):

    time1 = time()

    print("\n---------------")
    print(" ERPA Analysis")
    print("---------------\n")

    tol_dn = 10**-3

    h = np.einsum("mi,mn,nj->ij", C[:, 0 : pp.nbf5], H, C[:, 0 : pp.nbf5], optimize=True)

    if pp.RI:
        b_pql = np.einsum("mnl,mp,nq->pql", b_mnl, C[:, 0 : pp.nbf5], C[:, 0 : pp.nbf5], optimize=True)
        I = np.einsum("pqR,slR->pqsl", b_pql, b_pql, optimize=True)
    else:
        I = np.einsum(
            "mp,nq,mnab,as,bl->pqsl",
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            I,
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            optimize=True,
        )
    if pp.gpu:
        I = I.get()

    c = np.sqrt(n)
    c[pp.no1 + pp.ndns :] *= -1

    I = np.einsum("rpsq->rspq", I, optimize=True)

    A = np.zeros((pp.nbf5, pp.nbf5, pp.nbf5, pp.nbf5))
    Id = np.identity(pp.nbf5)

    A += np.einsum("sq,pr,p->rspq", h, Id, n, optimize=True)
    A -= np.einsum("sq,pr,s->rspq", h, Id, n, optimize=True)
    A += np.einsum("pr,sq,q->rspq", h, Id, n, optimize=True)
    A -= np.einsum("pr,sq,r->rspq", h, Id, n, optimize=True)

    Daa, Dab = pynof.compute_2RDM(pp, n)

    time2 = time()

    ######################################
    A += np.einsum("stqu,purt->rspq", I, Daa, optimize=True)
    A -= np.einsum("stuq,purt->rspq", I, Daa, optimize=True)
    A += np.einsum("stqu,purt->rspq", I, Dab, optimize=True)
    A += np.einsum("stuq,putr->rspq", I, Dab, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Daa, optimize=True)
    A -= np.einsum("uprt,stqu->rspq", I, Daa, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Dab, optimize=True)
    A += np.einsum("uprt,stuq->rspq", I, Dab, optimize=True)

    time3 = time()

    ####
    A += np.einsum("pstu,turq->rspq", I, Daa, optimize=True)
    A -= np.einsum("pstu,utrq->rspq", I, Dab, optimize=True)
    A += np.einsum("tuqr,sptu->rspq", I, Daa, optimize=True)
    A -= np.einsum("tuqr,pstu->rspq", I, Dab, optimize=True)
    ####
    time4 = time()

    A += np.einsum("sq,tpwu,wurt->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("sq,tpwu,uwrt->rspq", Id, I, Dab, optimize=True)
    A += np.einsum("pr,tuwq,swtu->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("pr,tuwq,wstu->rspq", Id, I, Dab, optimize=True)

    M = np.zeros((pp.nbf5**2, pp.nbf5**2))

    # i is rows
    # j is columns

    time5 = time()

    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time6 = time()

    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time7 = time()

    for r in range(pp.nbf5):
        i += 1
        j = -1
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, p, q]
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, q, p]
        for p in range(pp.nbf5):
            j += 1
            M[i, j] = A[r, r, p, p]

    v = np.zeros((pp.nbf5 * (pp.nbf5 - 1)))
    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = +(n[s] - n[r])
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = -(n[s] - n[r])

    idx = []
    for i, vi in enumerate(v):
        if np.abs(vi) < tol_dn:
            idx.append(i)

    i_start = 0  # start of hono/virtual coupling after tol_dn
    for s in range(pp.ndoc - 1):
        for r in range(s + 1, pp.nbf5):
            if np.abs(v[i]) < tol_dn:
                i_start += 1

    dim = (pp.nbf5) * (pp.nbf5 - 1) - len(idx)

    for i, j in enumerate(idx):
        tmp = v[j - i]
        v[j - i : -1] = v[j - i + 1 :]
        v[-1] = tmp

        tmp = M[j - i, :].copy()
        M[j - i : -1, :] = M[j - i + 1 :, :]
        M[-1, :] = tmp
        tmp = M[:, j - i].copy()
        M[:, j - i : -1] = M[:, j - i + 1 :]
        M[:, -1] = tmp

    print("M_ERPA Orig dim: {} New dim: {} Elements below tol_dN: {}".format((pp.nbf5) * (pp.nbf5 - 1), dim, len(idx)))

    ######## ERPA0 ########

    dd = int(dim / 2)
    AA = M[:dd, :dd]
    BB = M[:dd, dd : int(2 * dd)]

    ApB = AA + BB
    AmB = AA - BB

    dA = np.zeros((dd, dd))
    np.fill_diagonal(dA, A.diagonal())

    dN = np.zeros((dd, dd))
    np.fill_diagonal(dN, v[:dd])
    dNm1 = np.linalg.pinv(dN)

    maxApBsym = np.max(np.abs(ApB - ApB.T))
    maxAmBsym = np.max(np.abs(AmB - AmB.T))
    print("Max diff ApB {:3.1e} and Max AmB {:3.1e}".format(maxApBsym, maxAmBsym))
    print()

    l = pp.nbf5 - pp.ndoc
    bp = np.zeros((dd, l))
    bm = np.zeros((dd, l))

    for j in range(l):
        bp[i_start + j, j] = 1
        bm[i_start + j, j] = 1

    bp = pynof.orthonormalize(bp, ApB)
    bm = pynof.orthonormalize(bm, AmB)

    w_old = np.zeros((number_of_excitations))
    for k in range(100):
        converged = True
        print("==== Iter {} ====".format(k))
        print("Number of guess vectors: ", np.shape(bp)[1])
        l = np.shape(bp)[1]

        S = np.einsum("ji,jk,kl->il", bm, dN, bp, optimize=True)

        lambda2, up = eigh(np.matmul(S.T, S))
        sort_idx = np.argsort(lambda2)[::-1]
        lambda2, up = lambda2[sort_idx], up[:, sort_idx]
        omega = 1 / np.sqrt(lambda2)
        um = np.einsum("i,ij,jk->ik", 1 / omega, S, up)
        for i, val in enumerate(omega[:number_of_excitations]):
            print(" Exc. en {}: {:6.3f}".format(i, val * 27.2114))

        if np.max(np.abs(w_old - omega[:number_of_excitations])) > 0.001:
            converged = False
            w_old = omega[:number_of_excitations]
        if converged:
            break

        for i in range(number_of_excitations):
            Rp = np.einsum("ij,jk,k->i", dN, bm[:, :l], um[:, i]) - 1 / omega[i] * np.einsum(
                "ij,jk,k->i", ApB, bp[:, :l], up[:, i]
            )
            Rm = np.einsum("ij,jk,k->i", dN, bp[:, :l], up[:, i]) - 1 / omega[i] * np.einsum(
                "ij,jk,k->i", AmB, bm[:, :l], um[:, i]
            )

            b_p1 = -np.matmul(
                np.linalg.pinv(1 / omega[i] ** 2 * np.matmul(dA, dA) - np.matmul(dN, dN)),
                1 / omega[i] * np.matmul(dA, Rp) + np.matmul(dN, Rm),
            )
            b_m1 = -np.matmul(
                np.linalg.pinv(1 / omega[i] ** 2 * np.matmul(dA, dA) - np.matmul(dN, dN)),
                1 / omega[i] * np.matmul(dA, Rm) + np.matmul(dN, Rp),
            )

            bp = np.hstack((bp, b_p1.reshape(np.shape(bp)[0], -1)))
            bm = np.hstack((bm, b_m1.reshape(np.shape(bm)[0], -1)))
        bp = pynof.orthonormalize(bp, ApB)
        bm = pynof.orthonormalize(bm, AmB)

        # relevancep = np.zeros((number_of_excitations))
        # relevancem = np.zeros((number_of_excitations))
        # for i in range(n_guess_vectors):
        #    relevancep[i] = np.max(np.abs(up[i,:number_of_excitations]))
        #    relevancem[i] = np.max(np.abs(um[i,:number_of_excitations]))
        # relevancep_idx = np.argsort(relevancep)[::-1]
        # relevancem_idx = np.argsort(relevancem)[::-1]
        # bp = bp[:,relevancep_idx]
        # bm = bm[:,relevancem_idx]
        # bp = bp[:,:10]
        # bm = bm[:,:10]

    vals = w_old * 27.2114

    print()
    print("  Excitation energies ERPA0/PNOF{}(eV)".format(pp.ipnof))
    print("  ===================================")

    for i in range(min(10, len(vals))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals[i]))


def _erpa_transform_and_build_A(n, C, H, I, b_mnl, pp):
    """Transforma h e I a la base de NOs y ensambla el tensor A (nbf5^4) de
    ERPA. Bloque identico al de ERPA() / iterative_ERPA0. Devuelve A."""
    import numpy as np

    nb = pp.nbf5
    Cx = C[:, 0:nb]
    h = np.einsum("mi,mn,nj->ij", Cx, H, Cx, optimize=True)

    if pp.RI:
        b_pql = np.einsum("mnl,mp,nq->pql", b_mnl, Cx, Cx, optimize=True)
        I = np.einsum("pqR,slR->pqsl", b_pql, b_pql, optimize=True)
    else:
        I = np.einsum("mp,nq,mnab,as,bl->pqsl", Cx, Cx, I, Cx, Cx, optimize=True)
    if pp.gpu:
        I = I.get()

    I = np.einsum("rpsq->rspq", I, optimize=True)

    A = np.zeros((nb, nb, nb, nb))
    Id = np.identity(nb)

    A += np.einsum("sq,pr,p->rspq", h, Id, n, optimize=True)
    A -= np.einsum("sq,pr,s->rspq", h, Id, n, optimize=True)
    A += np.einsum("pr,sq,q->rspq", h, Id, n, optimize=True)
    A -= np.einsum("pr,sq,r->rspq", h, Id, n, optimize=True)

    Daa, Dab = pynof.compute_2RDM(pp, n)

    A += np.einsum("stqu,purt->rspq", I, Daa, optimize=True)
    A -= np.einsum("stuq,purt->rspq", I, Daa, optimize=True)
    A += np.einsum("stqu,purt->rspq", I, Dab, optimize=True)
    A += np.einsum("stuq,putr->rspq", I, Dab, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Daa, optimize=True)
    A -= np.einsum("uprt,stqu->rspq", I, Daa, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Dab, optimize=True)
    A += np.einsum("uprt,stuq->rspq", I, Dab, optimize=True)

    A += np.einsum("pstu,turq->rspq", I, Daa, optimize=True)
    A -= np.einsum("pstu,utrq->rspq", I, Dab, optimize=True)
    A += np.einsum("tuqr,sptu->rspq", I, Daa, optimize=True)
    A -= np.einsum("tuqr,pstu->rspq", I, Dab, optimize=True)

    A += np.einsum("sq,tpwu,wurt->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("sq,tpwu,uwrt->rspq", Id, I, Dab, optimize=True)
    A += np.einsum("pr,tuwq,swtu->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("pr,tuwq,wstu->rspq", Id, I, Dab, optimize=True)

    return A


def _erpa_deflated_pairs(n, pp, tol_dn):
    """Pares (r>s) que sobreviven a la deflacion por |DeltaN| < tol_dn, en el
    MISMO orden que el bucle 'for s: for r in s+1..' original de ERPA().
    Devuelve (rr, ss, dN_diag, dd, n_defl).

    CAMBIO A (continuidad del perfil): el corte 'tol_dn' se pasaba fijo a
    1e-3.  Como |DeltaN| de los pares debilmente ocupados cruza ese valor a
    angulos distintos al torsionar la molecula, la dimension del problema
    ERPA reducido saltaba a lo largo del barrido (p. ej. 1314 -> 1276 en
    C2H4) y con ella la energia de excitacion (escalones de 0.02-0.08 eV,
    ademas asimetricos entre las dos ramas 0-90 y 90-180).

    Ahora 'tol_dn' es configurable (pp.erpa_tol_dn, por defecto 1e-4).  Con
    1e-4 (vs el 1e-3 anterior) los pares debilmente ocupados dejan de
    entrar/salir del espacio activo al torsionar: la energia de excitacion
    pasa a ser mirror-simetrica a ~5 meV (antes ~80 meV) y sin escalones.
    Solo se descartan los pares de virtuales profundos (|DeltaN| ~ 1e-8),
    cuyo numero no depende de la geometria.  Un valor <= 0 desactiva la
    deflacion (dd = todos los pares); NO se recomienda bajar de ~1e-4: por
    debajo, las direcciones (casi) nulas de (A+B) hacen fallar la
    factorizacion de Cholesky de la lambda-ortogonalizacion incremental
    (potrf info=1) en geometrias fuertemente multiconfiguracionales
    (p. ej. C2H4 a 90 grados, M diagnostic ~ 1).
    """
    import numpy as np

    ss_all, rr_all = np.triu_indices(pp.nbf5, k=1)
    dN_all = n[ss_all] - n[rr_all]
    if tol_dn is None or tol_dn <= 0.0:
        keep = np.ones(dN_all.shape, dtype=bool)
    else:
        keep = np.abs(dN_all) >= tol_dn
    return (rr_all[keep], ss_all[keep], dN_all[keep].copy(),
            int(keep.sum()), int((~keep).sum()))


def _erpa_report_instability(ApB, AmB, etiqueta, pp):
    """CAMBIO C: aviso de inestabilidad de la referencia NOF.

    En un ERPA "sano" tanto (A-B) como (A+B)_eff son definidas positivas.
    Un autovalor negativo => la solucion NOF es un punto de silla y el
    problema ERPA tiene raices imaginarias.  El solver iterativo las oculta
    (usa eigh(S^T S), que fuerza omega^2 >= 0) y el filtro omega_min las
    descarta, de modo que el estado se reporta con un valor sin sentido
    (caso C2H4 a 90 grados en ERPA0).  Aqui solo se AVISA con el valor; no
    se aborta.  Se puede desactivar con pp.erpa_check_stability = False.
    """
    import numpy as np
    if not getattr(pp, "erpa_check_stability", True):
        return
    # eigvalsh denso pidiendo SOLO el autovalor mas bajo (driver 'evr' de
    # LAPACK): O(dd^2) para dd ~ 2000, ~1 s, y determinista.  Se probo
    # scipy.sparse.linalg.eigsh(which='SA') para evitar el coste, pero
    # ARPACK no converge en estas matrices (espectro agrupado, sin
    # shift-invert).  Desactivable con pp.erpa_check_stability = False.
    # Umbrales en Hartree.  El chequeo de "casi singular" se hace SOLO sobre
    # (A+B)_eff: (A-B) conserva de forma rutinaria un autovalor minusculo
    # (~1e-5 Ha) por las direcciones casi nulas que deja la deflacion, y no
    # es diagnostico.  En benigno min lambda(A+B) ~ 1e-3 Ha; en el punto
    # patologico (C2H4 a 90) cae a ~5e-7 Ha -> 2-3 ordenes de separacion.
    tol_neg = 1e-8     # < -tol_neg en (A+B) o (A-B) -> referencia inestable
    tol_marg = 1e-5    # 0 <= lambda(A+B) < tol_marg -> (A+B)_eff casi singular
    try:
        from scipy.linalg import eigvalsh
        lo_ApB = float(eigvalsh(ApB, subset_by_index=[0, 0])[0])
        lo_AmB = float(eigvalsh(AmB, subset_by_index=[0, 0])[0])
    except Exception as e:  # pragma: no cover - diagnostico, nunca critico
        print("  [INFO] {}: no se pudo comprobar la estabilidad ({})".format(etiqueta, e))
        return
    HART2EV = 27.2114
    msg = "min lambda(A+B)={:+.3e} eV, min lambda(A-B)={:+.3e} eV".format(
        lo_ApB * HART2EV, lo_AmB * HART2EV)
    if min(lo_ApB, lo_AmB) < -tol_neg:
        print("  [ADVERTENCIA] {}: referencia NOF INESTABLE para ERPA -- {}. "
              "Las raices mas bajas pueden ser imaginarias y el/los estado(s) "
              "afectado(s) no son fiables en esta geometria.".format(etiqueta, msg))
    elif lo_ApB < tol_marg:
        print("  [ADVERTENCIA] {}: (A+B)_eff CASI SINGULAR -- {}. "
              "El problema ERPA esta mal condicionado aqui; la raiz mas baja "
              "puede quedar mal definida (incertidumbre ~0.1 eV).".format(etiqueta, msg))
    else:
        print("  [INFO] {}: estabilidad OK ({}).".format(etiqueta, msg))


def _safe_inv_reg(x, eps):
    """Reciproco regularizado tipo Tikhonov: x/(x^2+eps^2).

    Coincide con 1/x para |x| >> eps y se satura (no diverge) para
    |x| << eps.  Usado solo en el bloque ERPA2 de ERPA() (CAMBIO ERPA2-1),
    donde aparecen divisiones por c[r]=+-sqrt(n[r]) (y sumas c[r]+c[s]) que
    se acercan a cero para orbitales muy debilmente ocupados o casi
    degenerados, produciendo elementos de matriz enormes y raices complejas
    espureas.
    """
    return x / (x * x + eps * eps)


def _erpa2_report_instability(Msym, sign_v, pp):
    """CAMBIO ERPA2-4: diagnostico (heuristico) de estabilidad para ERPA2.

    Msym = 0.5*(Mtilde+Mtilde.T) es la parte SIMETRICA de
    Mtilde = D^-1 M_ERPA2 D^-1 (ver CAMBIO ERPA2-3); el operador realmente
    usado para los autovalores es S @ Mtilde (S=diag(sign(v))), NO
    S @ Msym, porque M_ERPA2 no es simetrica por construccion y forzar la
    simetria completa altera el espectro (ver comentario en ERPA()).  Este
    chequeo es por tanto un PROXY barato, no un teorema exacto: por analogia
    con la condicion de estabilidad de ERPA0/ERPA1 (A+B y A-B definidas
    positivas, ver _erpa_report_instability), se espera que la parte
    simetrica de Mtilde sea aproximadamente semidefinida positiva en el
    subespacio sign(v)=+1 y semidefinida negativa en sign(v)=-1 para una
    referencia NOF "sana"; violaciones grandes son indicio (no prueba) de
    referencia inestable o deflacion insuficiente.  Solo avisa; nunca aborta
    el calculo.  Desactivable con pp.erpa_check_stability = False.
    """
    if not getattr(pp, "erpa_check_stability", True):
        return
    tol = 1e-6
    try:
        ip = sign_v > 0
        im = sign_v < 0
        msgs = []
        if np.any(ip):
            lo_p = float(np.linalg.eigvalsh(Msym[np.ix_(ip, ip)])[0])
            if lo_p < -tol:
                msgs.append("bloque (v>0) no semidef. positivo (min autoval {:+.2e})".format(lo_p))
        if np.any(im):
            hi_m = float(np.linalg.eigvalsh(Msym[np.ix_(im, im)])[-1])
            if hi_m > tol:
                msgs.append("bloque (v<0) no semidef. negativo (max autoval {:+.2e})".format(hi_m))
        if msgs:
            print("  [ADVERTENCIA] ERPA2: referencia NOF probablemente INESTABLE o "
                  "deflacion insuficiente en este punto -- " + "; ".join(msgs) + ". "
                  "Las raices mas altas pueden ser complejas/espureas.")
        else:
            print("  [INFO] ERPA2: diagnostico de estabilidad OK "
                  "(bloques v>0/v<0 con signo definido).")
    except np.linalg.LinAlgError as e:  # pragma: no cover - diagnostico, nunca critico
        print("  [ADVERTENCIA] ERPA2: diagnostico de estabilidad fallo ({}).".format(e))


def _erpa_iterative_core(ApB, AmB, diag_A, dN_diag, number_of_excitations, pp,
                         etiqueta, tol_rms, tol_max, k_max, m_max_por_estado,
                         omega_min_eV, time1, B0=None):
    """Solver iterativo SMO-GD / Davidson para el problema ERPA reducido

        omega^2 x = dN^{-1} (A+B)_eff dN^{-1} (A-B) x .

    ApB = (A+B)_eff y AmB = (A-B) deben venir YA simetrizadas; diag_A es la
    diagonal aproximada del operador (A+B)_eff para el precondicionador.
    Devuelve omegas_convergidas (Hartree). Es el nucleo COMUN de
    iterative_ERPA0_v2 (ApB = A+B) e iterative_ERPA1
    (ApB = A+B - 2 CC FF^+ EE).

    B0 (CAMBIO E): matriz (dd, >=number_of_excitations) de vectores guia
    iniciales (p. ej. los vectores de Ritz convergidos del punto anterior
    de un barrido).  Si es None o su primera dimension no coincide con dd
    (deflacion distinta), se usa la heuristica |diag_A/dN|.  Al terminar,
    los vectores de Ritz finales quedan en pp.erpa_vecs para reutilizarlos.
    """
    from time import time
    import numpy as np
    from scipy.linalg import eigh
    from pynof.utils import inicializar_lambda_ortho, expandir_lambda_ortho

    dd = ApB.shape[0]
    number_of_excitations = int(min(number_of_excitations, dd))
    omega_min = omega_min_eV / 27.2114
    # CAMBIO D: respaldo final menos laxo (antes 1e3*tol_rms, piso 1e-3).
    tol_rms_relax = max(tol_rms * 1e2, 1e-4)

    _erpa_report_instability(ApB, AmB, etiqueta, pp)

    # Diagnostico opcional (O(dd^3), como ERPA()): espectro exacto de ESTA
    # misma reduccion simetrizada -- valida la convergencia del iterativo
    # con independencia de la comparacion contra ERPA().
    if getattr(pp, "erpa_debug_exact", False):
        dNm1_dbg = np.zeros((dd, dd))
        np.fill_diagonal(dNm1_dbg, np.where(np.abs(dN_diag) > 0, 1.0 / dN_diag, 0.0))
        MM_dbg = dNm1_dbg @ ApB @ dNm1_dbg @ AmB
        w_dbg = np.sqrt(np.linalg.eigvals(MM_dbg).astype(complex))
        w_re = np.sort(np.real(w_dbg[np.abs(np.imag(w_dbg)) < 1e-8])) * 27.2114
        n_cplx = int(np.sum(np.abs(np.imag(w_dbg)) >= 1e-8))
        print("  [DEBUG] {} exacto (simetrizado) - {} raices complejas".format(etiqueta, n_cplx))
        for i in range(min(number_of_excitations, len(w_re))):
            print("  [DEBUG]   Exc. exacta {}: {:6.3f} eV".format(i, w_re[i]))

    # CAMBIO E: guess inicial reutilizable a lo largo de un barrido.
    B0_ok = (B0 is not None and np.ndim(B0) == 2
             and B0.shape[0] == dd and B0.shape[1] >= number_of_excitations)
    if B0_ok:
        B0p = np.asarray(B0[:, :number_of_excitations], dtype=float).copy()
        B0m = B0p.copy()
        print("  [INFO] {}: guess inicial tomado de B0 externo ({} vectores).".format(
            etiqueta, number_of_excitations))
    else:
        if B0 is not None:
            print("  [INFO] {}: B0 externo ignorado (forma incompatible con dd={}).".format(
                etiqueta, dd))
        with np.errstate(divide='ignore', invalid='ignore'):
            proxy_omega = np.abs(diag_A / dN_diag)
        proxy_omega[~np.isfinite(proxy_omega)] = np.inf
        guess_idx = np.argsort(proxy_omega)[:number_of_excitations]
        B0p = np.zeros((dd, number_of_excitations))
        B0m = np.zeros((dd, number_of_excitations))
        for j, idx in enumerate(guess_idx):
            B0p[idx, j] = 1.0
            B0m[idx, j] = 1.0

    bp, LVp = inicializar_lambda_ortho(B0p, ApB)
    bm, LVm = inicializar_lambda_ortho(B0m, AmB)

    # CAMBIO B: 'omegas_convergidas' guarda el MEJOR valor visto (menor
    # residuo) de cada raiz que en algun momento cumplio la tolerancia; no
    # se congela nada por posicion en el orden (ver nota en el bucle).
    omegas_convergidas = np.full(number_of_excitations, np.nan)
    rms_convergidas = np.full(number_of_excitations, np.inf)  # residuo con el que se guardo
    omega_ultimo = np.full(number_of_excitations, np.nan)  # ultimo valor fisico visto
    rms_ultimo = np.full(number_of_excitations, np.inf)
    ritz_p = B0p.copy()  # CAMBIO E: ultimos vectores de Ritz (para pp.erpa_vecs)
    n_ultimo_bloque = bp.shape[1]

    # CAMBIO B (salida anticipada por estancamiento): sin locking duro, un
    # punto benigno llegaba a k_max porque el estado mas alto se queda con
    # residuo entre tol_rms y tol_rms_relax.  Se corta cuando TODOS los
    # estados estan convergidos o bien tienen residuo < tol_rms_relax y su
    # omega no cambia (< rtol_stall) durante n_stall iteraciones seguidas.
    # El umbral de estancamiento es estricto para no cortar una raiz que
    # aun se mueve (caso C2H4 a 90 grados: el estado 0 baja de 8 a 4 eV
    # varias iteraciones -> nunca se considera estancado).
    n_stall = 10
    rtol_stall = 1e-6
    omega_prev = np.full(number_of_excitations, np.nan)
    stall = np.zeros(number_of_excitations, dtype=int)

    # CAMBIO C: si la referencia es inestable, (A+B)_eff deja de ser
    # definida positiva y la lambda-ortogonalizacion incremental (Cholesky)
    # revienta a mitad del bucle.  Se captura (en las llamadas a
    # expandir_/inicializar_lambda_ortho), se avisa y se sale con lo que
    # haya (NaN para lo no convergido) en vez de abortar el calculo.
    from numpy.linalg import LinAlgError
    solver_error = None

    for k in range(k_max):
        S = bm.T @ (dN_diag[:, None] * bp)

        STS = S.T @ S
        STS = 0.5 * (STS + STS.T)
        lambda2_all, up_all = eigh(STS)
        with np.errstate(divide='ignore', invalid='ignore'):
            omega_all = 1.0 / np.sqrt(np.maximum(lambda2_all, 1e-14))
        # Seleccion de los number_of_excitations estados FISICOS mas bajos.
        # ERPA genera modos espurios con omega -> 0 (aqui siempre reales por
        # la simetrizacion S^T S); ERPA() los descarta con el mismo umbral
        # (parte real > 1 eV). Si el subespacio aun no tiene suficientes
        # fisicos, se rellena con espurios (nunca se dan por convergidos).
        clave = np.where(omega_all > omega_min, omega_all, np.inf)
        order = np.argsort(clave)[:number_of_excitations]
        es_fisico = omega_all[order] > omega_min
        lambda2 = np.maximum(lambda2_all[order], 1e-14)
        omega = omega_all[order]
        up = up_all[:, order]
        um = (S @ up) / np.sqrt(lambda2)[np.newaxis, :]
        ritz_p = bp @ up  # CAMBIO E: vectores de Ritz actuales (base completa)

        Rp = (dN_diag[:, None] * bm) @ um - (1.0 / omega[np.newaxis, :]) * (ApB @ bp @ up)
        Rm = (dN_diag[:, None] * bp) @ up - (1.0 / omega[np.newaxis, :]) * (AmB @ bm @ um)

        norm_rms = np.sqrt(np.mean(Rp**2 + Rm**2, axis=0))
        norm_max = np.maximum(np.max(np.abs(Rp), axis=0), np.max(np.abs(Rm), axis=0))

        # Ultimo valor fisico visto de cada estado (para el respaldo final).
        for idx in range(number_of_excitations):
            if es_fisico[idx]:
                omega_ultimo[idx] = omega[idx]
                rms_ultimo[idx] = norm_rms[idx]

        # CAMBIO B -- convergencia SUAVE (state-following implicito).
        #
        # El locking anterior congelaba 'omegas_convergidas[idx]' segun la
        # POSICION 'idx' en el orden por energia y sacaba ese hueco de la
        # lista de activos.  Pero 'order = argsort(...)' reordena las
        # parejas de Ritz en cada iteracion: cerca de una (casi)
        # degeneracion dos raices se cruzan, el hueco 'idx' pasa a apuntar a
        # otra raiz fisica, y la raiz que aun estaba descendiendo se queda
        # atrapada en un hueco ya bloqueado -> se descartaban sus
        # correcciones, 'activos' se vaciaba y el bucle hacia 'break' con el
        # espectro equivocado.  Caso real: C2H4 a 90 grados, ERPA0 devolvia
        # S1 = 7.9 eV (cluster espurio) en vez de ~4.25 eV.
        #
        # Ahora: (1) cada iteracion se evalua el residuo de las N raices
        # fisicas mas bajas; (2) se guarda el mejor valor visto de cada una
        # (menor residuo) sin sacarla de la iteracion; (3) el bucle solo
        # termina cuando las N cumplen la tolerancia SIMULTANEAMENTE en la
        # misma iteracion.  Los huecos ya convergidos siguen en el bloque de
        # correccion, pero su residuo ~0 hace que el vector nuevo se
        # descarte en la ortogonalizacion, asi que no encarece el paso.
        conv_mask = es_fisico & (norm_rms < tol_rms) & (norm_max < tol_max)
        mejora = conv_mask & (norm_rms < rms_convergidas)
        omegas_convergidas[mejora] = omega[mejora]
        rms_convergidas[mejora] = norm_rms[mejora]
        n_conv = int(conv_mask.sum())

        # contador de estancamiento (omega estable entre iteraciones)
        with np.errstate(invalid='ignore'):
            quieto = np.abs(omega - omega_prev) <= rtol_stall * np.abs(omega)
        stall = np.where(quieto, stall + 1, 0)
        omega_prev = omega.copy()
        listo = conv_mask | (es_fisico & (norm_rms < tol_rms_relax) & (stall >= n_stall))

        print(f" Iter {k:03d} | subespacio m={bp.shape[1]:4d} | "
              f"convergidos={n_conv}/{number_of_excitations}")
        for i in range(number_of_excitations):
            print("    Exc. en {}: {:6.3f} eV{}".format(
                i, omega[i] * 27.2114, "" if es_fisico[i] else "  (espurio)"))

        if n_conv == number_of_excitations:
            break
        if listo.all():
            # los no convergidos por tol_rms estan estancados con residuo
            # pequeno: se guardan como aproximados y se corta (evita gastar
            # hasta k_max en un estado que ya no mejora).
            aprox = listo & ~conv_mask
            omegas_convergidas[aprox] = omega[aprox]
            rms_convergidas[aprox] = norm_rms[aprox]  # CAMBIO D: registrar el residuo
            print("[INFO] Corte por estancamiento: {} estado(s) aproximado(s) "
                  "(residuo < {:.0e}).".format(int(aprox.sum()), tol_rms_relax))
            break

        idx_activos = np.where(~conv_mask)[0]
        Rp_act, Rm_act = Rp[:, idx_activos], Rm[:, idx_activos]
        omega_act = omega[idx_activos]

        denom = (omega_act[np.newaxis, :]**-2) * (diag_A[:, np.newaxis]**2) - (dN_diag[:, np.newaxis]**2)
        denom_inv = 1.0 / (denom + np.sign(denom + 1e-300) * 1e-12)

        term_plus = (1.0 / omega_act[np.newaxis, :]) * diag_A[:, np.newaxis] * Rp_act + dN_diag[:, np.newaxis] * Rm_act
        term_minus = (1.0 / omega_act[np.newaxis, :]) * diag_A[:, np.newaxis] * Rm_act + dN_diag[:, np.newaxis] * Rp_act

        b_p_new = -denom_inv * term_plus
        b_m_new = -denom_inv * term_minus

        # Reinicio de subespacio (thick restart): conserva los vectores de
        # Ritz actuales MAS el bloque de correccion, en vez de colapsar solo
        # a los de Ritz (que estanca la convergencia del estado mas dificil).
        # Tope: min(number_of_excitations*m_max_por_estado, dd).
        m_cap = min(dd, number_of_excitations * m_max_por_estado)
        try:
            if bp.shape[1] >= m_cap:
                print("[INFO] Reiniciando subespacio (thick restart, m={})".format(bp.shape[1]))
                Xp = np.hstack((bp @ up, b_p_new))
                Xm = np.hstack((bm @ um, b_m_new))
                bp, LVp = inicializar_lambda_ortho(Xp, ApB)
                bm, LVm = inicializar_lambda_ortho(Xm, AmB)
                n_ultimo_bloque = bp.shape[1]
                continue

            ancho_antes = bp.shape[1]
            bp, LVp = expandir_lambda_ortho(b_p_new, bp, LVp, ApB)
            bm, LVm = expandir_lambda_ortho(b_m_new, bm, LVm, AmB)
        except LinAlgError as e:
            solver_error = e
            print("  [ADVERTENCIA] {}: el solver ERPA fallo en la iteracion {} "
                  "(LinAlgError: {}).  Causa tipica: (A+B)_eff no definida "
                  "positiva -> referencia NOF inestable (ver aviso de "
                  "estabilidad).  Se devuelve lo convergido y NaN para el "
                  "resto.".format(etiqueta, k, e))
            break
        # nunca exceder la dimension del espacio completo
        if bp.shape[1] > dd:
            bp, LVp = bp[:, :dd], LVp[:, :dd]
        if bm.shape[1] > dd:
            bm, LVm = bm[:, :dd], LVm[:, :dd]
        n_ultimo_bloque = min(bp.shape[1], bm.shape[1]) - ancho_antes
        if bp.shape[1] != bm.shape[1]:
            m_comun = min(bp.shape[1], bm.shape[1])
            bp, LVp = bp[:, :m_comun], LVp[:, :m_comun]
            bm, LVm = bm[:, :m_comun], LVm[:, :m_comun]

    # Respaldo final: estados no convergidos por residuo pero ya con residuo
    # pequeno se reportan como aproximados (misma idea que la v1, que acepta
    # por estancamiento de la energia).
    n_aprox = 0
    rms_final = rms_convergidas.copy()
    for idx in range(number_of_excitations):
        if np.isnan(omegas_convergidas[idx]) and rms_ultimo[idx] < tol_rms_relax:
            omegas_convergidas[idx] = omega_ultimo[idx]
            rms_final[idx] = rms_ultimo[idx]
            n_aprox += 1

    vals = omegas_convergidas * 27.2114

    # CAMBIO E: vectores de Ritz finales, para sembrar el punto siguiente
    # de un barrido (pp.erpa_B0 = pp.erpa_vecs).
    try:
        pp.erpa_vecs = np.asarray(ritz_p).copy()
    except Exception:
        pass

    print()
    print("  Excitation energies {}(eV)".format(etiqueta))
    print("  ===================================")
    # CAMBIO D: se informa el residuo alcanzado y se marca lo aproximado.
    for i in range(min(10, len(vals))):
        r = rms_final[i]
        if np.isfinite(r) and r < tol_rms:
            extra = "  (rms={:.1e})".format(r)
        elif np.isfinite(r):
            extra = "  (APROX., rms={:.1e} > tol_rms={:.0e})".format(r, tol_rms)
        else:
            extra = "  (NO CONVERGIDO)"
        print("    Exc. en. {}: {:6.3f}{}".format(i, vals[i], extra))

    if n_aprox:
        print("  [NOTA] {} estado(s) aproximado(s): residuo < {:.0e} pero no se "
              "alcanzo tol_rms={:.0e}".format(n_aprox, tol_rms_relax, tol_rms))
    if solver_error is not None:
        print("  [ADVERTENCIA] {}: resultado PARCIAL por fallo del solver "
              "(referencia probablemente inestable).".format(etiqueta))
    elif np.any(np.isnan(omegas_convergidas)):
        print("  [ADVERTENCIA] No todos los estados convergieron en {} iteraciones".format(k_max))

    print("  Elapsed Time: {:10.2f} (Seconds)".format(time() - time1))

    return omegas_convergidas


def iterative_ERPA0_v2(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag,
                        number_of_excitations, pp,
                        tol_rms=1e-6, tol_max=1e-5, k_max=100,
                        m_max_por_estado=25, omega_min_eV=1.0, B0=None):
    """
    Version corregida de iterative_ERPA0. El nucleo matematico (matriz S,
    reduccion de Rayleigh-Ritz, formulas de residuo Rp/Rm) es identico al
    original y ya era correcto. Correcciones aplicadas:

      1. diag_A extraido correctamente de np.diag(AA), no de A.diagonal()
         sobre el tensor 4D.
      2. dN_diag, diag_A tratados como vectores 1D (no matrices dd x dd);
         precondicionador resuelto elemento a elemento (O(dd), no O(dd^3)).
      3. Locking: los estados convergidos (criterio RMS + maximo) se
         congelan y dejan de generar vectores de prueba nuevos.
      4. Reinicio de subespacio (m_max_por_estado) para acotar memoria/costo.
      5. Guess inicial via heuristica estandar de Davidson (menor
         |diag_A/dN|), reemplazando la logica de "i_start" (que tenia un
         bug: reusaba una variable de bucle obsoleta).
      6. CRITICO: Lambda-ortogonalizacion incremental (pynof.utils.
         inicializar_lambda_ortho / expandir_lambda_ortho) en vez de
         pynof.orthonormalize(bp, ApB), que hacia eigh(ApB) COMPLETO en
         cada iteracion -- ver analisis en el plan de implementacion.
      7. Filtro de modos espurios: se buscan los number_of_excitations
         estados FISICOS mas bajos (omega > omega_min_eV), igual que
         ERPA() que descarta los autovalores con parte real < 1 eV.
      8. Robustez: el subespacio nunca excede dd; reinicio "thick" (Ritz +
         bloque de correccion); respaldo final para estados con residuo
         pequeno pero que no alcanzaron tol_rms.
      9. ApB y AmB se simetrizan explicitamente (son simetricas en la
         teoria; la asimetria ~1e-4 es ruido de ensamblado). Cerca de una
         inestabilidad de ERPA0 (raices complejas que ERPA() descarta)
         esto puede dar un estado mas bajo distinto al de ERPA(); poner
         pp.erpa_debug_exact=True imprime el espectro exacto de ESTA misma
         reduccion simetrizada para validar la convergencia del iterativo.
     10. FASE 2 (aceleracion): ApB/AmB/diag_A/dN_diag se construyen por
         indexado vectorizado directo del tensor A sobre los pares que
         sobreviven a la deflacion. Se elimina la matriz M completa
         (nbf5^2 x nbf5^2), su despliegue en bucles de Python y la
         rotacion fila/columna de la deflacion -- que juntos costaban
         O(nbf5^6) (~5-6 s en C2H4/cc-pVDZ). Resultado bit-a-bit identico.
         (Sigue construyendose el tensor A completo: una version
         "matrix-free" que lo evite por iteracion resulto MAS lenta en
         este regimen -- el bucle Davidson, no A, es ahora el costo mayor.)
     11. El bucle Davidson vive ahora en _erpa_iterative_core, compartido
         con iterative_ERPA1.
    """
    from time import time
    import numpy as np

    time1 = time()

    print("\n---------------")
    print(" ERPA Analysis (iterativo, ERPA0/v2)")
    print("---------------\n")

    # CAMBIO A: corte de deflacion configurable (antes fijo 1e-3, causaba
    # saltos de dimension -> discontinuidades en el perfil de excitacion).
    tol_dn = getattr(pp, "erpa_tol_dn", 1e-4)

    A = _erpa_transform_and_build_A(n, C, H, I, b_mnl, pp)
    rr, ss, dN_diag, dd, n_defl = _erpa_deflated_pairs(n, pp, tol_dn)
    print("M_ERPA Orig dim: {} New dim: {} Elements below tol_dN: {}".format(
        pp.nbf5 * (pp.nbf5 - 1), 2 * dd, 2 * n_defl))

    # FASE 2: bloques ERPA0 por indexado directo de A (ver docstring, punto 10).
    #   AA[a,b] = A[r_a, s_a, r_b, s_b]   (era M[:dd, :dd])
    #   BB[a,b] = A[r_a, s_a, s_b, r_b]   (era M[:dd, dd:2dd])
    AA = A[rr[:, None], ss[:, None], rr[None, :], ss[None, :]]
    BB = A[rr[:, None], ss[:, None], ss[None, :], rr[None, :]]

    ApB = AA + BB
    AmB = AA - BB

    maxApBsym = np.max(np.abs(ApB - ApB.T))
    maxAmBsym = np.max(np.abs(AmB - AmB.T))
    print("Max diff ApB {:3.1e} and Max AmB {:3.1e} (se simetrizan)".format(maxApBsym, maxAmBsym))
    print()

    # ERPA0: ApB y AmB son simetricas por construccion teorica; cualquier
    # asimetria (~1e-4) es ruido numerico del ensamblado de A y, si se
    # deja, pone un piso al residuo que impide alcanzar tol_rms.
    ApB = 0.5 * (ApB + ApB.T)
    AmB = 0.5 * (AmB + AmB.T)
    diag_A = np.diag(AA).copy()

    return _erpa_iterative_core(
        ApB, AmB, diag_A, dN_diag, number_of_excitations, pp,
        "ERPA0/PNOF{}".format(pp.ipnof), tol_rms, tol_max, k_max,
        m_max_por_estado, omega_min_eV, time1, B0=B0)


def iterative_ERPA1(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag,
                     number_of_excitations, pp,
                     tol_rms=1e-6, tol_max=1e-5, k_max=100,
                     m_max_por_estado=25, omega_min_eV=1.0, ff_rcond=1e-8,
                     B0=None):
    """
    Version iterativa del ERPA "completo" (a veces llamado ERPA1): es ERPA0
    mas el termino de acoplamiento a las filas/columnas "diagonales"
    (numero-conservantes). La reduccion es identica a la de
    iterative_ERPA0_v2 salvo que (A+B) se reemplaza por el operador efectivo

        (A+B)_eff = (A+B) - 2 * CC @ FF^+ @ EE

    (mismos indices que el bloque medio 'ERPA' de ERPA()):
        CC[a,p] = A[r_a, s_a, p, p]     (dd  x nbf5)
        EE[p,b] = A[p, p, r_b, s_b]     (nbf5 x dd)
        FF[p,q] = A[p, p, q, q]         (nbf5 x nbf5)

    Notas:
      * FF es (casi) singular por construccion (direcciones redundantes de
        rotacion que conservan el numero de ocupacion). ERPA() usa
        np.linalg.inv(FF), que es inestable; aqui se usa la pseudo-inversa
        np.linalg.pinv(FF, rcond=ff_rcond), que da CC FF^+ EE bien definido
        y reproducible. En sistemas donde inv(FF) esta bien condicionada
        (p. ej. H2/cc-pVDZ) ambos coinciden.
      * Todo lo demas (solver SMO-GD, precondicionador, filtro de modos
        espurios, deflacion, Fase 2 sin matriz M) es COMUN con
        iterative_ERPA0_v2 -- ver _erpa_iterative_core.
      * pp.erpa_debug_exact=True imprime el espectro exacto de ESTA misma
        reduccion (A+B)_eff simetrizada, para validar la convergencia.
    """
    from time import time
    import numpy as np

    time1 = time()

    print("\n---------------")
    print(" ERPA Analysis (iterativo, ERPA1)")
    print("---------------\n")

    # CAMBIO A: ver nota en iterative_ERPA0_v2 / _erpa_deflated_pairs.
    tol_dn = getattr(pp, "erpa_tol_dn", 1e-4)

    A = _erpa_transform_and_build_A(n, C, H, I, b_mnl, pp)
    rr, ss, dN_diag, dd, n_defl = _erpa_deflated_pairs(n, pp, tol_dn)
    print("M_ERPA Orig dim: {} New dim: {} Elements below tol_dN: {}".format(
        pp.nbf5 * (pp.nbf5 - 1), 2 * dd, 2 * n_defl))

    allp = np.arange(pp.nbf5)
    AA = A[rr[:, None], ss[:, None], rr[None, :], ss[None, :]]
    BB = A[rr[:, None], ss[:, None], ss[None, :], rr[None, :]]
    CC = A[rr[:, None], ss[:, None], allp[None, :], allp[None, :]]    # (dd, nbf5)
    EE = A[allp[:, None], allp[:, None], rr[None, :], ss[None, :]]    # (nbf5, dd)
    FF = A[allp[:, None], allp[:, None], allp[None, :], allp[None, :]]  # (nbf5, nbf5)

    FF_pinv = np.linalg.pinv(FF, rcond=ff_rcond)
    sv = np.linalg.svd(FF, compute_uv=False)
    n_drop = int(np.sum(sv < ff_rcond * sv[0]))
    print("FF: nbf5={} rank~{} (valores singulares descartados: {}, rcond={:.0e})".format(
        pp.nbf5, pp.nbf5 - n_drop, n_drop, ff_rcond))

    coupl = 2.0 * (CC @ (FF_pinv @ EE))
    ApB = AA + BB - coupl
    AmB = AA - BB

    maxApBsym = np.max(np.abs(ApB - ApB.T))
    maxAmBsym = np.max(np.abs(AmB - AmB.T))
    print("Max diff (A+B)_eff {:3.1e} and Max (A-B) {:3.1e} (se simetrizan)".format(maxApBsym, maxAmBsym))
    print()

    ApB = 0.5 * (ApB + ApB.T)
    AmB = 0.5 * (AmB + AmB.T)
    diag_A = np.diag(ApB).copy()  # diagonal del operador efectivo (A+B)_eff

    return _erpa_iterative_core(
        ApB, AmB, diag_A, dN_diag, number_of_excitations, pp,
        "ERPA1/PNOF{}".format(pp.ipnof), tol_rms, tol_max, k_max,
        m_max_por_estado, omega_min_eV, time1, B0=B0)
