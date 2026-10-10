import numpy as np
import psi4


class param:

    def __init__(self, mol, basname):

        psi4.set_options({"basis": basname})

        # Paramdetros del sistema
        self.wfn = wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))

        # Parámetros
        self.natoms = mol.natom()
        self.nbf = wfn.basisset().nbf()
        self.nbfaux = 0
        self.nalpha = wfn.nalpha()
        self.nbeta = wfn.nbeta()
        self.ne = self.nalpha + self.nbeta
        self.mul = mol.multiplicity()
        self.charge = mol.molecular_charge()
        no1 = 0  # Number of inactive doubly occupied orbitals | Se puede variar
        # for i in range(self.natoms):
        #    Z = mol.Z(i)
        #    if ( 1<=Z and Z<=  2):
        #        no1 += 0           # H-He
        #    elif ( 3<=Z and Z<= 10):
        #        no1 +=  1          # Li-Ne
        #    elif (11<=Z and Z<= 18):
        #        no1 +=  5          # Na-Ar
        #    elif(19<=Z and Z<= 36):
        #        no1 +=  9          # K-Kr
        #    elif(37<=Z and Z<= 49):
        #        no1 += 18          # Rb-In
        #    elif(50<=Z and Z<= 54):
        #        no1 += 23          # Sn-Xe
        #    elif(55<=Z and Z<= 71):
        #        no1 += 27          # Cs-Lu
        #    elif(72<=Z and Z<= 81):
        #        no1 += 30          # Hf-Tl
        #    elif(82<=Z and Z<= 86):
        #        no1 += 39          # Pb-Rn
        #    elif(87<=Z and Z<=109):
        #        no1 += 43          # Fr-Mt
        self.ndoc = self.nbeta - no1
        self.nsoc = self.nalpha - self.nbeta
        self.ndns = self.ndoc + self.nsoc
        self.nvir = self.nbf - self.nalpha

        ncwo = -1
        if self.ne == 2:
            ncwo = -1
        if self.ndns != 0:
            if self.ndoc > 0:
                if ncwo != 1:
                    if ncwo == -1 or ncwo > self.nvir / self.ndoc:
                        ncwo = int(self.nvir / self.ndoc)
            else:
                ncwo = 0

        self.closed = self.nbeta == (self.ne + self.mul - 1) / 2 and self.nalpha == (self.ne - self.mul + 1) / 2

        self.nac = self.ndoc * (1 + ncwo)
        self.nbf5 = no1 + self.nac + self.nsoc  # JFHLY warning: nbf must be >nbf5
        self.no0 = self.nbf - self.nbf5

        noptorb = self.nbf

        self.title = "pynof"
        self.maxit = 1000  # Número máximo de iteraciones de Occ-SCF
        self.no1 = no1  # Número de orbitales inactivos con ocupación 1
        self.thresheid = 10**-6  # 8 # Convergencia de la energía total
        self.maxitid = 30  # Número máximo de iteraciones externas en HF
        self.maxloop = 30  # Iteraciones internas en optimización orbital
        self.ipnof = 8  # PNOFi a calcular
        self.ista = 0  # PNOFi a calcular
        # SOLUCION 3: anchura del "agujero dinamico" de GNOF (ipnof=8).
        #   F(n) = exp(-((1-n_strong)/gnof_hcut)^2)  activa la correccion
        #   dinamica n_d solo cerca de capa cerrada.  Valor original
        #   0.02*sqrt(2) ~ 0.0283 -> transicion abrupta (favorece la
        #   estructura de dos ramas NOF al romper enlaces).  Aumentarlo
        #   (~0.05-0.15) suaviza la transicion y puede fusionar las ramas.
        self.gnof_hcut = 0.02 * np.sqrt(2.0)
        self.threshl = 10**-3  # Convergencia de los multiplicadores de Lagrange
        self.threshe = 10**-4  # Convergencia de la energía
        self.threshec = 10**-8  # Convergencia  de la energía en optimización orbital
        self.threshen = 10**-10  # Convergencia  de la energía en optimización de ocupaciones
        self.scaling = True  # Scaling for f
        self.nzeros = 0
        self.nzerosm = 5
        self.nzerosr = 2
        self.itziter = 10  # Iteraciones para scaling constante
        self.diis = True  # DIIS en optimización orbital
        self.thdiis = 10**-3  # Para iniciar DIIS
        self.ndiis = 5  # Número de ciclos para interpolar matriz de Fock generalizada en DIIS
        self.perdiis = True  # Aplica DIIS cada NDIIS (True) o después de NDIIS (False)
        self.ncwo = ncwo  # Número de orbitales débilmente ocupados acoplados a cada orbital fueremtente ocupado
        self.noptorb = noptorb  # Número de orbitales a optimizar Nbf5 <= Noptorb <= Nbf
        self.scaling = True
        self.nv = self.ncwo * self.ndoc
        self.nvar = int(self.nbf * (self.nbf - 1) / 2) - int(self.no0 * (self.no0 - 1) / 2)
        self.occupation_optimizer = "CG"
        self.orbital_optimizer = "CG"
        self.combined_optimizer = "CG"
        self.freeze_occ = False
        self.tol_gorb = 1e-3
        self.tol_gocc = 1e-2
        self.gpu = False
        self.RI = False
        self.jit = False

        self.HighSpin = False
        self.MSpin = 0

        self.lamb = 0.0
        # erpa_iterative: False = ERPA() completo (comportamiento actual);
        #   True / "ERPA0" = iterative_ERPA0_v2;  "ERPA1" = iterative_ERPA1.
        self.erpa_iterative = False
        self.n_erpa_states = 10      # numero de estados excitados mas bajos a buscar
        # --- Solver ERPA iterativo (CAMBIO F: knobs antes solo accesibles
        #     como kwargs de iterative_ERPA0_v2 / iterative_ERPA1) ---
        self.erpa_tol_dn = 1e-4          # corte de deflacion |Delta n| (CAMBIO A)
        self.erpa_tol_rms = 1e-6        # tolerancia RMS del residuo por estado
        self.erpa_tol_max = 1e-5        # tolerancia del residuo maximo por estado
        self.erpa_kmax = 100           # iteraciones maximas del solver
        self.erpa_omega_min_eV = 1.0    # umbral para descartar modos espurios
        self.erpa_check_stability = True  # avisar si la referencia es inestable (CAMBIO C)
        self.erpa_B0 = None            # guess inicial (dd, >=n_erpa_states) (CAMBIO E);
        self.erpa_vecs = None          #   el solver deja aqui los vectores de Ritz finales
        # --- ERPA2 (diagonalizacion completa, ERPA() con erpa_iterative=False) ---
        #     Soluciones de estabilizacion especificas de ERPA2 (ver
        #     et-iso/ERPA_full/CHANGELOG_ERPA2.md para el detalle).
        self.erpa2_tol_dn = 3e-3       # corte de deflacion propio de ERPA2
        self.erpa2_reg_eps = 1e-4      # eps de regularizacion Tikhonov en 1/c[r], 1/(c[p]*(c[r]+c[s])), etc.
                                        #   (empiricamente, valores >~5e-4 amortiguan tambien pares debilmente
                                        #   ocupados fisicos -no solo los casi-singulares- e introducen estados
                                        #   de baja energia espureos; ver et-iso/ERPA_full/CHANGELOG_ERPA2.md)
        self.occ_method = "Softmax"
        self.orb_method = "ADAM"
        self.alpha = 0.02
        # --- Multi-solucion de ocupaciones en NOF (para funcionales con
        #     varias soluciones NOF al romper enlaces, p. ej. GNOF/ipnof=8).
        #     Solo actua con occ_method="Softmax".  Coste: cada uno multiplica
        #     el numero de optimizaciones SCF por geometria (~2-4x). ---
        #   None => automatico: activo si ipnof == 8 (GNOF), inactivo si no.
        self.occ_multistart = None        # Solucion 1: repetir la optimizacion
                                          #   desde ocupaciones cerrada y abierta,
                                          #   quedarse con la de menor energia.
        self.occ_stability = None         # Solucion 2: tras converger, perturbar
                                          #   el par frontera (abrir/cerrar) y
                                          #   re-optimizar; adoptar si baja E.
        self.occ_multistart_npairs = 1    # nº de pares fuertes (HOMO, HOMO-1,...)
                                          #   que se abren en el guess "abierto".
        self.occ_multistart_tol = 1e-5    # mejora minima (Hartree) para cambiar
                                          #   de solucion.
        self.occ_multistart_level = 1     # 1 = guesses alt. solo desde orbitales
                                          #   de core (~3x); 2 = tambien desde los
                                          #   orbitales de restart (~5x).
        # --- Hessiano numerico (compute_energy(..., hessian=True), ver hessian.py) ---
        self.hess_step = 0.02             # desplazamiento cartesiano (bohr); >= 0.02 por el ruido del gradiente
        self.hess_polish = "lbfgs"        # pulido orbital: "lbfgs", "trust-krylov", "hessp" o "none"
        self.hess_polish_maxiter = 200    # iteraciones L-BFGS por ciclo de pulido
        self.hess_max_polish = None       # ciclos max. de pulido (None: 3 con lbfgs, 20 con trust-krylov)
        self.hess_refine = True           # refinar frecuencias con energias a lo largo de cada modo normal
        self.hess_refine_step = None      # desplazamiento cartesiano max. del refinamiento (None = 0.04; mas alla cambia la solucion NOF en isocianuros)
        self.hess_refine_de = 2e-4        # cambio de energia objetivo (Eh) para elegir el paso de cada modo
        self.hess_linear_tol = 5e-3       # desviacion relativa max. para tratar la molecula como lineal
        self.hess_linear = None           # None = detectar molecula lineal e imponer simetria cilindrica
        self.hess_verbose = False         # mostrar la salida de compute_energy en cada desplazamiento

    def autozeros(self, restart=False):
        if restart:
            if abs(np.log10(self.threshl)) <= 3:
                self.nzeros = 2
                self.nzerosr = 2
                self.nzerosm = 5
            else:
                self.nzeros = abs(int(np.log10(self.threshl))) - 1
                self.nzerosr = self.nzeros
                self.nzerosm = abs(int(np.log10(self.threshl))) + 2
        else:
            # if(abs(np.log10(self.threshl))>3):
            self.nzeros = 1
            self.nzerosr = 2
            self.nzerosm = abs(int(np.log10(self.threshl))) + 2

    def set_ncwo(self, ncwo):
        # if(self.ne==2):
        #    ncwo= -1
        if self.ndns != 0:
            if self.ndoc > 0:
                if ncwo != 1:
                    if ncwo == -1 or ncwo > self.nvir / self.ndoc:
                        ncwo = int(self.nvir / self.ndoc)
            else:
                ncwo = 0

        self.ncwo = ncwo

        self.nac = self.ndoc * (1 + ncwo)
        self.nbf5 = self.no1 + self.nac + self.nsoc  # JFHLY warning: nbf must be >nbf5
        self.no0 = self.nbf - self.nbf5
        self.nvar = int(self.nbf * (self.nbf - 1) / 2) - int(self.no0 * (self.no0 - 1) / 2)
        self.nv = self.ncwo * self.ndoc

    def remove_no1(self, gamma=None):

        no1_old = self.no1

        no1 = 0  # Number of inactive doubly occupied orbitals | Se puede variar
        self.ndoc = self.nbeta - no1
        self.ndns = self.ndoc + self.nsoc

        self.nac = self.ndoc * (1 + self.ncwo)
        self.nbf5 = no1 + self.nac + self.nsoc  # JFHLY warning: nbf must be >nbf5
        self.no0 = self.nbf - self.nbf5

        self.no1 = no1  # Número de orbitales inactivos con ocupación 1
        self.nv = self.ncwo * self.ndoc
        self.nvar = int(self.nbf * (self.nbf - 1) / 2) - int(self.no0 * (self.no0 - 1) / 2)

        if not gamma is None:
            gamma = np.concatenate((np.ones((no1_old)) * np.arccos(np.sqrt(2.0 * 0.999 - 1.0)), gamma))
            return gamma
