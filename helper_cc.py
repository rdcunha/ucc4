'''
HelperUCC4 class definition and function definitions
For Spin-orbital UCC(4) calculations
'''

import numpy as np
import psi4
from ndot import ndot

class HelperUCC4(object):
    def __init__(self, wfn):

        # Set energy and wfn from Psi4
        self.wfn = wfn

        # Get orbital coeffs from wfn
        C = self.wfn.Ca()
        basis = self.wfn.basisset()
        self.no_occ = 2 * self.wfn.doccpi()[0]
        self.no_mo = 2 * self.wfn.nmo()
        self.eps = np.asarray(self.wfn.epsilon_a())
        self.J = self.wfn.jk().J()[0].to_array()
        self.K = self.wfn.jk().K()[0].to_array()

        # Get No. of virtuals
        self.no_vir = self.no_mo - self.no_occ
        print("Checking dimensions of orbitals: no_occ: {}\t no_vir: {}\t total: {}".format(self.no_occ, self.no_vir, self.no_occ+self.no_vir))

        # Make slices
        o = slice(0, self.no_occ)
        v = slice(self.no_occ, self.no_mo)

        # Generate integrals
        mints = psi4.core.MintsHelper(basis)

        self.H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

        # Make MO integrals
        self.MO = np.asarray(mints.mo_spin_eri(C, C))

        self.H = np.einsum('uj, vi, uv', C, C, self.H)
        
        # Tile eps, H for alpha and beta spins
        self.eps = np.repeat(self.eps, 2)
        self.H = np.repeat(self.H, 2, axis=0)
        self.H = np.repeat(self.H, 2, axis=1)

        # Make H block diagonal
        spin_ind = np.arange(self.no_mo, dtype=np.int) % 2
        self.H *= (spin_ind.reshape(-1, 1) == spin_ind)

        # Need to change ERIs to physicist notation
        #self.MO = self.MO.swapaxes(1, 2)

        # Build Fock matrix
        self.F = self.H + np.einsum('pmqm->pq', self.MO[:, o, :, o])

        # Need F_occ and F_vir separate (will need F_vir for semi-canonical basis later)
        self.F_occ = self.F[o, o]
        self.F_vir = self.F[v, v]

        self.eps_occ = self.eps[:self.no_occ]
        self.eps_vir = self.eps[self.no_occ:]
        #self.eps_occ = np.diag(self.F_occ)
        #self.eps_vir = np.diag(self.F_vir)

        # Build denominators
        # note that occ.transpose(col) - vir(row) gives occ x vir matrix of differences
        self.d_ia = self.eps_occ.reshape(-1, 1) - self.eps_vir
        self.d_ijab = self.eps_occ.reshape(-1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1) - self.eps_vir.reshape(-1, 1) - self.eps_vir
        self.d_ijkabc = self.eps_occ.reshape(-1, 1, 1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1, 1) - self.eps_vir.reshape(-1, 1, 1) - self.eps_vir.reshape(-1, 1) - self.eps_vir

        # init T1s
        self.t_ia = np.zeros((self.no_occ, self.no_vir))

        # init T2s
        self.t_ijab = self.MO[o, o, v, v].copy()
        self.t_ijab /= self.d_ijab

        # init T3s
        self.t_ijkabc = np.zeros((self.no_occ, self.no_occ, self.no_occ, self.no_vir, self.no_vir, self.no_vir))

    # Update T amplitudes
    def update_ts(self, t_ia, t_ijab, t_ijkabc):
        
        # Make slices
        o = slice(0, self.no_occ)
        v = slice(self.no_occ, self.no_mo)

        # New T1s
        Ria = ndot('ikcd,akcd->ia', t_ijab, self.MO[v, o, v, v], prefactor=0.5)
        Ria -= ndot('klac,klic->ia', t_ijab, self.MO[o, o, o, v], prefactor=0.5)

        # New T2s
        # <ij||ab>
        Rijab = self.MO[o, o, v, v].copy()

        # P_ab P_ij t_ikac <kb||cj>
        Rijab += ndot('ikac,kbcj->ijab', t_ijab, self.MO[o, v, v, o])
        Rijab -= ndot('jkac,kbci->ijab', t_ijab, self.MO[o, v, v, o])
        Rijab -= ndot('ikbc,kacj->ijab', t_ijab, self.MO[o, v, v, o])
        Rijab += ndot('jkbc,kaci->ijab', t_ijab, self.MO[o, v, v, o])
        # 1/2 t_klab <kl||ij>
        Rijab += ndot('klab,klij->ijab', t_ijab, self.MO[o, o, o, o], prefactor=0.5)
        # 1/2 t_ijcd <ab||cd>
        Rijab += ndot('ijcd,abcd->ijab', t_ijab, self.MO[v, v, v, v], prefactor=0.5)

        # P_ij t_ic <ab||cj>
        Rijab += ndot('ic,abcj->ijab', t_ia, self.MO[v, v, v, o])
        Rijab -= ndot('jc,abci->ijab', t_ia, self.MO[v, v, v, o])
        # - P_ab t_ka <kb||ij>
        Rijab -= ndot('ka,kbij->ijab', t_ia, self.MO[o, v, o, o])
        Rijab += ndot('kb,kaij->ijab', t_ia, self.MO[o, v, o, o])

        # 1/2 P_ab t_ijkacd <bk||cd>
        Rijab += ndot('ijkacd,bkcd->ijab', t_ijkabc, self.MO[v, o, v, v], prefactor=0.5)
        Rijab -= ndot('ijkbcd,akcd->ijab', t_ijkabc, self.MO[v, o, v, v], prefactor=0.5)
        # - 1/2 P_ij t_iklabc <kl||jc>
        Rijab -= ndot('iklabc,kljc->ijab', t_ijkabc, self.MO[o, o, o, v], prefactor=0.5)
        Rijab += ndot('jklabc,klic->ijab', t_ijkabc, self.MO[o, o, o, v], prefactor=0.5)

        # P_ij P_ab t_ikac t_jlbd <kl||cd>
        Rijab += 0.5 * np.einsum('ikac,jlbd,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        Rijab -= 0.5 * np.einsum('jkac,ilbd,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        Rijab -= 0.5 * np.einsum('ikbc,jlad,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        Rijab += 0.5 * np.einsum('jkbc,ilad,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        # - 1/2 P_ij t_ikdc t_ljab <kl||cd>
        Rijab -= 0.5 * np.einsum('ikdc,ljab,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        Rijab += 0.5 * np.einsum('jkdc,liab,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        # - 1/2 P_ab t_klad t_ijcb <kl||cd>
        Rijab -= 0.5 * np.einsum('klad,ijcb,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])
        Rijab += 0.5 * np.einsum('klbd,ijca,klcd->ijab', t_ijab, t_ijab, self.MO[o, o, v, v])

        # P_ij P_ab t_ikac t_klcd <db||lj>
        Rijab += np.einsum('ikac,klcd,dblj->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab -= np.einsum('jkac,klcd,dbli->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab -= np.einsum('ikbc,klcd,dalj->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab += np.einsum('jkbc,klcd,dali->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        # 1/4 t_klcd t_ijcd <ab||kl>
        Rijab += 0.25 * np.einsum('klcd,ijcd,abkl->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        # 1/4 t_klcd t_klab <cd||ij>
        Rijab += 0.25 * np.einsum('klcd,klab,cdij->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        # - 1/2 P_ab t_klcd t_ijac <bd||kl>
        Rijab -= 0.5 * np.einsum('klcd,ijac,bdkl->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab += 0.5 * np.einsum('klcd,ijbc,adkl->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        # - 1/2 P_ij t_klcd t_ikab <cd||jl>
        Rijab -= 0.5 * np.einsum('klcd,ikab,cdjl->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab += 0.5 * np.einsum('klcd,jkab,cdil->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        # - 1/2 P_ab t_klcd t_klbd <ac||ij>
        Rijab -= 0.5 * np.einsum('klcd,klbd,acij->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab += 0.5 * np.einsum('klcd,klad,bcij->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        # - 1/2 P_ij t_klcd t_jlcd <ab||ik>
        Rijab -= 0.5 * np.einsum('klcd,jlcd,abik->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])
        Rijab += 0.5 * np.einsum('klcd,ilcd,abjk->ijab', t_ijab, t_ijab, self.MO[v, v, o, o])

        # New T3s
        # P^(a/bc)_(k/ij) t_ijad <bc||dk>
        Rijkabc = ndot('ijad,bcdk->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc -= ndot('kjad,bcdi->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc -= ndot('ikad,bcdj->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc -= ndot('ijbd,acdk->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc += ndot('kjbd,acdi->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc += ndot('ikbd,acdj->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc -= ndot('ijcd,badk->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc += ndot('kjcd,badi->ijkabc', t_ijab, self.MO[v, v, v, o])
        Rijkabc += ndot('ikcd,badj->ijkabc', t_ijab, self.MO[v, v, v, o])

        # - P^(c/ab)_(i/jk) t_ilab <lc||jk>
        Rijkabc -= ndot('ilab,lcjk->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc += ndot('jlab,lcik->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc += ndot('klab,lcji->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc += ndot('ilcb,lajk->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc -= ndot('jlcb,laik->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc -= ndot('klcb,laji->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc += ndot('ilac,lbjk->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc -= ndot('jlac,lbik->ijkabc', t_ijab, self.MO[o, v, o, o])
        Rijkabc -= ndot('klac,lbji->ijkabc', t_ijab, self.MO[o, v, o, o])

        # Apply denominators
        #new_tia =  t_ia.copy() 
        new_tia = Ria / self.d_ia
        #new_tijab = t_ijab.copy() 
        new_tijab = Rijab / self.d_ijab
        new_tijkabc = Rijkabc /self.d_ijkabc

        return new_tia, new_tijab, new_tijkabc

    # Compute UCC(4) correlation energy
    def corr_energy(self, t_ijab):
        # Make slices
        o = slice(0, self.no_occ)
        v = slice(self.no_occ, self.no_mo)
        E_corr = ndot('ijab,ijab->', t_ijab, self.MO[o, o, v, v], prefactor=0.25)
        temp = 0.5 * np.einsum('jlbd,ijab,ikac,cdkl->', t_ijab, t_ijab, t_ijab, self.MO[v, v, o, o])
        temp += 0.0625 * np.einsum('ijab,klcd,ijcd,abkl->', t_ijab, t_ijab, t_ijab, self.MO[v, v, o, o])
        temp -= 0.25 * np.einsum('klcd,ijab,ijac,bdkl->', t_ijab, t_ijab, t_ijab, self.MO[v, v, o, o])
        #temp -= 0.25 * np.einsum('klcd,ijab,ikab,cdjl->', t_ijab, t_ijab, t_ijab, self.MO[v, v, o, o])
        #temp += 0.5 * np.einsum('klcd,ijab,jkcd,bdil->', t_ijab, t_ijab, t_ijab, self.MO[v, v, o, o])
        #temp += 0.5 * np.einsum('klcd,ijab,ikcb,adjl->', t_ijab, t_ijab, t_ijab, self.MO[v, v, o, o])
        E_corr -= 0.25 * temp
        return E_corr

    def do_CC(self, e_conv=1e-8, r_conv=1e-7, maxiter=40):
        self.old_e = self.corr_energy(self.t_ijab)
        print('Iteration\t\t CCSD Correlation energy\t\tDifference\t\tRMS\nMP2\t\t\t {}'.format(self.old_e))

    # Iterate until convergence
        for i in range(maxiter):
            new_tia, new_tijab, new_tijkabc = self.update_ts(self.t_ia, self.t_ijab, self.t_ijkabc)
            new_e = self.corr_energy(new_tijab)
            rms = np.linalg.norm(new_tia - self.t_ia)
            rms += np.linalg.norm(new_tijab - self.t_ijab)
            print('CC Iteration: {}\t\t {}\t\t{}\t\t{}'.format(i, new_e, abs(new_e - self.old_e), rms))
            if(abs(new_e - self.old_e) < e_conv and abs(rms) < r_conv):
                print('Convergence reached.\n CCSD Correlation energy: {}\n'.format(new_e))
                self.t_ia = new_tia
                self.t_ijab = new_tijab
                self.t_ijkabc = new_tijkabc
                break

            self.t_ia = new_tia
            self.t_ijab = new_tijab
            self.t_ijkabc = new_tijkabc
            self.old_e = new_e

        return new_e
