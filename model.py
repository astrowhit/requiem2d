import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.io.ascii as asc
import pysynphot as S
import pyregion
import grizli
from grizli.multifit import GroupFLT, MultiBeam
from grizli.stack import StackFitter
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from utils import *
from tqdm import trange
import hickle as hkl


class Photometry(object):
    def __init__(self, id, global_photometry, im_root, region_file=None,
                sci_ext=0, aper_correct_file=None):
        """
        TBD
        """
        self.global_id = id
        self.resolved_ids=[]

        self.global_phot_dict = global_photometry.copy()
        self.bands = list(global_photometry.keys())

        self.im_root = im_root

        self.ref_phot_dict={}
        for band in self.bands:
            self.ref_phot_dict[band]={}

        self.reg_file = region_file
        self.regions = pyregion.open(region_file)
        self.sci_ext = sci_ext
        self.aper_correct_file = aper_correct_file

        self.process_global_photometry()
        self.process_image_files()
        self.process_resolved_photometry()
        self.write_updated_seg_file()

    def _isIR(self, band):
        """
        TBD
        """

        try:
            im_path = self.im_root+band+'_drz_sci.fits'
            im_hdu = pyfits.open(im_path)
            im_hdu.close()
            return True
        except FileNotFoundError:
            return False

    def process_global_photometry(self):
        """
        TBD
        """

        for band in self.bands:
            self.ref_phot_dict[band]['global']={}
            self.ref_phot_dict[band]['pysyn']=self.global_phot_dict[band]['pysyn']
            self.ref_phot_dict[band]['global']['flam']=self.global_phot_dict[band]['flam']
            self.ref_phot_dict[band]['global']['eflam']=self.global_phot_dict[band]['eflam']

    def process_image_files(self):
        """
        TBD
        """

        if self._isIR(self.bands[0]):
            im_path = self.im_root+self.bands[0]+'_drz_sci.fits'
            im_hdu = pyfits.open(im_path)
            wht_hdu = pyfits.open(self.im_root+self.bands[0]+'_drz_wht.fits')
        else:
            im_path = self.im_root+self.bands[0]+'_drc_sci.fits'
            im_hdu = pyfits.open(im_path)
            wht_hdu = pyfits.open(self.im_root+self.bands[0]+'_drc_wht.fits')

        seg_hdu = pyfits.open(self.im_root+self.bands[0]+'_seg.fits')
        self.ref_seg_path = self.im_root+self.bands[0]+'_seg.fits'

        self.ref_im_path=im_path
        self.ref_im = im_hdu[self.sci_ext].data*1.0
        self.ref_seg = seg_hdu[self.sci_ext].data*1
        self.ref_wht = wht_hdu[self.sci_ext].data*1.0

        self.ref_hdu=im_hdu
        self.ref_wcs = pywcs.WCS(im_hdu[self.sci_ext].header)


    def process_resolved_photometry(self):
        """
        TBD
        """

        seg_mask = np.zeros_like(self.ref_seg, dtype=np.bool)
        seg_mask[self.ref_seg==self.global_id]=True
        self.resolved_seg = np.zeros_like(self.ref_seg)

        # Create masks using segmentation map and user provided region reg_files
        # to calculate the initial photometry for each band
        for ireg, reg in enumerate(self.regions):
            self.resolved_ids.append(int(1e4+ireg+1))
            id_key = 'bin_{0}'.format(ireg+1)

            for iband, band in enumerate(self.bands):
                if band[:-1]=='IRAC':
                    continue
                self.ref_phot_dict[band][id_key] = {}
                if iband == 0:
                    im = self.ref_im*1.0
                    wht = self.ref_wht*1.0
                    reg_mask = self.regions.get_filter(self.ref_hdu[self.sci_ext].header)[ireg].mask(im.shape)
                    self.ref_phot_dict[band]['photflam']=self.ref_hdu[self.sci_ext].header['PHOTFLAM']*1.0
                    self.resolved_seg[seg_mask & reg_mask] = 1e4+ireg+1
                else:
                    if self._isIR(band):
                        im_path = self.im_root+band+'_drz_sci.fits'
                        wht_path = self.im_root+band+'_drz_wht.fits'
                    else:
                        im_path = self.im_root+band+'_drc_sci.fits'
                        wht_path = self.im_root+band+'_drc_wht.fits'
                    im_hdu = pyfits.open(im_path)
                    wht_hdu = pyfits.open(wht_path)
                    self.ref_phot_dict[band]['photflam']=im_hdu[self.sci_ext].header['PHOTFLAM']*1.0
                    im = im_hdu[self.sci_ext].data*1.0
                    wht = wht_hdu[self.sci_ext].data*1.0
                    reg_mask = self.regions.get_filter(im_hdu[self.sci_ext].header)[ireg].mask(im.shape)
                    im_hdu.close()
                    wht_hdu.close()

                mask = reg_mask & seg_mask
                flam_ = im[mask].sum()
                eflam_ = np.sqrt(np.sum(1.0/wht[mask]))
                Npix = mask.sum()

                self.ref_phot_dict[band][id_key]['flam']=flam_*self.ref_phot_dict[band]['photflam']
                self.ref_phot_dict[band][id_key]['eflam']=eflam_*self.ref_phot_dict[band]['photflam']
                self.ref_phot_dict[band][id_key]['Npix']=Npix

        # Implement aperture correction to fluxes and errors if necessary
        # files are provided by user
        band=self.bands[0]
        flam_=0.0
        for ireg in range(len(self.regions)):
            id_key = 'bin_{0}'.format(ireg+1)
            flam_+=self.ref_phot_dict[band][id_key]['flam']
        self.phot_scale = self.ref_phot_dict[band]['global']['flam']/flam_

        print('#####################')
        print('#####################')
        print('#####################')
        print('Aperture correction for resolved bins = {0:1.2f}'.format(self.phot_scale))

        for ireg in range(len(self.regions)):
            id_key = 'bin_{0}'.format(ireg+1)
            for iband, band in enumerate(self.bands):
                if band[:-1]=='IRAC':
                    continue
                self.ref_phot_dict[band][id_key]['flam']*=self.phot_scale
                self.ref_phot_dict[band][id_key]['eflam']*=self.phot_scale

        if self.aper_correct_file is not None:
            tab=asc.read(self.aper_correct_file)
            for ireg in range(len(self.regions)):
                id_key = 'bin_{0}'.format(ireg+1)
                for iband, band in enumerate(self.bands):
                    if band[:-1]=='IRAC':
                        continue
                    loc = tab['bands'] == band
                    alpha = tab['alpha'][loc][0]
                    beta = tab['beta'][loc][0]
                    Npix = self.ref_phot_dict[band][id_key]['Npix']
                    self.ref_phot_dict[band][id_key]['eflam'] = self.ref_phot_dict[band]['photflam']*alpha*(np.sqrt(Npix))**beta

        self.updated_seg = np.zeros_like(self.ref_seg)
        self.updated_seg[self.ref_seg==self.global_id]=self.resolved_seg[self.ref_seg==self.global_id]*1
        self.updated_seg[self.ref_seg!=self.global_id]=self.ref_seg[self.ref_seg!=self.global_id]*1


    def write_updated_seg_file(self, overwrite=True):
        """
        TBD
        """

        hdu_=pyfits.PrimaryHDU(data=self.updated_seg, header=self.ref_hdu[self.sci_ext].header)
        hdu_.writeto(self.im_root+self.bands[0]+'_updated_seg.fits', overwrite=overwrite)
        self.updated_seg_path = self.im_root+self.bands[0]+'_updated_seg.fits'



class ResolvedModel(Photometry):
    def __init__(self,  z, grism_flts, id, global_photometry, im_root, region_file,
                sci_ext=0, aper_correct_file=None, MW_EBV=0.0001, size=40, fcontam=0.1,
                 gname='res_model', remove_grism_contam=True):
        """
        TBD
        """

        # Make photometric catalogs
        Photometry.__init__(self, id, global_photometry, im_root, region_file,
                                sci_ext, aper_correct_file)

        # Initiate FLT containter
        self.grp = GroupFLT(grism_files=grism_flts, direct_files=[], cpu_count=4,
                            ref_file=self.ref_im_path, seg_file=self.updated_seg_path,
                            MW_EBV= MW_EBV)

        ref_catalog = asc.read(self.im_root+self.bands[0]+'.cat')
        self.grp.catalog = ref_catalog[ref_catalog['NUMBER']!=self.global_id].copy()

        if remove_grism_contam:
            self.reomve_grism_contam()

        # Initiate MultiBeam Object
        self.size=size
        self.MW_EBV = MW_EBV
        self.fcontam = fcontam
        self.gname = gname
        self.z = z

        beams=self.grp.get_beams(id=self.resolved_ids[0], size=self.size)
        self.mb=MultiBeam(beams, fcontam=self.fcontam, group_name=self.gname,
                        MW_EBV=self.MW_EBV)
        self.st={}
        self.master_kernel={}
        for bin_counter, id in enumerate(self.resolved_ids):
            id_key = 'bin_{0}'.format(bin_counter+1)
            for beam in self.mb.beams:
                beam.beam.id=id
            hdu, fig = self.mb.drizzle_grisms_and_PAs(fcontam=self.fcontam, flambda=False, size=self.size,
                                             kernel='point',usewcs=False,pixfrac=0.33, scale=1.0)
            hdu.writeto(self.im_root+'{0}_{1:05d}.stack.fits'.format(self.gname, id), overwrite=True)
            plt.close()

            self.st[id_key]=StackFitter(files=self.im_root+'{0}_{1:05d}.stack.fits'.format(self.gname, id),
                                      group_name=self.gname, sys_err=0.01, mask_min=0.1,
                                      fit_stacks=False, fcontam=self.fcontam, extensions=None,
                                      min_ivar=0.01, overlap_threshold=3, verbose=True,chi2_threshold=0.0)

            beam=self.st[id_key].beams[0]
            self.st[id_key].MW_EBV=self.MW_EBV

            self.master_kernel[id_key]=np.zeros((self.st[id_key].N,beam.kernel.shape[0],beam.kernel.shape[1]))
            self.st[id_key].initialize_masked_arrays()
            for ib, beam in enumerate(self.st[id_key].beams):
                beam.init_galactic_extinction(MW_EBV=self.MW_EBV)
                self.master_kernel[id_key][ib]=beam.kernel*1.0

    def reomve_grism_contam(self, save_data=False):
        """
        TBD
        """

        self.grp.compute_full_model()
        for i in range(5):
            self.grp.refine_list(mag_limits=[13, 25])
        if save_data:
            self.grp.save_full_data()

    def make_joint_models(self, phot_prior_dict, weights=None,
                                  PCA_keys=['logzsol', 'dust2'], PCA_nbox=[3, 4],
                                  Nbox=15, make_PCA_plot=True, save_data=True):
        """
        TBD
        """
        import fsps

        theta_labels=list(phot_prior_dict.keys())
        if weights is None:
            weights = np.ones_like(phot_prior_dict[theta_labels[0]])

        xN, yN=PCA_nbox
        xN+=1
        yN+=1

        cosmo = FlatLambdaCDM(H0=70.0 * u.km/u.s/u.Mpc, Om0=0.3)
        dL = cosmo.luminosity_distance(self.z).to(u.cm).value
        to_flamm=(3.8270e33/(4*np.pi*(dL**2)))

        sp_c = fsps.StellarPopulation(imf_type=1, logzsol=0, zcontinuous=1.0,
                                      dust_type=4,dust_index=0,
                                      dust1=0.5, dust2=0.3)

        wl, spec = sp_c.get_spectrum(tage=0.0, peraa=True)

        AGE = []
        SPEC = []
        STELLAR_MASS = []
        for ll in range(sp_c.log_age.shape[0]):
            if 10 ** sp_c.log_age[ll] / 1e9 > cosmo.age(self.z).value or 10 ** sp_c.log_age[ll] / 1e9 <= 0.001:
                continue
            AGE.append(10 ** sp_c.log_age[ll] / 1e9)
            SPEC.append(spec[ll])
            STELLAR_MASS.append(sp_c.stellar_mass[ll])
        WL = wl * 1.0
        AGE = np.asarray(AGE)
        SPEC = np.asarray(SPEC)
        STELLAR_MASS = np.asarray(STELLAR_MASS)
        AGE_edge=[1.08e-3]
        AGE_edge.extend((0.5*(AGE[1:]+AGE[:-1])).tolist())
        AGE_edge.append(cosmo.age(self.z).value)
        AGE_edge=np.asarray(AGE_edge)

        self.AGE=AGE
        self.Model={}
        for bin_counter, id in enumerate(self.resolved_ids):
            id_key = 'bin_{0}'.format(bin_counter+1)
            self.Model[id_key]={}
            for ib, beam in enumerate(self.st[id_key].beams):
                beam.kernel = self.master_kernel[id_key][ib] * 1.0
                beam.kernel *= self.ref_phot_dict[self.bands[0]]['global']['flam']
                beam._build_model()

            X_grid = np.linspace(wquantile(phot_prior_dict[PCA_keys[0]], weights, 0.0014),
                                wquantile(phot_prior_dict[PCA_keys[0]], weights, 0.9986),
                                     xN)
            Y_grid = np.linspace(wquantile(phot_prior_dict[PCA_keys[1]], weights, 0.0014),
                                    wquantile(phot_prior_dict[PCA_keys[1]], weights, 0.9986),
                                     yN)

            X_box_draws = np.zeros((X_grid.shape[0] - 1, Y_grid.shape[0] - 1, Nbox))
            Y_box_draws = X_box_draws*0.0
            draws_dict={}
            for lbl in theta_labels:
                draws_dict[lbl]=X_box_draws*0.0

            if make_PCA_plot:
                plt.figure(figsize=(8, 8))
                plt.scatter(phot_prior_dict[PCA_keys[0]], phot_prior_dict[PCA_keys[1]], color='black', s=1, alpha=0.1)

            for ind_mg in range(X_grid[:-1].shape[0]):

                mg_min = X_grid[ind_mg] * 1.0
                mg_max = X_grid[ind_mg + 1] * 1.0
                for ind_dg in range(Y_grid[:-1].shape[0]):
                    dg_min = Y_grid[ind_dg] * 1.0
                    dg_max = Y_grid[ind_dg + 1] * 1.0

                    inbox = (phot_prior_dict[PCA_keys[0]] >= mg_min) & (phot_prior_dict[PCA_keys[0]] < mg_max) & \
                            (phot_prior_dict[PCA_keys[1]] >= dg_min) & (phot_prior_dict[PCA_keys[1]] < dg_max)

                    weight_inbox = weights[inbox] * 1.0
                    if (weight_inbox > 0.0).sum() > Nbox:
                        replace = False
                    else:
                        replace = True
                    if weight_inbox.sum() != 0:
                        prob = weight_inbox / weight_inbox.sum()
                    else:
                        prob = np.ones_like(weight_inbox)
                        prob /= prob.sum()

                    ind_draws = np.cast[np.int32](np.random.choice(np.arange(inbox.sum()), Nbox,
                                                                   replace=replace, p=prob))

                    for lbl in theta_labels:
                        draws_dict[lbl][ind_mg, ind_dg]=phot_prior_dict[lbl][inbox][ind_draws]*1.0

                    if make_PCA_plot:
                        plt.scatter(phot_prior_dict[PCA_keys[0]][inbox][ind_draws] * 1.0,
                                    phot_prior_dict[PCA_keys[1]][inbox][ind_draws] * 1.0,
                                    s=weight_inbox.sum() * 2500, marker='*')

            if make_PCA_plot:
                for mg in X_grid:
                    plt.axvline(mg, color='crimson')
                for dg in Y_grid:
                    plt.axhline(dg, color='crimson')
                plt.xlim(X_grid.min(), X_grid.max())
                plt.ylim(Y_grid.min(), Y_grid.max())
                plt.ylabel(PCA_keys[1])
                plt.xlabel(PCA_keys[0])
                plt.savefig('{0}bin_{1}_PCA.pdf'.format(self.im_root, bin_counter+1))
                plt.close()

            A_spec = np.zeros((xN - 1, yN - 1, len(self.AGE), self.st[id_key].scif.shape[0]))
            A_mass = np.zeros((xN - 1, yN - 1, len(self.AGE)))
            A_phot = np.zeros((xN - 1, yN - 1, len(self.AGE), len(self.bands)))

            for ind_mg in range(xN - 1):
                for ind_dg in range(yN - 1):
                    print('\n Analyzing bin {4}, grid {0}/{1} x {2}/{3}...\n'.format(ind_mg + 1,
                                    xN - 1, ind_dg + 1, yN - 1, bin_counter+1))
                    for iiter in trange(Nbox):

                        for lbl in theta_labels:
                            sp_c.params[lbl] = draws_dict[lbl][ind_mg, ind_dg, iiter] * 1.0
                        sp_c.params['sfh'] = 1
                        sp_c.params['fburst'] = 0
                        sp_c.params['tau'] = 100
                        sp_c.params['const'] = 1.0

                        # Neb emissions
                        sp_c.params['gas_logu'] = -2.5
                        sp_c.params['add_neb_emission'] = True

                        SPEC = []
                        STELLAR_MASS = []
                        for ll in range(AGE.shape[0]):
                            sp_c.params['sf_start'] = cosmo.age(self.z).value - AGE_edge[ll + 1]
                            sp_c.params['sf_trunc'] = cosmo.age(self.z).value - AGE_edge[ll]
                            wl, spec = sp_c.get_spectrum(tage=cosmo.age(self.z).value, peraa=True)
                            SPEC.append(spec)
                            STELLAR_MASS.append(sp_c.stellar_mass)
                        WL = wl * 1.0
                        AGE = np.asarray(AGE)
                        SPEC = np.asarray(SPEC)
                        STELLAR_MASS = np.asarray(STELLAR_MASS)

                        if iiter == 0:
                            A_spec_tmp = np.zeros((Nbox, AGE.shape[0], self.st[id_key].scif.shape[0]))
                            A_mass_tmp = np.zeros((Nbox, AGE.shape[0]))
                            A_phot_tmp = np.zeros((Nbox, AGE.shape[0], len(self.bands)))

                        temp_holder = []
                        temp_sm_holder = []
                        bp = self.ref_phot_dict[self.bands[0]]['pysyn']
                        for ii in range(AGE.shape[0]):
                            sp = S.ArraySpectrum(wave=WL * (1 + self.z), flux=(SPEC[ii] / (1 + self.z)),
                                                 waveunits='angstrom', fluxunits='flam')
                            temp_fl = np.interp(bp.wave, sp.wave, sp.flux * to_flamm)
                            n_before = np.trapz(temp_fl * bp.wave * bp.throughput, bp.wave) / \
                                       np.trapz(bp.throughput / bp.wave, bp.wave) / bp.pivot() ** 2
                            sp = sp.renorm(self.ref_phot_dict[self.bands[0]][id_key]['flam'], 'flam', bp)
                            sp.convert('flam')
                            sp.convert('angstrom')
                            temp_fl = np.interp(bp.wave, sp.wave, sp.flux)
                            n_after = np.trapz(temp_fl * bp.wave * bp.throughput, bp.wave) / \
                                      np.trapz(bp.throughput / bp.wave, bp.wave) / bp.pivot() ** 2
                            sp.convert('flam')
                            sp.convert('angstrom')
                            temp_sm_holder.append((n_after / n_before) * STELLAR_MASS[ii])
                            temp_holder.append([sp.wave, sp.flux])
                        fsps_templates = np.asarray(temp_holder)
                        stellar_masses = np.asarray(temp_sm_holder)
                        A_mass_tmp[iiter] = stellar_masses

                        # Create spectroscopy
                        for ii in range(AGE.shape[0]):
                            i0 = 0
                            for j, beam in enumerate(self.st[id_key].beams):
                                d_px = int(beam.sh[0] * beam.sh[1])
                                A_spec_tmp[iiter, ii, i0:i0 + d_px] = beam.compute_model(spectrum_1d=fsps_templates[ii],
                                                                                        in_place=False, is_cgs=False)/self.ref_phot_dict[self.bands[0]][id_key]['flam']
                                i0 += d_px

                        # Create photometry
                        for ii in range(AGE.shape[0]):
                            for band_counter, band in enumerate(self.bands):
                                bandpass=self.ref_phot_dict[band]['pysyn']
                                templ_filter = np.interp(bandpass.wave, fsps_templates[ii, 0],
                                                         fsps_templates[ii, 1])
                                filter_norm = np.trapz(bandpass.throughput / bandpass.wave, bandpass.wave)
                                temp_int = np.trapz(bandpass.throughput * templ_filter * bandpass.wave, bandpass.wave) / \
                                           filter_norm
                                A_phot_tmp[iiter, ii, band_counter] = temp_int / bandpass.pivot() ** 2
                    A_spec[ind_mg, ind_dg] = np.median(A_spec_tmp, axis=0) * 1.0
                    A_phot[ind_mg, ind_dg] = np.median(A_phot_tmp, axis=0) * 1.0
                    A_mass[ind_mg, ind_dg] = np.median(A_mass_tmp, axis=0) * 1.0
            self.Model[id_key]['spec']=A_spec*1.0
            self.Model[id_key]['phot']=A_phot*1.0
            self.Model[id_key]['stellar_mass']=A_mass*1.0
        if save_data:
            hkl.dump(self.Model, self.im_root+'Model.hkl', mode='w')
