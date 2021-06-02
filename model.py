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
import scipy.ndimage as nd
import pymc3 as pm
import theano.tensor as tt
from theano import shared

LOW_PERC=0.0014
HIGH_PERC=0.9986

class Photometry(object):
    def __init__(self, id, RA, DEC, global_photometry, global_bands, semiresolved_bands, semiresolved_bins,
                im_root, region_file=None, sci_ext=0, aper_correct_file=None):
        """
        TBD
        """
        self.global_id = id
        self.resolved_ids=[]
        self.RA = RA
        self.DEC = DEC

        self.global_phot_dict = global_photometry.copy()
        self.bands = list(global_photometry.keys())
        self.global_bands=global_bands
        self.semiresolved_bands = semiresolved_bands
        self.semiresolved_bins = semiresolved_bins
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


        # Create masks using segmentation map and user provided region reg_files
        # to calculate the initial photometry for each band
        for ireg, reg in enumerate(self.regions):
            seg_mask = np.zeros_like(self.ref_seg, dtype=np.bool)
            seg_mask[self.ref_seg==self.global_id]=True
            self.resolved_seg = np.zeros_like(self.ref_seg)

            self.resolved_ids.append(int(1e4+ireg+1))
            id_key = 'bin_{0}'.format(ireg+1)

            for iband, band in enumerate(self.bands):
                if band in self.global_bands or band in self.semiresolved_bands:
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
                if band in self.global_bands or band in self.semiresolved_bands:
                    continue
                self.ref_phot_dict[band][id_key]['flam']*=self.phot_scale
                self.ref_phot_dict[band][id_key]['eflam']*=self.phot_scale

        if self.aper_correct_file is not None:
            tab=asc.read(self.aper_correct_file)
            for ireg in range(len(self.regions)):
                id_key = 'bin_{0}'.format(ireg+1)
                for iband, band in enumerate(self.bands):
                    if band in self.global_bands or band in self.semiresolved_bands:
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
    def __init__(self,  z, grism_flts, id, RA, DEC, global_photometry, global_bands, semiresolved_bands, semiresolved_bins,
                im_root, region_file, sci_ext=0, aper_correct_file=None, MW_EBV=0.0001, size=40, fcontam=0.1,
                 gname='res_model', remove_grism_contam=True):
        """
        TBD
        """

        # Make photometric catalogs
        Photometry.__init__(self, id, RA, DEC, global_photometry, global_bands, semiresolved_bands, semiresolved_bins,
                            im_root, region_file, sci_ext, aper_correct_file)

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
            try:
                self.st[id_key]=StackFitter(files=self.im_root+'{0}_{1:05d}.stack.fits'.format(self.gname, id),
                                      group_name=self.gname, sys_err=0.0, mask_min=0.1,
                                      fit_stacks=False, fcontam=self.fcontam, extensions=None,
                                      min_ivar=0.01, overlap_threshold=3, verbose=True,chi2_threshold=0.0)
            except:
                continue

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

    def init_grism_mask(self, low_lim=1.2, up_lim=1.6):
        """
        TBD
        """
        beam = self.st['bin_1'].beams[0]
        cc = np.ones(shape=(len(self.st['bin_1'].beams), beam.sh[0], beam.sh[1])).astype(bool)
        for ib, beam in enumerate(self.st['bin_1'].beams):
            ref_wave = beam.wave / 1e4
            cond = (ref_wave * 0.0).astype(bool)
            loc = (ref_wave > low_lim) & (ref_wave < up_lim)
            cond = cond | loc
            cond = cond.astype(np.int)
            cond = np.ones(beam.sh) * cond[np.newaxis, :]
            cond = cond.astype(bool)
            cc[ib] = cc[ib] & cond
        cc = cc.astype(np.bool).flatten()
        self.fcc = np.ones((cc.shape[0])).astype(np.bool)
        fit_mask = np.zeros_like(cc, dtype=np.bool)
        for bin_id in self.st.keys():
            fit_mask = fit_mask | self.st[bin_id].fit_mask
        self.fcc = cc & fit_mask

    def perform_fit(self, tune=1000, draws=1000, chains=2, target_accept=0.9, max_treedepth=None,
                    method='advi+adapt_diag', save_trace=True):
        """
        TBD
        """
        if max_treedepth is not None:
            with self.pymc3_model:
                trace = pm.sample(tune=tune, draws=draws, n_init=200000, chains=chains,
                cores=4, init=method, target_accept=target_accept, max_treedepth=max_treedepth)
        else:
            with self.pymc3_model:
                trace = pm.sample(tune=tune, draws=draws, n_init=200000, chains=chains,
                cores=4, init=method, target_accept=target_accept)

        self.trace = trace

        self.trace_values = {}
        for ky in self.trace.varnames:
            self.trace_values[ky] = self.trace.get_values(ky)

        self.ppc = pm.sample_posterior_predictive(self.trace, model=self.pymc3_model)

        if save_trace:
            hkl.dump(self.trace_values, self.im_root+'trace.hkl', mode='w', compression='lzf')
            hkl.dump(self.ppc, self.im_root+'ppc.hkl', mode='w', compression='lzf')
            pm.summary(self.trace).to_csv(self.im_root+'trace_summary.dat', sep=' ', mode='w')


    def load_fit(self, path_to_trace=None, path_to_ppc=None):
        """
        TBD
        """
        if path_to_trace is None:
            self.trace_values=hkl.load(self.im_root+'trace.hkl')
        else:
            self.trace_values=hkl.load(path_to_trace)

        if path_to_ppc is None:
            self.ppc=hkl.load(self.im_root+'ppc.hkl')
        else:
            self.ppc=hkl.load(path_to_ppc)

    def load_joint_models(self, phot_prior_dict, weights, path=None):
        """
        TBD
        """
        import fsps

        if path is None:
            self.Model=hkl.load(self.im_root+'Model.hkl')
        else:
            self.Model=hkl.load(path)

        theta_labels=list(phot_prior_dict.keys())
        if weights is None:
            weights = np.ones_like(phot_prior_dict[theta_labels[0]])

        xN=self.Model['bin_1']['phot'].shape[0]
        yN=self.Model['bin_1']['phot'].shape[1]
        xN+=1
        yN+=1
        self.xN=xN
        self.yN=yN

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
        self.AGE_edge=AGE_edge
        self.AGE_width = (AGE_edge[1:]-AGE_edge[:-1])

    def init_fitter(self, sfh_prior='linear_ar2', k=1, tau=400, include_spectroscopy=True,
                    include_global=True, low_pol=0, up_pol=2, regularize_old=False,
                    mixture_photometry=True, include_semiresolved=True, complex_dust_geometry=False):
        """
        TBD
        """

        self.init_grism_mask()

        A_spec_container=[]
        A_phot_container=[]
        A_mass_container=[]
        for bin_counter, bin_id in enumerate(self.st.keys()):
            A_spec_container.append(self.Model[bin_id]['spec'])
            A_phot_container.append(self.Model[bin_id]['phot'])
            A_mass_container.append(self.Model[bin_id]['stellar_mass'])
        A_spec_container=np.asarray(A_spec_container)
        A_phot_container=np.asarray(A_phot_container)
        A_mass_container=np.asarray(A_mass_container)

        A_spec_reduced=np.transpose((A_spec_container[:,:,:,:, self.fcc]*1.0).reshape((len(list(self.st.keys())),
                                    (self.xN-1)*(self.yN-1), self.AGE.shape[0],self.fcc.sum())), axes=(2, 0, 1, 3))
        A_phot_reduced = np.transpose((A_phot_container*1.0).reshape((len(list(self.st.keys())),(self.xN-1)*(self.yN-1),
                        self.AGE.shape[0],len(self.bands))), axes=(2, 3, 0, 1))

        data_s = self.st['bin_1'].scif[self.fcc]*1.0
        stdf_s = 1.0/self.st['bin_1'].sivarf[self.fcc]*1.0

        data_p_res = []
        stdf_p_res = []
        iloc_res=[]
        iloc_global = []
        iloc_semiresolved = []
        iloc_total = []
        for bin_counter, bin_id in enumerate(self.st.keys()):
            for iband, band in enumerate(self.bands):
                if band in self.global_bands:
                    if bin_counter==0:
                        iloc_global.append(iband)
                elif band in self.semiresolved_bands:
                    if bin_counter==0:
                        iloc_semiresolved.append(iband)
                else:
                    if bin_counter==0:
                        iloc_res.append(iband)
                    data_p_res.append(self.ref_phot_dict[band][bin_id]['flam'])
                    stdf_p_res.append(self.ref_phot_dict[band][bin_id]['eflam'])
        data_p_res = np.asarray(data_p_res)
        stdf_p_res = np.asarray(stdf_p_res)

        normal_phot = 1.0/np.mean(data_p_res)
        self.normal_phot = normal_phot

        data_p_global = []
        stdf_p_global = []
        if include_global:
            for band in self.bands:
                if band in self.global_bands:
                    data_p_global.append(self.ref_phot_dict[band]['global']['flam'])
                    stdf_p_global.append(self.ref_phot_dict[band]['global']['eflam'])

        data_p_global = np.asarray(data_p_global)
        stdf_p_global = np.asarray(stdf_p_global)

        data_p_semiresolved = []
        stdf_p_semiresolved = []
        if include_global:
            for band in self.bands:
                if band in self.semiresolved_bands:
                    data_p_semiresolved.append(self.ref_phot_dict[band]['global']['flam'])
                    stdf_p_semiresolved.append(self.ref_phot_dict[band]['global']['eflam'])

        data_p_semiresolved = np.asarray(data_p_semiresolved)
        stdf_p_semiresolved = np.asarray(stdf_p_semiresolved)

        sh_A_bg_reduced=shared(self.st['bin_1'].A_bg[:,self.fcc]*1.0)

        A_spec_reduced_flatten=np.transpose(A_spec_reduced,axes=[1,0,2,3]).reshape((len(list(self.st.keys())),
                                        self.AGE.shape[0]*(self.xN-1)*(self.yN-1),self.fcc.sum()))
        sh_A_spec_reduced=[]
        sh_A_spec_idx=[]
        for ii in range(len(list(self.st.keys()))):
            ind=np.argwhere(A_spec_container[ii][0,0,0,self.fcc]!=0).flatten()
            sh_A_spec_idx.append(ind)
            red_temp_arr=[]
            for jj in range(A_spec_reduced_flatten.shape[1]):
                red_temp_arr.append(A_spec_reduced_flatten[ii,jj][ind]*1.0)
            red_temp_arr=np.asarray(red_temp_arr)
            sh_A_spec_reduced.append(shared(red_temp_arr))

        sh_A_phot=shared(normal_phot*A_phot_reduced[:, iloc_res,:, :].reshape((self.AGE.shape[0],len(iloc_res),len(list(self.st.keys())),(self.xN-1)*(self.yN-1))))
        sh_A_phot_ir=shared(normal_phot*A_phot_reduced[:, iloc_global,:, :].reshape((self.AGE.shape[0],len(iloc_global),len(list(self.st.keys())),(self.xN-1)*(self.yN-1))))
        sh_A_phot_semiresolved=shared(normal_phot*A_phot_reduced[:, iloc_semiresolved,:, :].reshape((self.AGE.shape[0],len(iloc_semiresolved),len(list(self.st.keys())),(self.xN-1)*(self.yN-1))))

        contamf=[]
        for beam in self.st['bin_1'].beams:
            contamf.extend(beam.contamf)
        contamf=np.asarray(contamf)
        sh_contamf=shared(contamf[self.fcc]*1.0)

        N=self.AGE.shape[0]
        M=len(list(self.st.keys()))
        P=int(len(self.bands)-len(self.global_bands)-len(self.semiresolved_bands))
        L=A_spec_reduced.shape[-1]
        NxNy=(self.xN-1)*(self.yN-1)
        pol_sh = (len(list(self.st.keys())))*(up_pol-low_pol)
        Nbg=self.st['bin_1'].N

        self.N = N # Number of sps templates
        self.M = M # Number of analyzed resolved bins
        self.P = P # Number of resolved photometric bands
        self.L = L # Number of grism pixels

        sh_stdf_s=shared(stdf_s*1.0)
        sh_data_s=shared(data_s*1.0)
        sh_stdf_p=shared(normal_phot*stdf_p_res.reshape((M,P)).T.flatten()[:,None]*np.ones((P*M, NxNy)))
        sh_data_p=shared(normal_phot*data_p_res.reshape((M,P)).T.flatten())
        sh_flam=shared(normal_phot*data_p_global*1.0)
        sh_eflam=shared(normal_phot*stdf_p_global*1.0)
        sh_flam_semiresolved=shared(normal_phot*data_p_semiresolved*1.0)
        sh_eflam_semiresolved=shared(normal_phot*stdf_p_semiresolved*1.0)

        if include_global:
            semiresolved_ind = []
            for ii, id_key in enumerate(list(self.st.keys())):
                if id_key in self.semiresolved_bins:
                    semiresolved_ind.append(True)
                else:
                    semiresolved_ind.append(False)

        flat_flam=np.zeros((len(list(self.st.keys())),self.st['bin_1'].scif.shape[0]))
        for ii ,bin_id in enumerate(self.st.keys()):
            i0=0
            for ib, beam in enumerate(self.st[bin_id].beams):
                beam.kernel = self.master_kernel[bin_id][ib] * 1.0
                beam.kernel *= self.ref_phot_dict[self.bands[0]][bin_id]['flam']
                beam._build_model()
                d_px=int(beam.sh[0]*beam.sh[1])
                flat_flam[ii,i0:i0+d_px]=beam.compute_model()
                i0+=d_px
        xpf=[]
        for beam in self.st['bin_1'].beams:
            xpf.append(np.ones(beam.sh[0])[:,None]*(beam.wave-1e4)[None,:]/1e4)
        xpf=np.asarray(xpf).flatten()
        A_poly=[(xpf**order)[None,:]*flat_flam for order in np.arange(low_pol,up_pol)]
        A_poly=np.asarray(A_poly)
        sh_A_poly_reduced=shared(A_poly[:,:,self.fcc].reshape(((len(list(self.st.keys())))*(up_pol-low_pol),self.fcc.sum())))

        dt = self.AGE_width*1.0
        A_mass_interp=nd.gaussian_filter(np.median(A_mass_container.reshape((len(list(self.st.keys())),
                                                    (self.yN-1)*(self.xN-1),self.AGE.shape[0])),
                                                   axis=1),sigma=5.0).T
        sh_dt = shared(((dt[:,None]/A_mass_interp)/(dt[:,None]/A_mass_interp).max(axis=0)[None,:]))
        sh_reg=shared(np.exp(-self.AGE/5)[:,None]*np.ones((N,M)))

        young_age = self.AGE*0.0
        young_age[self.AGE<=0.01]=1.0
        sh_young_age=shared(young_age)
        pivot_wave = []
        for band in self.bands:
            pivot_wave.append(self.ref_phot_dict[band]['pysyn'].pivot())
        pivot_wave = np.asarray(pivot_wave)
        pwave = pivot_wave[iloc_res]*1.0
        pwave_ir = pivot_wave[iloc_global]*1.0
        pwave_semiresolved = pivot_wave[iloc_semiresolved]*1.0
        sh_pwave=shared(pwave)
        sh_pwave_ir=shared(pwave_ir)
        sh_pwave_semiresolved=shared(pwave_semiresolved)
        sh_grism_wave = shared(self.st['bin_1'].wavef[self.fcc]*1.0)
        if complex_dust_geometry:
            dust1_=[]
            dust2_=[]
            dust_index_=[]
            for id_bin in self.st.keys():
                dust1_.append(np.median(self.Model[id_bin]['dust1'],axis=-1).flatten())
                dust2_.append(np.median(self.Model[id_bin]['dust2'],axis=-1).flatten())
                dust_index_.append(np.median(self.Model[id_bin]['dust_index'],axis=-1).flatten())
            dust1_=np.asarray(dust1_)
            dust2_=np.asarray(dust2_)
            dust_index_=np.asarray(dust_index_)
            sh_dust1_=shared(dust1_)
            sh_dust2_=shared(dust2_)
            sh_dust_index_=shared(dust_index_)
            sh= (4, N, pwave.shape[0], M, NxNy)
            sh_ir= (4, N, pwave_ir.shape[0], M, NxNy)
            sh_semiresolved= (4, N, pwave_semiresolved.shape[0], M, NxNy)

            #corr_ = get_atten_curve(dust1_, dust2_, dust_index_, pwave, young_age)
            #self.corr_=corr_
            #corr_ir_ = get_atten_curve(dust1_, dust2_, dust_index_, pwave_ir, young_age)
            #self.corr_ir_=corr_ir_
            #corr_semiresolved_ = get_atten_curve(dust1_, dust2_, dust_index_, pwave_semiresolved, young_age)
            #self.corr_semiresolved_=corr_semiresolved_

        self.pymc3_model = pm.Model()

        if sfh_prior=='linear_ar2':
            with self.pymc3_model:
                BoundAR = pm.Bound(pm.AR, lower=tt.zeros((N, M)))
                rho=[2.0,-1.0]
                sfr0 = BoundAR('sfr0', rho=rho,tau=tau,shape=(N, M))
        elif sfh_prior=='log_ar2':
            with self.pymc3_model:
                rho=[2.0,-1.0]
                lsfr = pm.AR('lsfr', rho=rho, tau=tau, shape=(N, M))
                sfr0 = pm.Deterministic('sfr0', tt.power(10, lsfr))
        elif sfh_prior=='linear_ar1':
            with self.pymc3_model:
                BoundAR = pm.Bound(pm.AR, lower=tt.zeros((N, M)))
                rho=[k]
                sfr0 = BoundAR('sfr0',rho=rho,tau=tau,shape=(N, M))
        elif sfh_prior=='log_ar1':
            with self.pymc3_model:
                rho=[k]
                lsfr = pm.AR('lsfr', rho=rho,tau=tau,shape=(N, M))
                sfr0 = pm.Deterministic('sfr0', tt.power(10, lsfr))
        else:
            print('#####################')
            print('#####################')
            print('#####################')
            print('SFH prior not found! Current options: \n')
            print('linear_ar2: AR(2) \n')
            print('log_ar2: AR(2) in log space \n')
            print('linear_ar1: AR(1) \n')
            print('log_ar1: AR(1) in log space \n')
            print('See Akhshik et. al. (2020): https://arxiv.org/pdf/2008.02276.pdf')
            return None

        if regularize_old:
            with self.pymc3_model:
                sfr=pm.Deterministic('sfr',sfr0*sh_reg)
        else:
            with self.pymc3_model:
                sfr=pm.Deterministic('sfr',sfr0*1.0)

        with self.pymc3_model:
            x=pm.Deterministic('x', sfr*sh_dt)

            BoundNormal=pm.Bound(pm.Normal,lower=tt.zeros(M))

            alpha = pm.Gamma('alpha', alpha=(NxNy-1), beta=1.0, shape=(M,))
            dir_a = pm.Beta('dir_a', tt.ones((M,NxNy)), alpha.dimshuffle(0,'x')*tt.ones((M,NxNy)), shape=(M,NxNy))
            w = pm.Deterministic('w', stick_breaking(dir_a, M))

            if complex_dust_geometry:
                BoundedNormal=pm.Bound(pm.Normal,lower=-0.5*tt.ones((M,NxNy)),upper=0.5*tt.ones((M,NxNy)))
                dir_dust_a = pm.Beta('dir_dust_a',1.0, 3.0, shape=(M,4))
                w_dust_geo = pm.Deterministic('w_dust_geo', stick_breaking(dir_dust_a, M))

                ext_dust1 = pm.HalfNormal('ext_dust1',sd=1.5*tt.ones((M,NxNy)),shape=(M,NxNy))
                ext_dust2 = pm.HalfNormal('ext_dust2',sd=1.5*tt.ones((M,NxNy)),shape=(M,NxNy))
                ext_dust_index = BoundedNormal('ext_dust_index',mu=0,sd=0.5*tt.ones((M,NxNy)),shape=(M,NxNy))

                corr_ = get_atten_curve(sh_dust1_, sh_dust2_, sh_dust_index_, ext_dust1, ext_dust2, ext_dust_index, sh_pwave, sh_young_age, sh)
                corr_ir_ = get_atten_curve(sh_dust1_, sh_dust2_, sh_dust_index_, ext_dust1, ext_dust2, ext_dust_index, sh_pwave_ir, sh_young_age, sh_ir)
                corr_semiresolved_ = get_atten_curve(sh_dust1_, sh_dust2_, sh_dust_index_, ext_dust1, ext_dust2, ext_dust_index, sh_pwave_semiresolved, sh_young_age, sh_semiresolved)

                sh_A_phot_w = tt.zeros_like(sh_A_phot)
                sh_A_phot_ir_w = tt.zeros_like(sh_A_phot_ir)
                sh_A_phot_semiresolved_w = tt.zeros_like(sh_A_phot_semiresolved)
                for ii in range(M):
                    sh_A_phot_w = tt.set_subtensor(sh_A_phot_w[:,:,ii], tt.tensordot(w_dust_geo[ii],corr_[:,:,:,ii],axes=[[0],[0]])*sh_A_phot[:,:,ii])
                    sh_A_phot_ir_w = tt.set_subtensor(sh_A_phot_ir_w[:,:,ii],tt.tensordot(w_dust_geo[ii],corr_ir_[:,:,:,ii],axes=[[0],[0]])*sh_A_phot_ir[:,:,ii])
                    sh_A_phot_semiresolved_w = tt.set_subtensor(sh_A_phot_semiresolved_w[:,:,ii], tt.tensordot(w_dust_geo[ii],corr_semiresolved_[:,:,:,ii],axes=[[0],[0]])*sh_A_phot_semiresolved[:,:,ii])

            else:
                sh_A_phot_w = sh_A_phot
                sh_A_phot_ir_w = sh_A_phot_ir
                sh_A_phot_semiresolved_w = sh_A_phot_semiresolved

            contam_scale=pm.Normal('contam_scale', mu=0.0, sd=1.0)
            bg_scale = pm.Normal('bg_scale', mu=0.0, sd=1.0, shape=(Nbg))
            px = pm.Normal('px', mu=0.0, sd=1.0, shape=(pol_sh))
            pw = BoundNormal('pw',mu=1.0,sd=0.1,shape=(M,))
            scale_x = x.dimshuffle(0,1,'x')*w.dimshuffle('x',0,1)

            est_model_spec = tt.zeros(L)
            for ii in range(M):
                tmp=tt.zeros_like(est_model_spec)
                tmp=tt.set_subtensor(tmp[(sh_A_spec_idx[ii])],
                                tt.tensordot(tt.flatten(scale_x[:,ii]),sh_A_spec_reduced[ii],axes=[[0],[0]]))
                est_model_spec+=tmp

            est_model_poly = tt.tensordot(px,sh_A_poly_reduced, axes=[[0],[0]])
            est_model_others = contam_scale*sh_contamf+tt.tensordot(bg_scale, sh_A_bg_reduced,axes=[[0],[0]])
            full_model = est_model_spec +est_model_poly + est_model_others
            if include_spectroscopy:
                Spec_obs=pm.Normal('Spec_obs',mu=full_model,sd=sh_stdf_s,
                                      observed=sh_data_s, shape=(L,))

        if mixture_photometry:
            with self.pymc3_model:
                est_model_phot=tt.zeros((P, M, NxNy))
                for ii in range(M):
                    est_model_phot = tt.set_subtensor(est_model_phot[:, ii],pw[ii]*tt.tensordot(x.T[ii],
                                                sh_A_phot_w[:, :, ii],axes=[[0],[0]]))
                comps_phot = pm.Normal.dist(mu=est_model_phot.reshape((P*M,NxNy)), sd=sh_stdf_p, shape=(P*M,NxNy))
                Phot_obs=pm.Mixture('Phot_obs',w=(w.dimshuffle('x',0,1)*tt.ones((P,M,NxNy))).reshape((P*M,NxNy)),
                                    comp_dists=comps_phot, observed=sh_data_p, shape=(P*M))
        else:
            with self.pymc3_model:
                est_model_phot=tt.zeros((P, M))
                for ii in range(M):
                    est_model_phot = tt.set_subtensor(est_model_phot[:, ii],pw[ii]*tt.tensordot(x[:,ii].dimshuffle(0,'x')*w[ii].dimshuffle('x',0),
                                                sh_A_phot_w[:, :, ii],axes=[[0,1],[0,2]]))
                Phot_obs = pm.Normal('Phot_obs',mu=tt.flatten(est_model_phot), sd=sh_stdf_p[:,0], observed=sh_data_p)

        if include_global:
            with self.pymc3_model:
                est_model_phot_ir=tt.zeros(len(iloc_global))
                for ii in range(M):
                    est_model_phot_ir+=pw[ii]*tt.tensordot(x[:,ii].dimshuffle(0,'x')*w[ii].dimshuffle('x',0),
                                                    sh_A_phot_ir_w[:,:,ii,:],axes=[[0,1],[0,2]])
                Phot_obs_ir=pm.Normal('Phot_obs_ir',mu=est_model_phot_ir, sd=sh_eflam, observed=sh_flam)

            if len(self.semiresolved_bands)!=0 and include_semiresolved:
                with self.pymc3_model:
                    est_model_phot_semiresolved=tt.zeros(len(iloc_semiresolved))
                    est_model_phot_unresolved=tt.zeros(len(iloc_semiresolved))
                    for ii in range(M):
                        if semiresolved_ind[ii]:
                            est_model_phot_semiresolved+=pw[ii]*tt.tensordot(x[:,ii].dimshuffle(0,'x')*w[ii].dimshuffle('x',0),
                                                    sh_A_phot_semiresolved_w[:,:,ii,:],axes=[[0,1],[0,2]])
                        else:
                            est_model_phot_unresolved+=pw[ii]*tt.tensordot(x[:,ii].dimshuffle(0,'x')*w[ii].dimshuffle('x',0),
                                                    sh_A_phot_semiresolved_w[:,:,ii,:],axes=[[0,1],[0,2]])

                    Phot_obs_semiresolved=pm.Normal('Phot_obs_semiresolved',mu=est_model_phot_semiresolved,
                                    sd=sh_eflam_semiresolved, observed=sh_flam_semiresolved)

                    eps_censored = pm.HalfNormal('eps_censored', sd=sh_eflam_semiresolved/(2*M), shape=(len(iloc_semiresolved)))
                    Phot_obs_unresolved = pm.Potential('Phot_obs_unresolved', upper_limit_likelihood(est_model_phot_unresolved,
                                            eps_censored, len(iloc_semiresolved), 3*sh_eflam_semiresolved))

    def make_joint_models(self, phot_prior_dict, weights_dict=None,
                                  PCA_keys=['logzsol', 'dust2'], PCA_nbox=[3, 4],
                                  Nbox=15, make_PCA_plot=True, save_data=True):
        """
        TBD
        """
        import fsps

        theta_labels=list(phot_prior_dict['bin_1'].keys())

        xN, yN=PCA_nbox
        xN+=1
        yN+=1
        self.xN=xN
        self.yN=yN

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
        self.AGE_edge=AGE_edge
        self.AGE_width = (AGE_edge[1:]-AGE_edge[:-1])

        self.Model={}
        for bin_counter, id_key in enumerate(list(self.st.keys())):
            self.Model[id_key]={}
            if weights_dict is None:
                weights = np.ones_like(phot_prior_dict[id_key][theta_labels[0]])
            else:
                weights = weights_dict[id_key]*1.0

            for ib, beam in enumerate(self.st[id_key].beams):
                beam.kernel = self.master_kernel[id_key][ib] * 1.0
                beam.kernel *= self.ref_phot_dict[self.bands[0]][id_key]['flam']
                beam._build_model()

            if PCA_keys is None:
                from sklearn.decomposition import PCA

                first_lbl = theta_labels[0]
                X_PCA = np.zeros((len(theta_labels), phot_prior_dict[id_key][first_lbl].shape[0]))
                for ilbl, lbl in enumerate(theta_labels):
                    X_PCA[ilbl] = phot_prior_dict[id_key][lbl]*1.0

                pca_ = PCA(n_components=2)
                X_new = pca_.fit_transform(X_PCA.T)
                self.pca_ = pca_
                X_grid = np.linspace(wquantile(X_new[:,0], weights, LOW_PERC),
                                wquantile(X_new[:,0], weights, HIGH_PERC), xN)
                Y_grid = np.linspace(wquantile(X_new[:,1], weights, LOW_PERC),
                                    wquantile(X_new[:,1], weights, HIGH_PERC), yN)

            else:
                X_grid = np.linspace(wquantile(phot_prior_dict[id_key][PCA_keys[0]], weights, LOW_PERC),
                                wquantile(phot_prior_dict[id_key][PCA_keys[0]], weights, HIGH_PERC),
                                     xN)
                Y_grid = np.linspace(wquantile(phot_prior_dict[id_key][PCA_keys[1]], weights, LOW_PERC),
                                    wquantile(phot_prior_dict[id_key][PCA_keys[1]], weights, HIGH_PERC),
                                     yN)

            X_box_draws = np.zeros((X_grid.shape[0] - 1, Y_grid.shape[0] - 1, Nbox))
            Y_box_draws = X_box_draws*0.0
            draws_dict={}
            for lbl in theta_labels:
                draws_dict[lbl]=X_box_draws*0.0

            if make_PCA_plot:
                plt.figure(figsize=(8, 8))
                if PCA_keys is None:
                    plt.scatter(X_new[:,0], X_new[:,1], color='black', s=1, alpha=0.1)
                else:
                    plt.scatter(phot_prior_dict[id_key][PCA_keys[0]], phot_prior_dict[id_key][PCA_keys[1]], color='black', s=1, alpha=0.1)

            for ind_mg in range(X_grid[:-1].shape[0]):

                mg_min = X_grid[ind_mg] * 1.0
                mg_max = X_grid[ind_mg + 1] * 1.0
                for ind_dg in range(Y_grid[:-1].shape[0]):
                    dg_min = Y_grid[ind_dg] * 1.0
                    dg_max = Y_grid[ind_dg + 1] * 1.0

                    if PCA_keys is None:
                        inbox = (X_new[:,0] >= mg_min) & (X_new[:,0] < mg_max) & \
                            (X_new[:,1] >= dg_min) & (X_new[:,1] < dg_max)
                    else:
                        inbox = (phot_prior_dict[id_key][PCA_keys[0]] >= mg_min) & (phot_prior_dict[id_key][PCA_keys[0]] < mg_max) & \
                            (phot_prior_dict[id_key][PCA_keys[1]] >= dg_min) & (phot_prior_dict[id_key][PCA_keys[1]] < dg_max)

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
                        draws_dict[lbl][ind_mg, ind_dg]=phot_prior_dict[id_key][lbl][inbox][ind_draws]*1.0
                        self.Model[id_key][lbl]=draws_dict[lbl]

                    if make_PCA_plot:
                        if PCA_keys is None:
                            plt.scatter(X_new[:,0][inbox][ind_draws] * 1.0,
                                    X_new[:,1][inbox][ind_draws] * 1.0,
                                    s=weight_inbox.sum() * 2500, marker='*')

                        else:
                            plt.scatter(phot_prior_dict[id_key][PCA_keys[0]][inbox][ind_draws] * 1.0,
                                    phot_prior_dict[id_key][PCA_keys[1]][inbox][ind_draws] * 1.0,
                                    s=weight_inbox.sum() * 2500, marker='*')

            if make_PCA_plot:
                for mg in X_grid:
                    plt.axvline(mg, color='crimson')
                for dg in Y_grid:
                    plt.axhline(dg, color='crimson')
                plt.xlim(X_grid.min(), X_grid.max())
                plt.ylim(Y_grid.min(), Y_grid.max())
                if PCA_keys is None:
                    plt.ylabel('PC2')
                    plt.xlabel('PC1')
                else:
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
                            A_fsps_tmp = []


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
                        A_fsps_tmp.append(fsps_templates)
                        if ind_dg==0 and ind_mg==0:
                            A_fsps = np.zeros((xN-1, yN-1, fsps_templates.shape[0], fsps_templates.shape[1], fsps_templates.shape[2]))

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
                    A_fsps[ind_mg, ind_dg] = np.median(np.asarray(A_fsps_tmp),axis=0)

            self.Model[id_key]['spec']=A_spec*1.0
            self.Model[id_key]['phot']=A_phot*1.0
            self.Model[id_key]['stellar_mass']=A_mass*1.0
            self.Model[id_key]['fsps']=A_fsps*1.0
        if save_data:
            hkl.dump(self.Model, self.im_root+'Model.hkl', mode='w', compression='lzf')
