import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.io.ascii as asc
import pysynphot as S
import pyregion


class Photometry(object):
    def __init__(self, global_photometry, im_root, id, region_files, sci_ext=0
                aper_correct_file=None):
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

        self.reg_files=region_files
        self.sci_ext=sci_ext
        self.aper_correct_file = aper_correct_file

        self.process_global_photometry()
        self.process_image_files()
        self.process_resolved_photometry()

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
            self.ref_phot_dict[band]['pysyn']=self.global_phot_dict[band]['pysyn']
            self.ref_phot_dict[band]['global']['flam']=self.global_phot_dict['flam']
            self.ref_phot_dict[band]['global']['eflam']=self.global_phot_dict['eflam']

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

        self.ref_im = im_hdu[self.sci_ext].data*1.0
        self.ref_seg = seg_hdu[self.sci_ext].data*1.0
        self.ref_wht = wht_hdu[self.sci_ext].data*1.0

        self.ref_hdu=im_hdu
        self.ref_wcs = pywcs.WCS(im_hdu[self.sci_ext].header)

        seg_hdu.close()
        im_hdu.close()

    def process_resolved_photometry(self):
        """
        TBD
        """

        seg_mask = np.zeros_like(ref_seg, dtype=np.bool)
        seg_mask[self.ref_seg==self.global_id]=True

        # Create masks using segmentation map and user provided region reg_files
        # to calculate the initial photometry for each band
        for ireg, reg_file in enumerate(self.reg_files):
            reg_handle = pyregion.open(reg_file)
            reg_mask = reg_handle.get_mask(hdu=self.ref_hdu[self.sci_ext])
            mask = reg_mask & seg_mask

            self.resolved_ids.append(1e4+ireg+1)
            id_key = 'bin_{0}'.format(ireg+1)
            self.ref_phot_dict[band][id_key] = {}

            for iband, band in enumerate(self.bands):

                if iband == 0:
                    im = self.ref_im*1.0
                    wht = self.ref_wht*1.0
                else:
                    if self._isIR(band):
                        im_path = self.im_root+band+'_drc_sci.fits'
                        wht_path = self.im_root+band+'_drc_wht.fits'
                    else:
                        im_path = self.im_root+band+'_drc_sci.fits'
                        wht_path = self.im_root+band+'_drc_wht.fits'
                    im_hdu = pyfits.open(im_path)
                    wht_hdu = pyfits.open(wht_path)
                    im = im_hdu[self.sci_ext].data*1.0
                    wht = wht_hdu[self.sci_ext].data*1.0
                    im_hdu.close()
                    wht_hdu.close()

                flam_ = im[mask].sum()
                eflam_ = np.sqrt(np.sum(1.0/wht[mask]))
                Npix = mask.sum()

                self.ref_phot_dict[band][id_key]['flam']=flam_
                self.ref_phot_dict[band][id_key]['eflam']=eflam_
                self.ref_phot_dict[band][id_key]['Npix']=Npix

        # Implement aperture correction to fluxes and errors if necessary
        # files are provided by user
        band=self.bands[0]
        flam_=0.0
        for ireg in range(len(self.reg_files)):
            id_key = 'bin_{0}'.format(ireg+1)
            flam_+=self.ref_phot_dict[band][id_key]['flam']=flam_
        self.phot_scale = self.ref_phot_dict[band]['global']['flam']/flam_

        print('#####################')
        print('#####################')
        print('#####################')
        print('Aperture correction for resolved bins = {0:1.2f}'.format(self.phot_scale))

        for ireg in range(len(self.reg_files)):
            for iband, band in enumerate(self.bands):
                self.ref_phot_dict[band][id_key]['flam']*=self.phot_scale
                self.ref_phot_dict[band][id_key]['eflam']*=self.phot_scale

        if self.aper_correct_file is not None:
            tab=asc.read(self.aper_correct_file)
            for ireg in range(len(self.reg_files)):
                id_key = 'bin_{0}'.format(ireg+1)
                    for iband, band in enumerate(self.bands):
                        loc = tab['bands'] == band
                        alpha = tab['alpha'][loc][0]
                        beta = tab['beta'][loc][0]
                        Npix = self.ref_phot_dict[band][id_key]['Npix']
                        self.ref_phot_dict[band][id_key]['eflam']=alpha*(np.sqrt(Npix))**beta
