
#!/usr/bin/env python
"""
ITL R02-S02 spot projector data analysis

This script is adapted from a notebook by Theo Schutt (Stanford/SLAC) that investigated PSF distortions (i.e. 2nd moment distortions) in the ITL R02-S02 sensor.
The original notebooks can be found in /u/ki/schutt20/notebooks/ on the SLAC cluster.

The main change made on the original script was to upload the dataset using Butler.
"""
__author__ = "Johnny Esteves"

#-------------------------------------------------------------------------------

# Load in data
import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, binned_statistic_2d

from lsst.daf.butler import Butler

import sys
sys.path.append('/gpfs/slac/kipac/fs1/u/esteves/codes/treeRingAnalysis/mixcoatl/python/')
## MixCOATL imports
from mixcoatl.sourcegrid import DistortedGrid


#### YU
from astropy.stats import mad_std
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
import progressbar
#### YU


#-------------------------------------------------------------------------------
import os

class SpotgridCatalog():
    """
    Class for spotgrid catalog.

    input:
    repository: str : path to the Butler Repo
    catalog_collection: str : collection name
    catalog_collection: str : calibration collection name
    sensor: str: e2v or ITL
    """
    def __init__(self, repository, catalog_collection, calib_collection):
        self.butler = Butler(repository)
        self.registry = self.butler.registry
        
        self.catalog_collection = str(catalog_collection)
        self.calib_collection   = str(calib_collection)
        
        # retrieve metadata from an image
        aref = list(self.registry.queryDatasets('postISRCCD', collections=self.calib_collection))[0]
        animg=self.butler.get(aref,collections=self.calib_collection)
        self.header = animg.getMetadata().toDict()
        self.sensorbay=f'{self.header["RAFTBAY"].lower()}_{self.header["CCDSLOT"].lower()}'
        self.save=f'tmp/{self.header["RAFTBAY"].lower()}_{self.header["CCDSLOT"].lower()}'
        
        self.spot_size = 49*49
#        print(self.header["LSST_NUM"][:3], self.header["CCD_MANU"])
        self.sensor    = "e2v" if self.header["LSST_NUM"][:3] == "E2V" else "ITL"


        if not os.path.isdir('tmp'): os.makedirs('tmp')
            
        
        print('Repository        : %s'%(repository))
        print('catalog collection: %s'%(catalog_collection))
        print('calib collection  : %s'%(calib_collection))
        print('\n')

        # self.get_calibration_table()
        # self.load_data()
        # self.compute_statistics()
        # #self.compute_spotgrid()
        # self.filter_spots()
        # self.calibrate()

    def get_calibration_table(self):
        
        ## Get calibration table and grid
        self.ref = list(self.registry.queryDatasets('gridCalibration', collections=self.calib_collection))[0]
        self.calib_table = self.butler.get(self.ref, collections=self.calib_collection)
        self.calib_grid = DistortedGrid.from_astropy(self.calib_table)
        
        ## Get source catalogs
        self.catalog_refs = list(self.registry.queryDatasets('gridSpotSrc', collections=self.catalog_collection))#[:200]
        self.num_catalogs = len(self.catalog_refs)
    
    def save_data(self):
        x_arr = np.empty((self.spot_size, self.num_catalogs))
        y_arr = np.empty((self.spot_size, self.num_catalogs))
        
        xx_arr = np.empty((self.spot_size, self.num_catalogs))
        yy_arr = np.empty((self.spot_size, self.num_catalogs))
        xy_arr = np.empty((self.spot_size, self.num_catalogs))

        instFlux_arr = np.empty((self.spot_size, self.num_catalogs))
        instFluxErr_arr = np.empty((self.spot_size, self.num_catalogs))
        
        dx_arr = np.empty((self.spot_size, self.num_catalogs))
        dy_arr = np.empty((self.spot_size, self.num_catalogs))
        
        dxx_arr = np.empty((self.spot_size, self.num_catalogs))
        dyy_arr = np.empty((self.spot_size, self.num_catalogs))
        dxy_arr = np.empty((self.spot_size, self.num_catalogs))
        
        x_center= np.empty((self.spot_size,self.num_catalogs))
        y_center= np.empty((self.spot_size,self.num_catalogs))
        
        dFlux_arr = np.empty((self.spot_size, self.num_catalogs))
        dg1_arr   = np.empty((self.spot_size, self.num_catalogs))
        dg2_arr   = np.empty((self.spot_size, self.num_catalogs))

        for i, ref in progressbar.progressbar(enumerate(self.catalog_refs), redirect_stdout=True):       
#        for i, ref in enumerate(self.catalog_refs):
#            print(f'Loading catalog {i+1}/{self.num_catalogs}.')
            
            catalog = self.butler.get(ref, collections=self.catalog_collection)
            grid = DistortedGrid.from_astropy(catalog.asAstropy())
            
            xx    = np.full(self.spot_size, np.nan)
            yy    = np.full(self.spot_size, np.nan)
            xy    = np.full(self.spot_size, np.nan)
            dFlux = np.full(self.spot_size, np.nan)
            dg1   = np.full(self.spot_size, np.nan)
            dg2   = np.full(self.spot_size, np.nan)

            # Get Center Point
            x_center[:,i] = np.full(self.spot_size,catalog.getMetadata()['GRID_X0'])
            y_center[:,i] = np.full(self.spot_size,catalog.getMetadata()['GRID_Y0'])
            
            ## Get centroid shifts
            norm_dy = grid.norm_dy - self.calib_grid.norm_dy
            norm_dx = grid.norm_dx - self.calib_grid.norm_dx
            dy, dx = grid.convert_normalized_shifts(norm_dy, norm_dx)

            ## Get source positions
            y, x = grid.get_centroids()
            calib_dy, calib_dx = grid.convert_normalized_shifts(self.calib_grid.norm_dy, self.calib_grid.norm_dx)
            y += calib_dy
            x += calib_dx

            ## Get shape distortions
            indices = catalog['spotgrid_index']
            select  = indices >= 0
            spot_idx= indices[select]
        
            # Identify which spots are and are not included in the catalog
            # We will set those not included in the catalog to np.nan
            nan_idx  = np.setdiff1d(range(self.spot_size), spot_idx)

            # Set the included spots to their respective values
            xx[spot_idx] = catalog['base_SdssShape_xx'][select]
            yy[spot_idx] = catalog['base_SdssShape_yy'][select]
            xy[spot_idx] = catalog['base_SdssShape_xy'][select]
            
            dxx = xx - self.calib_table['base_SdssShape_xx']
            dyy = yy - self.calib_table['base_SdssShape_yy']
            dxy = xy - self.calib_table['base_SdssShape_xy']

            instFlux_arr[spot_idx, i]    = catalog['base_SdssShape_instFlux'][select]
            instFluxErr_arr[spot_idx, i] = catalog['base_SdssShape_instFluxErr'][select]

            dFlux[spot_idx] = catalog['base_SdssShape_instFlux'][select] - self.calib_table['base_SdssShape_instFlux'][spot_idx]
            #### What's the purpose of those perc_error? These seem to complicate residuals of shears, by not letting the average to be zero
#            dg1[spot_idx]   = get_perc_error(catalog['ext_shapeHSM_HsmShapeKsb_g1'][select], self.calib_table['ext_shapeHSM_HsmShapeKsb_g1'][spot_idx])
#            dg2[spot_idx]   = get_perc_error(catalog['ext_shapeHSM_HsmShapeKsb_g2'][select], self.calib_table['ext_shapeHSM_HsmShapeKsb_g2'][spot_idx])
            dg1[spot_idx]   = catalog['ext_shapeHSM_HsmShapeKsb_g1'][select] - self.calib_table['ext_shapeHSM_HsmShapeKsb_g1'][spot_idx]
            dg2[spot_idx]   = catalog['ext_shapeHSM_HsmShapeKsb_g2'][select] - self.calib_table['ext_shapeHSM_HsmShapeKsb_g2'][spot_idx]
            
            # Set the not included spots to np.nan
            instFlux_arr[nan_idx, i]    = np.nan
            instFluxErr_arr[nan_idx, i] = np.nan

            dFlux_arr[:,i] = dFlux
            dg1_arr[:,i]   = dg1
            dg2_arr[:,i]   = dg2

            x_arr[:, i] = x
            y_arr[:, i] = y
            dx_arr[:, i] = dx
            dy_arr[:, i] = dy

            xx_arr[:, i] = xx
            yy_arr[:, i] = yy
            xy_arr[:, i] = xy
            
            dxx_arr[:, i] = dxx
            dyy_arr[:, i] = dyy
            dxy_arr[:, i] = dxy
        
        print(f'Saving arrays for {self.sensor} {self.save}.')
        np.save(f'{self.save}_x_arr.npy', x_arr)
        np.save(f'{self.save}_y_arr.npy', y_arr)
        np.save(f'{self.save}_xx_arr.npy', xx_arr)
        np.save(f'{self.save}_xy_arr.npy', xy_arr)
        np.save(f'{self.save}_yy_arr.npy', yy_arr)
        np.save(f'{self.save}_dx_arr.npy', dx_arr)
        np.save(f'{self.save}_dy_arr.npy', dy_arr)
        np.save(f'{self.save}_dxx_arr.npy', dxx_arr)
        np.save(f'{self.save}_dxy_arr.npy', dxy_arr)
        np.save(f'{self.save}_dyy_arr.npy', dyy_arr)
        np.save(f'{self.save}_instFlux_arr.npy', instFlux_arr)
        np.save(f'{self.save}_instFluxErr_arr.npy', instFluxErr_arr)
        np.save(f'{self.save}_x_center_arr.npy', x_center)
        np.save(f'{self.save}_y_center_arr.npy', y_center)
        
        np.save(f'{self.save}_dFlux_arr.npy', dFlux_arr)
        np.save(f'{self.save}_dg1_arr.npy', dg1_arr)
        np.save(f'{self.save}_dg2_arr.npy', dg2_arr)

        # np.save(f'{self.save}_date_arr.npy', date_arr)
        # np.save(f'{self.save}_expnum_arr.npy', expnum_arr)
        
    def load_data(self):
        for m in ['x', 'y', 'xx', 'xy', 'yy', 'dx', 'dy', 'dxx', 'dyy', 'dxy', 'instFlux', 'instFluxErr', 'x_center', 'y_center', 'dFlux','dg1','dg2']:
            if os.path.exists(f'{self.save}_{m}_arr.npy'):
                print(f'Found {self.save}_{m}_arr.npy')
                continue
            else:
                self.save_data()

        print(f'Loading data for {self.sensor} {self.save}.')
        self.x_arr = np.load(f'{self.save}_x_arr.npy')
        self.y_arr = np.load(f'{self.save}_y_arr.npy')
        self.xx_arr = np.load(f'{self.save}_xx_arr.npy')
        self.xy_arr = np.load(f'{self.save}_xy_arr.npy')
        self.yy_arr = np.load(f'{self.save}_yy_arr.npy')
        self.dx_arr = np.load(f'{self.save}_dx_arr.npy')
        self.dy_arr = np.load(f'{self.save}_dy_arr.npy')
        self.dxx_arr = np.load(f'{self.save}_dxx_arr.npy')
        self.dxy_arr = np.load(f'{self.save}_dxy_arr.npy')
        self.dyy_arr = np.load(f'{self.save}_dyy_arr.npy')
        self.instFlux_arr = np.load(f'{self.save}_instFlux_arr.npy')
        self.instFluxErr_arr = np.load(f'{self.save}_instFluxErr_arr.npy')
        self.x_center = np.load(f'{self.save}_x_center_arr.npy')
        self.y_center = np.load(f'{self.save}_y_center_arr.npy')
        self.dFlux_arr= np.load(f'{self.save}_dFlux_arr.npy')
        self.dg1_arr  = np.load(f'{self.save}_dg1_arr.npy')
        self.dg2_arr  = np.load(f'{self.save}_dg2_arr.npy')   

        # self.date_arr = np.load(f'{self.save}_date_arr.npy')
        # self.expnum_arr = np.load(f'{self.save}_expnum_arr.npy')

    #-------------------------------------------------------------------------------
    # Computation Routines
    # - Still need to sort out a few of the _2 quantities, mean vs med, etc.
    #-------------------------------------------------------------------------------

    def compute_statistics(self):
        print(f'Computing statistics for {self.sensor} {self.save}.')

        #averages all exposures together of each spot # average of row
        self.xx_mean = np.nanmean(self.xx_arr, axis=1)
        self.xx_med = np.nanmedian(self.xx_arr, axis=1)
        self.xx_std = np.nanstd(self.xx_arr, axis=1)
        
        self.yy_mean = np.nanmean(self.yy_arr, axis=1)
        self.yy_med = np.nanmedian(self.yy_arr, axis=1)
        self.yy_std = np.nanstd(self.yy_arr, axis=1)
        
        self.xy_mean = np.nanmean(self.xy_arr, axis=1)
        self.xy_med = np.nanmedian(self.xy_arr, axis=1)
        self.xy_std = np.nanstd(self.xy_arr, axis=1)
        
        self.xxyy_err = np.sqrt(self.xx_std**2 + self.yy_std**2)

        #averages all exposures together of each spot 
        self.dxx_mean = np.nanmean(self.dxx_arr, axis=1)
        self.dxx_med = np.nanmedian(self.dxx_arr, axis=1)
        self.dxx_std = np.nanstd(self.dxx_arr, axis=1)
        
        self.dyy_mean = np.nanmean(self.dyy_arr, axis=1)
        self.dyy_med = np.nanmedian(self.dyy_arr, axis=1)
        self.dyy_std = np.nanstd(self.dyy_arr, axis=1)
        
        self.dxy_mean = np.nanmean(self.dxy_arr, axis=1)
        self.dxy_med = np.nanmedian(self.dxy_arr, axis=1)
        self.dxy_std = np.nanstd(self.dxy_arr, axis=1)
        
        self.dxxyy_err = np.sqrt(self.dxx_std**2 + self.dyy_std**2)

        self.instFlux_mean = np.nanmean(self.instFlux_arr, axis=1)
        self.instFlux_med = np.nanmedian(self.instFlux_arr, axis=1)

        self.dFlux_med = np.nanmean(self.dFlux_arr, axis=1)
        self.dg1_med   = np.nanmedian(self.dg1_arr, axis=1)
        self.dg2_med   = np.nanmedian(self.dg1_arr, axis=1)

        #averages all exposures together for each spot
        self.xx_med_2 = np.nanmedian(self.xx_arr, axis=0)
        self.yy_med_2 = np.nanmedian(self.yy_arr, axis=0)
        
        #averages all spots together for each exposure # average of column
        self.xx_mean_2 = np.nanmean(self.xx_arr, axis=0)
        self.yy_mean_2 = np.nanmean(self.yy_arr, axis=0)

    def compute_spotgrid(self,idx=None):
        nsize = len(self.x_arr[0])
        self.nmising = np.array([np.count_nonzero(np.isnan(self.xx_arr[:,i])) for i in range(nsize)])
        
        if idx is None: idx = np.argmin(self.nmising)
        
        self.x = self.x_arr[:,idx]
        self.y = self.y_arr[:,idx]
        self.x0= self.x_center[0][idx]
        self.y0= self.y_center[0][idx]
        
    def filter_spots(self, value=0.06):
        print(f'Computing filter spots for {self.sensor} {self.save}.')


        self.spot_filter = (
                (self.xxyy_err < value )
                & (self.instFlux_med > 0.2*np.nanmax(self.instFlux_med) )
            )
        
        
        self.xfltr = self.x_arr[self.spot_filter]
        self.yfltr = self.y_arr[self.spot_filter]
        self.xxfltr = self.xx_arr[self.spot_filter]
        self.yyfltr = self.yy_arr[self.spot_filter]
        self.xyfltr = self.xy_arr[self.spot_filter]

        self.xx_med_3 = np.nanmedian(self.xxfltr, axis=1)
        self.yy_med_3 = np.nanmedian(self.yyfltr, axis=1)

    ## Now we can plot the distribution of mean 2nd moments for the spots. We see the tail above ~5.5 px^2 is diminished from before.
    #
    ## #averages all spots together for each exposure
    #xx_mean_2 = np.nanmedian(xxfltr, axis=1)
    #yy_mean_2 = np.nanmedian(yyfltr, axis=1)
    #print(len(xx_mean_2))
    def clean( self, x ):
        """
        compute median of x for each exposure
        and subtract median(x) from each spot on the exposure"""
        stdxx=np.nanstd(self.dxx_arr[self.spot_filter],axis=0)
        stdyy=np.nanstd(self.dyy_arr[self.spot_filter],axis=0)
        stdxy=np.nanstd(self.dxy_arr[self.spot_filter],axis=0)

#        x[:, (stdxx/stdxy>2) | (stdyy/stdxy>2) ] = np.nan
        sigclip = SigmaClip(sigma=3, maxiters=3)
        return (x-np.nanmedian(sigclip(x,masked=False,axis=0),axis=0))
#        return x

    def clean2( self, x ):
        """
        compute median of x for each exposure
        and subtract median(x) from each spot on the exposure
        """
        def function(data, a, b, c ):
            x = data[0]
            y = data[1]
            return a * x + b * y + c
        
        stdxx=np.nanstd(self.dxx_arr[self.spot_filter],axis=0)
        stdyy=np.nanstd(self.dyy_arr[self.spot_filter],axis=0)
        stdxy=np.nanstd(self.dxy_arr[self.spot_filter],axis=0)

#        x[:, (stdxx/stdxy>2) | (stdyy/stdxy>2) ] = np.nan
        sigclip = SigmaClip(sigma=3, maxiters=3)
        z= (x-np.nanmedian(sigclip(x,masked=False,axis=0),axis=0))
        self.tmp = []
        for i in range(len(stdxx)):
            zorig=z[:,i][self.spot_filter]
            finite=np.isfinite(zorig)
            zsel=zorig[finite]
            x=self.xfltr[:,i][finite]
            y=self.yfltr[:,i][finite]
            parameters, covariance = curve_fit(function, [x , y], zsel)
            Z = function(np.array([self.xfltr[:,i], self.yfltr[:,i]]), *parameters)
            self.tmp.append(parameters)
            z[:,i][self.spot_filter] = (zorig - Z)
        return z

    def calibrate(self):
        # Apply calibration to all stars
        self.deltaX  = self.clean2(self.dx_arr)[self.spot_filter]
        self.deltaY  = self.clean2(self.dy_arr)[self.spot_filter]
        self.deltaR  = np.sqrt(self.deltaX**2+self.deltaY**2)
        
        #subtract "calibrated" grid I_xx+yy (median) moment from each star
        self.deltaXX = self.clean2(self.dxx_arr)[self.spot_filter]
        self.deltaYY = self.clean2(self.dyy_arr)[self.spot_filter]
        self.deltaXY = self.clean2(self.dxy_arr)[self.spot_filter]   
        self.deltaT   = self.deltaXX + self.deltaYY

        self.deltaF  = self.clean2(self.dFlux_arr/self.instFlux_arr)[self.spot_filter]
        self.deltag1 = self.clean2(self.dg1_arr)[self.spot_filter]
        self.deltag2 = self.clean2(self.dg2_arr)[self.spot_filter]

        # spot_filter = np.load('r02_s02_spotfilter.npy')
        
        #self.xxmedfltr = self.xx_med[self.spot_filter]
        #self.yymedfltr = self.yy_med[self.spot_filter]
        #self.xymedfltr = self.xy_med[self.spot_filter]
        #
        #self.deltaXX = self.xxfltr - np.reshape(self.xxmedfltr, (-1,1))
        #self.deltaXY = self.xyfltr - np.reshape(self.xymedfltr, (-1,1))
        #self.deltaYY = self.yyfltr - np.reshape(self.yymedfltr, (-1,1))

        # self.deltaXX = self.xx_arr[self.spot_filter] - np.reshape(self.xx_med[self.spot_filter], (-1,1))
        # self.deltaXY = self.xy_arr[self.spot_filter] - np.reshape(self.xy_med[self.spot_filter], (-1,1))
        # self.deltaYY = self.yy_arr[self.spot_filter] - np.reshape(self.yy_med[self.spot_filter], (-1,1))

        #self.dT = self.deltaT.flatten()

        ##---------------------------------

        ##get radial distance from tree ring center (assumed to be positioned at (4072,4000))
        ##TO-DO: determine actual TR center
        #
        rad  = np.sqrt((4072 - self.xfltr)**2 + (4000 - self.yfltr)**2)
        #rad = np.sqrt((self.x_center[self.spot_filter] - self.xfltr)**2 + (self.y_center[self.spot_filter] - self.yfltr)**2)
        
        #remove NaN entries from data
        nanmask = np.isfinite(self.deltaXX) #masks all the NaN entries created when a catalog didn't have 2401 entries
        self.xfltr_flat = self.xfltr[nanmask].flatten()
        self.yfltr_flat = self.yfltr[nanmask].flatten()
        
        self.dXX = self.deltaXX[nanmask].flatten()
        self.dYY = self.deltaYY[nanmask].flatten()
        self.dXY = self.deltaXY[nanmask].flatten()
        
        self.r  = rad[nanmask].flatten()
        self.dT = self.deltaT[nanmask].flatten()

        self.dX = self.deltaX[nanmask].flatten()
        self.dY = self.deltaY[nanmask].flatten()
        self.dR= self.deltaR[nanmask].flatten()

        self.dF = self.deltaF[nanmask].flatten() 
        self.dg1= self.deltag1[nanmask].flatten()
        self.dg2= self.deltag2[nanmask].flatten()

def get_perc_error(x,y):
    return (x-y)/np.max(np.abs(np.vstack([x,y])),0)
    #return (x-y)/((np.abs(x)+np.abs(y))/2.)
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('repository'        ,type=str, help='Repository path for the Bulter')
    parser.add_argument('catalog_collection',type=str, help='Catalog collection name')
    parser.add_argument('calib_collection'  ,type=str, help='Calibration collection name')
    parser.add_argument('--sensor',type=str ,required=False,default='ITL',
                        help='CCD type to investigate [str].')
    args = parser.parse_args()

    if args.sensor not in ['ITL', 'e2v']:
        parser.error('Please specify type [ITL, e2v].')

    # The command line is not working
    # The butler is not finding the collection name.

    # spotgrid = SpotgridCatalog(args.repository, args.catalog_collection, args.calib_collection, 
    #                            sensor=args.sensor)

    # spotgrid.get_calibration_table()
    # spotgrid.load_data()
    # spotgrid.compute_statistics()
    # #spotgrid.compute_spotgrid()
    # spotgrid.filter_spots()
    # spotgrid.calibrate()


#    import pdb;pdb.set_trace()
