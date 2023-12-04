
import numpy as np
import nibabel as nib
import math
import csv
import matplotlib.cm as cm
import SimpleITK as sitk
import csv
from copy import deepcopy
import matplotlib.colors as mcolors
import nibabel as nib 
from matplotlib import pyplot as plt


class heatmapPlotter():
    def __init__(self, seed=None):
        self.seed = seed
        #self.shape = test_mri_nonorm[0].shape

    #ATTEMPT AT VISUALIZE_SALIENCY:
    #for j in range(len(test_data[0])):
    #    grads = netCNN.make_vis_saliency(test_data,j)
    #    plt.imshow(grads,alpha=0.6)
    
    def plot_idv_brain(self, heat_map, brain_img, ref_scale, fig=None, ax=None, contour_areas=[],
              x_idx=slice(0, 91), y_idx=slice(0, 109), z_idx=slice(0, 91),
              vmin=90, vmax=99.5, set_nan=True, cmap=None, c=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 12))
        img = deepcopy(heat_map)
        #if set_nan:
            #img[nmm_mask==0]=np.nan
        if cmap is None:
            cmap = mcolors.LinearSegmentedColormap.from_list(name='alphared',colors=[(1, 0, 0, 0),"darkred", "red", "darkorange", "orange", "yellow"],N=5000)    
        grey_vmin, grey_vmax = np.min(brain_img), np.max(brain_img)
        if brain_img is not None:
            brain = deepcopy(brain_img)
            ax.imshow(np.squeeze(brain[x_idx, y_idx, z_idx],-1), cmap="gray",   #was .T before (but I dont need to transpose the indices I dont think)
                     vmin=grey_vmin, vmax=grey_vmax ) #,alpha=.9
        vmin, vmax = np.percentile(ref_scale, vmin), np.percentile(ref_scale, vmax)
        im = ax.imshow(np.squeeze(img[x_idx, y_idx, z_idx],-1), cmap=cmap,     #was .T before (but I dont need to transpose the indices I dont think)
                   vmin=vmin, vmax=vmax, interpolation="gaussian", alpha=.7)
        ax.axis('off') 
        #plot_contours(contour_areas, x_idx, y_idx, z_idx, fig=fig, ax=ax, c=c)
        plt.gca().invert_yaxis()
        return fig, ax, im

    ##GRAD-CAM
    def GuidedGradCAM(self, test_data, test_mri_nonorm, model_filepath, netCNN, test_predsCNN):
        last_conv_layer_name = "features"  #maybe supposed to be fc1?
        classifier_layer_names = "CNNclass_output"   #supposed to have 2 layers??
        shape = test_mri_nonorm[0].shape
        cases = ["AD", "NC", "TP", "TN", "FP", "FN"]
        case_maps_GGC = {case: np.zeros(shape) for case in cases}
        mean_maps_GGC = {case: np.zeros(shape) for case in cases}
        counts = {case: 0 for case in cases}
        j=53   #CHANGE START POINT FOR NC DATA 
        while j < len(test_data[0]):  #CHANGE END POINT FOR NC DATA = len(test_data[0]), for AD data = len(test_data[0])/2 
            #sitk_mri = sitk.GetImageFromArray(test_mri_nonorm[j], isVector=True)  #use the non normalized image array
            #sitk.WriteImage(sitk_mri,model_filepath+'/figures/mri_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[4][j])+'_'+str(test_data[3][j])+'.nii') 
            #sitk_mri_normed = sitk.GetImageFromArray(test_data[0][j],isVector=True)  #check out the normalized image
            #sitk.WriteImage(sitk_mri_normed,model_filepath+'/figures/mri_normed_'+str(seed)+'_'+str(j)+'_'+test_data[4][j]+'_'+test_data[3][j]+'.nii')
            
            CNN_gradcam_map = netCNN.make_gradcam_heatmap2(test_data,j)
            #CNN_gradcam[j] = CNN_gradcam_map
            #CNN_sitk_gradcam = sitk.GetImageFromArray(CNN_gradcam_map, isVector=True)
            #CNN_sitk_gradcam.CopyInformation(sitk_mri)
            #sitk.WriteImage(CNN_sitk_gradcam,model_filepath+'/figures/CNN_gradcam_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[4][j])+'_'+str(test_data[3][j])+'.nii')
        
        #GUIDED BACKPROP
            CNN_gb_map = netCNN.guided_backprop(test_data,j)
            #CNN_gb[j] = CNN_gb_map
            #CNN_sitk_gb = sitk.GetImageFromArray(CNN_gb_map)
            #sitk.WriteImage(CNN_sitk_gb,model_filepath+'/figures/CNN_gb_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[4][j])+'_'+str(test_data[3][j])+'.nii')#

        #GUIDED GRAD-CAM
            CNN_guided_gradcam_map = CNN_gb_map * CNN_gradcam_map
            #CNN_guided_gradcam[j] = CNN_guided_gradcam_map
            #CNN_sitk_guided_gradcam = sitk.GetImageFromArray(CNN_guided_gradcam_map)
            #sitk.WriteImage(CNN_sitk_guided_gradcam,model_filepath+'/figures/CNN_guided_gradcam_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[4][j])+'_'+str(test_data[3][j])+'.nii')
            """ Just for now for memory purposes
        #Plot middle slice of each
            subplot_args = { 'nrows': 1, 'ncols': 5, 'figsize': (12, 4),
                         'subplot_kw': {'xticks': [], 'yticks': []} }
            f, ax = plt.subplots(**subplot_args)
            ax[0].set_title('Original Image', fontsize=11)
            ax[0].imshow(test_mri_nonorm[j][:,:,45,0],cmap='gray')
            ax[1].set_title('Guided Backprop overlay', fontsize=11)
            ax[1].imshow(test_mri_nonorm[j][:,:,45,0],cmap='gray')
            ax[1].imshow(CNN_gb_map[:,:,45,0],cmap='jet', alpha=0.4)
            ax[2].set_title('GRAD-CAM', fontsize=11)
            ax[2].imshow(CNN_gradcam_map[:,:,45,0],cmap='jet')
            ax[3].set_title('Guided GRAD-CAM', fontsize=11)
            ax[3].imshow(CNN_guided_gradcam_map[:,:,45,0],cmap='jet')
            ax[4].set_title('Guided GRAD-CAM overlay', fontsize=11)
            ax[4].imshow(test_mri_nonorm[j][:,:,45,0],cmap='gray')
            ax[4].imshow(CNN_guided_gradcam_map[:,:,45,0],cmap='jet', alpha=0.4)
            plt.savefig(model_filepath+'/figures/CNN_grad_maps_z45_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[4][j])+'_'+str(test_data[3][j])+'.png')
            #plt.show()
            fig.clf()
            plt.close(f)
            """
            #Sort maps by cases
            true_case = "AD" if test_data[3][j]==0 else "NC"
            if np.argmax(test_predsCNN[j])==0 and true_case=="AD":
                case = "TP"
            elif np.argmax(test_predsCNN[j])==0 and true_case!="AD":
                case = "FP"
            elif np.argmax(test_predsCNN[j])==1 and true_case=="NC":
                case = "TN"
            elif np.argmax(test_predsCNN[j])==1 and true_case!="NC":
                case = "FN"
            """
            #for Guided Grad Cam
            case_maps_GGC[case] += CNN_guided_gradcam_map
            counts[case] += 1
            case_maps_GGC[true_case] += CNN_guided_gradcam_map
            counts[true_case] += 1
            """
            #for Grad Cam
            case_maps_GGC[case] += CNN_gradcam_map
            counts[case] += 1
            case_maps_GGC[true_case] += CNN_gradcam_map
            counts[true_case] += 1
            
            print('counts: ',counts) 
            j+=1
        """ 
        #Plot INDIVIDUAL heatmaps - can't do this anymore because I removed CNN_gradcam, CNN_gb, CNN_guided_gradcam in order to save memory
        mean_maps_GGC["AD"] = case_maps_GGC["AD"]/counts["AD"]
        for j in range(len(test_data[0])):
            subplot_args = { 'nrows': 4, 'ncols': 1, 'figsize': (12, 12), 'sharey':True, 'sharex':True,
                         'subplot_kw': {'xticks': [], 'yticks': []} }
            fig, axes = plt.subplots(**subplot_args)
            vmin, vmax = 50, 99.5  #NOT SURE I WANT THIS (READ PAPER) - might be what is creating the 'mask' effect
            for ax, idx in zip(axes[:],[30, 40, 50, 60]):
                ax.text(-25, 22, "Slice " + str(idx), rotation="vertical", fontsize=20)
                fig, ax, im = self.plot_idv_brain(CNN_guided_gradcam[j], test_mri_nonorm[j], mean_maps_GGC["AD"],x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                                    vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
            ax.text(5, -20, "class: "+str(test_data[3][j])+", prediction: "+str(np.argmax(test_predsCNN[j])), fontsize=20)
            fig.tight_layout()
            fig.subplots_adjust(right=0.8, top=0.95, hspace=0.05, wspace=0.05)
            fig.suptitle("LRP for Patient "+str(test_data[4][j])+", ImageID: "+str(test_data[5][j]), fontsize=22, x=.41)
            cbar_ax = fig.add_axes([0.6, 0.15, 0.025, 0.7])
            cbar = fig.colorbar(im, shrink=0.5, ticks=[vmin, vmax], cax=cbar_ax)
            vmin_val, vmax_val = np.percentile(mean_maps_GGC["AD"], vmin), np.percentile(mean_maps_GGC["AD"], vmax)
            cbar.set_ticks([vmin_val, vmax_val])
            cbar.ax.set_yticklabels(['{0:.1f}%'.format(vmin), '{0:.1f}%'.format(vmax)],
                                   fontsize=16)
            cbar.set_label('Percentile of average AD patient values', rotation=270, fontsize=18)
            fig.savefig(model_filepath+'/figures/CNN_GGC_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[5][j])+'_'+str(test_data[3][j])+'.png')
            fig.clf()
            plt.close(fig)
        """
        return case_maps_GGC, counts   #Removed CNN_gradcam, CNN_gb, CNN_guided_gradcam to save memory
    
    #LAYERWISE RELEVANCE PROPAGATION    #https://github.com/moboehle/Pytorch-LRP/blob/master/Plotting%20brain%20maps.ipynb
    def LRP(self, test_data, test_mri_nonorm, model_filepath, netCNN, test_predsCNN):
        shape = test_mri_nonorm[0].shape
        print('length of test_data[3]: ',len(test_data[3]))
        #Run LRP for each test image
        cases = ["AD", "NC", "TP", "TN", "FP", "FN"]
        case_maps_LRP = {case: np.zeros(shape) for case in cases}
        mean_maps_LRP = {case: np.zeros(shape) for case in cases}
        counts = {case: 0 for case in cases}
        j=53   #CHANGE START POINT FOR NC DATA 
        while j < len(test_data[0]):    #CHANGE END POINT FOR NC DATA = len(test_data[0]), for AD data = len(test_data[0])/2 
            #sitk_mri = sitk.GetImageFromArray(test_mri_nonorm[j], isVector=True)  #use the non normalized image array
            #sitk.WriteImage(sitk_mri,model_filepath+'/figures/mri_'+str(seed)+'_'+str(j)+'_'+str(test_data[5][j])+'_'+str(test_data[3][j])+'.nii') 

            LRP_analysis = netCNN.LRP_heatmap(test_data, j) 
            CNN_LRP = LRP_analysis
            #CNN_sitk_LRP = sitk.GetImageFromArray(CNN_LRP[j], isVector=True)
            #CNN_sitk_LRP.CopyInformation(sitk_mri)
            #sitk.WriteImage(CNN_sitk_LRP,model_filepath+'/figures/CNN_LRP_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[5][j])+'_'+str(test_data[3][j])+'.nii')
        
            #Sort maps by cases
            true_case = "AD" if test_data[3][j]==0 else "NC"
            if np.argmax(test_predsCNN[j])==0 and true_case=="AD":
                case = "TP"
            elif np.argmax(test_predsCNN[j])==0 and true_case!="AD":
                case = "FP"
            elif np.argmax(test_predsCNN[j])==1 and true_case=="NC":
                case = "TN"
            elif np.argmax(test_predsCNN[j])==1 and true_case!="NC":
                case = "FN"
            #case_maps_LRP[case] += CNN_LRP[j]
            case_maps_LRP[case] += CNN_LRP
            counts[case] += 1
            #case_maps_LRP[true_case] += CNN_LRP[j]
            case_maps_LRP[true_case] += CNN_LRP
            counts[true_case] += 1
            print('counts: ',counts) 
            j+=1
        """ 
        #Plot INDIVIDUAL heatmaps - can't do this anymore because I removed CNN_LRP in order to save memory
        mean_maps_LRP["AD"] = case_maps_LRP["AD"]/counts["AD"]
        for j in range(len(test_data[0])):
            subplot_args = { 'nrows': 4, 'ncols': 1, 'figsize': (12, 12), 'sharey':True, 'sharex':True,
                         'subplot_kw': {'xticks': [], 'yticks': []} }
            fig, axes = plt.subplots(**subplot_args)
            vmin, vmax = 50, 99.5  #NOT SURE I WANT THIS (READ PAPER) - might be what is creating the 'mask' effect
            for ax, idx in zip(axes[:],[30, 40, 50, 60]):
                ax.text(-25, 22, "Slice " + str(idx), rotation="vertical", fontsize=20)
                fig, ax, im = self.plot_idv_brain(CNN_LRP[j], test_mri_nonorm[j], mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                                    vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
            ax.text(5, -20, "class: "+str(test_data[3][j])+", prediction: "+str(np.argmax(test_predsCNN[j])), fontsize=20)
            fig.tight_layout()
            fig.subplots_adjust(right=0.8, top=0.95, hspace=0.05, wspace=0.05)
            fig.suptitle("LRP for Patient "+str(test_data[4][j])+", ImageID: "+str(test_data[5][j]), fontsize=22, x=.41)
            cbar_ax = fig.add_axes([0.6, 0.15, 0.025, 0.7])
            cbar = fig.colorbar(im, shrink=0.5, ticks=[vmin, vmax], cax=cbar_ax)
            vmin_val, vmax_val = np.percentile(mean_maps_LRP["AD"], vmin), np.percentile(mean_maps_LRP["AD"], vmax)
            cbar.set_ticks([vmin_val, vmax_val])
            cbar.ax.set_yticklabels(['{0:.1f}%'.format(vmin), '{0:.1f}%'.format(vmax)],
                                   fontsize=16)
            cbar.set_label('Percentile of average AD patient values', rotation=270, fontsize=18)
            fig.savefig(model_filepath+'/figures/CNN_LRP_'+str(self.seed)+'_'+str(j)+'_'+str(test_data[5][j])+'_'+str(test_data[3][j])+'.png')
            fig.clf()
            plt.close(fig)
            
        """
        return case_maps_LRP, counts  #Removed CNN_LRP to save memory

    #Create AVERAGE heatmaps
    def plot_avg_maps(self, case_maps_LRP, counts, map_type, test_mri_nonorm, model_filepath, mean_map_AD):
        shape = test_mri_nonorm[0].shape
        cases = ["AD", "NC", "TP", "TN", "FP", "FN"]
        mean_maps_LRP = {case: np.zeros(shape) for case in cases}
        mean_maps_LRP["AD"] = mean_map_AD
        #Get the PET template
        proxy_image = nib.load(model_filepath + '/rbet_TEMPLATE_FDGPET_100.Resampled.nii')
        template = np.asarray(proxy_image.dataobj)
        PETtemplate = np.asarray(np.expand_dims(template, axis = -1))
        print('PET template shape: ', PETtemplate.shape)
        #Calculate the mean maps
        CNN_sitk_mean_maps_LRP = {case: np.zeros(shape) for case in cases}
        print('counts: ',counts) 
        for case in cases:
            is_all_0 = np.all((mean_maps_LRP[case]==0))
            if is_all_0:
                mean_maps_LRP[case] = case_maps_LRP[case]/counts[case]
                sitk_mri = sitk.GetImageFromArray(test_mri_nonorm[0], isVector=True)
                CNN_sitk_mean_maps_LRP[case] = sitk.GetImageFromArray(mean_maps_LRP[case], isVector=True)
                CNN_sitk_mean_maps_LRP[case].CopyInformation(sitk_mri)
                sitk.WriteImage(CNN_sitk_mean_maps_LRP[case],model_filepath+'/figures/CNN_mean_'+str(map_type)+'_'+str(case)+'_'+str(self.seed)+'.nii')
        
        #Plot average heatmaps for AD vs NC
        subplot_args = { 'nrows': 3, 'ncols': 2, 'figsize': (12,12), 'sharey':True, 'sharex':True,
                     'subplot_kw': {'xticks': [], 'yticks': []},'constrained_layout':True }
        fig, axes = plt.subplots(**subplot_args)
        vmin, vmax = 50, 99.5  #NOT SURE I WANT THIS (READ PAPER) - might be what is creating the 'mask' effect
        #Plot all three views (matching ADRP format):
        ax = axes[0,0]
        idx = 36
        ax.text(-25, 20, "Slice " + str(idx), rotation="vertical", fontsize=20)
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["AD"], PETtemplate, mean_maps_LRP["AD"], x_idx=idx,y_idx=slice(0, shape[1]),z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[1,0]
        idx = 58
        ax.text(-25, 20, "Slice " + str(idx), rotation="vertical", fontsize=20)
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["AD"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=idx,z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[2,0]
        idx = 58
        ax.text(-25, 20, "Slice " + str(idx), rotation="vertical", fontsize=20)
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["AD"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(45, -20, "AD", fontsize=20)
        ax = axes[0,1]
        idx = 36
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["NC"], PETtemplate, mean_maps_LRP["AD"], x_idx=idx,y_idx=slice(0, shape[1]),z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[1,1]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["NC"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=idx,z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[2,1]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["NC"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(45, -20, "NC", fontsize=20)
        #Plot several slices along z axis:   (matching slices from Boehle paper (https://github.com/moboehle/Pytorch-LRP/blob/master/Plotting%20brain%20maps.ipynb)
    #    for ax, idx in zip(axes[:, 0], [30, 40, 50, 60]):
    #        ax.text(-25, 20, "Slice " + str(idx), rotation="vertical", fontsize=20)
    #        fig, ax, im = plot_idv_brain(mean_maps_LRP["AD"], PETtemplate, mean_maps_LRP["AD"], z_idx=idx, contour_areas=[],
    #                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
    #    ax.text(45, -20, "AD", fontsize=20)
    #    for ax, idx in zip(axes[:, 1], [30, 40, 50, 60]):
    #        fig, ax, im = plot_idv_brain(mean_maps_LRP["NC"], PETtemplate, mean_maps_LRP["AD"], z_idx=idx, contour_areas=[],
    #                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
    #    ax.text(45, -20, "NC", fontsize=20)
        #fig.tight_layout()
        fig.subplots_adjust(right=0.8, top=0.95, hspace=0.05, wspace=0.05)
        fig.suptitle("Average "+str(map_type)+" for AD and NC patients", fontsize=22, x=.41)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
        cbar = fig.colorbar(im, shrink=0.5, ticks=[vmin, vmax], cax=cbar_ax)
        vmin_val, vmax_val = np.percentile(mean_maps_LRP["AD"], vmin), np.percentile(mean_maps_LRP["AD"], vmax)
        cbar.set_ticks([vmin_val, vmax_val])
        cbar.ax.set_yticklabels(['{0:.1f}%'.format(vmin), '{0:.1f}%'.format(vmax)],
                               fontsize=16)
        cbar.set_label('Percentile of average AD patient values', rotation=270, fontsize=18) 
        fig.savefig(model_filepath+'/figures/CNN_'+str(map_type)+'_avg_ADvNC_'+str(self.seed)+'.png', bbox_inches='tight')
        plt.close(fig)

        """    
        #Plot average heatmaps for TP, FP, TN, FN
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharey=True, sharex=True)
        vmin, vmax = 50, 99.5
        for ax, idx in zip(axes[:, 0], [30, 40, 50, 60]):
            ax.text(-25, 20, "Slice " + str(idx), rotation="vertical", fontsize=18)
            fig, ax, im = plot_idv_brain(mean_maps_LRP["TP"], PETtemplate, mean_maps_LRP["AD"], z_idx=idx, contour_areas=[],
                                         vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "True positives", fontsize=18)
        for ax, idx in zip(axes[:, 1], [30, 40, 50, 60]):
            fig, ax, im = plot_idv_brain(mean_maps_LRP["FP"], PETtemplate, mean_maps_LRP["AD"], z_idx=idx, contour_areas=[],
                                         vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "False positives", fontsize=18)
        for ax, idx in zip(axes[:, 2], [30, 40, 50, 60]):
            fig, ax, im = plot_idv_brain(mean_maps_LRP["TN"], PETtemplate, mean_maps_LRP["AD"], z_idx=idx, contour_areas=[],
                                         vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "True negatives", fontsize=18)
        for ax, idx in zip(axes[:, 3], [30, 40, 50, 60]):
            fig, ax, im = plot_idv_brain(mean_maps_LRP["FN"], PETtemplate, mean_maps_LRP["AD"], z_idx=idx, contour_areas=[],
                                         vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "False negatives", fontsize=18)
        """
        #Plot average heatmaps for TP, FP, TN, FN
        fig, axes = plt.subplots(3, 4, figsize=(12, 12), sharey=True, sharex=True, constrained_layout=True)
        vmin, vmax = 50, 99.5
        ax = axes[0,0]
        idx = 36
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["TP"], PETtemplate, mean_maps_LRP["AD"], x_idx=idx,y_idx=slice(0, shape[1]),z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[1,0]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["TP"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=idx,z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[2,0]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["TP"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "True positives", fontsize=18)
        ax = axes[0,1]
        idx = 36
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["FP"], PETtemplate, mean_maps_LRP["AD"], x_idx=idx,y_idx=slice(0, shape[1]),z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[1,1]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["FP"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=idx,z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[2,1]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["FP"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "False positives", fontsize=18)
        ax = axes[0,2]
        idx = 36
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["TN"], PETtemplate, mean_maps_LRP["AD"], x_idx=idx,y_idx=slice(0, shape[1]),z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[1,2]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["TN"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=idx,z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[2,2]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["TN"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "True negatives", fontsize=18)
        ax = axes[0,3]
        idx = 36
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["FN"], PETtemplate, mean_maps_LRP["AD"], x_idx=idx,y_idx=slice(0, shape[1]),z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[1,3]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["FN"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=idx,z_idx=slice(0, shape[2]), contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax = axes[2,3]
        idx = 58
        fig, ax, im = self.plot_idv_brain(mean_maps_LRP["FN"], PETtemplate, mean_maps_LRP["AD"], x_idx=slice(0, shape[0]),y_idx=slice(0, shape[1]),z_idx=idx, contour_areas=[],
                            vmin=vmin, vmax=vmax, fig=fig, ax=ax, set_nan=False, cmap="hot");
        ax.text(10, -20, "False negatives", fontsize=18)
        fig.suptitle("Average "+str(map_type)+" for varying cases", fontsize=24, x=.42)
    #    fig.tight_layout()
        fig.subplots_adjust(top=0.95, right=0.8, hspace=0.05, wspace=0.05)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, shrink=0.5, ticks=[vmin, vmax], cax=cbar_ax)
        vmin_val, vmax_val = np.percentile(mean_maps_LRP["AD"], vmin), np.percentile(mean_maps_LRP["AD"], vmax)
        cbar.set_ticks([vmin_val, vmax_val])
        cbar.ax.set_yticklabels(['{0:.1f}%'.format(vmin), '{0:.1f}%'.format(vmax)],
                                fontsize=16)
        cbar.set_label('Percentile of average AD patient values', rotation=270, fontsize=20)
        fig.savefig(model_filepath+'/figures/CNN_'+str(map_type)+'_avg_TPvFPvTNvFN_'+str(self.seed)+'.png', bbox_inches='tight')
        plt.close(fig)
        
        return mean_maps_LRP