# Plot models

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.tri as tri

def PlotModelRes_2lay(res, depth):
    # Arrays for plotting
    npos = np.shape(res)[0]
    depthmax = 10
    ny = 101
    y = np.linspace(0, depthmax, ny)
    resy = np.zeros((npos, ny))
    xx = np.linspace(0,npos+1,npos+1, endpoint=False)

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth[i, 1]:
            resy[i, y1] = res[i, 0]
            y1 += 1
            y2=y1
        resy[i, y2:] = res[i, 1]
    
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow((resy).T, cmap='viridis', interpolation='none', 
                    extent=[0,npos,10,0], norm = colors.LogNorm(vmin=1, vmax=100))
    plt.step(np.hstack((xx, xx[-1])), np.hstack((depth[0,1],depth[:,1], depth[-2,1])), ':r')
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Resistivity [Ohm.m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')
    
def PlotModelRes_3lay(res, depth):
    npos = np.shape(res)[0]
    xx = np.linspace(0,npos+1,npos+1, endpoint=False)
    depthmax = 10
    ny = 101
    y = np.linspace(0, depthmax, ny)
    # Depths to be plotted
    resy = np.zeros((npos, ny))

    # Resistivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth[i, 1]:
            resy[i, y1] = res[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth[i, 2]:
            resy[i, y2] = res[i, 1]
            y2 += 1
            if y2 == 50:
                break
        resy[i, y2:] = res[i, 2]
        
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow(resy.T, cmap='viridis', interpolation='none', extent=[0,npos,10,0], 
                    norm = colors.LogNorm(vmin=1, vmax=100))
    plt.step(np.hstack((xx, xx[-1])), np.hstack((depth[0,1],depth[:,1], depth[-2,1])), ':r')
    plt.step(np.hstack((xx, xx[-1])), np.hstack((depth[0,2],depth[:,2], depth[-2,2])), ':r')
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Resistivity [Ohm.m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')

def PlotModelCond_2lay(sig, depth, depth_true, vmin=1, vmax=1000):
    # Arrays for plotting
    npos = np.shape(sig)[0]
    depthmax = 10
    ny = 71
    y = np.linspace(0, depthmax, ny)
    sigy = np.zeros((npos, ny))
    xx = np.linspace(0,npos+1,npos+1, endpoint=False)

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth[i, 1]:
            sigy[i, y1] = sig[i, 0]
            y1 += 1
            y2=y1
        sigy[i, y2:] = sig[i, 1]
    
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow((sigy*1000).T, cmap='viridis', interpolation='none', 
                    extent=[0,npos,10,0], norm = colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.step(np.hstack((xx, xx[-1])), np.hstack((depth_true[0,1],depth_true[:,1], depth_true[-2,1])), ':r')
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Electrical conductivity [mS/m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')
    return sigy
    
def PlotModelCond_2lay_elev(sig, depth, elevation):
    # Arrays for plotting
    npos = np.shape(sig)[0]
    depthmax = -np.min(elevation)+14
    ny = 101
    y = np.linspace(0, depthmax, ny)
    sigy = np.zeros((npos, ny))
    #xx = np.linspace(0,npos+1,npos+1, endpoint=False)
    
    depth[:,1] = depth[:,1] -(elevation)

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= -elevation[i]: # - sign because the elevations are negative!!
            sigy[i, y1] = 0
            y1 +=1
        while y[y1] <= depth[i, 1]:
            sigy[i, y1] = sig[i, 0]
            y1 += 1
            y2=y1
        sigy[i, y2:] = sig[i, 1]
    
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow((sigy*1000).T, cmap='viridis', interpolation='none', 
                    extent=[0,37,depthmax,0], norm = colors.LogNorm(vmin=1, vmax=1000))
    #plt.step(np.hstack((xx, xx[-1])), np.hstack((depth[0,1],depth[:,1], depth[-2,1])), ':r')
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Electrical conductivity [mS/m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')
    
def PlotModelCond_3lay(sig, depth, vmin=1, vmax=1000):
    npos = np.shape(sig)[0]
    xx = np.linspace(0,npos+1,npos+1, endpoint=False)
    depthmax = 10
    ny = 101
    y = np.linspace(0, depthmax, ny)
    # Depths to be plotted
    sigy = np.zeros((npos, ny))

    # Resistivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth[i,0]:
            sigy[i, y1] = sig[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth[i,1]:
            sigy[i, y2] = sig[i, 1]
            y2 += 1
            if y2 == (ny-1):
                break
        sigy[i, y2:] = sig[i, 2]
        
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow((sigy*1000).T, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax))

    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Electrical conductivity [mS/m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')
    return sigy
    
def PlotModelCond_3lay_elev(sig, depth, elevation):
    npos = np.shape(sig)[0]
    #xx = np.linspace(0,npos+1,npos+1, endpoint=False)
    depthmax = -np.min(elevation)+14
    ny = 101
    y = np.linspace(0, depthmax, ny)
    # Depths to be plotted
    sigy = np.zeros((npos, ny))
    
    depth[:,1] = depth[:,1] -(elevation)
    depth[:,2] = depth[:,2] -(elevation)

    # Resistivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= -elevation[i]: # - sign because the elevations are negative!!
            sigy[i, y1] = 0
            y1 +=1
        while y[y1] <= depth[i, 1]:
            sigy[i, y1] = sig[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth[i, 2]:
            sigy[i, y2] = sig[i, 1]
            y2 += 1
            if y2 == 50:
                break
        sigy[i, y2:] = sig[i, 2]
        
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow((sigy*1000).T, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    norm = colors.LogNorm(vmin=10, vmax=1000))
    #plt.step(np.hstack((xx, xx[-1])), np.hstack((true_depth[0,1],true_depth[:,1], true_depth[-2,1])), ':r')
    #plt.step(np.hstack((xx, xx[-1])), np.hstack((true_depth[0,2],true_depth[:,2], true_depth[-2,2])), ':r')
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Electrical conductivity [mS/m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')
    
# Plot true and estimated model in same figure

def rmse_a(p, o):
    diff = (p - o)**2 /(((p + o)/2)**2)
    error = (np.sqrt(np.sum(diff)/len(p)))
    return error

def Plot2Models_2lay(model1, model2, conds, npos=20, depthmax=10, ny=50):

    sigma1 = model1[:,:2]
    depth1 = model1[:,2]
    
    sigma2 = model2[:,:2]
    depth2 = model2[:,2]
    
    # Arrays for plotting
    y = np.linspace(0, depthmax, ny)
    sigy1 = np.zeros((npos, ny))
    sigy2 = np.zeros((npos, ny))
    
    xx = np.linspace(0,npos+1,npos+1,endpoint=False)

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth1[i]:
            sigy1[i, y1] = sigma1[i, 0]
            y1 += 1
            y2=y1
        sigy1[i, y2:] = sigma1[i, 1]
    
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth2[i]:
            sigy2[i, y1] = sigma2[i, 0]
            y1 += 1
            y2=y1
        sigy2[i, y2:] = sigma2[i, 1]
    
    fig, axs = plt.subplots(2,1, figsize=(6,4), constrained_layout=True)
    
    for i in range(len(axs)):
        ax=axs[i]
        if i==0:
            pos = ax.imshow(sigy1.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    vmin = np.min(conds)*1000, vmax=np.max(conds)*1000, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Horizontal distance [m]')
            ax.set_title('True Model')
        else:
            pos = ax.imshow(sigy2.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    vmin = np.min(conds*1000), vmax=np.max(conds*1000), norm='log' )
            
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Horizontal distance [m]')
            ax.set_title('Estimated Model')
            ax.text(0,9.5,'$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)),                    np.log10(np.hstack(sigma2)))*100)+ '% \n' +
                '$h$ RMSE: %2.2f' %(rmse_a(depth1[:], depth2[:])*100)+'%', color='w',fontsize=10)
    clb = fig.colorbar(pos, ax=axs, )
    clb.set_label('Electrical Conductivity $\sigma$ [mS/m]',  )
    
def Plot3Models_2lay(model1, model2, model3, conds, npos=20, depthmax=10, ny=50):

    sigma1 = model1[:,:2]
    depth1 = model1[:,2]
    
    sigma2 = model2[:,:2]
    depth2 = model2[:,2]
    
    sigma3 = model3[:,:2]
    depth3 = model3[:,2]
    
    # Arrays for plotting
    y = np.linspace(0, depthmax, ny)
    sigy1 = np.zeros((npos, ny))
    sigy2 = np.zeros((npos, ny))
    sigy3 = np.zeros((npos, ny))
    
    xx = np.linspace(0,npos+1,npos+1,endpoint=False)

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth1[i]:
            sigy1[i, y1] = sigma1[i, 0]
            y1 += 1
            y2=y1
        sigy1[i, y2:] = sigma1[i, 1]
    
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth2[i]:
            sigy2[i, y1] = sigma2[i, 0]
            y1 += 1
            y2=y1
        sigy2[i, y2:] = sigma2[i, 1]
        
        # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth3[i]:
            sigy3[i, y1] = sigma3[i, 0]
            y1 += 1
            y2=y1
        sigy3[i, y2:] = sigma3[i, 1]
    
    fig, axs = plt.subplots(1,3, figsize=(6,3), constrained_layout=True)
    
    for i in range(len(axs)):
        ax=axs[i]
        if i==0:
            pos = ax.imshow(sigy1.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    vmin = np.min(conds)*1000, vmax=np.max(conds)*1000, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=7)
            ax.set_xlabel('Horizontal distance [m]', fontsize=7)
            ax.set_title('True Model', fontsize=7)
            ax.tick_params(labelsize=8)
        elif i==1:
            pos = ax.imshow(sigy2.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    vmin = np.min(conds*1000), vmax=np.max(conds*1000), norm='log' )
            
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=7)
            ax.set_xlabel('Horizontal distance [m]', fontsize=7)
            ax.set_title('Estimated Model Q + IP', fontsize=7)
            ax.text(0,9.5,'$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)),            
                    np.log10(np.hstack(sigma2)))*100)+ '% \n' + '$h$ RMSE: %2.2f' %(rmse_a(depth1[:], 
                    depth2[:])*100)+'%', color='w',fontsize=7)
            ax.tick_params(labelsize=8)
            
        else:
            pos = ax.imshow(sigy3.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,depthmax,0], 
                    vmin = np.min(conds*1000), vmax=np.max(conds*1000), norm='log' )
            
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=7)
            ax.set_xlabel('Horizontal distance [m]', fontsize=7)
            ax.set_title('Estimated Model Q', fontsize=7)
            ax.text(0,9.5,'$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)),            
                    np.log10(np.hstack(sigma3)))*100)+ '% \n' + '$h$ RMSE: %2.2f' %(rmse_a(depth1[:], 
                    depth3[:])*100)+'%', color='w',fontsize=7)
            ax.tick_params(labelsize=8)
    clb = fig.colorbar(pos, ax=axs, )
    clb.ax.tick_params(labelsize=7)
    clb.set_label('Electrical Conductivity $\sigma$ [mS/m]', fontsize=7 )

def PlotErrorSpace(model, model_est, pos, err, models_err, depthmax=10):
    
    # Arrays to plot
    depth_true = np.array([0, model[pos,2], depthmax])
    depth_est = np.array([0, model_est[pos,2], depthmax])

    sigma_true = np.hstack([model[pos,:2], model[pos,1]])
    sigma_est = np.hstack([model_est[pos,:2], model_est[pos,1]])
       
    fig, ax = plt.subplots(1,2)
    
    ax[0].step(sigma_true*1000, depth_true, 'r', label = 'True')
    ax[0].step(sigma_est*1000, depth_est, 'g', label='Estimated')
    ax[0].invert_yaxis()
    ax[0].set_ylabel('Depth [m]')
    ax[0].set_xlabel('$\sigma$ [mS/m]')
    ax[0].set_title('Model at receiver Rx: ' +str(pos))
    ax[0].set_xscale('log')
    ax[0].legend()

    x = np.log10((models_err[:,1])*1000) # conductivities of second layer
    y = models_err[:,2]                  # thickness of first layer
    z = err

    ngridx = 100
    ngridy = 200
    
    # Create grid values first.
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    
    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    # from scipy.interpolate import griddata
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    ax[1].contour(xi, yi, zi*100, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax[1].contourf(xi, yi, zi*100, levels=14, cmap="RdBu_r")
    ax[1].plot(x, y, '.k', ms=1)
    ax[1].set(xlim=(0,1.2), ylim=(3,5))
    ax[1].scatter(np.log10((model_est[pos,1])*1000), model_est[pos,2],
                 marker ='^', c='y', label='Estimated model')
    #ax1.scatter( np.log10((model_cm_pos[1])*1000), model_cm_pos[2],
    #             marker ='^', c='k', label='Closest')
    ax[1].set_xlabel('$log10(\sigma_2)$')
    ax[1].set_ylabel('Thickness $h_1$ [m]')
    ax[1].legend()
    clb = fig.colorbar(cntr1, ax=ax[1])
    clb.ax.set_title('RMS Error %')
    
def Plot3Models(model1, model2, model3, conds, npos=20, vmin=1, vmax=1000):
    
    sigma1 = model1[:,:2]
    sigma2 = model2[:,:2]
    sigma3 = model3[:,:2]
    depth1 = model1[:,2]
    depth2 = model2[:,2]
    depth3 = model3[:,2]
    
    xx = np.linspace(0,npos+1,npos+1,endpoint=False)
    
    # Arrays for plotting
    depthmax=10
    ny = 50
    y = np.linspace(0, depthmax, ny)
    sigy1 = np.zeros((npos, ny))
    sigy2 = np.zeros((npos, ny))
    sigy3 = np.zeros((npos, ny))

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth1[i]:
            sigy1[i, y1] = sigma1[i, 0]
            y1 += 1
            y2=y1
        sigy1[i, y2:] = sigma1[i, 1]
    
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth2[i]:
            sigy2[i, y1] = sigma2[i, 0]
            y1 += 1
            y2=y1
        sigy2[i, y2:] = sigma2[i, 1]
        
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth3[i]:
            sigy3[i, y1] = sigma3[i, 0]
            y1 += 1
            y2=y1
        sigy3[i, y2:] = sigma3[i, 1]
    
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(6,4), constrained_layout=True)
    
    for i in range(len(axs)):
        ax=axs[i]
        if i==0:
            pos = ax.imshow(sigy1.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], vmin = vmin, vmax= vmax, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=8)
            #ax.set_xlabel('Horizontal distance [m]')
            ax.set_title('True Model', fontsize=8)
            ax.tick_params(labelsize=8)
            
        elif i==1:
            pos = ax.imshow(sigy2.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], vmin = vmin, vmax= vmax, norm='log' )
            
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=8)
            #ax.set_xlabel('Horizontal distance [m]')
            ax.text(0,9.9,'$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)), np.log10(np.hstack(sigma2)))*100)+ '% \n' +
                '$h$ RMSE: %2.2f' %(rmse_a(depth1, depth2)*100)+'%',color='w',fontsize=8)
            ax.set_title('Estimated SA Model', fontsize=8)
            ax.tick_params(labelsize=8)
        else:
            pos = ax.imshow(sigy3.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], vmin = vmin, vmax= vmax, norm='log' )
            
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0],depth1[:], depth1[-1])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=7)
            ax.set_xlabel('Horizontal distance [m]', fontsize=8)
            ax.set_title('Estimated LIN Model', fontsize=8)
            ax.text(0,9.9,'$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)), np.log10(np.hstack(sigma3)))*100)+ '% \n' +
                '$h$ RMSE: %2.2f' %(rmse_a(depth1, depth3)*100)+'%',color='w',fontsize=8)
            ax.tick_params(labelsize=8)
            
    clb = fig.colorbar(pos, ax=axs)
    clb.ax.tick_params(labelsize=7)
    clb.set_label('Electrical Conductivity $\sigma$ [mS/m]', fontsize=8 )
   # fig.suptitle('Model $\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)), np.log10(np.hstack(sigma2)))*100)+ '% \n' +
   #             'Model $h$ RMSE: %2.2f' %(rmse_a(depth1[:,1], depth2[:,1])*100)+'%')
    
def Plot3Datas(data_true, data_est, data_est_LIN):
    
    fig, ax = plt.subplots(3,2, sharex=True,)

    ax[0,0].plot(data_true[:,0]*1e3, 'b', label = 'H2 True', markersize=4)
    ax[0,0].plot(data_est[:,0]*1e3, '*b', label = 'H2 SA', markersize=4)
    ax[0,0].plot(data_est_LIN[:,0]*1e3, '.b', label ='H2 LIN', markersize=4)
    ax[0,0].plot(data_true[:,3]*1e3, 'g', label = 'V2 True', markersize=4)
    ax[0,0].plot(data_est[:,3]*1e3, '*g', label = 'V2 SA', markersize=4)
    ax[0,0].plot(data_est_LIN[:,3]*1e3, '.g', label ='V2 LIN', markersize=4)
    ax[0,0].plot(data_true[:,6]*1e3, 'r', label = 'P2.1 True', markersize=4)
    ax[0,0].plot(data_est[:,6]*1e3, '*r', label = 'P2.1 SA', markersize=4)
    ax[0,0].plot(data_est_LIN[:,6]*1e3, '.r', label ='P2.1 LIN', markersize=4)
    ax[0,0].tick_params(labelsize=8)
    ax[0,0].set_ylabel('Q [ppt]', fontsize=8)

    ax[1,0].plot(data_true[:,1]*1e3, 'b', label = 'H4 True', markersize=4)
    ax[1,0].plot(data_est[:,1]*1e3, '*b', label = 'H4 SA', markersize=4)
    ax[1,0].plot(data_est_LIN[:,1]*1e3, '.b', label ='H4 LIN', markersize=4)
    ax[1,0].plot(data_true[:,4]*1e3, 'g', label = 'V4 True', markersize=4)
    ax[1,0].plot(data_est[:,4]*1e3, '*g', label = 'V2 SA', markersize=4)
    ax[1,0].plot(data_est_LIN[:,4]*1e3, '.g', label ='V4 LIN', markersize=4)
    ax[1,0].plot(data_true[:,7]*1e3, 'r', label = 'P4.1 True', markersize=4)
    ax[1,0].plot(data_est[:,7]*1e3, '*r', label = 'P2.1 SA', markersize=4)
    ax[1,0].plot(data_est_LIN[:,7]*1e3, '.r', label ='P4.1 LIN', markersize=4)
    ax[1,0].tick_params(labelsize=8)
    ax[1,0].set_ylabel('Q [ppt]', fontsize=8)

    ax[2,0].plot(data_true[:,2]*1e3, 'b', label = 'H8 True', markersize=4)
    ax[2,0].plot(data_est[:,2]*1e3, '*b', label = 'H8 SA', markersize=4)
    ax[2,0].plot(data_est_LIN[:,2]*1e3, '.b', label ='H8 LIN', markersize=4)
    ax[2,0].plot(data_true[:,5]*1e3, 'g', label = 'V8 True', markersize=4)
    ax[2,0].plot(data_est[:,5]*1e3, '*g', label = 'V8 SA', markersize=4)
    ax[2,0].plot(data_est_LIN[:,5]*1e3, '.g', label ='V8 LIN', markersize=4)
    ax[2,0].plot(data_true[:,8]*1e3, 'r', label = 'P8.1 True', markersize=4)
    ax[2,0].plot(data_est[:,8]*1e3, '*r', label = 'P8.1 SA', markersize=4)
    ax[2,0].plot(data_est_LIN[:,8]*1e3, '.r', label ='P8.1 LIN', markersize=4)
    ax[2,0].tick_params(labelsize=8)
    ax[2,0].set_xlabel('Horizontal distance [m]', fontsize=8)
    ax[2,0].set_ylabel('Q [ppt]', fontsize=8)

    ax[0,1].plot(data_true[:,9]*1e3, 'b', label = 'H2 True', markersize=4)
    ax[0,1].plot(data_est[:,9]*1e3, '*b', label = 'H2 SA', markersize=4)
    ax[0,1].plot(data_est_LIN[:,9]*1e3, '.b', label ='H2 LIN', markersize=4)
    ax[0,1].plot(data_true[:,12]*1e3, 'g', label = 'V2 True', markersize=4)
    ax[0,1].plot(data_est[:,12]*1e3, '*g', label = 'V2 SA', markersize=4)
    ax[0,1].plot(data_est_LIN[:,12]*1e3, '.g', label ='V2 LIN', markersize=4)
    ax[0,1].plot(data_true[:,15]*1e3, 'r', label = 'P2.1 True', markersize=4)
    ax[0,1].plot(data_est[:,15]*1e3, '*r', label = 'P2.1 SA', markersize=4)
    ax[0,1].plot(data_est_LIN[:,15]*1e3, '.r', label ='P2.1 LIN', markersize=4)
    ax[0,1].tick_params(labelsize=8)
    ax[0,1].set_ylabel('IP [ppt]', fontsize=8)


    ax[1,1].plot(data_true[:,10]*1e3, 'b', label = 'H4 True', markersize=4)
    ax[1,1].plot(data_est[:,10]*1e3, '*b', label = 'H4 SA', markersize=4)
    ax[1,1].plot(data_est_LIN[:,10]*1e3, '.b', label ='H4 LIN', markersize=4)
    ax[1,1].plot(data_true[:,13]*1e3, 'g', label = 'V4 True', markersize=4)
    ax[1,1].plot(data_est[:,13]*1e3, '*g', label = 'V4 SA', markersize=4)
    ax[1,1].plot(data_est_LIN[:,13]*1e3, '.g', label ='V4 LIN', markersize=4)
    ax[1,1].plot(data_true[:,16]*1e3, 'r', label = 'P4.1 True', markersize=4)
    ax[1,1].plot(data_est[:,16]*1e3, '*r', label = 'P4.1 SA', markersize=4)
    ax[1,1].plot(data_est_LIN[:,16]*1e3, '.r', label ='P4.1 LIN', markersize=4)
    ax[1,1].tick_params(labelsize=8)
    ax[1,1].set_ylabel('IP [ppt]', fontsize=8)

    ax[2,1].plot(data_true[:,11]*1e3, 'b', label = 'H8 True', markersize=4)
    ax[2,1].plot(data_est[:,11]*1e3, '*b', label = 'H8 SA', markersize=4)
    ax[2,1].plot(data_est_LIN[:,11]*1e3, '.b', label ='H8 LIN', markersize=4)
    ax[2,1].plot(data_true[:,14]*1e3, 'g', label = 'V8 True', markersize=4)
    ax[2,1].plot(data_est[:,14]*1e3, '*g', label = 'V8 SA', markersize=4)
    ax[2,1].plot(data_est_LIN[:,14]*1e3, '.g', label ='V8 LIN', markersize=4)
    ax[2,1].plot(data_true[:,17]*1e3, 'r', label = 'P8.1 True', markersize=4)
    ax[2,1].plot(data_est[:,17]*1e3, '*r', label = 'P8.1 SA', markersize=4)
    ax[2,1].plot(data_est_LIN[:,17]*1e3, '.r', label ='P8.1 LIN', markersize=4)
    ax[2,1].set_ylabel('IP [ppt]', fontsize=8)
    ax[2,1].set_xlabel('Horizontal distance [m]', fontsize=8)
    ax[2,1].tick_params(labelsize=8)

    ax[0,1].legend(bbox_to_anchor=(1.2, 1.0), loc='upper left', fontsize=7)
    ax[1,1].legend(bbox_to_anchor=(1.2, 1.0), loc='upper left',fontsize=7)
    ax[2,1].legend(bbox_to_anchor=(1.2, 1.0), loc='upper left',fontsize=7)

    plt.tight_layout()
    

    
def Plot2Models_3lay(sigma1, sigma2, thk11, thk12, thk21, thk22, vmin=1, vmax=1000):
    npos = len(thk11)
    surface = np.zeros(npos)
    depth1 = np.zeros((npos,4))
    depth2 = np.zeros((npos,4))
    depthmax = 10
    ny = 50
    y = np.linspace(0, depthmax, ny)
    xx = np.linspace(0,npos+1,npos+1,endpoint=False)
    
    # Depths to be plotted for sigma1
    for i in range(npos):
        depth1[i,0] = 0
        depth1[i,1] = thk11[i]
        depth1[i,2] = thk11[i] + thk12[i]
        depth1[i,3] = thk11[i] + thk12[i]
        
    # Depths to be plotted for sigma1
    for i in range(npos):
        depth2[i,0] = 0
        depth2[i,1] = thk21[i]
        depth2[i,2] = thk21[i] + thk22[i]
        depth2[i,3] = thk21[i] + thk22[i]
        
    sigy1 = np.zeros((npos, ny))
    sigy2 = np.zeros((npos, ny))

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth1[i, 1]:
            sigy1[i, y1] = sigma1[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth1[i, 2]:
            sigy1[i, y2] = sigma1[i, 1]
            y2 += 1
            if y2 == 50:
                break
        sigy1[i, y2:] = sigma1[i, 2]
        
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth2[i, 1]:
            sigy2[i, y1] = sigma2[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth2[i, 2]:
            sigy2[i, y2] = sigma2[i, 1]
            y2 += 1
            if y2 == 50:
                break
        sigy2[i, y2:] = sigma2[i, 2]
    
    fig, axs = plt.subplots(2,1, sharex=True, figsize=(6,4), constrained_layout=True)
    
    for i in range(len(axs)):
        ax=axs[i]
        if i==0:
            pos = ax.imshow(sigy1.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], 
                    vmin = vmin, vmax=vmax, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,1],depth1[:,1], depth1[-1,1])), ':r')
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,2],depth1[:,2], depth1[-1,2])), ':r')
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Horizontal distance [m]')
            ax.set_title('True Model')
            ax.tick_params(labelsize=8)
        else:
            pos = ax.imshow(sigy2.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], 
                    vmin = vmin, vmax=vmax, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,1],depth1[:,1], depth1[-1,1])), ':r')
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,2],depth1[:,2], depth1[-1,2])), ':r')
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Horizontal distance [m]')
            ax.set_title('Estimated Model')
            ax.text(0, 9.9, '$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)), 
                np.log10(np.hstack(sigma2)))*100)+ '% \n' + 
                 '$t$ RMSE: %2.2f' %(rmse_a(depth1[:,1], depth2[:,1])*100)+'%', color='w',fontsize=8)
            ax.tick_params(labelsize=8)
    clb = fig.colorbar(pos, ax=axs, )
    clb.set_label('Electrical Conductivity $\sigma$ [mS/m]',  )
    
def PlotErrorSpace_3lay(model, model_est, pos, err, models_err):
    
    # Arrays to plot
    depth_true = np.array([0, model[pos,3], model[pos,3]+ model[pos,4], 10])
    depth_est = np.array([0, model_est[pos,3], model_est[pos,3]+ model_est[pos,4], 10])

    sigma_true = np.hstack([model[pos,:3], model[pos,2]])
    sigma_est = np.hstack([model_est[pos,:3], model_est[pos,2]])
       
    fig, ax = plt.subplots(1,2)
    
    ax[0].step(sigma_true*1000, depth_true, 'r', label = 'True')
    ax[0].step(sigma_est*1000, depth_est, 'g', label='Estimated')
    ax[0].invert_yaxis()
    ax[0].set_ylabel('Depth [m]')
    ax[0].set_xlabel('$\sigma$ [mS/m]')
    ax[0].set_title('Model at receiver Rx: ' +str(pos))
    ax[0].set_xscale('log')
    ax[0].legend()

    x = np.log10((models_err[:,1])*1000) # conductivities of second layer
    y = models_err[:,4]                  # thickness of second layer
    z = err

    ngridx = 100
    ngridy = 200
    
    # Create grid values first.
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    
    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    # from scipy.interpolate import griddata
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    ax[1].contour(xi, yi, zi*100, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax[1].contourf(xi, yi, zi*100, levels=14, cmap="RdBu_r")
    ax[1].plot(x, y, '.k', ms=1)
    ax[1].set(xlim=(0,1.2), ylim=(2,3.5))
    ax[1].scatter(np.log10((model_est[pos,1])*1000), model_est[pos,3],
                 marker ='^', c='y', label='Estimated model')
    #ax1.scatter( np.log10((model_cm_pos[1])*1000), model_cm_pos[2],
    #             marker ='^', c='k', label='Closest')
    ax[1].set_xlabel('$log10(\sigma_2)$')
    ax[1].set_ylabel('Thickness $h_1$ [m]')
    ax[1].legend()
    clb = fig.colorbar(cntr1, ax=ax[1])
    clb.ax.set_title('RMS Error %')


def Plot3Models_3lay(sigma1, sigma2, sigma3, thk11, thk12, thk21, thk22, thk31, thk32, title1='True Model', title2 ='Estimated Model 1', title3 = 'Estimated Model 2', vmin=1, vmax=1000, depthmax=10, ny=50):
    npos = len(thk11)
    surface = np.zeros(npos)
    depth1 = np.zeros((npos,4))
    depth2 = np.zeros((npos,4))
    depth3 = np.zeros((npos,4))

    y = np.linspace(0, depthmax, ny)
    xx = np.linspace(0,npos+1,npos+1,endpoint=False)
    
    # Depths to be plotted for sigma1
    for i in range(npos):
        depth1[i,0] = 0
        depth1[i,1] = thk11[i]
        depth1[i,2] = thk11[i] + thk12[i]
        depth1[i,3] = thk11[i] + thk12[i]
        
    # Depths to be plotted for sigma2
    for i in range(npos):
        depth2[i,0] = 0
        depth2[i,1] = thk21[i]
        depth2[i,2] = thk21[i] + thk22[i]
        depth2[i,3] = thk21[i] + thk22[i]
        
    # Depths to be plotted for sigma3
    for i in range(npos):
        depth3[i,0] = 0
        depth3[i,1] = thk31[i]
        depth3[i,2] = thk31[i] + thk32[i]
        depth3[i,3] = thk31[i] + thk32[i]
        
    sigy1 = np.zeros((npos, ny))
    sigy2 = np.zeros((npos, ny))
    sigy3 = np.zeros((npos, ny))

    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth1[i, 1]:
            sigy1[i, y1] = sigma1[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth1[i, 2]:
            sigy1[i, y2] = sigma1[i, 1]
            y2 += 1
            if y2 == 50:
                break
        sigy1[i, y2:] = sigma1[i, 2]
        
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth2[i, 1]:
            sigy2[i, y1] = sigma2[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth2[i, 2]:
            sigy2[i, y2] = sigma2[i, 1]
            y2 += 1
            if y2 == 50:
                break
        sigy2[i, y2:] = sigma2[i, 2]
        
    # Conductivities array to be plotted
    for i in range(npos):
        y1=0
        while y[y1] <= depth3[i, 1]:
            sigy3[i, y1] = sigma3[i, 0]
            y1 += 1
            y2=y1
        while y[y2] <= depth3[i, 2]:
            sigy3[i, y2] = sigma3[i, 1]
            y2 += 1
            if y2 == 50:
                break
        sigy3[i, y2:] = sigma3[i, 2]
    
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(6,4), constrained_layout=True)
    
    for i in range(len(axs)):
        ax=axs[i]
        if i==0:
            pos = ax.imshow(sigy1.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], 
                    vmin = vmin, vmax=vmax, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,1],depth1[:,1], depth1[-1,1])), ':r')
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,2],depth1[:,2], depth1[-1,2])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=8)
            ax.set_xlabel('Horizontal distance [m]', fontsize=8)
            ax.set_title(title1, fontsize=8)
            ax.tick_params(labelsize=8)
        elif i==1:
            pos = ax.imshow(sigy2.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], 
                    vmin = vmin, vmax=vmax, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,1],depth1[:,1], depth1[-1,1])), ':r')
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,2],depth1[:,2], depth1[-1,2])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=8)
            ax.set_xlabel('Horizontal distance [m]', fontsize=8)
            ax.set_title(title2, fontsize=8)
            ax.text(0, 9.9, '$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)), 
                np.log10(np.hstack(sigma2)))*100)+ '% \n' + 
                 '$t$ RMSE: %2.2f' %(rmse_a(depth1[:,1], depth2[:,1])*100)+'%', color='w',fontsize=8)
            ax.tick_params(labelsize=8)
        else:
            pos = ax.imshow(sigy3.T*1000, cmap='viridis', interpolation='none', extent=[0,npos,10,0], 
                    vmin = vmin, vmax=vmax, norm='log' )
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,1],depth1[:,1], depth1[-1,1])), ':r')
            ax.step(np.hstack((xx, xx[-1])), np.hstack((depth1[0,2],depth1[:,2], depth1[-1,2])), ':r')
            ax.set_ylabel('Depth [m]', fontsize=8)
            ax.set_xlabel('Horizontal distance [m]', fontsize=8)
            ax.set_title(title3, fontsize=8)
            ax.text(0, 9.9, '$\sigma$ RMSE: %2.2f' %(rmse_a(np.log10(np.hstack(sigma1)), 
                np.log10(np.hstack(sigma2)))*100)+ '% \n' + 
                 '$t$ RMSE: %2.2f' %(rmse_a(depth1[:,1], depth2[:,1])*100)+'%', color='w',fontsize=8)
            ax.tick_params(labelsize=8)
    clb = fig.colorbar(pos, ax=axs, )
    clb.ax.tick_params(labelsize=7)
    clb.set_label('Electrical Conductivity $\sigma$ [mS/m]',  fontsize=8)
    
def PlotLine(model, elevation, dist, vmin=1, vmax=500):
    """ Function to plot the estimated positions in a stitched 2D section
    
    Parameters:
    1. model: Estimated model in each position from the global search 
       m = [sigma_1, sigma_2, h_1]
    2. elevation: Elevation values [m]
    3. dist = Horizontal distance of the section [m]
    4. vmin = minimum value for colorbar
    5. vmax = maximum value for colorbar
    
    """
    
    # Arrays for plotting
    
    npos = len(model)
    depthmax = -np.min(elevation)+14
    
    ny = 101
    y = np.linspace(0, depthmax, ny)
    sigy = np.zeros((len(model), ny))
    
    sigma_1 = model[:,0]
    sigma_2 = model[:,1]
    thick_1 = model[:,2]
    
    depth = thick_1 -(elevation)
    
    # Conductivities array to be plotted
    for i in range(npos):
        y0 = 0
        while y[y0] <= -elevation[i]: # - sign because the elevations are negative!!
            sigy[i, y0:] = 0
            y0 += 1       
        while y[y0] <= depth[i]:
            sigy[i, y0:] = sigma_1[i]
            y0 += 1
        sigy[i, y0:] = sigma_2[i]
            
    fig, ax = plt.subplots(figsize = (7,6))
    pos = ax.imshow((sigy*1000).T, cmap='viridis', interpolation='none', 
                    extent= [0,dist,depthmax+2,0], vmin = vmin, vmax=vmax, norm='log' )
    
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Electrical Conductivity $\sigma$ [mS/m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Distance [m]')
    
def PlotData(data_true, ax=None):
    # Changed to match data EMforward_2lay format
    if ax.any() == None:
        fig, ax = plt.subplots(3,2, sharex=True,)
    
    ax[0,0].plot(data_true[:,0]*1e3, 'b', label = 'H2 True', markersize=4)
    ax[0,0].plot(data_true[:,3]*1e3, 'g', label = 'V2 True', markersize=4)
    ax[0,0].plot(data_true[:,6]*1e3, 'r', label = 'P2.1 True', markersize=4)
    ax[0,0].tick_params(labelsize=8)
    ax[0,0].set_ylabel('Q [ppt]', fontsize=8)

    ax[1,0].plot(data_true[:,1]*1e3, 'b', label = 'H4 True', markersize=4)
    ax[1,0].plot(data_true[:,4]*1e3, 'g', label = 'V4 True', markersize=4)
    ax[1,0].plot(data_true[:,7]*1e3, 'r', label = 'P4.1 True', markersize=4)
    ax[1,0].tick_params(labelsize=8)
    ax[1,0].set_ylabel('Q [ppt]', fontsize=8)

    ax[2,0].plot(data_true[:,2]*1e3, 'b', label = 'H8 True', markersize=4)
    ax[2,0].plot(data_true[:,5]*1e3, 'g', label = 'V8 True', markersize=4)
    ax[2,0].plot(data_true[:,8]*1e3, 'r', label = 'P8.1 True', markersize=4)
    ax[2,0].tick_params(labelsize=8)
    ax[2,0].set_xlabel('Horizontal distance [m]', fontsize=8)
    ax[2,0].set_ylabel('Q [ppt]', fontsize=8)

    ax[0,1].plot(data_true[:,9]*1e3, 'b', label = 'H2 True', markersize=4)
    ax[0,1].plot(data_true[:,12]*1e3, 'g', label = 'V2 True', markersize=4)
    ax[0,1].plot(data_true[:,15]*1e3, 'r', label = 'P2.1 True', markersize=4)
    ax[0,1].tick_params(labelsize=8)
    ax[0,1].set_ylabel('IP [ppt]', fontsize=8)

    ax[1,1].plot(data_true[:,10]*1e3, 'b', label = 'H4 True', markersize=4)
    ax[1,1].plot(data_true[:,13]*1e3, 'g', label = 'V4 True', markersize=4)
    ax[1,1].plot(data_true[:,16]*1e3, 'r', label = 'P4.1 True', markersize=4)
    ax[1,1].tick_params(labelsize=8)
    ax[1,1].set_ylabel('IP [ppt]', fontsize=8)

    ax[2,1].plot(data_true[:,11]*1e3, 'b', label = 'H8 True', markersize=4)
    ax[2,1].plot(data_true[:,14]*1e3, 'g', label = 'V8 True', markersize=4)
    ax[2,1].plot(data_true[:,17]*1e3, 'r', label = 'P8.1 True', markersize=4)
    ax[2,1].set_ylabel('IP [ppt]', fontsize=8)
    ax[2,1].set_xlabel('Horizontal distance [m]', fontsize=8)
    ax[2,1].tick_params(labelsize=8)

    ax[0,1].legend(bbox_to_anchor=(1.2, 1.0), loc='upper left', fontsize=7)
    ax[1,1].legend(bbox_to_anchor=(1.2, 1.0), loc='upper left',fontsize=7)
    ax[2,1].legend(bbox_to_anchor=(1.2, 1.0), loc='upper left',fontsize=7)

    plt.tight_layout()
