import numpy as np
from scipy.integrate import odeint
import pdb
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import seaborn as sns
import math
import pandas as pd
from scipy.signal import savgol_filter
def color20():
# These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
           r, g, b = tableau20[i]
           tableau20[i] = (r / 255., g / 255., b / 255.)
    return tableau20

class Invasion_plot(object):
        def __init__(self, par):
            self.file_input=par[0]
            #self.flag_condition=par[0]
            self.initial_type='steady_initial'
            self.df = pd.read_csv(self.file_input)
            self.df=self.df[self.df['Initial_type']==self.initial_type]
        def _plot_richness_distribution(self,fig_name):
            data=self.df.loc[self.df['richness']>0,['richness','Resource Type']]
            sns.set(style="white")
            x, y, hue = "richness", "counts", "Resource Type"
            hue_order = ['quadratic','linear','constant']
            f, ax = plt.subplots()
            #sns.countplot(x=x, hue=hue, data=df, ax=axes[0])
            prop_df = (data[x]
                       .groupby(data[hue])
                       .value_counts(normalize=True)
                       .rename(y)
                       .reset_index())
            sns.barplot(x=x, y=y, hue=hue, hue_order=hue_order,data=prop_df, ax=ax)
            ax.legend(loc='upper right', frameon=False, fontsize=15)
            ax.set_ylabel('renormalized probability')
            f.set_size_inches(8, 6)
            f.savefig(fig_name, bbox_inches='tight',dpi=100)
            return f 
        def _plot_richness_step(self, fig_name,ticker=20):
            data=self.df.loc[self.df['richness']>0,['step','richness','Resource Type']]
            sns.set(style="white")
            hue_order = ['quadratic','linear','constant']
            g = sns.factorplot(x="step", y="richness", hue="Resource Type", hue_order=hue_order, data=data,size=5, aspect=1.2)
            # iterate over axes of FacetGrid
            for ax in g.axes.flat:
                labels = ax.get_xticklabels() # get x labels
                for i,l in enumerate(labels):
                    if(i%ticker!= 0): labels[i] = '' # skip even labels
                ax.set_xticklabels(labels) # set new labels
            g.savefig(fig_name,bbox_inches='tight',dpi=100)
            return g
        def _plot_richness_power(self, fig_name,plot_type=None):
            data=self.df.loc[self.df['richness']>0,['richness','consumed power','Resource Type']]
            sns.set(style="white")
            hue_order = ['quadratic','linear','constant']
            if not plot_type:
                g = sns.factorplot(x="richness", y="consumed power", hue="Resource Type", hue_order=hue_order, data=data,size=5, aspect=1.2)
            elif plot_type=="violin":  
                g = sns.factorplot(x="richness", y="consumed power", hue="Resource Type", hue_order=hue_order, data=data,size=5, aspect=1.2,kind="violin")
            # iterate over axes of FacetGrid
            for ax in g.axes.flat:
                labels = ax.get_xticklabels() # get x labels
                for i,l in enumerate(labels):
                    if(i%2!= 0): labels[i] = '' # skip even labels
                ax.set_xticklabels(labels) # set new labels 
            g.savefig(fig_name,bbox_inches='tight',dpi=100)
            return g            
        def _plot_step_power(self, fig_name,ticker=20):
            data=self.df.loc[self.df['richness']>0,['step','consumed power','Resource Type']]
            sns.set(style="white")
            hue_order = ['quadratic','linear','constant']
            g = sns.factorplot(x="step", y="consumed power", hue="Resource Type", hue_order=hue_order, data=data,size=5, aspect=1.2,kind="violin")
            # iterate over axes of FacetGrid
            for ax in g.axes.flat:
                labels = ax.get_xticklabels() # get x labels
                for i,l in enumerate(labels):
                    if(i%ticker!= 0): labels[i] = '' # skip even labels
                ax.set_xticklabels(labels) # set new labels 
            g.savefig(fig_name,bbox_inches='tight',dpi=100)
            return g  
        def _plot_community_richness(self, fig_name,xright=30, step=1000):   
            tableau20=color20()
            sns.set(style="white")
            fig, ax = plt.subplots(1,3, figsize=(28, 8))
            axs = ax.ravel()
            i=0
            for ax,R_type in zip(axs, ['quadratic','linear', 'constant']):
                data=self.df[self.df['Resource Type']==R_type ]
                data=data[data['richness']>0 ]
                data=data[data['step']<step ]
                columns=['richness','Community augmentation','Replacement','Indirect failure','Rejection failure']
                def func_pro(situation, richness):
                    return list(data.loc[data['richness']==richness, [situation]].sum()/len(data.loc[data['richness']==richness, [situation]]))[0]
                data_pro= pd.DataFrame(columns=columns)
                for richness in set(data['richness']):
                    para_df_current = pd.DataFrame([[richness, func_pro(columns[1],richness), func_pro(columns[2],richness),func_pro(columns[3],richness),func_pro(columns[4],richness)]],columns=columns)
                    data_pro =pd.concat([data_pro,para_df_current],ignore_index=True)   
                data_pro['Indirect failure']=1-data_pro['Community augmentation']-data_pro['Replacement']-data_pro['Rejection failure']
                #data_pro.set_index('richness').plot(kind='bar', stacked=True, ax=ax) 
                for column in columns[1:]:
                    ax.scatter(data_pro['richness'],data_pro[column])
                    ax.plot(data_pro['richness'], savgol_filter(data_pro[column],5,2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=0 )
                ax.set_ylim([0,1.01])
                ax.set_xlim([0,xright])
                ax.set_ylabel('normalized probability',fontsize=15)
                ax.set_xlabel('richness',fontsize=15)
                ax.set_title(R_type)
                ax.legend(bbox_to_anchor=(0.5,1.1),loc='center', mode="expand", shadow=True, fancybox=True, ncol=2)
                ax1=ax.twinx()
                data_1=data.groupby('richness')['consumed power'].mean()
                ax1.plot(data_1.loc[:],c=tableau20[10],label='consumed power')
                ax1.set_ylabel('consumed power',fontsize=15)
                if i==0:
                    ax1.legend(loc='upper right')
                ax1.scatter(data['richness'],data['consumed power'],label='consumed power',c=tableau20[11], alpha=0.1,s=10)    
                i=i+1  
                ax1.set_xlim([0,xright])
            fig.savefig(fig_name,bbox_inches='tight',dpi=50) 
            return fig 
        def _plot_community_step(self, fig_name,xright=300):
            tableau20=color20()
            sns.set(style="white")
            fig, ax = plt.subplots(1,3, figsize=(28, 8))
            axs = ax.ravel()
            i=0
            for ax,R_type in zip(axs, ['quadratic','linear', 'constant']):
                data=self.df[self.df['Resource Type']==R_type ]
                data=data[data['step']<1000 ]
                data=data[data['richness']>0 ]
                columns=['step','Community augmentation','Replacement','Indirect failure','Rejection failure']
                def func_pro(situation, step):
                    return list(data.loc[data['step']==step, [situation]].sum()/len(data.loc[data['step']==step, [situation]]))[0]
                    #return list(data.loc[data['step']==step, [situation]].sum())[0]
                data_pro= pd.DataFrame(columns=columns)
                for step in set(data['step']):
                    para_df_current = pd.DataFrame([[step, func_pro(columns[1],step), func_pro(columns[2],step),func_pro(columns[3],step),func_pro(columns[4],step)]],columns=columns)
                    data_pro =pd.concat([data_pro,para_df_current],ignore_index=True) 
                for column in columns[1:]:
                    ax.scatter(data_pro['step'],data_pro[column])  
                ax.set_ylim([0,1])
                ax.set_ylabel('normalized probability',fontsize=15)
                ax.set_xlabel('step',fontsize=15)
                ax.set_title(R_type)
                ax.legend(bbox_to_anchor=(0.5,1.1),loc='center', mode="expand", shadow=True, fancybox=True, ncol=2)
                ax.set_xlim([0,xright])
                ax1=ax.twinx()
                data_1=data.groupby('step')['consumed power'].mean()
                ax1.plot(data_1.loc[:],c=tableau20[10],label='consumed power')
                if i==0:
                    ax1.legend(loc='upper right')
                ax1.scatter(data['step'],data['consumed power'],label='consumed power',c=tableau20[11], alpha=0.1,s=10)
                ax1.set_ylabel('consumed power',fontsize=15)
                ax1.set_xlim([0,xright])
                i=i+1
            fig.savefig(fig_name,bbox_inches='tight',dpi=50) 
            return fig 

def Efficiency_plot(fig_name, TT, Eff, Survive, Ntotal, N_entropy, Rtotal,R_entropy):
    tableau20=color20()
    f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax1.set_xlim([0, np.amax(TT)])
    ax1.scatter(TT, Eff, color=tableau20[2], label='Efficiency')
    ax1.axhline(y=1.0, linestyle='--', c ='r', linewidth = 5)
    #ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylim([0, 1.2])
    ax1.set_xlabel('Steps', fontsize=15)
    ax1.set_ylabel('Energy utilization efficiency', fontsize=15)

    ax11 = ax1.twinx()
    ax11.scatter(TT, Survive, color=tableau20[4], label='Survivor')
    ax11.set_ylim([0, 2*np.amax(Survive)])
    ax11.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax11.set_ylabel('Num of survivors', fontsize=15)
    legend = ax1.legend(loc='upper left', frameon=False, fontsize=15)
    legend = ax11.legend(loc='upper right', frameon=False, fontsize=15)



    ax2.scatter(TT, Ntotal, color=tableau20[0], label='Total speicies')
    ax2.set_xlim([0, 30])
#ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylim([0, 1.3*np.amax(Ntotal)])
    ax2.set_xlabel('Steps', fontsize=15)
    ax2.set_ylabel('Total speicies', fontsize=15)

    ax21 = ax2.twinx()
    ax21.scatter(TT, N_entropy, color=tableau20[1], label='Entropye')
    ax21.set_ylim([0,2*np.amax(N_entropy)])
    ax21.set_ylabel('Entropy of speicies', fontsize=15)
#ax21.set_yscale('log')
    legend = ax2.legend(loc='upper left', frameon=False, fontsize=15)
    legend = ax21.legend(loc='upper right', frameon=False, fontsize=15)


    ax3.scatter(TT, Rtotal, color=tableau20[6], label='Total resources')
    ax3.set_xlim([0, 30])
#ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.set_ylim([0, 1.3*np.amax(Rtotal)])
    ax3.set_xlabel('Steps', fontsize=15)
    ax3.set_ylabel('Total resources', fontsize=15)

    ax31 = ax3.twinx()
    ax31.scatter(TT, R_entropy, color=tableau20[7], label='Entropy')
    ax31.set_ylim([0, 2*np.amax(R_entropy)])
#ax31.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax31.set_ylabel('Entropy of resources', fontsize=15)
#ax31.set_yscale('log')
    legend = ax3.legend(loc='upper left', frameon=False, fontsize=15)
    legend = ax31.legend(loc='upper right', frameon=False, fontsize=15)

    f.tight_layout()
    plt.subplots_adjust(hspace = .1,wspace=0.4)
    f.set_size_inches(20, 6)
    f.savefig(fig_name, bbox_inches='tight',dpi=100)
    return f