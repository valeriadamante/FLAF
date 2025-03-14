import ROOT as rt
import CMS_lumi, tdrstyle
import array

import mplhep as hep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as patches
hep.style.use("CMS")

# https://matplotlib.org/stable/users/explain/colors/colormaps.html
# https://mplhep.readthedocs.io/en/latest/api.html
# https://www.desy.de/~tadej/tutorial/matplotlib_tutorial.html

#plt.rcParams.update({
#    "text.usetex": True,
    #"font.family": "sans-serif",
    #"font.sans-serif": "Helvetica",
#})

def TH1F2np(histo, overflow=False):
    content = histo.GetArray()
    content.SetSize(histo.GetNbinsX() + 2)
    if overflow:
        content = np.array(content)[1:]
    else:
        content = np.array(content)[1:-1]
    binning = np.array(histo.GetXaxis().GetXbins())
    if overflow:
        binning = np.append(binning, [np.inf])
    return content, binning

period_dict = {
    "Run2_2018" : "59.83",
    "Run2_2017" : "41.48",
    "Run2_2016" : "16.8",
    "Run2_2016_HIPM" : "19.5",
    "Run2_all": "138.8",

}
def plot_2D_histogram(histogram, title, xlabel, ylabel, bin_labels, filename, period, rectangle_coordinates):
    hep.style.use("CMS")
    plt.figure(figsize=(25, 15))
    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel(ylabel, fontsize=40)
    hep.hist2dplot(histogram, cmap='Blues', cmax=histogram.GetMaximum())
    #hep.cms.text("work in progress")
    #hep.cms.text("")
    #hep.cms.label("Simulation Preliminary")
    # labels https://mplhep.readthedocs.io/en/latest/api.html#experiment-label-helpers
    hep.cms.label("Preliminary", lumi=period_dict[period], year=period.split('_')[1], fontsize=40)
    if bin_labels:
        textstr = '\n'.join([f"{i}. {label}" for i, label in enumerate(bin_labels)])
        plt.gcf().text(0.8, 0.5, textstr, fontsize=14)


    #plt.grid(True)
    # rectangle: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
    fig = plt.gcf()
    ax = fig.gca()
    #print(rectangle_coordinates)
    x1,y1,lenx,leny = rectangle_coordinates
    rect = patches.Rectangle((x1, y1), lenx, leny, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.gca().set_aspect('auto')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", bbox_inches="tight")
    plt.show()


def plot_1D_histogram(histograms, labels, bins,title, bin_labels, filename, period, mass):
    fig, ax = plt.subplots(figsize=(16,10))
    #plt.xlabel(xlabel, fontsize=40)
    plt.ylabel('Events', fontsize=25)
    plt.xlabel('$M_{HH}$(GeV)', fontsize=25)
    alpha=0.8
    linewidth=2
    content, binning
    colors = ["cornflowerblue", "salmon"]
    for histogram,label,color in zip(histograms,labels,colors):
        bins_x = []
        bins_y = []
        #print(histogram.GetTitle(), label)
        good_binnum = 0
        for binnum in range(1,histogram.GetNbinsX()+1):
            strint_val = str(int(histogram.GetXaxis().GetBinCenter(binnum)))
            if strint_val == mass :
                #print(f"good binnum is {binnum}")
                good_binnum = binnum
            #print(f"binCenter = {strint_val}, mass = {mass}")
            bins_x.append(histogram.GetXaxis().GetBinCenter(binnum))
            bins_y.append(histogram.GetBinContent(binnum))
        hep.histplot(
            np.array(bins_y),
            bins=np.array(bins),
            histtype="fill",
            color=color,
            # alpha=alpha,
            # edgecolor=color,
            label=label,
            # linewidth=linewidth,
            ax=ax,
        )
        alpha-=0.2
        linewidth+=1
        #print(good_binnum)
    #ax.axvline(x = mass, color = 'black', linestyle = '--', label=f'm(res) = {mass} GeV')
    ax.axvline(x=bins_x[good_binnum-1], color='navy', linestyle='--',linewidth=2, label=f'$m_{{X}}$ = {mass} GeV')
    hep.cms.label("Preliminary", lumi=period_dict[period], year=period.split('_')[1], fontsize=25)
    ax.legend(fontsize=25)
    plt.show()
    plt.gca().set_aspect('auto')
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", format="png", bbox_inches="tight")


def plot_1D_histogram_SV(histograms, labels,title, filename, period):
    fig, ax = plt.subplots(figsize=(16,10))
    #plt.xlabel(xlabel, fontsize=40)
    plt.ylabel('Events', fontsize=25)
    plt.xlabel('$m_{\\tau\\tau}$(GeV)', fontsize=25)
    colors = ["blue","red"]
    # colors = ["cornflowerblue", "salmon"]
    for histogram,label,color in zip(histograms,labels,colors):
        # histogram.Scale(1/histogram.Integral()) #0,histogram.GetNbinsX()+1
        bins_y = []
        bins_x = []
        # print(histogram.Sumw2())
        sumw2= histogram.Sumw2()
        for binnum in range(1,histogram.GetNbinsX()+1):
            # bins_x.append(histogram.GetBinLowEdge(binnum))
            bins_y.append(histogram.GetBinContent(binnum))
            # print(binnum, histogram.GetBinCenter(binnum), histogram.GetBinContent(binnum))
            # print(len(bins_x), len(bins_y))
        # print(bins_x)
        # print(bins_y)
        print(len(bins_x), len(bins_y))
        # bins_x.append(histogram.GetBinLowEdge(histogram.GetNbinsX() + 1))  # Aggiungi l'ultimo bordo superiore
        bins_x = [histogram.GetBinLowEdge(b) for b in range(1, histogram.GetNbinsX() + 2)]


        hep.histplot(
            np.array(bins_y),
            bins=np.array(bins_x),
            yerr=True,
            xerr=True,
            # w2=sumw2,
            density=True,
            histtype="errorbar",
            color=color,
            label=label,
            ax=ax,
            markersize = 10,
        )
    hep.cms.label("Preliminary", lumi=period_dict[period], year=period.split('_')[1], fontsize=25)
    ax.legend(fontsize=25)
    plt.show()
    plt.gca().set_aspect('auto')
    print(filename)
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", format="png", bbox_inches="tight")