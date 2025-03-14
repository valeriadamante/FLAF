import ROOT
import os
import sys
import yaml
import numpy as np

ROOT.gStyle.SetOptStat(0)

#ROOT.gStyle.SetPalette(109)
ROOT.gStyle.SetPalette(51)
ROOT.TColor.InvertPalette()

# kCubehelix=58, # 5/10 ma inverted è poco colorblind proven
# kCMYK=73, # 6/10 ma inverted è o cess
# kWaterMelon=108, # 3/10 ma inverted è meglio
# kCividis=113, # 7/10 # invertito è 2/10
# kTemperatureMap=104, # 6.5 /10 # invertito è 2/10
# kColorPrintableOnGrey=62, # 5 /10
# kDeepSea=51, # 8 /10 (inverted è top)
# kBlueYellow= 54, # 7 /10
# kCool=109, # 8 /10 (inverted anche è top)
# kBlueGreenYellow=71, # 6/10
# kBird=57, # 8/10
# kRainBow=55, # /10
# kThermometer=105, # 6 /10
# kViridis=112, # /10

if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

from Studies.MassCuts.DrawPlots import plot_1D_histogram_SV


def RebinHisto(hist_initial, new_binning, sample, wantOverflow=True, verbose=False):
    new_binning_array = array.array('d', new_binning)
    new_hist = hist_initial.Rebin(len(new_binning)-1, sample, new_binning_array)
    if sample == 'data' : new_hist.SetBinErrorOption(ROOT.TH1.kPoisson)
    if wantOverflow:
        n_finalbin = new_hist.GetBinContent(new_hist.GetNbinsX())
        n_overflow = new_hist.GetBinContent(new_hist.GetNbinsX()+1)
        new_hist.SetBinContent(new_hist.GetNbinsX(), n_finalbin+n_overflow)
        err_finalbin = new_hist.GetBinError(new_hist.GetNbinsX())
        err_overflow = new_hist.GetBinError(new_hist.GetNbinsX()+1)
        new_hist.SetBinError(new_hist.GetNbinsX(), math.sqrt(err_finalbin*err_finalbin+err_overflow*err_overflow))

    if verbose:
        for nbin in range(0, len(new_binning)):
            print(f"nbin = {nbin}, content = {new_hist.GetBinContent(nbin)}, error {new_hist.GetBinError(nbin)}")
    fix_negative_contributions,debug_info,negative_bins_info = FixNegativeContributions(new_hist)
    if not fix_negative_contributions:
        print("negative contribution not fixed")
        print(fix_negative_contributions,debug_info,negative_bins_info)
        for nbin in range(0,new_hist.GetNbinsX()+1):
            content=new_hist.GetBinContent(nbin)
            if content<0:
               print(f"for {sample}, bin {nbin} content is < 0:  {content}")

    return new_hist

def getLabels(cat,year,channel):
    cat_name = cat.split('_')[0]
    # if cat_name == 'baseline':
    #     cat_name == cat
    outDir = f"output/Masses_histograms_SVFit/Run2_{year}/{cat_name}"
    os.makedirs(outDir, exist_ok=True)
    outFileName = f"{outDir}/{channel}_{cat}"
    channelnames = {
        "eTau":"bbe$\\tau$",
        "muTau":"bb$\\mu\\tau$",
        "tauTau":"bb$\\tau\\tau$",
    }
    channelname = channelnames[channel]
    title = f"{channelname} {cat_name}"
    return cat_name,channelname,title,outFileName


def PlotMass(df, hist_cfg_dict, global_cfg_dict,filter_str,cat,channel='tauTau', year='2018', return_hists=False):
    saveFile = not return_hists
    bins = hist_cfg_dict["SVfit_m"]["x_rebin"][cat] if cat in hist_cfg_dict["SVfit_m"]["x_rebin"].keys() else hist_cfg_dict["SVfit_m"]["x_rebin"]["other"]
    cat_name,channelname,title,outFileName = getLabels(cat,"all",channel)
    hist_list = []
    total_weight_expression = GetWeight(channel,cat,global_cfg_dict['boosted_categories']) #if sample_type!='data' else "1"
    btag_weight = GetBTagWeight(global_cfg_dict,cat,applyBtag=False)
    total_weight_expression = "*".join([total_weight_expression,btag_weight])

    hist_SVfit_m = df.Filter(filter_str).Define("final_weight", total_weight_expression).Histo1D(GetModel1D(bins), "SVfit_m", "final_weight").GetValue()
    hist_list.append(hist_SVfit_m)
    hist_mttvis = df.Filter(filter_str).Define("final_weight", total_weight_expression).Histo1D(GetModel1D(bins),  "tautau_m_vis", "final_weight").GetValue()
    hist_list.append(hist_mttvis)
    # if saveFile:
    #     plot_1D_histogram(hist_list,labels, bins,title, [], outFileName, f"Run2_{year}")
    #     print(outFileName+".pdf")
    if return_hists:
        return hist_list



if __name__ == "__main__":
    import argparse
    import yaml
    import Common.Utilities as Utilities
    from Analysis.HistHelper import *
    from Analysis.hh_bbtautau import *
    import GetIntervals
    import GetIntervalsSimultaneously
    parser = argparse.ArgumentParser()
    # parser.add_argument('--year', required=False, type=str, default='all')
    parser.add_argument('--cat', required=False, type=str, default='')
    parser.add_argument('--channels', required=False, type=str, default = '')
    args = parser.parse_args()

    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')
    ROOT.gInterpreter.Declare(f'#include "include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')
    ROOT.gInterpreter.Declare(f'#include "include/pnetSF.h"')
    ROOT.gROOT.ProcessLine('#include "include/AnalysisTools.h"')

    global_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/HH_bbtautau/global.yaml'
    global_cfg_dict = {}
    with open(global_cfg_file, 'r') as f:
        global_cfg_dict = yaml.safe_load(f)


    hist_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/plot/histograms.yaml'
    hist_cfg_dict = {}
    with open(hist_cfg_file, 'r') as f:
        hist_cfg_dict = yaml.safe_load(f)


    hists = {}
    channels = ['eTau', 'muTau','tauTau'] if args.channels == '' else args.channels.split(',')
    global_cfg_dict['channels_to_consider']=channels
    cats = ['baseline', 'inclusive','res1b_cat3', 'res2b_cat3'] if args.cat == '' else args.cat.split(',') #
    global_cfg_dict["categories"] = cats
    years_list = args.year.split(",")
    if args.year == 'all':
        years_list = ["2016_HIPM","2016","2017","2018"]
    labels = ["$H\\rightarrow \\tau\\tau$", "DY"]
    inputFile_SV = "/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_MassCut_2p1/SC/SR/Run2_all/merged/SVfit_m/SVfit_m.root"
    inputFile_mvis = "/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_MassCut_2p1/SC/SR/Run2_all/merged/tautau_m_vis/tautau_m_vis.root"
    inputfileroot_SV = ROOT.TFile(inputFile_SV)
    inputfileroot_mvis = ROOT.TFile(inputFile_mvis)
    histograms = {}
    new_binning = hist_cfg_dict["SVfit_m"]["x_rebin"][cat] if cat in hist_cfg_dict["SVfit_m"]["x_rebin"].keys() else hist_cfg_dict["SVfit_m"]["x_rebin"]["other"]
    print(f"Filled histograms")
    for infileroot,massname in zip([inputfileroot_SV, inputfileroot_mvis],["SV","mvis"]):
        histograms[massname] = {}
        for channel in channels:
            histograms[massname][channel] = {}
            dir_0=infileroot.Get(channel)
            dir_1=dir_0.Get("OS_Iso")
            for cat in cats:
                histograms[massname][channel][cat] = {}
                dir_2 = dir_1.Get(cat)
                hist_initial_DY= dir_2.Get("DY")
                hist_initial_DY.SetDirectory(0)
                hist_initial_Signal=dir_2.Get("GluGluToBulkGravitonToHHTo2B2Tau_M-700")
                hist_initial_Signal.SetDirectory(0)
                hist_DY = RebinHisto(hist_initial_DY, new_binning, "DY", wantOverflow=True, verbose=False)
                hist_Signal = RebinHisto(hist_initial_Signal, new_binning, "GluGluToBulkGravitonToHHTo2B2Tau_M-700", wantOverflow=True, verbose=False)
                if "DY" not in histograms[massname][channel][cat].keys():
                    histograms[massname][channel][cat]["DY"] = []
                histograms[massname][channel][cat]["DY"].append(hist_initial_DY)

                if "Signal" not in histograms[massname][channel][cat].keys():
                    histograms[massname][channel][cat]["Signal"] = []
                histograms[massname][channel][cat]["Signal"].append(hist_initial_Signal)

                cat_name,channelname,title,outFileName= getLabels(cat, "all", channel)
        for channel in channels:
            for cat in cats:
                plot_1D_histogram_SV([histograms["SV"][channel][cat]["Signal"],histograms["SV"][channel][cat]["DY"]], labels, f"{title} SVfit",  f"{outFileName}_SVfit", "Run2_all")
                plot_1D_histogram_SV([histograms["mvis"][channel][cat]["Signal"],histograms["mvis"][channel][cat]["DY"]], labels, f"{title} m vis",  f"{outFileName}_mvis", "Run2_all")
                print(outFileName + "_SVfit.pdf")
                print(outFileName + "_mvis.pdf")
