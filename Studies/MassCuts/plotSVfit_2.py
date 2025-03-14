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

def getDataFramesFromFile(infile, res=None, mass=None,printout=False):
    my_file = open(infile, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    new_data = []
    if printout:
        print(data_into_list)
    if res and mass:
        for line in data_into_list:
            if res in line and mass in line:
                new_data.append(line)
                if printout:print(line)
    else: new_data = data_into_list
    # print(new_data)
    inFiles = Utilities.ListToVector(new_data)
    df_initial = ROOT.RDataFrame("Events", inFiles)
    return df_initial


def buildDfWrapped(df_initial,global_cfg_dict,year,df_initial_cache=None):
    dfBuilder_init = DataFrameBuilderForHistograms(df_initial,global_cfg_dict, period=f"Run2_{year}", region="SR",isData=False)
    dfBuilder_cache_init = DataFrameBuilderForHistograms(df_initial_cache,global_cfg_dict, period=f"Run2_{year}", region="SR",isData=False)
    # print(dfBuilder_init.df.Count().GetValue())
    # print(dfBuilder_cache_init.df.Count().GetValue())\
    if df_initial_cache:
        AddCacheColumnsInDf(dfBuilder_init, dfBuilder_cache_init, "cache_map")
    dfWrapped = PrepareDfForHistograms(dfBuilder_init)
    particleNet_mass = 'particleNet_mass' if 'SelectedFatJet_particleNet_mass_boosted' in dfWrapped.df.GetColumnNames() else 'particleNetLegacy_mass'
    return dfWrapped

def createCacheQuantities(dfWrapped_cache, cache_map_name):
    df_cache = dfWrapped_cache.df
    map_creator_cache = ROOT.analysis.CacheCreator(*dfWrapped_cache.colTypes)()
    df_cache = map_creator_cache.processCache(ROOT.RDF.AsRNode(df_cache), Utilities.ListToVector(dfWrapped_cache.colNames), cache_map_name)
    return df_cache

def AddCacheColumnsInDf(dfWrapped_central, dfWrapped_cache,cache_map_name='cache_map_placeholder'):
    col_names_cache =  dfWrapped_cache.colNames
    col_tpyes_cache =  dfWrapped_cache.colTypes
    #print(col_names_cache)
    #if "kinFit_result" in col_names_cache:
    #    col_names_cache.remove("kinFit_result")
    dfWrapped_cache.df = createCacheQuantities(dfWrapped_cache, cache_map_name)
    if dfWrapped_cache.df.Filter(f"{cache_map_name} > 0").Count().GetValue() <= 0 : raise RuntimeError("no events passed map placeolder")
    dfWrapped_central.AddCacheColumns(col_names_cache,col_tpyes_cache)

def createCentralQuantities(df_central, central_col_types, central_columns):
    map_creator = ROOT.analysis.MapCreator(*central_col_types)()
    df_central = map_creator.processCentral(ROOT.RDF.AsRNode(df_central), Utilities.ListToVector(central_columns), 1)
    #df_central = map_creator.getEventIdxFromShifted(ROOT.RDF.AsRNode(df_central))
    return df_central


def GetModel1D(x_bins):#hist_cfg, var1, var2):
    #x_bins = hist_cfg[var1]['x_bins']
    #y_bins = hist_cfg[var2]['x_bins']
    if type(x_bins)==list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        model = ROOT.RDF.TH1DModel("", "", x_bins_vec.size()-1, x_bins_vec.data())
    else:
        n_x_bins, x_bin_range = x_bins.split('|')
        x_start,x_stop = x_bin_range.split(':')
        model = ROOT.RDF.TH1DModel("", "",int(n_x_bins), float(x_start), float(x_stop))
    return model


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
    parser.add_argument('--year', required=False, type=str, default='all')
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
    for year in years_list:
        key_filter_dict = createKeyFilterDict(global_cfg_dict, f"Run2_{year}")
        # print(key_filter_dict)
        httFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/HTauTauSamples_{year}.txt"
        httCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/HTauTauCaches_{year}.txt"
        signalFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/SignalSamples_{year}.txt"
        signalCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/SignalCaches_{year}.txt"

        DYFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/DYSamples_{year}.txt"
        DYCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/DYCaches_{year}.txt"

        df_sig = getDataFramesFromFile(signalFiles, res="Radion", mass='700',printout=False)
        df_sig_cache = getDataFramesFromFile(signalCaches, res="Radion", mass='700',printout=False)
        dfWrapped_sig = buildDfWrapped(df_sig,global_cfg_dict,year,df_sig_cache)

        # df_bckg = getDataFramesFromFile(httFiles)
        # df_bckg_cache = getDataFramesFromFile(httCaches)
        df_bckg = getDataFramesFromFile(DYFiles)
        df_bckg_cache = getDataFramesFromFile(DYCaches)
        dfWrapped_bckg = buildDfWrapped(df_bckg,global_cfg_dict,year,df_bckg_cache)
        reduce_size=False


        for channel in channels:
            for cat in cats:
                filter_str = key_filter_dict[(channel, 'OS_Iso', cat)]
                if (channel, cat) not in hists.keys():
                    hists[(channel, cat)] = {}
                if year not in hists[(channel, cat)].keys():
                    hists[(channel, cat)][year] = {}
                hists_sig = PlotMass(dfWrapped_sig.df, hist_cfg_dict, global_cfg_dict, filter_str, cat, channel, args.year, return_hists=True)
                hists_bckg = PlotMass(dfWrapped_bckg.df, hist_cfg_dict, global_cfg_dict, filter_str, cat, channel, args.year, return_hists=True)
                hists[(channel, cat)][year]["svFit"] = [hists_sig[0],hists_bckg[0]]
                hists[(channel, cat)][year]["mtt"] = [hists_sig[1],hists_bckg[1]]

    print(f"Filled histograms")
    for channel in channels:
        for cat in cats:
            hist_cat_list_svfit = hists[(channel, cat)][years_list[0]]["svFit"]
            hist_cat_list_mtt = hists[(channel, cat)][years_list[0]]["mtt"]
            for year_idx in range(1, len(years_list)):
                year_value = years_list[year_idx]
                hist_cat_list_svfit_2 = hists[(channel, cat)][year_value]["svFit"]
                hist_cat_list_mtt_2 = hists[(channel, cat)][year_value]["mtt"]
                for hist1_svfit, hist2_svfit in zip(hist_cat_list_svfit, hist_cat_list_svfit_2):
                    hist1_svfit.Add(hist2_svfit)
                for hist1_mtt, hist2_mtt in zip(hist_cat_list_mtt, hist_cat_list_mtt_2):
                    hist1_mtt.Add(hist2_mtt)
            cat_name,channelname,title,outFileName= getLabels(cat, "all", channel)
            bins = hist_cfg_dict["SVfit_m"]["x_rebin"][cat] if cat in hist_cfg_dict["SVfit_m"]["x_rebin"].keys() else hist_cfg_dict["SVfit_m"]["x_rebin"]["other"]
            plot_1D_histogram_SV(hist_cat_list_svfit, labels, f"{title} SVfit",  f"{outFileName}_SVfit", "Run2_all")
            plot_1D_histogram_SV(hist_cat_list_mtt, labels, f"{title} m vis", f"{outFileName}_mvis", "Run2_all")
            print(outFileName + "_SVfit.pdf")
            print(outFileName + "_mvis.pdf")
