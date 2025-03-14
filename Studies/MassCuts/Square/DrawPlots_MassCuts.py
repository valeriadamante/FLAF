import ROOT
import os
import sys
import math
import yaml
import numpy as np

import argparse
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

import Common.Utilities as Utilities
from Analysis.HistHelper import *
from Analysis.hh_bbtautau import *
from Studies.MassCuts.Square.GetSquareIntervals import *
from Studies.MassCuts.Square.SquarePlot import *

import Studies.MassCuts.Ellypse.plotWithEllypse as Ellypse

channel_names = {
    "eTau" : "e \\tau_h",
    "muTau" : "\\mu \\tau_h",
    "tauTau" : "\\tau_h \\tau_h",
}
def createCacheQuantities(dfWrapped_cache, cache_map_name):
    df_cache = dfWrapped_cache.df
    map_creator_cache = ROOT.analysis.CacheCreator(*dfWrapped_cache.colTypes)()
    df_cache = map_creator_cache.processCache(ROOT.RDF.AsRNode(df_cache), Utilities.ListToVector(dfWrapped_cache.colNames), cache_map_name)
    return df_cache


def AddCacheColumnsInDf(dfWrapped_central, dfWrapped_cache,cache_map_name='cache_map_placeholder'):
    col_names_cache =  dfWrapped_cache.colNames
    col_types_cache =  dfWrapped_cache.colTypes
    dfWrapped_cache.df = createCacheQuantities(dfWrapped_cache, cache_map_name)
    if dfWrapped_cache.df.Filter(f"{cache_map_name} > 0").Count().GetValue() <= 0 : raise RuntimeError("no events passed map placeolder")
    dfWrapped_central.AddCacheColumns(col_names_cache,col_types_cache)


def FilterForbJets(cat,dfWrapper_s):
    if cat == 'boosted':
        dfWrapper_s.df = dfWrapper_s.df.Define("FatJet_atLeast1BHadron",
        "SelectedFatJet_nBHadrons>0").Filter("SelectedFatJet_p4[FatJet_atLeast1BHadron].size()>0")
    else:
        dfWrapper_s.df = dfWrapper_s.df.Filter("b1_hadronFlavour==5 && b2_hadronFlavour==5 ")
    return dfWrapper_s

def GetModel2D(x_bins, y_bins):#hist_cfg, var1, var2):
    #x_bins = hist_cfg[var1]['x_bins']
    #y_bins = hist_cfg[var2]['x_bins']
    if type(x_bins)==list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        if type(y_bins)==list:
            y_bins_vec = Utilities.ListToVector(y_bins, "double")
            model = ROOT.RDF.TH2DModel("", "", x_bins_vec.size()-1, x_bins_vec.data(), y_bins_vec.size()-1, y_bins_vec.data())
        else:
            n_y_bins, y_bin_range = y_bins.split('|')
            y_start,y_stop = y_bin_range.split(':')
            model = ROOT.RDF.TH2DModel("", "", x_bins_vec.size()-1, x_bins_vec.data(), int(n_y_bins), float(y_start), float(y_stop))
    else:
        n_x_bins, x_bin_range = x_bins.split('|')
        x_start,x_stop = x_bin_range.split(':')
        if type(y_bins)==list:
            y_bins_vec = Utilities.ListToVector(y_bins, "double")
            model = ROOT.RDF.TH2DModel("", "",int(n_x_bins), float(x_start), float(x_stop), y_bins_vec.size()-1, y_bins_vec.data())
        else:
            n_y_bins, y_bin_range = y_bins.split('|')
            y_start,y_stop = y_bin_range.split(':')
            model = ROOT.RDF.TH2DModel("", "",int(n_x_bins), float(x_start), float(x_stop), int(n_y_bins), float(y_start), float(y_stop))
    return model

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
    dfBuilder_init = DataFrameBuilderForHistograms(df_initial,global_cfg_dict, f"Run2_{year}")
    dfBuilder_cache_init = DataFrameBuilderForHistograms(df_initial_cache,global_cfg_dict, f"Run2_{year}")
    if df_initial_cache:
        AddCacheColumnsInDf(dfBuilder_init, dfBuilder_cache_init, "cache_map")
    dfWrapped = PrepareDfForHistograms(dfBuilder_init)
    particleNet_mass = 'particleNet_mass' if 'SelectedFatJet_particleNet_mass_boosted' in dfWrapped.df.GetColumnNames() else 'particleNetLegacy_mass'
    return dfWrapped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=False, type=str, default='2018')
    parser.add_argument('--cat', required=False, type=str, default='res2b_cat3')
    parser.add_argument('--channels', required=False, type=str, default='tauTau')
    parser.add_argument('--res', required=False, type=str, default="Radion")
    parser.add_argument('--mass', required=False, type=str, default=None)
    parser.add_argument('--compute_bckg', required=False, type=bool, default=False)
    parser.add_argument('--ell_params', required=False, type=str, default="")
    parser.add_argument('--square_params', required=False, type=str, default="")

    args = parser.parse_args()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')
    ROOT.gInterpreter.Declare(f'#include "include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')
    ROOT.gROOT.ProcessLine('#include "include/AnalysisTools.h"')
    ROOT.gInterpreter.Declare(f'#include "include/pnetSF.h"')

    signalFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/SignalSamples_{args.year}.txt"
    signalCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/SignalCaches_{args.year}.txt"

    TTFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/TTSamples_{args.year}.txt"
    TTCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/TTCaches_{args.year}.txt"


    global_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/HH_bbtautau/global.yaml'
    global_cfg_dict = {}
    with open(global_cfg_file, 'r') as f:
        global_cfg_dict = yaml.safe_load(f)
    global_cfg_dict['channels_to_consider']=args.channels.split(',')

    hist_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/plot/histograms.yaml'
    hist_cfg_dict = {}
    with open(hist_cfg_file, 'r') as f:
        hist_cfg_dict = yaml.safe_load(f)

    if args.res and args.mass:
        df_sig = getDataFramesFromFile(signalFiles,args.res, args.mass)
        df_sig_cache = getDataFramesFromFile(signalCaches,args.res, args.mass)
        dfWrapped_sig = buildDfWrapped(df_sig,global_cfg_dict,args.year,df_sig_cache)

    if args.compute_bckg:
        df_bckg = getDataFramesFromFile(TTFiles)
        df_bckg_cache = getDataFramesFromFile(TTCaches)
        dfWrapped_bckg = buildDfWrapped(df_bckg,global_cfg_dict,args.year,df_bckg_cache)

    reduce_size=True
    tt_mass = "SVfit_m"

    if args.square_params:
        masses = args.square_params.split(",")
        mtt_min, mtt_max, mbb_min, mbb_max = masses[0],masses[1],masses[2],masses[3]
        print("mtt_min, mtt_max, mbb_min, mbb_max")
        print(mtt_min, mtt_max, mbb_min, mbb_max)
    else:
        mtt_min, mtt_max, mbb_min, mbb_max = 80,170,70,150 #80,190,70,150

    if args.ell_params:
        ellypse_params = args.ell_params.split(",")
        A, B_final, C, D_final = ellypse_params[0],ellypse_params[1],ellypse_params[2],ellypse_params[3]
    else:
        A, B_final, C, D_final = 121,26,115,36

    new_par_A = (math.ceil(A*100)/100)  # -> 2.36
    new_par_B = (math.ceil(B_final*100)/100)  # -> 2.36
    new_par_C = (math.ceil(C*100)/100)  # -> 2.36
    new_par_D = (math.ceil(D_final*100)/100)  # -> 2.36
    # print(f"{tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} && {bb_mass}< {mbb_max} && {bb_mass} > {mbb_min}")

    x_bins = hist_cfg_dict[tt_mass]['x_rebin']['other']
    rectangle_coordinates = mbb_max, mbb_min, mtt_max, mtt_min
    ellypse_par=  new_par_C, new_par_D,new_par_A, new_par_B


    for cat in  args.cat.split(','):
        bb_mass = "bb_m_vis" if cat != 'boosted_cat3' else "bb_m_vis_softdrop"
        y_bins = hist_cfg_dict[bb_mass]['x_rebin']['other']
        outFile_prefix = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Square/MassCut2DPlots/Run2_{args.year}/{cat}/"
        text_coordinates = [180, 270, 180, 260]
        for channel in global_cfg_dict['channels_to_consider']:
            channel_name = channel_names[channel]
            if args.compute_bckg:
                dfWrapped_bckg.df = dfWrapped_bckg.df.Filter(f"SVfit_valid>0 && OS_Iso && {channel} && {cat} && SVfit_m > 0")
                dfWrapped_bckg = FilterForbJets(cat,dfWrapped_bckg)
                df_bckg_new = dfWrapped_bckg.df
                if reduce_size: df_bckg_new = df_bckg_new.Range(100000)
                # df_bckg_new = df_bckg_new.Filter(f"(({tt_mass} - {new_par_A})*({tt_mass} - {new_par_A}) / ({new_par_B}*{new_par_B}) + ({bb_mass} -  {new_par_C})*({bb_mass} - {new_par_C}) / ({new_par_D}*{new_par_D})) < 1")
                hist_bckg=df_bckg_new.Histo2D(GetModel2D(x_bins, y_bins),bb_mass, tt_mass).GetValue()
                outFile_prefix+=f"TT/"
                os.makedirs(outFile_prefix,exist_ok=True)
                finalFileName = f"{outFile_prefix}{channel}_TT"
                Ellypse.plot_2D_histogram(hist_bckg, "$m_{bb}$", f"$m^{{SV}}_{{{channel_name}}}$", None, f"{finalFileName}_square", f"Run2_{args.year}", cat.split('_')[0], f"$bb{channel_name}$", None,rectangle_coordinates, text_coordinates)
                print(f"{finalFileName}_square.png")
                # Ellypse.plot_2D_histogram(hist_bckg, "$m_{bb}$", f"$m^{{SV}}_{{{channel_name}}}$", None, f"{finalFileName}_ellypse_square", f"Run2_{args.year}", cat.split('_')[0], f"$bb{channel_name}$", ellypse_par,rectangle_coordinates, text_coordinates)
                # print(f"{finalFileName}_ellypse_square.png")
                # Ellypse.plot_2D_histogram(hist_bckg, "$m_{bb}$", f"$m^{{SV}}_{{{channel_name}}}$", None, f"{finalFileName}_ellypse", f"Run2_{args.year}", cat.split('_')[0], f"$bb{channel_name}$", ellypse_par,None, text_coordinates)
                # print(f"{finalFileName}_ellypse.png")

            if args.res and args.mass:
                dfWrapped_sig.df = dfWrapped_sig.df.Filter(f"SVfit_valid >0 && OS_Iso && {channel} && {cat} && SVfit_m > 0")
                dfWrapped_sig = FilterForbJets(cat,dfWrapped_sig)
                df_sig_new = dfWrapped_sig.df
                if reduce_size :
                    df_sig_new = df_sig_new.Range(100000)
                # df_sig_new =  df_sig_new.Filter(f"(({tt_mass} - {new_par_A})*({tt_mass} - {new_par_A}) / ({new_par_B}*{new_par_B}) + ({bb_mass} - {new_par_C})*({bb_mass} - {new_par_C}) / ({new_par_D}*{new_par_D})) < 1")
                outFile_prefix = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Square/MassCut2DPlots/Run2_{args.year}/{cat}/"
                hist_sig=df_sig_new.Histo2D(GetModel2D(x_bins, y_bins),bb_mass, tt_mass).GetValue()
                outFile_prefix+=f"{args.res}/{args.mass}/"
                os.makedirs(outFile_prefix,exist_ok=True)
                finalFileName = f"{outFile_prefix}{channel}_{args.res}_M-{args.mass}"
                Ellypse.plot_2D_histogram(hist_sig, "$m_{bb}$", f"$m^{{SV}}_{{{channel_name}}}$", None, f"{finalFileName}_square", f"Run2_{args.year}", cat.split('_')[0], f"bb${channel_name}$", None,rectangle_coordinates, text_coordinates)
                print(f"{finalFileName}_square.png")
                # Ellypse.plot_2D_histogram(hist_sig, "$m_{bb}$", f"$m^{{SV}}_{{{channel_name}}}$", None, f"{finalFileName}_ellypse_square", f"Run2_{args.year}", cat.split('_')[0], f"bb${channel_name}$", ellypse_par,rectangle_coordinates, text_coordinates)
                # print(f"{finalFileName}_ellypse_square.png")
                # Ellypse.plot_2D_histogram(hist_sig, "$m_{bb}$", f"$m^{{SV}}_{{{channel_name}}}$", None, f"{finalFileName}_ellypse", f"Run2_{args.year}", cat.split('_')[0], f"bb${channel_name}$", ellypse_par,None, text_coordinates)
                # print(f"{finalFileName}_ellypse.png")



