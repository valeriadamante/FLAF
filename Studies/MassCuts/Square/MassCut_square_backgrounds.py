
import ROOT
import os
import sys
import math
import yaml
import numpy as np

import argparse
import csv


if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

import Common.Utilities as Utilities
from Analysis.HistHelper import *
from Analysis.hh_bbtautau import *
from Studies.MassCuts.Square.GetSquareIntervals import *
from Studies.MassCuts.Square.SquarePlot import *

import Studies.MassCuts.Ellypse.plotWithEllypse as Ellypse


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


def getDataFramesFromFile(infile):
    my_file = open(infile, "r")
    data = my_file.read()
    data_into_list = []
    for linentry in data.split("\n"):
        data_into_list.append(linentry.replace("CC/","SC/"))
    inFiles = Utilities.ListToVector(data_into_list)
    df_initial = ROOT.RDataFrame("Events", inFiles)
    return df_initial

def buildDfWrapped(df_initial,global_cfg_dict,year,df_initial_cache=None):
    dfBuilder_init = DataFrameBuilderForHistograms(df_initial,global_cfg_dict, f"Run2_{year}")
    dfBuilder_cache_init = DataFrameBuilderForHistograms(df_initial_cache,global_cfg_dict, f"Run2_{year}")
    # print(dfBuilder_init.df.Count().GetValue())
    # print(dfBuilder_cache_init.df.Count().GetValue())
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
    parser.add_argument('--calculate_square_intervals', required=False, type=bool, default=False)
    parser.add_argument('--square_params', required=False, type=str, default="")

    args = parser.parse_args()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')
    ROOT.gInterpreter.Declare(f'#include "include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')
    ROOT.gROOT.ProcessLine('#include "include/AnalysisTools.h"')
    ROOT.gInterpreter.Declare(f'#include "include/pnetSF.h"')

    DYFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/DYSamples_{args.year}.txt"
    DYCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/DYCaches_{args.year}.txt"

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
    df_DY = getDataFramesFromFile(DYFiles)
    df_DY_cache = getDataFramesFromFile(DYCaches)
    dfWrapped_DY = buildDfWrapped(df_DY,global_cfg_dict,args.year,df_DY_cache)
    df_TT = getDataFramesFromFile(TTFiles)
    df_TT_cache = getDataFramesFromFile(TTCaches)
    dfWrapped_TT = buildDfWrapped(df_TT,global_cfg_dict,args.year,df_TT_cache)
    reduce_size=False
    tt_mass = "SVfit_m"
    x_bins = hist_cfg_dict[tt_mass]['x_rebin']['other']
    for cat in  args.cat.split(','):
        # print(cat)
        bb_mass = "bb_m_vis" if cat != 'boosted_cat3' else "bb_m_vis_softdrop"
        y_bins = hist_cfg_dict[bb_mass]['x_rebin']['other']
        # print(bb_mass)
        # print(x_bins)
        for channel in global_cfg_dict['channels_to_consider']:
            # print(channel,cat)
            dfWrapped_DY.df = dfWrapped_DY.df.Filter(f"SVfit_valid >0 && OS_Iso && {channel} && {cat} && SVfit_m > 0")
            dfWrapped_DY = FilterForbJets(cat,dfWrapped_DY)
            df_DY_new = dfWrapped_DY.df
            if reduce_size : df_DY_new = df_DY_new.Range(100000)
            dfWrapped_TT.df = dfWrapped_TT.df.Filter(f"SVfit_valid>0 && OS_Iso && {channel} && {cat} && SVfit_m > 0")#.Filter(f"!({tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} && {bb_mass}< {mbb_max} && {bb_mass} > {mbb_min})")
            dfWrapped_TT = FilterForbJets(cat,dfWrapped_TT)
            df_TT_new = dfWrapped_TT.df
            if reduce_size: df_TT_new = df_TT_new.Range(100000)
            # INITIALLY
            n_in_DY = df_DY_new.Count().GetValue()
            print(f"inizialmente {n_in_DY} eventi di DY")
            n_in_TT = df_TT_new.Count().GetValue()
            print(f"inizialmente {n_in_DY} eventi di DY e {n_in_TT} eventi di TT")
            # SQUARE CUT
            if args.calculate_square_intervals:
                # min_tt_TT1, max_tt_TT1 = GetMassesQuantiles(df_TT_new, tt_mass, 0.68)
                # print("including 68% TT")
                # print(f"{tt_mass}> {min_tt_TT1} && {tt_mass} < {max_tt_TT1}")
                min_tt_DY, max_tt_DY = GetMassesQuantiles(df_DY_new, tt_mass, 0.68)
                print("including 90% DY")
                print(f"{tt_mass}> {min_tt_DY} && {tt_mass} < {max_tt_DY}")

                # min_tt_TT2, max_tt_TT2 = GetMassesQuantiles(df_TT_new, tt_mass, 0.90)
                # print("including 90% TT")
                # print(f"{tt_mass}> {min_tt_TT2} && {tt_mass} < {max_tt_TT2}")

            if args.square_params:
                masses = args.square_params.split(",")
                mtt_min, mtt_max, mbb_min, mbb_max = masses[0],masses[1],masses[2],masses[3]
                print("mtt_min, mtt_max, mbb_min, mbb_max")
                print(mtt_min, mtt_max, mbb_min, mbb_max)
            else:
                mtt_min, mtt_max, mbb_min, mbb_max = 75,155,90,150

            n_after_DY_lin = df_DY_new.Filter(f"!({tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} && {bb_mass}< {mbb_max} && {bb_mass} > {mbb_min})").Count().GetValue()#.Filter(f"{tt_mass}> {min_tt_TT1} && {tt_mass} < {max_tt_TT1}").Count().GetValue()
            # percentage_DY_lin =  0 if n_in_DY==0 else n_after_DY_lin/n_in_DY
            percentage_DY_lin =  n_after_DY_lin/n_in_DY if n_in_DY!=0 else 0
            n_after_TT_lin = 0
            percentage_TT_lin = 0
            print(f"percentage_DY_lin = {percentage_DY_lin} ")

            n_after_TT_lin = df_TT_new.Filter(f"!({tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} && {bb_mass}< {mbb_max} && {bb_mass} > {mbb_min})").Count().GetValue()#.Filter(f"{tt_mass}> {min_tt_TT1} && {tt_mass} < {max_tt_TT1}").Count().GetValue()
            print(f"dopo taglio lineare {n_after_DY_lin} eventi di DY e {n_after_TT_lin} eventi di TT")
            percentage_TT_lin =  n_after_TT_lin/n_in_TT if n_in_TT!=0 else 0
            print(f"percentage_TT_lin = {percentage_TT_lin} ")