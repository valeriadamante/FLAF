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
    data_into_list = []
    for linentry in data.split("\n"):
        data_into_list.append(linentry.replace("SC/","CC/"))
    inFiles = Utilities.ListToVector(data_into_list)
    df_initial = ROOT.RDataFrame("Events", inFiles)
    return df_initial

def buildDfWrapped(df_initial,global_cfg_dict,year):
    dfBuilder_init = DataFrameBuilderForHistograms(df_initial,global_cfg_dict, f"Run2_{year}")
    dfWrapped = PrepareDfForHistograms(dfBuilder_init)
    particleNet_mass = 'particleNet_mass' if 'SelectedFatJet_particleNet_mass_boosted' in dfWrapped.df.GetColumnNames() else 'particleNetLegacy_mass'
    return dfWrapped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=False, type=str, default='2018')
    parser.add_argument('--cat', required=False, type=str, default='res2b_cat3')
    parser.add_argument('--channels', required=False, type=str, default='tauTau')
    parser.add_argument('--bckgtype', required=False, type=str, default='DY')

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


    BckgFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/{args.bckgtype}Samples_{args.year}.txt"
    global_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/HH_bbtautau/global.yaml'
    global_cfg_dict = {}
    with open(global_cfg_file, 'r') as f:
        global_cfg_dict = yaml.safe_load(f)
    global_cfg_dict['channels_to_consider']=args.channels.split(',')

    hist_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/plot/histograms.yaml'
    hist_cfg_dict = {}
    with open(hist_cfg_file, 'r') as f:
        hist_cfg_dict = yaml.safe_load(f)


    df_bckg = getDataFramesFromFile(BckgFiles)
    dfWrapped_bckg = buildDfWrapped(df_bckg,global_cfg_dict,args.year)
    # quantile_bckg = 0.9
    wantSequential=False
    tt_mass = "tautau_m_vis"
    for cat in  args.cat.split(','):
        print(cat)
        bb_mass = "bb_m_vis" if cat != 'boosted_cat3' else "bb_m_vis_softdrop"
        # print(bb_mass)
        y_bins = hist_cfg_dict[bb_mass]['x_rebin']['other']
        x_bins = hist_cfg_dict[tt_mass]['x_rebin']['other']
        # print(x_bins)
        for channel in global_cfg_dict['channels_to_consider']:
            print(channel,cat)
            sel_str = f" OS_Iso && {channel} && {cat}"
            dfWrapped_bckg.df = dfWrapped_bckg.df.Filter(sel_str)
            dfWrapped_bckg = FilterForbJets(cat,dfWrapped_bckg)
            df_bckg_new = dfWrapped_bckg.df



            not_in_DYZone = f"{tt_mass} < 100 && {tt_mass} > 80"
            if args.bckgtype == 'TT':
                print("adding not in dy zone")
                df_bckg_new = df_bckg_new.Filter(f"!({not_in_DYZone})")
            df_bckg_new = df_bckg_new.Range(100000)
            print(df_bckg_new.Count().GetValue())
            # SQUARE CUT
            if args.calculate_square_intervals:
                # print(f"{tt_mass}> {mtt_min} && {tt_mass} < {mtt_max}")
                min_tt_bckg1, max_tt_bckg1 = GetMassesQuantiles(df_bckg_new, tt_mass, 0.68)
                print(f"including 68% bckg : {tt_mass}> {min_tt_bckg1} && {tt_mass} < {max_tt_bckg1}")
                min_tt_bckg2, max_tt_bckg2 = GetMassesQuantiles(df_bckg_new, tt_mass, 0.90)
                print(f"including 90% bckg : {tt_mass}> {min_tt_bckg2} && {tt_mass} < {max_tt_bckg2}")
                print()

            if args.square_params:
                masses = args.square_params.split(",")
                mtt_min, mtt_max = masses[0],masses[1]
            elif args.calculate_square_intervals:
                mtt_min, mtt_max = min_tt_bckg1,max_tt_bckg1
                # mtt_min, mtt_max = min_tt_bckg2,max_tt_bckg2
            else:
                mtt_min, mtt_max = 80,100

            print(f"{tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} ")
            n_after_bckg_lin = 0
            percentage_bckg_lin = 0
            ssqrtb_lin = 0

            n_before_bckg_lin = df_bckg_new.Count().GetValue()

            n_after_bckg_lin = df_bckg_new.Filter(f"{tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} ").Count().GetValue()
            print(f"{tt_mass}< {mtt_max} && {tt_mass} > {mtt_min} ")
            percentage_bckg_lin =  n_after_bckg_lin/n_before_bckg_lin if n_before_bckg_lin!=0 else 0
            print(f"percentage_bckg_lin = {percentage_bckg_lin} ")