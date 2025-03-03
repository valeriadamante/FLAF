import ROOT
import array
import sys
import os

import mplhep as hep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hep.style.use("CMS")

if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

# import Common.Utilities as Utilities
# from Analysis.HistHelper import *
from Analysis.hh_bbtautau import *

period_dict = {
    "Run2_2018" : "59.83",
    "Run2_2017" : "41.48",
    "Run2_2016" : "16.8",
    "Run2_2016_HIPM" : "19.5",
    "all": "138.8 fb^{-1} (13 TeV)",

}
# "CMS_QCD_norm_{year}{scale}",
unc_names = [ "CMS_btag_HF{scale}", "CMS_btag_LF{scale}", "CMS_btag_cferr1{scale}", "CMS_btag_cferr2{scale}", "CMS_eff_t_id_syst_alleras{scale}", "CMS_eff_t_id_syst_highpT_bin1{scale}", "CMS_eff_t_id_syst_highpT_bin2{scale}", "CMS_eff_t_id_syst_highpT_extrap{scale}", "CMS_scale_j_Abs{scale}", "CMS_scale_j_BBEC1{scale}", "CMS_scale_j_EC2{scale}", "CMS_scale_j_FlavQCD{scale}", "CMS_scale_j_HF{scale}", "CMS_scale_j_RelBal{scale}", "CMS_btag_hfstats1_{year}{scale}", "CMS_btag_hfstats2_{year}{scale}", "CMS_btag_lfstats1_{year}{scale}", "CMS_btag_lfstats2_{year}{scale}", "CMS_eff_j_PUJET_id_{year}{scale}", "CMS_eff_m_id_iso_{year}{scale}", "CMS_eff_m_id_{year}{scale}", "CMS_eff_e_{year}{scale}", "CMS_eff_t_id_stat1_DM0_{year}{scale}", "CMS_eff_t_id_stat1_DM10_{year}{scale}", "CMS_eff_t_id_stat1_DM11_{year}{scale}", "CMS_eff_t_id_stat1_DM1_{year}{scale}", "CMS_eff_t_id_stat2_DM0_{year}{scale}", "CMS_eff_t_id_stat2_DM10_{year}{scale}", "CMS_eff_t_id_stat2_DM11_{year}{scale}", "CMS_eff_t_id_stat2_DM1_{year}{scale}", "CMS_eff_t_id_syst_{year}{scale}", "CMS_eff_t_id_syst_{year}_DM0{scale}", "CMS_eff_t_id_syst_{year}_DM10{scale}", "CMS_eff_t_id_syst_{year}_DM11{scale}", "CMS_eff_t_id_syst_{year}_DM1{scale}", "CMS_eff_t_id_etauFR_barrel_{year}{scale}", "CMS_eff_t_id_etauFR_endcaps_{year}{scale}", "CMS_eff_t_id_mutauFR_eta0p4to0p8_{year}{scale}", "CMS_eff_t_id_mutauFR_eta0p8to1p2_{year}{scale}", "CMS_eff_t_id_mutauFR_eta1p2to1p7_{year}{scale}", "CMS_eff_t_id_mutauFR_etaGt1p7_{year}{scale}", "CMS_eff_t_id_mutauFR_etaLt0p4_{year}{scale}", "CMS_eff_t_id_stat_highpT_bin1_{year}{scale}", "CMS_eff_t_id_stat_highpT_bin2_{year}{scale}", "CMS_eff_m_id_reco_{year}{scale}", "CMS_eff_m_id_highpt_{year}{scale}", "CMS_eff_m_id_reco_highpt_{year}{scale}", "CMS_l1_prefiring_{year}{scale}", "CMS_pu_lumi_MC_{year}{scale}", "CMS_scale_t_DM0_{year}{scale}","CMS_scale_t_DM1_{year}{scale}", "CMS_scale_t_3prong_{year}{scale}", "CMS_scale_t_eFake_DM0_{year}{scale}", "CMS_scale_t_eFake_DM1_{year}{scale}", "CMS_scale_t_muFake_{year}{scale}", "CMS_res_j_{year}{scale}", "CMS_scale_j_Abs_{year}{scale}", "CMS_scale_j_BBEC1_{year}{scale}", "CMS_scale_j_EC2_{year}{scale}", "CMS_scale_j_HF_{year}{scale}", "CMS_scale_j_RelSample_{year}{scale}", "CMS_scale_qcd_{year}{scale}",  "CMS_norm_qcd_{year}{scale}", "CMS_pnet_{year}{scale}", "CMS_bbtt_trig_MET_{year}{scale}", "CMS_bbtt_trig_singleTau_{year}{scale}", "CMS_bbtt_trig_ele_{year}{scale}", "CMS_bbtt_trig_mu_{year}{scale}", "CMS_bbtt_trig_tau_{year}{scale}"]
def RebinHisto(hist_initial, new_binning):
    new_binning_array = array.array('d', new_binning)
    new_hist = hist_initial.Rebin(len(new_binning)-1, "new_hist", new_binning_array)
    return new_hist
processes= ["QCD","W","TT","DY","EWK_ZTo2L","EWK_ZTo2Nu","EWK_WplusToLNu","EWK_WminusToLNu","ggHToZZTo2L2Q","GluGluHToTauTau_M125","VBFHToTauTau_M125","GluGluHToWWTo2L2Nu_M125","VBFHToWWTo2L2Nu_M125","WplusHToTauTau_M125","WminusHToTauTau_M125","ZHToTauTau_M125","ZH_Hbb_Zll","ZH_Hbb_Zqq","HWplusJ_HToWW_M125","HWminusJ_HToWW_M125","HZJ_HToWW_M125","WW","WZ","ZZ","WWW_4F","WWZ_4F","WZZ","ZZZ","ST_t-channel_antitop_4f_InclusiveDecays","ST_t-channel_top_4f_InclusiveDecays","ST_tW_antitop_5f_InclusiveDecays","ST_tW_top_5f_InclusiveDecays","TTWW","TTWH","TTZH","TTZZ","TTWZ","TTTT","TTTW","TTTJ","TTGG","TTGJets","TT4b","ttHTobb_M125","ttHToTauTau_M125"]
signals=["GluGluToRadionToHHTo2B2Tau_M-250","GluGluToRadionToHHTo2B2Tau_M-260","GluGluToRadionToHHTo2B2Tau_M-270","GluGluToRadionToHHTo2B2Tau_M-280","GluGluToRadionToHHTo2B2Tau_M-300","GluGluToRadionToHHTo2B2Tau_M-320","GluGluToRadionToHHTo2B2Tau_M-350","GluGluToRadionToHHTo2B2Tau_M-400","GluGluToRadionToHHTo2B2Tau_M-450","GluGluToRadionToHHTo2B2Tau_M-500","GluGluToRadionToHHTo2B2Tau_M-550","GluGluToRadionToHHTo2B2Tau_M-600","GluGluToRadionToHHTo2B2Tau_M-650","GluGluToRadionToHHTo2B2Tau_M-700","GluGluToRadionToHHTo2B2Tau_M-750","GluGluToRadionToHHTo2B2Tau_M-800","GluGluToRadionToHHTo2B2Tau_M-850","GluGluToRadionToHHTo2B2Tau_M-900","GluGluToRadionToHHTo2B2Tau_M-1000","GluGluToRadionToHHTo2B2Tau_M-1250","GluGluToRadionToHHTo2B2Tau_M-1500","GluGluToRadionToHHTo2B2Tau_M-1750","GluGluToRadionToHHTo2B2Tau_M-2000","GluGluToRadionToHHTo2B2Tau_M-2500","GluGluToRadionToHHTo2B2Tau_M-3000","GluGluToBulkGravitonToHHTo2B2Tau_M-250","GluGluToBulkGravitonToHHTo2B2Tau_M-260","GluGluToBulkGravitonToHHTo2B2Tau_M-270","GluGluToBulkGravitonToHHTo2B2Tau_M-280","GluGluToBulkGravitonToHHTo2B2Tau_M-300","GluGluToBulkGravitonToHHTo2B2Tau_M-320","GluGluToBulkGravitonToHHTo2B2Tau_M-350","GluGluToBulkGravitonToHHTo2B2Tau_M-400","GluGluToBulkGravitonToHHTo2B2Tau_M-450","GluGluToBulkGravitonToHHTo2B2Tau_M-500","GluGluToBulkGravitonToHHTo2B2Tau_M-550","GluGluToBulkGravitonToHHTo2B2Tau_M-600","GluGluToBulkGravitonToHHTo2B2Tau_M-650","GluGluToBulkGravitonToHHTo2B2Tau_M-700","GluGluToBulkGravitonToHHTo2B2Tau_M-750","GluGluToBulkGravitonToHHTo2B2Tau_M-800","GluGluToBulkGravitonToHHTo2B2Tau_M-850","GluGluToBulkGravitonToHHTo2B2Tau_M-900","GluGluToBulkGravitonToHHTo2B2Tau_M-1000","GluGluToBulkGravitonToHHTo2B2Tau_M-1250","GluGluToBulkGravitonToHHTo2B2Tau_M-1500","GluGluToBulkGravitonToHHTo2B2Tau_M-1750","GluGluToBulkGravitonToHHTo2B2Tau_M-2000","GluGluToBulkGravitonToHHTo2B2Tau_M-2500","GluGluToBulkGravitonToHHTo2B2Tau_M-3000",]


years = ["2016", "2016_HIPM", "2016", "2018"]
histograms_final = []
histograms_initials = {
    "2016":"/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_massCut/SC/SR/Run2_2016/merged/kinFit_m/kinFit_m.root",
    "2016_HIPM":"/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_massCut/SC/SR/Run2_2016_HIPM/merged/kinFit_m/kinFit_m.root",
    "2017":"/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_massCut/SC/SR/Run2_2017/merged/kinFit_m/kinFit_m.root",
    "2018":"/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_massCut/SC/SR/Run2_2018/merged/kinFit_m/kinFit_m.root"
}


# Funzione per estrarre dati e incertezze da un TH1
def extract_hist_data(hist):
    bins = hist.GetNbinsX()
    edges = np.array([hist.GetBinLowEdge(i + 1) for i in range(bins + 1)])
    values = np.array([hist.GetBinContent(i + 1) for i in range(bins)])
    errors = np.array([hist.GetBinError(i + 1) for i in range(bins)])
    return edges, values, errors



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=False, type=str, default='2018')
    parser.add_argument('--unc', required=False, type=str, default='CMS_eff_j_PUJET_id')
    parser.add_argument('--cat', required=False, type=str, default='res2b_cat3_masswindow')
    parser.add_argument('--channel', required=False, type=str, default='tauTau')
    # parser.add_argument('--wantPlots', required=False, type=bool, default=False)
    # parser.add_argument('--resonance', required=False, type=str, default="Radion")
    args = parser.parse_args()

    outdir = f"output/plotuncs/{args.year}/{args.channel}/{args.cat}"
    os.makedirs(outdir, exist_ok = True)
    # histo_initial = histograms_initials[args.year]
    inFile = ROOT.TFile.Open(f"/eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/New_massCut/SC/SR/Run2_{args.year}/merged/kinFit_m/kinFit_m.root", "READ")
    dir_0 = inFile.Get(args.channel)
    keys_categories = [str(key.GetName()) for key in dir_0.GetListOfKeys()]
    # print(keys_categories)
    dir_1 = dir_0.Get(args.cat)
    keys_cat = [str(key.GetName()) for key in dir_1.GetListOfKeys()]
    # print(keys_cat)
    hist_central = dir_1.Get(processes[0])
    hist_central.SetDirectory(0)
    for process in processes[1:]:
        # print(process)
        hist_to_add_central = dir_1.Get(process)
        hist_to_add_central.SetDirectory(0)
        hist_central.Add(hist_to_add_central)
    for unc in unc_names:
        # print(unc)
        full_name_unc_up = unc.format(scale='Up', year=args.year)
        full_name_unc_down = unc.format(scale='Down', year=args.year)
        print(f"{processes[0]}_{full_name_unc_up}")
        hist_up = dir_1.Get(f"{processes[0]}_{full_name_unc_up}")
        hist_up.SetDirectory(0)
        hist_down = dir_1.Get(f"{processes[0]}_{full_name_unc_down}")
        hist_down.SetDirectory(0)
        for process in processes[1:]:
            hist_to_add_up = dir_1.Get(f"{process}_{full_name_unc_up}")
            hist_to_add_up.SetDirectory(0)
            hist_up.Add(hist_to_add_up)
            hist_to_add_down = dir_1.Get(f"{process}_{full_name_unc_down}")
            hist_to_add_down.SetDirectory(0)
            hist_down.Add(hist_to_add_down)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(24, 18), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        plt.ylabel('Events', fontsize=35)
        plt.xlabel('$M_{HH}$(GeV)', fontsize=35)
        alpha=0
        linewidth=0.8
        new_bins = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000] #, 1250, 1500, 1750, 2000, 2500, 3000 ] #[250, 300, 350, 400, 450, 500, 600, 800, 1000]
        histogram_central = RebinHisto(hist_central, new_bins)
        histogram_up = RebinHisto(hist_up, new_bins)
        histogram_down = RebinHisto(hist_down, new_bins)
        #### ottieni i ratio e come fillare gli hists ####
        edges, values_central, errors_central = extract_hist_data(histogram_central)
        _, values_up, errors_up = extract_hist_data(histogram_up)
        _, values_down, errors_down = extract_hist_data(histogram_down)
        ratio_up = np.divide(
        values_up, values_central, out=np.zeros_like(values_central, dtype=float), where=(values_central != 0)
        )
        ratio_down = np.divide(
            values_down, values_central, out=np.zeros_like(values_central, dtype=float), where=(values_central != 0)
        )
        ratio_up_err = np.zeros_like(values_central, dtype=float)
        ratio_down_err = np.zeros_like(values_central, dtype=float)

        non_zero_mask = (values_central > 0) & (values_up > 0)
        ratio_up_err[non_zero_mask] = ratio_up[non_zero_mask] * np.sqrt(
            (errors_up[non_zero_mask] / values_up[non_zero_mask]) ** 2
            + (errors_central[non_zero_mask] / values_central[non_zero_mask]) ** 2
        )

        non_zero_mask = (values_central > 0) & (values_down > 0)
        ratio_down_err[non_zero_mask] = ratio_down[non_zero_mask] * np.sqrt(
            (errors_down[non_zero_mask] / values_down[non_zero_mask]) ** 2
            + (errors_central[non_zero_mask] / values_central[non_zero_mask]) ** 2
        )

        hep.histplot(values_central, edges, label="Central", histtype="step", color="black", linewidth=3,ax=ax1)
        # hep.histplot(values_central, edges, label="Central", histtype="step", color="cornflowerblue", linewidth=5,ax=ax1)
        hep.histplot(values_up, edges, label="Up", histtype="step", color="blue", linewidth=2,ax=ax1)
        # hep.histplot(values_up, edges, label="Up", histtype="step", color="salmon", linewidth=3,ax=ax1)
        # hep.histplot(values_down, edges, label="Down", histtype="step", color="green", linewidth=3,ax=ax1)
        hep.histplot(values_down, edges, label="Down", histtype="step", color="red", linewidth=2,ax=ax1)


        # ax1.fill_between(
        #     0.5 * (edges[1:] + edges[:-1]),
        #     values_central - errors_central,
        #     values_central + errors_central,
        #     step="mid",
        #     color="gray",
        #     alpha=0.3,
        #     label="unc",
        # )
        # Etichette e legenda
        ax1.set_ylabel("Entries")
        ax1.legend(loc="best")
        ax1.grid(True)
        hep.cms.label("Preliminary", lumi=period_dict[f"Run2_{args.year}"], year=args.year, fontsize=35, ax=ax1)
        ax1.legend(fontsize=35)

        # === Pannello della ratio (sotto) ===
        hep.histplot(ratio_up, edges, label="Up/Nominal", histtype="step", color="blue",linewidth=3, ax=ax2)
        hep.histplot(ratio_down, edges, label="Down/Nominal", histtype="step", color="red", linewidth=3,ax=ax2)

        # # Aggiungi bande di incertezza sulle ratio
        # ax2.fill_between(
        #     0.5 * (edges[1:] + edges[:-1]),
        #     ratio_up - ratio_up_err,
        #     ratio_up + ratio_up_err,
        #     step="mid",
        #     color="blue",
        #     alpha=0.3,
        #     label="Up (unc.)",
        # )
        # ax2.fill_between(
        #     0.5 * (edges[1:] + edges[:-1]),
        #     ratio_down - ratio_down_err,
        #     ratio_down + ratio_down_err,
        #     step="mid",
        #     color="red",
        #     alpha=0.3,
        #     label="Down (unc.)",
        # )
        maxv = max(max(ratio_down), max(ratio_up))
        minv = min(min(ratio_down), min(ratio_up))
        # print(maxv,minv)
        range_up = maxv-1
        range_down = 1-minv
        # print(range_up)
        # print(range_down)
        symm_range = 1.1*abs(range_up) if abs(range_up) > abs(range_down) else 1.1*abs(range_down)
        # print(symm_range)
        # Etichette della ratio
        ax2.axhline(1, color="black", linestyle="--", linewidth=2)
        # ax2.set_xlabel("Variable")
        ax2.set_ylabel("Ratio")
        ax2.set_ylim(1-symm_range, 1+symm_range)  # Adatta il range per il rapporto
        # ax2.legend(loc="best")
        ax2.grid(True)




        plt.show()
        plt.gca().set_aspect('auto')
        plt.savefig(f"output/plotuncs/{args.year}/{args.channel}/{args.cat}/{full_name_unc_up}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"output/plotuncs/{args.year}/{args.channel}/{args.cat}/{full_name_unc_up}.png", format="png", bbox_inches="tight")
        print(f"output/plotuncs/{args.year}/{args.channel}/{args.cat}/{full_name_unc_up}.png")