import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import correlations as corrs
from create_dictionary import Create_dictionary
from target_compensation import target_compensation
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda.additions import rank_preferences
from visualizations import *
from normalizations import targeted_minmax_normalization

from topsis import TOPSIS, TOPSIS_TARGETED
from pyrepo_mcda.mcda_methods import SPOTIS, VIKOR


def main():
    #start
    path = 'DATASET/'

    #------------------------------------------------------------------------------------------------
    # Additional loop for multi-target TOPSIS for subsequent years
    years = ['2016', '2017', '2018', '2019', '2020']
    flag_C10 = False
    rank_results = pd.DataFrame()
    
    for year in years:
        data = pd.read_csv(os.path.join(path, 'data_' + year + '.csv'), index_col = 'Country')
        data_target = pd.read_csv(os.path.join(path, 'target.csv'), index_col = 'Country')

        # :-1 without C10 columns
        if flag_C10 == False:
            df_data = data.iloc[:len(data) - 1, :-1]
            types = data.iloc[len(data) - 1, :-1].to_numpy()
            data_target = data_target.iloc[:, :-1]
            idx_nan = []
            lastC = 'C9'
        else:
            df_data = data.iloc[:len(data) - 1, :]
            types = data.iloc[len(data) - 1, :].to_numpy()
            idx_nan = list(np.where(data_target['C10'].isnull())[0])
            lastC = 'C10'

        list_alt_names = []
        for i in range(0, len(df_data)):
            if i not in idx_nan:
                list_alt_names.append(r'$A_{' + str(i + 1) + '}$')


        if flag_C10 == True:
            df_data = df_data.reset_index()
            df_data = df_data.drop(idx_nan)
            df_data = df_data.set_index('Country')

            data_target = data_target.reset_index()
            data_target = data_target.drop(idx_nan)
            data_target = data_target.set_index('Country')

        matrix = df_data.to_numpy()
        matrix_target_raw = data_target.to_numpy()
        matrix_target = target_compensation(matrix, matrix_target_raw, types)

        weights = mcda_weights.critic_weighting(matrix)

        #TOPSIS targeted
        topsis_targeted = TOPSIS_TARGETED(normalization_method = targeted_minmax_normalization)
        pref = topsis_targeted(matrix, matrix_target, weights, types)
        rank = rank_preferences(pref, reverse = True)
        rank_results[year] = rank

    rank_results['Ai'] = list(list_alt_names)
    rank_results = rank_results.set_index('Ai')
    rank_results.to_csv('./results/MT_TOPSIS_rank_years_' + lastC + '.csv')

    #=====================================================================================================

    # Main part of research: assessment for 2020
    year = '2020'
    data = pd.read_csv(os.path.join(path, 'data_' + year + '.csv'), index_col = 'Country')
    data_target = pd.read_csv(os.path.join(path, 'target.csv'), index_col = 'Country')
    # is criterion C10 taken into account?
    flag_C10 = False

    # :-1 without C10 columns
    if flag_C10 == False:
        df_data = data.iloc[:len(data) - 1, :-1]
        types = data.iloc[len(data) - 1, :-1].to_numpy()
        data_target = data_target.iloc[:, :-1]
        idx_nan = []
        lastC = 'C9'
    else:
        df_data = data.iloc[:len(data) - 1, :]
        types = data.iloc[len(data) - 1, :].to_numpy()
        idx_nan = list(np.where(data_target['C10'].isnull())[0])
        lastC = 'C10'


    lan = pd.read_csv('./DATASET/symbols.csv', index_col='Country')
    list_alt_names = lan['Symbol']

    if flag_C10 == True:
        df_data = df_data.reset_index()
        df_data = df_data.drop(idx_nan)
        df_data = df_data.set_index('Country')

        data_target = data_target.reset_index()
        data_target = data_target.drop(idx_nan)
        data_target = data_target.set_index('Country')


    matrix = df_data.to_numpy()
    matrix_target_raw = data_target.to_numpy()
    matrix_target = target_compensation(matrix, matrix_target_raw, types)

    # criteria weights
    weights = mcda_weights.critic_weighting(matrix)

    pref_results = pd.DataFrame()
    pref_results['Ai'] = list(list_alt_names)

    rank_results = pd.DataFrame()
    rank_results['Ai'] = list(list_alt_names)
    
    #TOPSIS targeted
    topsis_targeted = TOPSIS_TARGETED(normalization_method = targeted_minmax_normalization)
    pref = topsis_targeted(matrix, matrix_target, weights, types)
    rank = rank_preferences(pref, reverse = True)
    pref_results['MT-TOPSIS'] = pref
    rank_results['MT-TOPSIS'] = rank

    # classic MCDA methods
    # TOPSIS
    topsis = TOPSIS(normalization_method = norms.minmax_normalization)
    pref = topsis(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    pref_results['TOPSIS'] = pref
    rank_results['TOPSIS'] = rank

    #SPOTIS
    bounds_min = np.amin(matrix, axis = 0)
    bounds_max = np.amax(matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))
    spotis = SPOTIS()
    pref = spotis(matrix, weights, types, bounds)
    rank = rank_preferences(pref, reverse = False)
    pref_results['SPOTIS'] = pref
    rank_results['SPOTIS'] = rank

    #VIKOR
    vikor = VIKOR(normalization_method = norms.minmax_normalization)
    pref = vikor(matrix, weights, types)
    rank = rank_preferences(pref, reverse = False)
    pref_results['VIKOR'] = pref
    rank_results['VIKOR'] = rank

    pref_results = pref_results.set_index('Ai')
    pref_results.to_csv('./results/final_preferences_' + lastC + '_' + year + '.csv')

    rank_results = rank_results.set_index('Ai')
    rank_results.to_csv('./results/final_rankings_' + lastC + '_' + year + '.csv')
    
    
    #=============================================================================
    # comparison of multi-target TOPSIS and TOPSIS
    # scatter plot
    
    names = ['TOPSIS']
    model_compare = []
    for name in names:
        model_compare.append([name, 'MT-' + name])

    data = copy.deepcopy(rank_results)
    sns.set_style("darkgrid")
    plot_scatter(data = data, model_compare = model_compare, lastC = lastC, year = year)
    #=============================================================================
    
    
    #bar chart
    df_plot = copy.deepcopy(rank_results)
    colors = ['#b41f77', '#1f77b4', '#77b41f', '#FFBF00']
    plot_barplot(df_plot, df_plot.index, colors = colors, lastC = lastC, year = year)

    
    # ======================================================================================
    # Correlations
    data = copy.deepcopy(rank_results)
    method_types = list(data.columns)

    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_pearson = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(corrs.weighted_spearman(data[i], data[j]))
            dict_new_heatmap_pearson[j].append(corrs.pearson_coeff(data[i], data[j]))
        

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_pearson = pd.DataFrame(dict_new_heatmap_pearson, index = method_types[::-1])
    df_new_heatmap_pearson.columns = method_types


    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$', lastC, year = year)

    # correlation matrix with Pearson coefficient
    draw_heatmap(df_new_heatmap_pearson, r'$Pearson$', lastC, year = year)

    df_new_heatmap_pearson.to_csv('./results/df_new_heatmap_pearson.csv')
    


if __name__ == "__main__":
    main()
