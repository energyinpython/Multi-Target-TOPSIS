import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# bar chart for basic version - used here
def plot_barplot(df_plot, list_alt_names, colors, lastC, year):
    step = 2
    list_rank = np.arange(1, len(df_plot) + 1, step)

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (11,5), color = colors)
    ax.set_xlabel('Alternatives, year: ' + year, fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(list_alt_names, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=5, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'MCDA methods', fontsize = 12)

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./output/' + 'bar_chart_' + lastC + '_' + year + '.pdf')
    plt.savefig('./output/' + 'bar_chart_' + lastC + '_' + year + '.png')
    plt.show()


# bar chart for sensitivity analysis in basic version - not used here--------------
def plot_barplot_sensitivity(df_plot, list_alt_names, method_name, criterion_name):
    step = 1
    list_rank = np.arange(1, len(df_plot) + 1, step)

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (11,5))
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(list_alt_names, rotation = 'horizontal')
    ax.tick_params(axis='both', labelsize=12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)
    ax.set_title(method_name + ', modification of ' + criterion_name + ' weights')

    plt.legend(bbox_to_anchor=(1.0, 0.82, 0.3, 0.2), loc='upper left', title = 'Weights change', edgecolor = 'black', fontsize = 12)

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./output/sensitivity_analysis_results/' + 'sens_' + 'hist_' + method_name + '_' + criterion_name + '.png')
    plt.show()


# plot line chart for sensitivity analysis in basic version - not used here---------------
def plot_lineplot_sensitivity(data_sens, list_alt_names, method_name, criterion_name):
    #plt.figure(figsize = (7, 4))
    step = 1
    list_rank = np.arange(1, len(data_sens) + 1, step)
    for j in range(data_sens.shape[0]):
        
        plt.plot(data_sens.iloc[j, :], linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(list_alt_names[j], (x_max, data_sens.iloc[j, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel("Weight modification", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.yticks(list_rank, fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.title(method_name + ', modification of ' + criterion_name + ' weights')
    plt.grid(linestyle = ':')
    plt.tight_layout()
    plt.savefig('./output/sensitivity_analysis_results/' + 'sens_' + 'lineplot_' + method_name + '_' + criterion_name + '.png')
    plt.show()


# heat maps with correlations for basic version - used here
def draw_heatmap(df_new_heatmap, title, lastC, year):
    plt.figure(figsize = (8,5))
    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="PuBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title + ', year: ' + year)
    plt.tight_layout()
    plt.savefig('./output/' + 'correlations_' + title + '_' + lastC + '_' + year + '.pdf')
    plt.savefig('./output/' + 'correlations_' + title + '_' + lastC + '_' + year + '.png')
    plt.show()


# radar plot for basic version - not used here---------------------------
def plot_radar(data):
    fig=plt.figure()
    ax = fig.add_subplot(111, polar = True)

    for col in list(data.columns):
        labels=np.array(list(data.index))
        stats = data.loc[labels, col].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(data.index)
        lista.append(data.index[0])
        labels=np.array(lista)

        ax.plot(angles, stats, '-D', linewidth=1)
        ax.fill_between(angles, stats, alpha=0.05)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    plt.tight_layout()
    plt.savefig('./output/' + 'radar_chart' + '.png')
    plt.show()


# simulations

# plot box chart of results obtained in simulations - not used here-----------------
def plot_boxplot(data, x, y, xtitle, ytitle, title, filename, flag_rotation = True):
    plt.figure(figsize = (9,5))
    ax = sns.boxplot(x = x, y = y, hue = 'Weight change', palette = 'Blues', data = data)
    if flag_rotation:
        ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)
    else:
        ax.tick_params(axis = 'x', labelsize = 12)
    ax.set_xlabel(xtitle, fontsize = 12)
    ax.set_ylabel(ytitle, fontsize = 12)
    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Weights change', fontsize = 12)

    plt.tight_layout()
    plt.savefig('./output/' + filename + '.png')
    plt.show()



# bar chart for simulations - not used here-----------------
def plot_barplot_simulations(df_plot, xtitle, ytitle, title, filename, wider = False, flag_rotation = True):
    
    ax = df_plot.plot(kind='bar', width = 0.6, stacked=False, edgecolor = 'black', figsize = (9,5), colormap = 'Blues')
    ax.set_xlabel(xtitle, fontsize = 12)
    ax.set_ylabel(ytitle, fontsize = 12)

    if flag_rotation == True:
        ax.set_xticklabels(df_plot.index, rotation = 90)
    else:
        ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    #ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)
    ax.tick_params(axis='both', labelsize=12)
    y_ticks = ax.yaxis.get_major_ticks()

    if wider:
        x_offset = -0.115
    else:
        x_offset = -0.075
    y_offset = 0.04

    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.1f}".format(b.y1 + b.y0)        
        ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize = 8)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Weights change', fontsize = 12)


    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./output/' + filename + '.png')
    plt.show()


# bar chart for simulations dla 1 metody - used here
def plot_barplot_sensitivity_coeff(df_plot, xtitle, ytitle, filename, colors, lastC, year):
    
    ax = df_plot.plot(kind='bar', width = 0.6, stacked=False, edgecolor = 'black', figsize = (9,5), color = colors)
    ax.set_xlabel(xtitle, fontsize = 12)
    ax.set_ylabel(ytitle, fontsize = 12)
    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    #ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)
    ax.tick_params(axis='both', labelsize=12)
    y_ticks = ax.yaxis.get_major_ticks()

    
    x_offset = -0.125
    y_offset = 0.01

    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.2f}".format(b.y1 + b.y0)        
        ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize = 8)

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=2, mode="expand", borderaxespad=0., edgecolor = 'black', title = 'Weight modification', fontsize = 12)
    plt.tight_layout()
    plt.savefig('./output/' + filename + 'sens_coeff_' + lastC + '_' + year + '.pdf')
    plt.show()


# bar chart for simulations dla kilku metod - used here
def plot_barplot_sensitivity_coeff3(df_plot, xtitle, ytitle, title, filename, lastC, year):
    crit_list = list(df_plot.columns)
    
    new_crit_list = []
    for el in crit_list:
        for n in range(df_plot.shape[0]):
            new_crit_list.append(el)
    #print(new_crit_list)
    ax = df_plot.plot(kind='bar', width = 0.9, stacked=False, edgecolor = 'black', figsize = (9,5), legend = None)
    ax.set_xlabel(xtitle, fontsize = 12)
    ax.set_ylabel(ytitle, fontsize = 12)
    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    #ax.tick_params(axis = 'x', labelsize = 12, rotation = 90)
    ax.tick_params(axis='both', labelsize=12)
    y_ticks = ax.yaxis.get_major_ticks()

    
    x_offset = -0.04
    y_offset = 0.01

    counter = 0
    for p in ax.patches:
        b = p.get_bbox()
        
        #val = "{:.2f}".format(b.y1 + b.y0) val zamiast 1 wartosci ax.annoatte bylo jako wartosc y        
        ax.annotate(new_crit_list[counter], ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
        counter += 1

    ax.grid(True, linestyle = ':')
    ax.set_axisbelow(True)
    plt.title(title, fontsize = 12)
    plt.tight_layout()
    plt.savefig('./output/' + filename + 'sens_coeff3_' + lastC + '_' + year + '.pdf')
    plt.show()


# used here
def plot_scatter(data, model_compare, lastC, year):
    list_rank = np.arange(1, len(data) + 1, 2)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        # print('We compare: ', el[0], ' + ', el[1])
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(5, 5))
        #fig, ax = plt.subplots()
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 12)
        ax.set_ylabel(el[1], fontsize = 12)
        #ax.set_title(el[0], fontsize = 12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        y_ticks = ax.yaxis.get_major_ticks()
        #y_ticks[0].label1.set_visible(False)

        x_ticks = ax.xaxis.get_major_ticks()
        #x_ticks[0].label1.set_visible(False)
        ax.set_xlim(-1.5, len(data) + 1)
        ax.set_ylim(0, len(data) + 2)

        #ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, linestyle = ':')
        ax.set_axisbelow(True)
    
        plt.title('Year: ' + year)
        plt.tight_layout()
        plt.savefig('./output/scatter_' + el[0] + '_' + lastC + '_' + year + '.pdf')
        plt.savefig('./output/scatter_' + el[0] + '_' + lastC + '_' + year + '.png')
        plt.show()

