import numpy as np
import matplotlib.pyplot as plt
import test_error
import importlib
import matplotlib
import datasets
import seaborn as sns
import matplotlib.gridspec as gridspec
from mingpt.utils import set_seed
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
def data_to_line(datem, length, label, axis, color = None, style = None):
    m = np.mean(datem, axis = 0)
    std_err = np.std(datem, axis = 0)
    # Confidence interval (95% CI)
    ci = 1.96 * std_err
    line = axis.plot((m,)*length, label = label, color = color, linestyle = style)
    error = axis.fill_between(range(length), m - ci, m + ci,  alpha=0.2,color = color)
    return line, error


def data_to_curve(datem, label, axis, color = None):
    m = np.mean(datem, axis = 0)
    std_err = np.std(datem, axis = 0) / np.sqrt(len(datem))
    # Confidence interval (95% CI)
    ci = 1.96 * std_err
    error = axis.fill_between(range(len(m)), m - ci, m + ci,  alpha=0.2,color=color)
    line = axis.plot(m, label=label,color=color)
    return line, error

@torch.no_grad()
def pos_encode_graph(model_history, datem, config, points = [5,100,199]):
    colors = sns.color_palette()
    # points = [5,100,199]
    seed = -1
    data = datem[seed]
    # Plotting
    # plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(8, 3))  # Adjust the size as needed
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax0 = plt.subplot(gs[0])
    labels = ["uniform", "unigram", "bigram", "trigram", "tetragram", "pentagram"]
    for i in range(1,len(data[1])):
        dats = [data[1][i] for dat in datem]
        data_to_line(dats, len(data[0]), labels[i], ax0, colors[i])
    # ax0.legend()

    # data_to_curve([t[0].numpy() for t in datem], "Test Loss", colors[0])
    # for dat in datem[1:]:
    #     plt.plot(dat[0].numpy(), color = colors[0])
    ax0.plot(data[0].numpy(), color = colors[0], label="transformer")
    label_colors = [colors[3], colors[4], colors[9]]
    for i in range(len(points)):
        color = label_colors[i]
        ax0.axvline(points[i], color = color, linestyle="--")
        label = f't={int((points[i])*config.batch_size / 1000*config.max_iters/(len(model_history)-1))}'
        if points[i] > 180:
            ax0.text(points[i]-40, ax0.get_ylim()[1]*.8+.03 * (i%2), label, fontsize=12, color=color)
        else:
            ax0.text(points[i] + 1, ax0.get_ylim()[1]*.8+.03 * (i%2), label, fontsize=12, color=color)
    ax0.set_xlim(-1, len(datem[-1][2][0])+1)
    # fig.suptitle(f'{config.model_type}: {config.vocab_size} Symbols')
    ax0.set_title(f'Test Loss')
    time = len(datem[-1][0])
    tick_n = 5
    ticks = np.linspace(0, time, tick_n, dtype=int)
    conversion = ((config.max_iters * config.batch_size/1000)/(tick_n-1))
    ax0.set_xticks(ticks, np.round(np.arange(tick_n) * conversion,2))
    # axis.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    
    # ax0.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    ax0.set_xlabel('Training Sequences Seen (Thousands)')
    ax0.set_ylabel('KL-Div Loss')

    pos_dat = test_error.pos_enc(model_history, config)
    # pos = torch.nn.functional.softmax(pos_dat[1]+pos_dat[0], dim=1)
    pos = torch.mean(pos_dat, axis=0)
    if config.n_head > 1:
        pos = torch.mean(pos, axis=-1)
    # pos = pos_dat[0]
    # Create subplots on the right
    gs_right = gridspec.GridSpecFromSubplotSpec(len(points), 1, subplot_spec=gs[1], hspace=0.5)
    axes_right = []
    # axes_right = [plt.subplot(gs_right[i], sharex=axes_right[0] if i else None) for i in range(len(points))]

    temp=0
    for i in range(len(points)):
        if i == 0:
            ax = plt.subplot(gs_right[i])
            ax.set_title(f'Positional Encoding')
        else:
            ax = plt.subplot(gs_right[i], sharex=axes_right[0])
        
        color = label_colors[i]
        label = f't={int((points[i])*config.batch_size / 1000*config.max_iters/(len(model_history)-1))}'
        ax.text(1.01, .5, label, fontsize=12, color=color,transform=ax.transAxes)
        axes_right.append(ax)
        ax.bar(np.arange(len(pos)-temp), pos[temp:,points[i]], width =0.8,edgecolor='none', color = colors[0])
        # set max y value to 1
        # ax.set_ylim(0, 1)
        # ax.plot(np.fft.fft(pos[:,points[i]])[1:])
        ax.yaxis.labelpad = 5
        # ax.set_xlim(-1, len(pos))
        # ax.set_ylabel("weight")
        boundary = 1
        ax.patch.set_edgecolor(color)  # Set the color
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color) 
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)

        ax.patch.set_linewidth(boundary)  # Set the line width
        if i < len(points) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Position")
    plt.tight_layout()
    return ax, gs



# def out_of_distribution(datem, unigrams, bigrams, config):
#     # matplotlib.rcParams.update({'font.size': 20})
#     colors = sns.color_palette()
#     # Plotting
#     plt.figure(figsize=(8, 6))
#     # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey="row")
#     # data_to_curve([t[0].numpy() for t in datem], "Test Loss", colors[0])
#     # for dat in datem[1:]:
#     #     plt.plot(dat[0].numpy(), color = colors[0])
#     datasets = [datem, unigrams, bigrams]
#     names = ["", "Unigram ", "Uniform Stationary "]
#     labels = ["Uniform", "Unigram", "Bigram"]
#     fig.suptitle(f'{config.model_type}: {config.vocab_size} Symbols')
#     for i in range(3):
#         data = datasets[i][-1]
#         # print(data[0])
#         for j in range(1,3):
#             dats = [data[1][j]]
#             data_to_line(dats, len(data[0]), labels[j], axes[i], colors[1+j])
#         axes[i].set_xlim(-1, len(datem[-1][2][0])+1)
#         axes[i].plot(data[0].numpy(), color = colors[0], label="transformer")
#         axes[i].set_xlim(-1, 201)
#         axes[i].set_title(f'Test Loss in {names[i]}Distribution')
#         axes[i].set_xticks(range(0, len(datem[-1][0]) + 1, len(datem[-1][0])//10), range(0,config.max_iters + 1, config.max_iters//10))
#         axes[i].set_xlabel('Iterations')
#         axes[i].set_ylabel('KL-Div Loss')


#         # Enable the y-axis labels and ticks for each subplot
#         axes[i].tick_params(axis='y', which='both', labelleft=True)


#     return fig, axes

@torch.no_grad()
def test_loss(datem, config, axis = None):
    # matplotlib.rcParams.update({'font.size': 20})

    matplotlib.rcParams.update({'font.size': 12})
    colors = sns.color_palette()
    # Plotting
    if axis is None:
        fig = plt.figure(figsize=(5,3))
        axis = fig.add_subplot(111)
    labels = ["Uniform", "Unigram", "Bigram", "Trigram", "Tetragram"]
    styles = ["-", "--", "-.", ":", "-"]
    for i in range(1, config.n + 1):
        data_to_line([datem[-1][1][i]], len(datem[-1][0]), labels[i], axis, colors[i], style = styles[i])
    for data in datem:
        axis.plot(data[0].numpy())#, color = colors[0])
        # axis.plot([data[1][1].numpy()]* len(data[0]), color=colors[i])
    # axis.set_xlim(-1, len(datem[-1][0])+1)
    axis.set_xlim(-1, 201)
    # axis.set_title(f'{config.model_type}: {config.vocab_size} Symbols\n Test Loss in Distribution For 10 Random Seeds')
    # axis.set_title(f'{config.model_type}: {config.vocab_size} Symbols')
    axis.set_title(f"{config.model_type} {config.n_layer} layers:\n test loss on 2-state trigrams")
    # if config.n == 3:
    #     axis.set_title("Attention-only transformer:\ntest loss on 3-state ICL-Tetragrams")
    # axis.legend()
    time = len(datem[-1][0])
    tick_n = 5
    ticks = np.linspace(0, time, tick_n, dtype=int)
    conversion = ((config.max_iters * config.batch_size/1000)/(tick_n-1))
    axis.set_xticks(ticks, np.round(np.arange(tick_n) * conversion,2))
    # axis.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    # axis.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    axis.set_xlabel('Training Sequences Seen (Thousands)')
    axis.set_ylabel('KL-Div Loss on last token')


    # Enable the y-axis labels and ticks for each subplot
    axis.tick_params(axis='y', which='both', labelleft=True)


    return axis

@torch.no_grad()
def out_of_distribution(datem, unigrams, bigrams, config):
    # matplotlib.rcParams.update({'font.size': 20})
    colors = sns.color_palette()
    # Plotting
    fig = plt.figure(figsize=(6,4))
    axis = fig.add_subplot(111)
    datasets = [datem, unigrams, bigrams]
    names = ["Full", "Unigram", "Uniform Stationary"]
    labels = ["Uniform", "Unigram", "Bigram"]
    # data_to_line([datem[-1][1][2]], len(datem[-1][0]), labels[1], axis, colors[4])
    for i in range(3):
        data = datasets[i][-1]
        
        axis.plot(data[0].numpy(), color = colors[i], label=names[i])
        # axis.plot([data[1][1].numpy()]* len(data[0]), color=colors[i])
    axis.set_xlim(-1, len(datem[-1][2][0])+1)
    axis.set_xlim(-1, 201)
    axis.set_title(f'{config.model_type}: {config.vocab_size} Symbols\n Test Loss on Distributions')

    time = len(datem[-1][0])
    tick_n = 5
    ticks = np.linspace(0, time, tick_n, dtype=int)
    conversion = ((config.max_iters * config.batch_size/1000)/(tick_n-1))
    axis.set_xticks(ticks, np.round(np.arange(tick_n) * conversion,2))
    # axis.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    axis.set_xlabel('Training Sequences Seen (Thousands)')
    axis.legend(loc="upper right")
    axis.set_ylabel('KL-Div Loss')


    # Enable the y-axis labels and ticks for each subplot
    axis.tick_params(axis='y', which='both', labelleft=True)


    return fig, axis

@torch.no_grad()
def similarity(datem, config, axis = None):
    colors = sns.color_palette()
    
    if axis is None:
        matplotlib.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(5, 3))
        axis = fig.add_subplot(111)
    labels = ["Uniform", "Unigram", "Bigram", "Trigram", "Tetragram"] + [f"{n}gram" for n in range(5, 10)]

    curves = [curve.numpy() for curve in datem[-1][2]]  # Extract curve data
    x = np.arange(len(curves[0]))

    # Find the index of the lowest curve at each x
    lowest_curve_indices = np.argmin(np.vstack(curves), axis=0)
    y_min, y_max = axis.get_ylim()
    # Plotting the curves
    for i in range(len(curves)):

        # Define the shading regions more precisely
        is_lowest = lowest_curve_indices == i
        start = None
        for j in range(len(x)):
            if is_lowest[j] and start is None:
                start = j  # start of a new shaded area
            elif (not is_lowest[j:].any() or j == len(x)-1) and start is not None:
                end = min(j+1, len(x)-1)
                axis.fill_between(x[start:end], y_min, y_max, color=colors[i], alpha=0.2)
                break
    for i in range(len(curves)):

        axis.plot(x, curves[i], color=colors[i], label=labels[i])
        # # Highlighting the lowest sections
        # is_lowest = lowest_curve_indices == i
        # axis.fill_between(x, curves[i], where=is_lowest, color=colors[i], alpha=0.3)

    # Your existing axis setup code...
    # # axis.set_xlim(-1, len(curves[0])+1)
    # axis.set_xlim(x[0], x[-1])  # Adjusting the x-limits to the logarithmic scale
    axis.set_ylim(0, np.max(curves)) 
    # # axis.set_title(f'{config.model_type} KL-Divergence: {config.vocab_size} Symbols')
    axis.set_title(f'Distance between model predictions\n and candidate strategies\n{config.model_type} {config.n_layer} layers')
    tick_n = 5
    time = len(datem[-1][0])
    ticks = np.linspace(0, time, tick_n, dtype=int)
    conversion = ((config.max_iters * config.batch_size/1000)/(tick_n-1))
    axis.set_xticks(ticks, np.round(np.arange(tick_n) * conversion,2))
    # axis.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    # axis.set_xticks(ticks, ticks * conversion)
    axis.set_xlabel('Training Sequences Seen (Thousands)')
    axis.set_ylabel('KL-Div(Distribution||model)')
    # Setting the x-axis to logarithmic scale
    # axis.set_xscale('log')
    axis.legend(fancybox=True, framealpha=0.5)
    axis.tick_params(axis='y', which='both', labelleft=True)

    return axis

def mixtures(config, num):
    import seaborn as sns
    import training_pipeline
    colors = sns.color_palette()
    colormap = matplotlib.cm.viridis
    # Plotting
    fig = plt.figure(figsize=(5, 4))
    # fig.set_layout_engine("tight") 
    axis = fig.add_subplot(111)
    base = [datasets.doubly_stochastic_3('train', 1, 2, config.block_size+1, num_symbols = config.vocab_size,), datasets.unigram_3('train', 1, 2, config.block_size+1, num_symbols=config.vocab_size)]
    # base = [datasets.ngrams('train',2, config.block_size+1, num_symbols = config.vocab_size,), datasets.unigram_3('train', 1, 2, config.block_size+1, num_symbols=config.vocab_size)]

    test_dataset = datasets.ngrams("test", 2, config.block_size+1, config.vocab_size)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    # Create a Scalar Mappable for the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # You need to set the array to an empty list
    labels = ["Uniform", "Unigram", "Bigram"]
    name = config.model_type
    # fig.suptitle(f'{config.model_type}: {config.vocab_size} Symbols')
    seeds = range(10)
    def_max = config.max_iters
    for i in range(num-2, num):
        if i == num -2:
            continue
        datem = []
        for seed in seeds:
            set_seed(seed)
            config.seed = seed
            if i > 0:
                config.max_iters = def_max * (num - 1) / i
            
                config.dataset = datasets.mixture(base[0], base[1], def_max, config.max_iters, i / (num-1))
            else:
                config.dataset = base[1]
            model_history, _ = training_pipeline.train(config)
            data = test_error.test_last_token(model_history, test_dataset, config.device)
            datem.append(data[0])
        # data_to_curve(datem, f"{i/(num-1)}", axis, colormap(i/(num-1)))
        if i == num-1:e
    seeds = range(2)
    # matplotlib.rcParams.update({'font.size': 20})
    # assert config.vocab_size == 2
    # Create a colormap
    colormap = matplotlib.cm.viridis
    # Plotting
    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)
    sets = [datasets.interpolate_unigram("train", p, config.block_size+1, config.vocab_size) for p in np.arange(0,1+1/(num-1),1/(num-1))]
    base = [datasets.doubly_stochastic_3('train', 1, 2, config.block_size+1, num_symbols = config.vocab_size,), datasets.unigram_3('train', 1, 2, config.block_size+1, num_symbols=config.vocab_size)]
    
    sets = [datasets.mixture(base[0], base[1], p) for p in np.arange(0,1+1/(num-1),1/(num-1))]
    test_dataset = datasets.ngrams("test", 2, config.block_size+1, config.vocab_size)
    # dataset_labels = ["Doubly Stochastic", "Unigrams", "Mixture"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    # Create a Scalar Mappable for the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # You need to set the array to an empty list

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8), sharey="row")
    # data_to_curve([t[0].numpy() for t in datem], "Test Loss", colors[0])
    # for dat in datem[1:]:
    #     plt.plot(dat[0].numpy(), color = colors[0])
    # names = ["", "Unigram ", "Uniform Stationary "]
    labels = ["Uniform", "Unigram", "Bigram"]
    name = config.model_type
    # fig.suptitle(f'{config.model_type}: {config.vocab_size} Symbols')
    for i in range(len(sets)):
        datem = []
        config.dataset = sets[i]
        for seed in seeds:
            config.seed == seed
            model_history, _ = training_pipeline.train(config)
            data = test_error.test_last_token(model_history, test_dataset, config.device)
            datem.append(data[0])
        # print(data[0])
        data_to_curve(datem, f"{i/(num-1)}", axis, colormap(i/(num-1)))
        # axis.plot(data[0], color = colormap(i/(num-1)), label = )
        # axis.plot(data[0], label = dataset_labels[i])

    colors = sns.color_palette()
    # for i in range(1,3):
    #     plt.axhline(data[1][i].numpy(), color = colors[i])
    # axis.legend()

    axis.set_xlim(-1, len(data[0])+1)
    # axis.plot(data[0].numpy(), color = colors[0], label="transformer")
    axis.set_title(f'Test Loss in Original Distribution')
    # axis.set_xticks(range(0, len(data[0]) + 1, len(data[0])//10), range(0,config.max_iters + 1, config.max_iters//10))
    ticks = 5
    time = len(data[0])
    axis.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    axis.set_xlabel('Training Sequences Seen (Thousands)')
    axis.set_ylabel('KL-Div Loss')
    axis.tick_params(axis='y', which='both', labelleft=True)
    plt.colorbar(sm, ax=axis)
    return datem

@torch.no_grad()
def attention(config, model_history, points, data, idx = None):
    if idx is None:
        set_seed(0)
        test_dataset = datasets.ngrams('test', 2, config.block_size+1, config.vocab_size)
        idx = torch.stack([test_dataset[0][0][:16]]).to("cuda")
    colors = sns.color_palette()
    fig = plt.figure(figsize=(3 * (len(points) + 1)+2, 5))  # Adjust the size as needed
    gs = gridspec.GridSpec(2, ncols=len(points)+1, width_ratios = [2] + [1]*len(points))
    ax0 = plt.subplot(gs[:,0])
    fig.suptitle(f'Effective Positional Encoding')
    labels = ["uniform", "unigram", "bigram"]
    for i in range(1, len(data[1])):
        dats = [data[1][i]]
        data_to_line(dats, len(data[0]), labels[i], ax0, colors[1+i])
    # ax0.legend()
    ax0.plot(data[0].numpy(), color = colors[0], label="transformer")
    for i in range(len(points)):
        ax0.axvline(points[i], color = colors[5+i], linestyle="--")
        label = f't={int((points[i])*config.batch_size / 1000*config.max_iters/(len(model_history)-1))}'
        ax0.text(points[i] + 1, ax0.get_ylim()[1]*.9+.01 * (i%2), label, fontsize=12, color=colors[5+i])
    ax0.set_xlim(-1, len(data[2][0])+1)
    fig.suptitle(f'{config.model_type}: {config.vocab_size} Symbols')
    ax0.set_title(f'Test Loss')
    time = len(data[0])
    ticks = 5
    ax0.set_xticks(range(0, time + 1, time//ticks), range(0, (config.max_iters + 1) * config.batch_size // 1000, config.max_iters//ticks * config.batch_size // 1000))
    ax0.set_xlabel('Training Sequences Seen (Thousands)')
    ax0.set_ylabel('KL-Div Loss')


    temp=0
    
    for i in range(len(points)):
        for layer in [0,1]:
            ax = plt.subplot(gs[layer,i+1])
            if layer == 0:
                ax.set_title(f"t={int((points[i])*config.batch_size / 1000*config.max_iters/(len(model_history)-1))}")
                if i == 0:
                    ax.text(-.1, 0.5, 'First Layer', va='center', ha='right', transform=ax.transAxes, rotation=90)
            if i == 0 and layer ==1:
                    ax.text(-.1, 0.5, 'Second Layer', va='center', ha='right', transform=ax.transAxes, rotation=90)
            ax.tick_params(direction='out', length=6, width=2, colors='k')

            m = model_history[points[i]]
            attn, sm_attn = m.visualize_attention(idx)
            to_graph = sm_attn[layer].squeeze().cpu()
            # attention_weights(idx.squeeze().cpu().numpy(), to_graph)
            sns.heatmap(to_graph, cmap="viridis", ax = ax, mask = np.triu(np.ones_like(to_graph), k = 1), square = True, xticklabels=False, yticklabels=False)
            # ax.yaxis.labelpad = 10
            # ax.set_xlim(-1, len(pos))
            # ax.set_ylabel("weight")
            ax.patch.set_edgecolor(colors[5+i])  # Set the color
            boundary = 4
            ax.patch.set_linewidth(boundary)  # Set the line width
            # Adjust the position of the axes spines
            for spine in ax.spines.values():
                spine.set_position(('outward', boundary-1))  # Move the spines outward

    plt.tight_layout()
    return fig, gs

from matplotlib.collections import LineCollection
@torch.no_grad()
def attention_weights(x, attn, ax, top = True, points = None, arrows = True, color = "xkcd:azure"):
    attn = attn / attn.max(axis=-1)[0]
    # Tokens
    colors = sns.color_palette()
    top_tokens = x
    bottom_tokens = x
    if points is None:
        points = range(len(top_tokens))
        if top:
            points = [len(top_tokens)-1]
    for top_pos, top_token in enumerate(top_tokens):
        if arrows:
            ax.arrow(top_pos, 0, 0, 1, color="k",
                    head_width=0.2, head_length=0.03)
        else:
            ax.plot([top_pos]*2, [0,1], color="k", zorder = 1)

    # Draw a gray rounded rectangle between the top and bottom tokens
    offset = .05
    rectangle = matplotlib.patches.FancyBboxPatch((0, 0.4-offset), len(x)-1, .4- 2 * offset, color="lightgray", ec="none", alpha=.9)
    ax.add_patch(rectangle)
    # segs is cartesian product of points with range(len(x))
    segs = np.array([[(j,offset), (i, 1- offset)] for i in points for j in range(len(x))])
    widths = np.array([attn[i,j] for i in points for j in range(len(x))]) * 3
    
    alpha = widths / 3 * .9/2
    lines = LineCollection(segs, linewidths=widths, alpha=alpha, color=color)
    lines.set_clip_path(rectangle)
    ax.add_collection(lines)

    # lines = []
    # for top_pos in points:
    #     top_token = top_tokens[top_pos]
    #     for bottom_pos, bottom_token in enumerate(bottom_tokens):
    #         weight = attn[top_pos,bottom_pos]  # Attention weight
    #         linewidth = weight * 3  # Line thickness based on the attention weight
    #         l = ax.plot((bottom_pos, top_pos), (offset, 1-offset), color=color, linewidth=linewidth, alpha=float(weight)*.9)
    #         lines.append((l[0], top_pos, bottom_pos))
    #         # l = ax.plot((bottom_pos, top_pos), (offset, 1-offset), color=colors[bottom_token], linewidth=linewidth, alpha=float(weight))
    #         l[0].set_clip_path(rectangle)
    
    # Add the tokens to the top and bottom of the plot
    if top:
        for pos, token in enumerate(top_tokens):
            ax.text(pos, 1.05, str(token), ha='center', va='bottom', color = colors[token], fontweight='bold')

    for pos, token in enumerate(bottom_tokens):
        ax.text(pos, -0.05, str(token), ha='center', va='top', color = colors[token], fontweight='bold')
    # Remove all axes and spines as they are not needed for this visualization
    ax.axis('off')

    return lines

@torch.no_grad()
def adjust(attn):
    # mini = (attn + (torch.ones_like(attn) * float("inf")).triu(diagonal = 1)).min(axis=1)[0][None].T
    attn = attn.tril()
    # attn = attn - mini
    # attn = F.normalize(attn, p=1, dim=1)
    attn /= attn.max(axis=1)[0][None].T
    attn = torch.torch.nan_to_num(attn)
    return attn.tril()

@torch.no_grad()
def flow(m, x):    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharey="row")
    attn, sm_attn = m.visualize_attention(x)
    l1, l2 = sm_attn[0].squeeze().cpu(), sm_attn[1].squeeze().cpu()
    l1, l2 = adjust(l1), adjust(l2)
    # print(l1[7])
    # print(l2)
    # print(l2[13])
    # print(l1[14])
    # for i in range(l1.shape[0]):
    #     l2[i] *= l1[15,i]
    # print(l2.shape)
    
    attention_weights(x.squeeze().cpu().numpy(), l1, axes[1], top = False)
    attention_weights(x.squeeze().cpu().numpy(), l2, axes[0], top = True, points = [len(x.squeeze())-1])
    # attention_weights(x.squeeze().cpu().numpy(), l2, axes[0], pos=[, top = True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()

@torch.no_grad()
def attention_at(model_history, points, x):
    matplotlib.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=2, ncols=len(points), figsize=(6 * len(points), 4))
    for i, point in enumerate(points):
        if len(points) > 1:
            axes[0, i].set_title(f"t={int(point*64 / 1000*2000/(len(model_history)-1))}", pad=15)
        else:
            axes[i].set_title(f"t={int(point*64 / 1000*2000/(len(model_history)-1))}", pad=15)
        attn, sm_attn = model_history[point].visualize_attention(x)
        l1, l2 = sm_attn[0].squeeze().cpu(), sm_attn[1].squeeze().cpu()
        xcpu = x.squeeze().cpu().numpy()
        # print(l2[-1])
        l1, l2 = adjust(l1), adjust(l2)
        if len(points) > 1:
            attention_weights(xcpu, l1, axes[1, i], top = False)
            attention_weights(xcpu, l2, axes[0, i], top = True)
        else:
            attention_weights(xcpu, l1, axes[1], top = False)
            attention_weights(xcpu, l2, axes[0], top = True)
    
    if len(points) > 1:
        axes[0, 0].text(.04, 0.5, 'Second Layer', va='center', ha='right', transform=axes[0, 0].transAxes, rotation=90)
        axes[1, 0].text(.04, 0.5, 'First Layer', va='center', ha='right', transform=axes[1, 0].transAxes, rotation=90)
    else:
        axes[0].text(.04, 0.5, 'Second Layer', va='center', ha='right', transform=axes[0].transAxes, rotation=90)
        axes[1].text(.04, 0.5, 'First Layer', va='center', ha='right', transform=axes[1].transAxes, rotation=90)
    plt.tight_layout()
    plt.suptitle("Attention Patterns", y=1.02)
    # Adjust subplot parameters
    plt.subplots_adjust(hspace=0.1)

@torch.no_grad()
def multi_head_attention_at(model_history, x, axes = None, frame = -1):
    colors = ["xkcd:azure", "xkcd:mango", "xkcd:lime"]
    attn, sm_attn = model_history[frame].visualize_attention(x)
    sm_attn = sm_attn[0][0].cpu(), sm_attn[1][0].cpu()
    num_heads = sm_attn[0].shape[0]
    x = x.squeeze().cpu().numpy()
    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
    # axes[0].set_title(f"Attention", pad=15)
    lines = [[[] for j in range(num_heads)] for i in range(len(sm_attn))]
    for i in range(len(sm_attn)):
        layer = sm_attn[i]
        for j in range(len(layer)):
            head = layer[j]
            # head = adjust(head)
            lines[i][j].append(attention_weights(x, head, axes[1-i], top = i == 1, color=colors[j]))
    
        axes[0].text(.04, 0.5, 'Second Layer', va='center', ha='right', transform=axes[0].transAxes, rotation=90)
        axes[1].text(.04, 0.5, 'First Layer', va='center', ha='right', transform=axes[1].transAxes, rotation=90)
    axes[0].set_title("Attention pattern on example sequence", y=1.1)
    plt.tight_layout()
    return lines


@torch.no_grad()
def multi_head_attention_at_seperate(model_history, x, axes = None, frame = -1):
    
    attn, sm_attn = model_history[frame].visualize_attention(x)
    sm_attn = sm_attn[0][0].cpu(), sm_attn[1][0].cpu()
    num_heads = sm_attn[0].shape[0]
    x = x.squeeze().cpu().numpy()
    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=num_heads, figsize=(6*num_heads, 4))
    # axes[0].set_title(f"Attention", pad=15)
    lines = [[] for j in range(num_heads)]
    for i in range(len(sm_attn)):
        layer = sm_attn[i]
        for j in range(len(layer)):
            head = layer[j]
            head = adjust(head)
            lines[j].append(attention_weights(x, head, axes[1-i,j], top = i == 1, arrows = False))
    
        axes[0,i].text(.04, 0.5, 'Second Layer', va='center', ha='right', transform=axes[0,i].transAxes, rotation=90)
        axes[1,i].text(.04, 0.5, 'First Layer', va='center', ha='right', transform=axes[1,i].transAxes, rotation=90)
    plt.tight_layout()
    return lines


@torch.no_grad()
def pos_encode(pos, config, ax, frame = 0):
    
    colors = ["xkcd:azure", "xkcd:mango", "xkcd:aqua green"]
    # matplotlib.rcParams.update({'font.size': 20})
    ax.set_title(f'First Layer Positional Encoding\nt={frame*64/100}' , fontsize=10)
    for i in range(config.n_head):
        bar = ax.bar(np.arange(len(pos)), pos[:,frame, i], width = 0.8, edgecolor='none', alpha = .9, color = colors[i])
    ax.set_xlabel("Position")

    # ax.set_yscale('log')
    # change minimum y axis value to 1e-2
    # ymin, ymax = ax.get_ylim()
    # ax.set_ylim(ymin, max(ymax, 1.1e-2))
    return bar


@torch.no_grad()
def mini_attention(model_history, x, axes, frame = 0):
    axes[0].set_title(f"Attention", pad=15)
    attn, sm_attn = model_history[frame].visualize_attention(x)
    l1, l2 = sm_attn[0].squeeze().cpu(), sm_attn[1].squeeze().cpu()
    xcpu = x.squeeze().cpu().numpy()
    # print(l2[-1])
    l1, l2 = adjust(l1), adjust(l2)
    lines = []
    lines.append(attention_weights(xcpu, l1, axes[1], top = False))
    lines.append(attention_weights(xcpu, l2, axes[0], top = True))
    
    axes[0].text(.04, 0.5, 'Second Layer', va='center', ha='right', transform=axes[0].transAxes, rotation=90)
    axes[1].text(.04, 0.5, 'First Layer', va='center', ha='right', transform=axes[1].transAxes, rotation=90)

    return lines

@torch.no_grad()
def animation(config, model_history, data, x, pos_dat = None):
    if pos_dat is None:
        pos_dat = test_error.pos_enc(model_history, config)
        pos_dat = torch.mean(pos_dat, axis=0)
        pos_dat = pos_dat.reshape((config.block_size, len(model_history), config.n_head))
    # create 2x2 gridspec for modular placement of plots
    fig = plt.figure(figsize=(9, 8), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    # top left: test loss
    ax0 = plt.subplot(gs[0, 0])
    test_loss([data], config, ax0)

    # top right: similarity
    ax1 = plt.subplot(gs[0, 1])
    similarity([data], config, ax1)


    # vertical bar at x = 0
    bars = [ax0.axvline(0, color='black', lw=2), ax1.axvline(0, color='black', lw=2)]

    # bottom left: positional encoding
    ax2 = plt.subplot(gs[1:, 0])
    pos_encodings = pos_encode(pos_dat, config, ax2)

    # bottom right: attention
    ax3 = plt.subplot(gs[1, 1]), plt.subplot(gs[2, 1])
    lines = multi_head_attention_at(model_history, x, ax3, frame=0)


    plt.tight_layout()
    # reduce vertical padding between ax3[0] and ax3[1]
    pos = ax3[0].get_position()
    ax3[1].set_position([pos.x0, pos.y0 - 0.172, pos.width, pos.height])

    def update(frame, idx=x):
        for bar in bars:
            bar.set_xdata(frame)
        ax2.clear()
        poses = pos_encode(pos_dat, config, ax2, frame)

        _, attn = model_history[frame].visualize_attention(idx)
        attn = attn[0][0].cpu(), attn[1][0].cpu()
        for i in [0, 1]:
            for j in range(len(attn[i])):
                head = attn[i][j]
                head = adjust(head)
                if i == 1:
                    points = [len(idx[0])-1]
                else:
                    points = list(range(len(idx[0])))
                widths = np.array([head[x, y] for x in points for y in range(len(idx[0]))]) * 3
                alpha = widths / 3 * 0.9
                lines[i][j][0].set_linewidth(widths)
                lines[i][j][0].set_alpha(alpha)

                    # ax3[1-i].draw_artist(lines[i][j][0])


                    # # print(lines[i][j])
                    # for line, x, y in lines[i][j][0]:
                    #     weight = head[x,y]
                    #     line.set_linewidth(weight * 3)
                    #     line.set_alpha(float(weight)*.95)

                #     lines[i],[j].append(attention_weights(x, head, axes[1-i], top = i == 1, color=colors[j]))
                # ax3[0].clear()
                # ax3[1].clear()
                # multi_head_attention_at(model_history, x, ax3, frame = frame)
                # # reduce vertical padding between ax3[0] and ax3[1]
                # pos = ax3[0].get_position()
                # ax3[1].set_position([pos.x0, pos.y0 - 0.172, pos.width, pos.height])


    
    # anim = matplotlib.animation.FuncAnimation(fig, update, frames=range(len(data[0])), blit=True)
    # anim = FuncAnimation(fig, update, frames=range(0, len(data[0])), interval=75)
    anim = FuncAnimation(fig, update, frames=range(0, len(data[0])), interval=75)
    return anim


@torch.no_grad()
def animation_nopos(config, model_history, data, x):

    # create 2x2 gridspec for modular placement of plots
    fig = plt.figure(figsize=(9, 8), dpi=200)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    # top left: test loss
    ax0 = plt.subplot(gs[0, 0])
    test_loss([data], config, ax0)

    # top right: similarity
    ax1 = plt.subplot(gs[0, 1])
    similarity([data], config, ax1)


    # vertical bar at x = 0
    bars = [ax0.axvline(0, color='black', lw=2), ax1.axvline(0, color='black', lw=2)]
    # bottom right: attention
    ax3 = plt.subplot(gs[1, 0:]), plt.subplot(gs[2, 0:])
    lines = multi_head_attention_at(model_history, x, ax3, frame=0)
    
    plt.tight_layout()
    # reduce vertical padding between ax3[0] and ax3[1]
    pos = ax3[0].get_position()
    ax3[1].set_position([pos.x0, pos.y0 - 0.17, pos.width, pos.height])

    def update(frame, idx=x):
        for bar in bars:
            bar.set_xdata(frame)
        _, attn = model_history[frame].visualize_attention(idx)
        attn = attn[0][0].cpu(), attn[1][0].cpu()
        for i in [0, 1]:
            for j in range(len(attn[i])):
                head = attn[i][j]
                head = adjust(head)
                if i == 1:
                    points = [len(idx[0])-1]
                else:
                    points = list(range(len(idx[0])))
                widths = np.array([head[x, y] for x in points for y in range(len(idx[0]))]) * 3
                alpha = widths / 3 * 0.9
                lines[i][j][0].set_linewidth(widths)
                lines[i][j][0].set_alpha(alpha)

    anim = FuncAnimation(fig, update, frames=range(0, len(data[0])), interval=25)
    return anim


@torch.no_grad()
def animate_attn(config, model_history, data, x):
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 4), dpi=200)

    lines = multi_head_attention_at(model_history, x, axes, frame=0)


    plt.tight_layout()
    # reduce vertical padding between ax3[0] and ax3[1]
    pos = axes[1].get_position()
    axes[1].set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])

    def update(frame, idx=x):
        
        _, attn = model_history[frame].visualize_attention(idx)
        attn = attn[0][0].cpu(), attn[1][0].cpu()
        for i in [0, 1]:
            head = attn[i][0]
            head = adjust(head)
            if i == 1:
                points = [len(idx[0])-1]
            else:
                points = list(range(len(idx[0])))
            widths = np.array([head[x, y] for x in points for y in range(len(idx[0]))]) * 3
            alpha = widths / 3 * 0.9
            lines[i][0][0].set_linewidth(widths)
            lines[i][0][0].set_alpha(alpha)



                # # print(lines[i][j])
                # for line, x, y in lines[i][j][0]:
                #     weight = head[x,y]
                #     line.set_linewidth(weight * 3)
                #     line.set_alpha(float(weight)*.95)

            #     lines[i],[j].append(attention_weights(x, head, axes[1-i], top = i == 1, color=colors[j]))
            # ax3[0].clear()
            # ax3[1].clear()
            # multi_head_attention_at(model_history, x, ax3, frame = frame)
            # # reduce vertical padding between ax3[0] and ax3[1]
            # pos = ax3[0].get_position()
            # ax3[1].set_position([pos.x0, pos.y0 - 0.172, pos.width, pos.height])
        temper = [line for temp in lines for line in temp if line != []]
        temper = [line[0] for line in temper]


    
    # anim = matplotlib.animation.FuncAnimation(fig, update, frames=range(len(data[0])), blit=True)
    # anim = FuncAnimation(fig, update, frames=range(0, len(data[0]), 50), interval=75)
    anim = FuncAnimation(fig, update, frames=range(0, len(data[0]),50), interval=75)
    return anim


@torch.no_grad()
def animation_attn_loss(config, model_history, data, x):

    # create 2x2 gridspec for modular placement of plots
    fig = plt.figure(figsize=(9, 8), dpi=200)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # top: test loss
    ax0 = plt.subplot(gs[0, 0])
    test_loss([data], config, ax0)



    # vertical bar at x = 0
    bars = [ax0.axvline(0, color='black', lw=2)]
    # bottom right: attention
    ax3 = plt.subplot(gs[1, 0:]), plt.subplot(gs[2, 0:])
    lines = multi_head_attention_at(model_history, x, ax3, frame=0)


    plt.tight_layout()
    # reduce vertical padding between ax3[0] and ax3[1]
    pos = ax3[0].get_position()
    ax3[1].set_position([pos.x0, pos.y0 - 0.19, pos.width, pos.height])

    def update(frame, idx=x):
        for bar in bars:
            bar.set_xdata(frame)
        _, attn = model_history[frame].visualize_attention(idx)
        attn = attn[0][0].cpu(), attn[1][0].cpu()
        for i in [0, 1]:
            for j in range(len(attn[i])):
                head = attn[i][j]
                head = adjust(head)
                if i == 1:
                    points = [len(idx[0])-1]
                else:
                    points = list(range(len(idx[0])))
                widths = np.array([head[x, y] for x in points for y in range(len(idx[0]))]) * 3
                alpha = widths / 3 * 0.9
                lines[i][j][0].set_linewidth(widths)
                lines[i][j][0].set_alpha(alpha)

    anim = FuncAnimation(fig, update, frames=range(0, len(data[0])), interval=75)
    return anim