def _format_x_axis(ax, x_centres, x_width):
    """
    Format x-axis tick marks and labels for better readability.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    x_centres : ndarray
        X position centers of bins
    x_width : float
        Width of each bin
    """
    # Set ticks at bin centers
    ax.set_xticks(x_centres)
    
    # Format tick labels - round to reasonable precision
    tick_labels = []
    for val in x_centres:
        if abs(val) < 0.001 or abs(val) > 10000:
            # Use scientific notation for very small/large numbers
            tick_labels.append(f'{val:.2e}')
        elif abs(val - int(val)) < 1e-10:
            # Integer values
            tick_labels.append(f'{int(val)}')
        else:
            # Decimal values - use 2 significant figures after decimal
            tick_labels.append(f'{val:.3f}')
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add minor gridlines for better readability
    ax.grid(True, axis='x', alpha=0.2, linestyle='--', which='major')
    ax.grid(True, axis='y', alpha=0.1, linestyle=':', which='minor')


def plot_boxplot(results,
                       separate_by='both',
                       metric_name=None,
                       x_col=None,
                       colors=None,
                       alphas=None,
                       figsize=(12, 6),
                       ax=None,
                       fig=None,
                       showmeans=True,
                       showfliers=True):
    """
    Plot boxplot statistics from boxplot_metvx_ref output using matplotlib's bxp.
    
    Parameters
    ----------
    results : dict
        Output from boxplot_metvx_ref with structure:
        {x_col_name: {metric_name: {'box_stats': [...], 'x_edge': [...], ...}}}
    
    separate_by : {'both', 'x_col', 'metric_name'}, default='both'
        How to separate plots:
        - 'both': Create subplot grid with x_col as rows and metric_name as columns
        - 'x_col': Create separate plots for each x_col (all metrics in each plot)
        - 'metric_name': Create separate plots for each metric (all x_cols in each plot)
    
    metric_name : str, optional
        Filter to plot only a specific metric. If None, plot all metrics.
    
    x_col : str, optional
        Filter to plot only a specific x_col. If None, plot all x_cols.
    
    colors : list or None, default=None
        List of colors for box fill. Cycled through each box.
        If None, uses matplotlib default colors.
    
    alphas : list or None, default=None
        List of transparencies (0-1) for boxes. Cycled through each box.
        If None, uses opaque boxes (alpha=1).
    
    figsize : tuple, default=(12, 6)
        Figure size (width, height) if creating new figure.
    
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. Used only when separate_by='both' fails
        or for single subplot cases.
    
    fig : matplotlib.figure.Figure, optional
        Existing figure to use. If provided with ax=None, creates subplots within fig.
    
    showmeans : bool, default=True
        Show mean markers on boxplots (from box_stats if available).
    
    showfliers : bool, default=True
        Show outlier points (from box_stats if available).
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'fig': matplotlib Figure object
        - 'axes': dict mapping (x_col, metric_name) to Axes, or simplified dict if only one plot
        - 'axes_flat': list of all Axes objects created
    """
    # Validate separate_by
    if separate_by not in ['both', 'x_col', 'metric_name']:
        raise ValueError("separate_by must be 'both', 'x_col', or 'metric_name'")
    
    # Filter results based on metric_name and x_col
    filtered_results = {}
    for xcol, metrics_dict in results.items():
        if x_col is not None and xcol != x_col:
            continue
        filtered_metrics = {}
        for mname, data in metrics_dict.items():
            if metric_name is not None and mname != metric_name:
                continue
            filtered_metrics[mname] = data
        if filtered_metrics:
            filtered_results[xcol] = filtered_metrics
    
    if not filtered_results:
        raise ValueError("No data matches the provided filters (metric_name, x_col)")
    
    # Get unique x_cols and metric names
    unique_xcols = sorted(filtered_results.keys())
    unique_metrics = sorted(set(m for metrics_dict in filtered_results.values() for m in metrics_dict.keys()))
    
    # Setup colors and alphas cycles
    if colors is None:
        colors = plt.cm.Set1.colors
    color_cycle = cycle(colors)
    
    if alphas is None:
        alphas = [1.0]
    alpha_cycle = cycle(alphas)
    
    # Create figure and axes based on separate_by
    axes_dict = {}
    axes_list = []
    
    if separate_by == 'both':
        n_rows = len(unique_xcols)
        n_cols = len(unique_metrics)
        if fig is None:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows / 2))
        
        global_color_idx = 0
        global_alpha_idx = 0
        
        for i, xcol in enumerate(unique_xcols):
            for j, mname in enumerate(unique_metrics):
                ax_idx = i * n_cols + j + 1
                ax = fig.add_subplot(n_rows, n_cols, ax_idx)
                axes_list.append(ax)
                axes_dict[(xcol, mname)] = ax
                
                # Plot if data exists
                if xcol in filtered_results and mname in filtered_results[xcol]:
                    # Get single color for this panel
                    panel_color = colors[global_color_idx % len(colors)]
                    panel_alpha = alphas[global_alpha_idx % len(alphas)]
                    
                    _plot_single_boxplot(
                        filtered_results[xcol][mname],
                        ax=ax,
                        color=panel_color,
                        alpha=panel_alpha,
                        showmeans=showmeans,
                        showfliers=showfliers,
                        title=f'{xcol} - {mname}'
                    )
                    # Format x-axis
                    data = filtered_results[xcol][mname]
                    _format_x_axis(ax, data['x_centre'], data['x_width'])
                    
                    # Increment color indices for next panel
                    global_color_idx += 1
                    global_alpha_idx += 1
    
    elif separate_by == 'x_col':
        if fig is None:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(unique_xcols) / 2))
        
        # Global color cycling across all panels
        global_color_idx = 0
        global_alpha_idx = 0
        
        for i, xcol in enumerate(unique_xcols):
            ax = fig.add_subplot(len(unique_xcols), 1, i + 1)
            axes_list.append(ax)
            axes_dict[xcol] = ax
            
            metric_colors = {}
            
            # Plot each metric separately on the same axes with different colors
            for mname in unique_metrics:
                if xcol in filtered_results and mname in filtered_results[xcol]:
                    data = filtered_results[xcol][mname]
                    box_stats = data['box_stats']
                    x_centres = data['x_centre']
                    x_width = data['x_width']
                    
                    # Get color and alpha from global cycling
                    color = colors[global_color_idx % len(colors)]
                    alpha = alphas[global_alpha_idx % len(alphas)]
                    metric_colors[mname] = (color, alpha)
                    
                    # Convert color to RGBA tuple if it's a string or list
                    try:
                        rgba_color = list(to_rgba(color))
                    except (ValueError, AttributeError):
                        # Assume it's already a list like [r, g, b] or [r, g, b, a]
                        if len(color) == 3:
                            rgba_color = list(color) + [1.0]
                        else:
                            rgba_color = list(color)
                    
                    # Set box face and edge colors
                    box_fc = rgba_color.copy()
                    box_fc[-1] = alpha  # Set alpha for face color
                    box_ec = rgba_color.copy()
                    box_ec[-1] = 1.0  # Edge always opaque
                    
                    # Define properties for this metric
                    boxprops = {'fc': tuple(box_fc), 'ec': tuple(box_ec), 'lw': 1.5}
                    medianprops = {'c': tuple(rgba_color[:3] + [1.0]), 'lw': 2.0}
                    meanprops = {'marker': 'D', 'mfc': tuple(rgba_color[:3] + [1.0]), 
                                'mec': tuple(rgba_color[:3] + [1.0]), 'markersize': 5}
                    
                    # Plot this metric's boxes
                    ax.bxp(box_stats, positions=x_centres, widths=x_width * 0.6,
                          boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,
                          showmeans=showmeans, showfliers=showfliers, patch_artist=True)
                    
                    # Increment global color indices
                    global_color_idx += 1
                    global_alpha_idx += 1
            
            ax.set_title(f'{xcol}')
            ax.set_xlabel('Bin Center')
            ax.set_ylabel('Metric Value')
            
            # Format x-axis for readability
            _format_x_axis(ax, x_centres, x_width)
            
            # Add legend if multiple metrics
            if len(unique_metrics) > 1:
                legend_elements = [
                    plt.Line2D([0], [0], color=metric_colors[mname][0], lw=4, 
                             label=mname, alpha=metric_colors[mname][1])
                    for mname in unique_metrics if mname in metric_colors
                ]
                ax.legend(handles=legend_elements, loc='best')
    
    elif separate_by == 'metric_name':
        if fig is None:
            fig = plt.figure(figsize=(figsize[0] * len(unique_metrics) / 2, figsize[1]))
        
        # Global color cycling across all panels
        global_color_idx = 0
        global_alpha_idx = 0
        
        for i, mname in enumerate(unique_metrics):
            ax = fig.add_subplot(1, len(unique_metrics), i + 1)
            axes_list.append(ax)
            axes_dict[mname] = ax
            
            xcol_colors = {}
            
            # Plot each x_col separately on the same axes with different colors
            for xcol in unique_xcols:
                if xcol in filtered_results and mname in filtered_results[xcol]:
                    data = filtered_results[xcol][mname]
                    box_stats = data['box_stats']
                    x_centres = data['x_centre']
                    x_width = data['x_width']
                    
                    # Get color and alpha from global cycling
                    color = colors[global_color_idx % len(colors)]
                    alpha = alphas[global_alpha_idx % len(alphas)]
                    xcol_colors[xcol] = (color, alpha)
                    
                    # Convert color to RGBA tuple if it's a string or list
                    try:
                        rgba_color = list(to_rgba(color))
                    except (ValueError, AttributeError):
                        # Assume it's already a list like [r, g, b] or [r, g, b, a]
                        if len(color) == 3:
                            rgba_color = list(color) + [1.0]
                        else:
                            rgba_color = list(color)
                    
                    # Set box face and edge colors
                    box_fc = rgba_color.copy()
                    box_fc[-1] = alpha  # Set alpha for face color
                    box_ec = rgba_color.copy()
                    box_ec[-1] = 1.0  # Edge always opaque
                    
                    # Define properties for this x_col
                    boxprops = {'fc': tuple(box_fc), 'ec': tuple(box_ec), 'lw': 1.5}
                    medianprops = {'c': tuple(rgba_color[:3] + [1.0]), 'lw': 2.0}
                    meanprops = {'marker': 'D', 'mfc': tuple(rgba_color[:3] + [1.0]), 
                                'mec': tuple(rgba_color[:3] + [1.0]), 'markersize': 5}
                    
                    # Plot this x_col's boxes
                    ax.bxp(box_stats, positions=x_centres, widths=x_width * 0.6,
                          boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,
                          showmeans=showmeans, showfliers=showfliers, patch_artist=True)
                    
                    # Increment global color indices
                    global_color_idx += 1
                    global_alpha_idx += 1
            
            ax.set_title(f'{mname}')
            ax.set_xlabel('Bin Center')
            ax.set_ylabel('Metric Value')
            
            # Format x-axis for readability (use data from last plotted x_col)
            if len(unique_xcols) > 0 and xcol in filtered_results and mname in filtered_results[xcol]:
                data = filtered_results[xcol][mname]
                _format_x_axis(ax, data['x_centre'], data['x_width'])
            
            # Add legend if multiple x_cols
            if len(unique_xcols) > 1:
                legend_elements = [
                    plt.Line2D([0], [0], color=xcol_colors[xcol][0], lw=4,
                             label=xcol, alpha=xcol_colors[xcol][1])
                    for xcol in unique_xcols if xcol in xcol_colors
                ]
                ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    return {
        'fig': fig,
        'axes': axes_dict,
        'axes_flat': axes_list
    }


def _plot_single_boxplot(data, ax, color, alpha, showmeans, showfliers, title=None):
    """
    Plot a single boxplot on given axes with a single color for all boxes.
    
    Parameters
    ----------
    data : dict
        Single metric data with keys 'box_stats', 'x_centre', 'x_width', 'x_edge'
    ax : matplotlib.axes.Axes
        Axes to plot on
    color : str or tuple
        Single color for all boxes in this panel
    alpha : float
        Alpha transparency for box faces (0-1)
    showmeans : bool
        Whether to show means
    showfliers : bool
        Whether to show fliers
    title : str, optional
        Title for the plot
    """
    box_stats = data['box_stats']
    x_centres = data['x_centre']
    x_width = data['x_width']
    
    # Convert color to RGBA tuple
    try:
        rgba_color = list(to_rgba(color))
    except (ValueError, AttributeError):
        if isinstance(color, (list, tuple)) and len(color) == 3:
            rgba_color = list(color) + [1.0]
        else:
            rgba_color = list(color) if isinstance(color, (list, tuple)) else [0, 0, 0, 1.0]
    
    # Set box face and edge colors
    box_fc = rgba_color.copy()
    box_fc[-1] = alpha  # Set alpha for face color
    box_ec = rgba_color.copy()
    box_ec[-1] = 1.0  # Edge always opaque
    
    # Define properties using this panel's color
    boxprops = {'fc': tuple(box_fc), 'ec': tuple(box_ec), 'lw': 1.5}
    medianprops = {'c': tuple(rgba_color[:3] + [1.0]), 'lw': 2.0}
    meanprops = {'marker': 'D', 'mfc': tuple(rgba_color[:3] + [1.0]), 
                'mec': tuple(rgba_color[:3] + [1.0]), 'markersize': 5}
    
    # Call bxp with patch_artist=True
    ax.bxp(box_stats, positions=x_centres, widths=x_width * 0.6,
           boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,
           showmeans=showmeans, showfliers=showfliers, patch_artist=True)
    
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    if title:
        ax.set_title(title)
