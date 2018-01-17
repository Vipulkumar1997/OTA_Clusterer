import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
import math;
from pyclustering.utils.color import color as color_list;
from ota_clusterer.clusterer.kmedoid.CanvasClusterDescription import CanvasClusterDescription


class ClusterVisualizer:
    """!
    @brief Common visualizer of clusters on 2D or 3D surface.

    """

    def __init__(self, number_canvases=1, size_row=1, titles=None, data=None, labels=None):
        """!
        @brief Constructor of cluster visualizer.

        @param[in] number_canvases (uint): Number of canvases that is used for visualization.
        @param[in] size_row (uint): Amount of canvases that can be placed in one row.
        @param[in] titles (list): List of canvas's titles.

        @param[in] data = tsne_model data -- added by sandroci
        @param[in] labels = data point labels for visulization purposes -- added by sandroci

        """

        self.__number_canvases = number_canvases;
        self.__size_row = size_row;
        self.__canvas_clusters = [[] for _ in range(number_canvases)];
        self.__canvas_dimensions = [None for _ in range(number_canvases)];
        self.__canvas_titles = [None for _ in range(number_canvases)];
        self.__data = data;
        self.__labels = labels;

        if (titles is not None):
            self.__canvas_titles = titles;

        self.__default_2d_marker_size = 5;
        self.__default_3d_marker_size = 30;

    def append_cluster(self, cluster, data=None, canvas=0, marker='.', markersize=None, color=None):
        """!
        @brief Appends cluster to canvas for drawing.

        @param[in] cluster (list): cluster that may consist of indexes of objects from the data or object itself.
        @param[in] data (list): If defines that each element of cluster is considered as a index of object from the data.
        @param[in] canvas (uint): Number of canvas that should be used for displaying cluster.
        @param[in] marker (string): Marker that is used for displaying objects from cluster on the canvas.
        @param[in] markersize (uint): Size of marker.
        @param[in] color (string): Color of marker.

        @return Returns index of cluster descriptor on the canvas.

        """

        if (len(cluster) == 0):
            return;

        if (canvas > self.__number_canvases):
            raise NameError('Canvas does ' + canvas + ' not exists.');

        if (color is None):
            index_color = len(self.__canvas_clusters[canvas]) % len(color_list.TITLES);
            color = color_list.TITLES[index_color];

        added_canvas_descriptor = CanvasClusterDescription(cluster, data, marker, markersize, color);
        self.__canvas_clusters[canvas].append(added_canvas_descriptor);

        dimension = 0;
        if (data is None):
            dimension = len(cluster[0]);
            if (self.__canvas_dimensions[canvas] is None):
                self.__canvas_dimensions[canvas] = dimension;
            elif (self.__canvas_dimensions[canvas] != dimension):
                raise NameError('Only clusters with the same dimension of objects can be displayed on canvas.');

        else:
            dimension = len(data[0]);
            if (self.__canvas_dimensions[canvas] is None):
                self.__canvas_dimensions[canvas] = dimension;
            elif (self.__canvas_dimensions[canvas] != dimension):
                raise NameError('Only clusters with the same dimension of objects can be displayed on canvas.');

        if ((dimension < 1) and (dimension > 3)):
            raise NameError(
                'Only objects with size dimension 1 (1D plot), 2 (2D plot) or 3 (3D plot) can be displayed.');

        if (markersize is None):
            if ((dimension == 1) or (dimension == 2)):
                added_canvas_descriptor.markersize = self.__default_2d_marker_size;
            elif (dimension == 3):
                added_canvas_descriptor.markersize = self.__default_3d_marker_size;

        return len(self.__canvas_clusters[canvas]) - 1;

    def append_cluster_attribute(self, index_canvas, index_cluster, data, marker=None, markersize=None):
        """!
        @brief Append cluster attribure for cluster on specific canvas.
        @details Attribute it is data that is visualized for specific cluster using its color, marker and markersize if last two is not specified.

        @param[in] index_canvas (uint): Index canvas where cluster is located.
        @param[in] index_cluster (uint): Index cluster whose attribute should be added.
        @param[in] data (list): List of points (data) that represents attribute.
        @param[in] marker (string): Marker that is used for displaying objects from cluster on the canvas.
        @param[in] markersize (uint): Size of marker.

        """

        cluster_descr = self.__canvas_clusters[index_canvas][index_cluster];
        attribute_marker = marker;
        if (attribute_marker is None):
            attribute_marker = cluster_descr.marker;

        attribure_markersize = markersize;
        if (attribure_markersize is None):
            attribure_markersize = cluster_descr.markersize;

        attribute_color = cluster_descr.color;

        added_attribute_cluster_descriptor = CanvasClusterDescription(data, None, attribute_marker, attribure_markersize,
                                                                  attribute_color);
        self.__canvas_clusters[index_canvas][index_cluster].attributes.append(added_attribute_cluster_descriptor);

    def append_clusters(self, clusters, data=None, canvas=0, marker='.', markersize=None):
        """!
        @brief Appends list of cluster to canvas for drawing.

        @param[in] clusters (list): List of clusters where each cluster may consist of indexes of objects from the data or object itself.
        @param[in] data (list): If defines that each element of cluster is considered as a index of object from the data.
        @param[in] canvas (uint): Number of canvas that should be used for displaying clusters.
        @param[in] marker (string): Marker that is used for displaying objects from clusters on the canvas.
        @param[in] markersize (uint): Size of marker.

        """

        for cluster in clusters:
            self.append_cluster(cluster, data, canvas, marker, markersize);

    def set_canvas_title(self, text, canvas=0):
        """!
        @brief Set title for specified canvas.

        @param[in] text (string): Title for canvas.
        @param[in] canvas (uint): Index of canvas where title should be displayed.

        """

        if (canvas > self.__number_canvases):
            raise NameError('Canvas does ' + canvas + ' not exists.');

        self.__canvas_titles[canvas] = text;

    def get_cluster_color(self, index_cluster, index_canvas):
        """!
        @brief Returns cluster color on specified canvas.

        """
        return self.__canvas_clusters[index_canvas][index_cluster].color;

    def show(self, figure=None, visible_axis=True, visible_grid=True, display=True, shift=None, k=None, tolerance=None):
        """!
        @brief Shows clusters (visualize).

        @param[in] figure (fig): Defines requirement to use specified figure, if None - new figure is created for drawing clusters.
        @param[in] visible_axis (bool): Defines visibility of axes on each canvas, if True - axes are invisible.
        @param[in] visible_grid (bool): Defines visibility of axes on each canvas, if True - grid is displayed.
        @param[in] display (bool): Defines requirement to display clusters on a stage, if True - clusters are displayed, if False - plt.show() should be called by user."
        @param[in] shift (uint): Force canvas shift value - defines canvas index from which custers should be visualized.

        @return (fig) Figure where clusters are shown.
        s
        """

        cluster_figure = None;

        canvas_shift = shift;
        if (canvas_shift is None):
            if (figure is not None):
                canvas_shift = len(figure.get_axes());
            else:
                canvas_shift = 0;

        if (figure is not None):
            cluster_figure = figure;
        else:
            cluster_figure = plt.figure();

        maximum_cols = self.__size_row;
        maximum_rows = math.ceil((self.__number_canvases + canvas_shift) / maximum_cols);

        grid_spec = gridspec.GridSpec(maximum_rows, maximum_cols);

        for index_canvas in range(len(self.__canvas_clusters)):
            canvas_data = self.__canvas_clusters[index_canvas];
            if (len(canvas_data) == 0):
                continue;

            dimension = self.__canvas_dimensions[index_canvas];

            # ax = axes[real_index];
            if ((dimension == 1) or (dimension == 2)):
                ax = cluster_figure.add_subplot(grid_spec[index_canvas + canvas_shift]);
            else:
                ax = cluster_figure.add_subplot(grid_spec[index_canvas + canvas_shift], projection='3d');

            if (len(canvas_data) == 0):
                plt.setp(ax, visible=False);

            for cluster_descr in canvas_data:
                self.__draw_canvas_cluster(ax, dimension, cluster_descr);

                for attribute_descr in cluster_descr.attributes:
                    self.__draw_canvas_cluster(ax, dimension, attribute_descr);

            if (visible_axis is True):
                ax.xaxis.set_ticklabels([]);
                ax.yaxis.set_ticklabels([]);

                if (dimension == 3):
                    ax.zaxis.set_ticklabels([]);

            if (self.__canvas_titles[index_canvas] is not None):
                ax.set_title(self.__canvas_titles[index_canvas]);

            ax.grid(visible_grid);

        if (display is True):
            # Annotate the data points on the plot -- added by sandroci
            for i, txt in zip(self.__data, self.__labels):
                plt.annotate(txt, (i[0], i[1]), xytext=(0, -8), textcoords="offset points", va="center", ha="left")

            plt.title(
                'K-Mediod clustering with ' + str(k) + ' tolerance= ' + str(tolerance) + ' on the t-sne reduced data')
            plt.show();

        return cluster_figure;

    """!
    @brief Draw canvas cluster descriptor.

    @param[in] ax (Axis): Axis of the canvas where canvas cluster descriptor should be displayed.
    @param[in] dimension (uint): Canvas dimension.
    @param[in] cluster_descr (canvas_cluster_descr): Canvas cluster descriptor that should be displayed.
    @return (fig) Figure where clusters are shown.

    """

    def __draw_canvas_cluster(self, ax, dimension, cluster_descr):
        cluster = cluster_descr.cluster;
        data = cluster_descr.data;
        marker = cluster_descr.marker;
        markersize = cluster_descr.markersize;
        color = cluster_descr.color;

        for item in cluster:
            if (dimension == 1):
                if (data is None):
                    ax.plot(item[0], 0.0, color=color, marker=marker, markersize=markersize);
                else:
                    ax.plot(data[item][0], 0.0, color=color, marker=marker, markersize=markersize);

            elif (dimension == 2):
                if (data is None):
                    ax.plot(item[0], item[1], color=color, marker=marker, markersize=markersize);
                else:
                    ax.plot(data[item][0], data[item][1], color=color, marker=marker, markersize=markersize);

            elif (dimension == 3):
                if (data is None):
                    ax.scatter(item[0], item[1], item[2], c=color, marker=marker, s=markersize);
                else:
                    ax.scatter(data[item][0], data[item][1], data[item][2], c=color, marker=marker, s=markersize);
