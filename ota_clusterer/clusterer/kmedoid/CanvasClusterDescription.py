class CanvasClusterDescription:
    """!
    @brief Description of cluster for representation on canvas.

    """

    def __init__(self, cluster, data, marker, markersize, color):
        """!
        @brief Constructor of cluster representation on the canvas.

        @param[in] cluster (list): Single cluster that consists of objects or indexes from data.
        @param[in] data (list): Objects that should be displayed, can be None if clusters consist of objects instead of indexes.
        @param[in] marker (string): Type of marker that is used for drawing objects.
        @param[in] markersize (uint): Size of marker that is used for drawing objects.
        @param[in] color (string): Color of the marker that is used for drawing objects.

        """
        ## Cluster that may consist of objects or indexes of objects from data.
        self.cluster = cluster;

        ## Data where objects are stored. It can be None if clusters consist of objects instead of indexes.
        self.data = data;

        ## Marker that is used for drawing objects.
        self.marker = marker;

        ## Size of marker that is used for drawing objects.
        self.markersize = markersize;

        ## Color that is used for coloring marker.
        self.color = color;

        ## Attribures of the clusters - additional collections of data points that are regarded to the cluster.
        self.attributes = [];