import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import networkx as nx
import random

from io import BytesIO
from itertools import chain
from collections import namedtuple, OrderedDict


Sentence = namedtuple("Sentence", "words tags")

def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)


def model2png(model, filename="", overwrite=False, show_ends=False):
    """Convert a Pomegranate model into a PNG image

    The conversion pipeline extracts the underlying NetworkX graph object,
    converts it to a PyDot graph, then writes the PNG data to a bytes array,
    which can be saved as a file to disk or imported with matplotlib for display.

        Model -> NetworkX.Graph -> PyDot.Graph -> bytes -> PNG

    Parameters
    ----------
    model : Pomegranate.Model
        The model object to convert. The model must have an attribute .graph
        referencing a NetworkX.Graph instance.

    filename : string (optional)
        The PNG file will be saved to disk with this filename if one is provided.
        By default, the image file will NOT be created if a file with this name
        already exists unless overwrite=True.

    overwrite : bool (optional)
        overwrite=True allows the new PNG to overwrite the specified file if it
        already exists

    show_ends : bool (optional)
        show_ends=True will generate the PNG including the two end states from
        the Pomegranate model (which are not usually an explicit part of the graph)
    """
    nodes = model.graph.nodes()
    if not show_ends:
        nodes = [n for n in nodes if n not in (model.start, model.end)]
    g = nx.relabel_nodes(model.graph.subgraph(nodes), {n: n.name for n in model.graph.nodes()})
    pydot_graph = nx.drawing.nx_pydot.to_pydot(g)
    pydot_graph.set_rankdir("LR")
    png_data = pydot_graph.create_png(prog='dot')
    img_data = BytesIO()
    img_data.write(png_data)
    img_data.seek(0)
    if filename:
        if os.path.exists(filename) and not overwrite:
            raise IOError("File already exists. Use overwrite=True to replace existing files on disk.")
        with open(filename, 'wb') as f:
            f.write(img_data.read())
        img_data.seek(0)
    return mplimg.imread(img_data)
