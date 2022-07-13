.. _optimization_vscode:

Using Visual Studio Code for Optimization
=========================================

volume / arith intensity

local view

transforming

adding custom transformations


SDFGs can be optimized using transformations from within the editor.
The SDFG Optimization sidepanel lists applicable transformations for the currently
opened grpah in the top right. This list of transformations is categorized and sorted by relevance
to what SDFG elements are currently in view and/or selected.

Hovering over trainsformations highlights the graph elements that are affected by them.

By clicking a transformation, the transformation description and properties appear in the details
panel, and the properties can be adjusted arbitrarily. From here, a button zooms to the
affected graph nodes, the transformation can be previewed on the graph, or the transformation can
be applied using the provided properties. A transformation can be applied using the default
properties with a single click by selecting `Quick Apply` in the transformation list.