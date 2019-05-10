## Installation

Install python >= 3.6, flask, parglare, dace, nodejs, npm
git checkout this repository
do npm install
do npm start



## SDFG Rendering

### Shapes

Each shape has a label, defined by the properties of the shape, e.g., the
iteration domain and variables of a Map and the level of detail. Thus when
we render the individual nodes of an SDFG our goals are:

* Shapes need to be easy do discrern
* Shapes need to embed the label of the node
* Shapes need to convay the "meaning", i.e., semantics of the code
* Shapes need to be aesthetically pleasing and lead to a nice drawing overall 

In the following discusssion we assume that the label is a rectangular area,
i.e., the bounding box of the rendered label text, of size w * h. The label is
always drawn horizontally and the symmetry axis of the label box coincides with
the symmetry axis of the node shape.


#### Array Nodes

Are drawn as a box of height h with two half-circles of diameter h attached on
the sides.

#### Map-Entry

A box of height h with two isocles right-angled triangles attached on the sides,
such that the right angles coincide with the bottom left and right corner of 
the label box and the catheti are of length h.

#### Map-Exit

Similar to Map-Entry except the right angles of the triangles coincide with the
top left and top right corners of the label box.

#### Tasklet

An octagon of which two parallel sides are defined by the label box (horizontal,
length w, distance h). The other sides are drawn such that their vertical
projection is h/3, the middle segment is vertical and the distance from the
middle segment to the label box is h/2.

#### Conflict Resolution

Conflict Resolution is drawn as a isosceles triangle with one horizontal leg
coinciding with the top side of the label box. We fix the angles in the left
and right corner to be 20 degrees. 


