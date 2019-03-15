/*
* Copyright 1997, Regents of the University of Minnesota
*
* Extracted from Metis io.c and some of it's headers http://glaros.dtc.umn.edu/gkhome/metis/metis/download
*
* This file contains routines related to I/O
*
* Started 8/28/94
* George
*
* $Id: io.c 11932 2012-05-10 18:18:23Z dominique $
*
*/

#ifndef PARSER_H
#define PARSER_H


#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <cstddef>
#include "../../../dace/runtime/include/dace/dace.h"

#define SCIDX  "ld"
#define PRIDX  "I32d"

typedef uint32_t idx_t;
typedef float real_t;
typedef uint64_t size_t;

/*************************************************************************/
/*! This data structure stores cut-based k-way refinement info about an
adjacent subdomain for a given vertex. */
/*************************************************************************/
typedef struct cnbr_t {
    idx_t pid;            /*!< The partition ID */
    idx_t ed;             /*!< The sum of the weights of the adjacent edges
                          that are incident on pid */
} cnbr_t;


/*************************************************************************/
/*! The following data structure stores holds information on degrees for k-way
partition */
/*************************************************************************/
typedef struct ckrinfo_t {
    idx_t id;              /*!< The internal degree of a vertex (sum of weights) */
    idx_t ed;                /*!< The total external degree of a vertex */
    idx_t nnbrs;              /*!< The number of neighboring subdomains */
    idx_t inbr;            /*!< The index in the cnbr_t array where the nnbrs list
                           of neighbors is stored */
} ckrinfo_t;


/*************************************************************************/
/*! This data structure stores volume-based k-way refinement info about an
adjacent subdomain for a given vertex. */
/*************************************************************************/
typedef struct vnbr_t {
    idx_t pid;            /*!< The partition ID */
    idx_t ned;            /*!< The number of the adjacent edges
                          that are incident on pid */
    idx_t gv;             /*!< The gain in volume achieved by moving the
                          vertex to pid */
} vnbr_t;


/*************************************************************************/
/*! The following data structure holds information on degrees for k-way
vol-based partition */
/*************************************************************************/
typedef struct vkrinfo_t {
    idx_t nid;             /*!< The internal degree of a vertex (count of edges) */
    idx_t ned;                /*!< The total external degree of a vertex (count of edges) */
    idx_t gv;                /*!< The volume gain of moving that vertex */
    idx_t nnbrs;              /*!< The number of neighboring subdomains */
    idx_t inbr;            /*!< The index in the vnbr_t array where the nnbrs list
                           of neighbors is stored */
} vkrinfo_t;


/*************************************************************************/
/*! The following data structure holds information on degrees for k-way
partition */
/*************************************************************************/
typedef struct nrinfo_t {
    idx_t edegrees[2];
} nrinfo_t;


/*************************************************************************/
/*! This data structure holds a graph */
/*************************************************************************/
typedef struct graph_t {
    idx_t nvtxs, nedges;    /* The # of vertices and edges in the graph */
    idx_t ncon;        /* The # of constrains */
    idx_t *xadj;        /* Pointers to the locally stored vertices */
    idx_t *vwgt;        /* Vertex weights */
    idx_t *vsize;        /* Vertex sizes for min-volume formulation */
    idx_t *adjncy;        /* Array that stores the adjacency lists of nvtxs */
    idx_t *adjwgt;        /* Array that stores the weights of the adjacency lists */

    idx_t *tvwgt;         /* The sum of the vertex weights in the graph */
    real_t *invtvwgt;     /* The inverse of the sum of the vertex weights in the graph */

    bool readvw; // did the source file contain vertex weights
    bool readew; // did the source file contain edge weights

    /* These are to keep track control if the corresponding fields correspond to
    application or library memory */
    int free_xadj, free_vwgt, free_vsize, free_adjncy, free_adjwgt;

    idx_t *label;

    idx_t *cmap;

    /* Partition parameters */
    idx_t mincut, minvol;
    idx_t *where, *pwgts;
    idx_t nbnd;
    idx_t *bndptr, *bndind;

    /* Bisection refinement parameters */
    idx_t *id, *ed;

    /* K-way refinement parameters */
    ckrinfo_t *ckrinfo;   /*!< The per-vertex cut-based refinement info */
    vkrinfo_t *vkrinfo;   /*!< The per-vertex volume-based refinement info */

    /* Node refinement information */
    nrinfo_t *nrinfo;

    struct graph_t *coarser, *finer;
} graph_t;

FILE *gk_fopen(const char *fname, const char *mode, const char *msg);
void gk_fclose(FILE *fp);
ptrdiff_t gk_getline(char **lineptr, size_t *n, FILE *stream);

DACE_EXPORTED graph_t *ReadGraph(char* filename);
DACE_EXPORTED graph_t *ReadGraphGR(char* filename);

DACE_EXPORTED void FreeGraph(graph_t *r_graph);

#endif // PARSER_H
