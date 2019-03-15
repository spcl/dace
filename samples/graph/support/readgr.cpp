/*
* Copyright 1997, Regents of the University of Minnesota
*
* Extracted from Metis io.c http://glaros.dtc.umn.edu/gkhome/metis/metis/download
*
* This file contains routines related to I/O
*
* Started 8/28/94
* George
*
* $Id: io.c 11932 2012-05-10 18:18:23Z dominique $
*
*/

#include "readgr.h"


//#include <crtdefs.h>
#include <cstddef>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cassert>

/*************************************************************************/
/*! This function initializes a graph_t data structure */
/*************************************************************************/
void InitGraph(graph_t *graph)
{
    memset((void *)graph, 0, sizeof(graph_t));

    /* graph size constants */
    graph->nvtxs = -1;
    graph->nedges = -1;
    graph->ncon = -1;
    graph->mincut = -1;
    graph->minvol = -1;
    graph->nbnd = -1;

    /* memory for the graph structure */
    graph->xadj = NULL;
    graph->vwgt = NULL;
    graph->vsize = NULL;
    graph->adjncy = NULL;
    graph->adjwgt = NULL;
    graph->label = NULL;
    graph->cmap = NULL;
    graph->tvwgt = NULL;
    graph->invtvwgt = NULL;

    graph->readvw = false;
    graph->readew = false;

    /* by default these are set to true, but the can be explicitly changed afterwards */
    graph->free_xadj = 1;
    graph->free_vwgt = 1;
    graph->free_vsize = 1;
    graph->free_adjncy = 1;
    graph->free_adjwgt = 1;


    /* memory for the partition/refinement structure */
    graph->where = NULL;
    graph->pwgts = NULL;
    graph->id = NULL;
    graph->ed = NULL;
    graph->bndptr = NULL;
    graph->bndind = NULL;
    graph->nrinfo = NULL;
    graph->ckrinfo = NULL;
    graph->vkrinfo = NULL;

    /* linked-list structure */
    graph->coarser = NULL;
    graph->finer = NULL;
}

/*************************************************************************/
/*! This function creates and initializes a graph_t data structure */
/*************************************************************************/
graph_t *CreateGraph(void)
{
    graph_t *graph;

    graph = (graph_t *)malloc(sizeof(graph_t));

    InitGraph(graph);

    return graph;
}

/*************************************************************************/
/*! This function deallocates any memory stored in a graph */
/*************************************************************************/
void FreeGraph(graph_t *graph)
{
    
    /* free graph structure */
    if (graph->free_xadj)
        free((void *)graph->xadj);
    if (graph->free_vwgt)
        free((void *)graph->vwgt);
    if (graph->free_vsize)
        free((void *)graph->vsize);
    if (graph->free_adjncy)
        free((void *)graph->adjncy);
    if (graph->free_adjwgt)
        free((void *)graph->adjwgt);

    /* free partition/refinement structure */
    //FreeRData(graph);

    free((void *)graph->tvwgt);
    free((void *)graph->invtvwgt);
    free((void *)graph->label);
    free((void *)graph->cmap);
    free((void *)graph);

}

//static int exit_on_error = 1;

/*************************************************************************/
/*! This function prints an error message and exits
*/
/*************************************************************************/
void errexit(const char *f_str, ...)
{
    va_list argp;

    va_start(argp, f_str);
    vfprintf(stderr, f_str, argp);
    va_end(argp);

    if (strlen(f_str) == 0 || f_str[strlen(f_str) - 1] != '\n')
        fprintf(stderr, "\n");
    fflush(stderr);

    if (/*exit_on_error*/ 1)
        exit(-2);

    /* abort(); */
}

/*************************************************************************
* This function opens a file
**************************************************************************/
FILE *gk_fopen(const char *fname, const char *mode, const char *msg)
{
    FILE *fp;
    char errmsg[8192];

    fp = fopen(fname, mode);
    if (fp != NULL)
        return fp;

    sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    errexit("Failed on gk_fopen()\n");

    return NULL;
}


/*************************************************************************
* This function closes a file
**************************************************************************/
void gk_fclose(FILE *fp)
{
    fclose(fp);
}


/*************************************************************************/
/*! This function is the GKlib implementation of glibc's getline()
function.
\returns -1 if the EOF has been reached, otherwise it returns the
number of bytes read.
*/
/*************************************************************************/
ptrdiff_t gk_getline(char **lineptr, size_t *n, FILE *stream)
{
#ifdef HAVE_GETLINE
    return getline(lineptr, n, stream);
#else
    size_t i;
    int ch;

    if (feof(stream))
        return -1;

    /* Initial memory allocation if *lineptr is NULL */
    if (*lineptr == NULL || *n == 0) {
        *n = 1024;
        *lineptr = (char*)malloc((*n)*sizeof(char));
    }

    /* get into the main loop */
    i = 0;
    while ((ch = getc(stream)) != EOF) {
        (*lineptr)[i++] = (char)ch;

        /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
        if (i + 1 == *n) {
            *n = 2 * (*n);
            *lineptr = (char*)realloc(*lineptr, (*n)*sizeof(char));
        }

        if (ch == '\n')
            break;
    }
    (*lineptr)[i] = '\0';

    return (i == 0 ? -1 : i);
#endif
}

/*************************************************************************/
/*! This function reads in a sparse graph */
/*************************************************************************/
graph_t *ReadGraph(char* filename)
{
    idx_t i, j, k, l, fmt, ncon, nfields, readew, readvw, readvs, edge, ewgt;
    idx_t *xadj, *adjncy, *vwgt, *adjwgt, *vsize;
    char *line = NULL, fmtstr[256], *curstr, *newstr;
    size_t lnlen = 0;
    FILE *fpin;
    graph_t *graph;

    graph = CreateGraph();

    fpin = gk_fopen(filename, "r", "ReadGRaph: Graph");

    /* Skip comment lines until you get to the first valid line */
    do {
        if (gk_getline(&line, &lnlen, fpin) == -1)
            errexit("Premature end of input file: file: %s\n", filename);
    } while (line[0] == '%');


    fmt = ncon = 0;
    nfields = sscanf(line, "%" SCIDX " %" SCIDX " %" SCIDX " %" SCIDX,
        &(graph->nvtxs), &(graph->nedges), &fmt, &ncon);

    if (nfields < 2)
        errexit("The input file does not specify the number of vertices and edges.\n");

    if (graph->nvtxs <= 0 || graph->nedges <= 0)
        errexit("The supplied nvtxs:%" PRIDX " and nedges:%" PRIDX " must be positive.\n",
        graph->nvtxs, graph->nedges);

    if (fmt > 111)
        errexit("Cannot read this type of file format [fmt=%" PRIDX "]!\n", fmt);

    sprintf(fmtstr, "%03" PRIDX , fmt % 1000);
    readvs = (fmtstr[0] == '1');
    readvw = (fmtstr[1] == '1');
    readew = (fmtstr[2] == '1');

    graph->readew = readew;

    /*printf("%s %" PRIDX " %" PRIDX " %" PRIDX "\n", fmtstr, readvs, readvw, readew); */


    if (ncon > 0 && !readvw)
        errexit(
        "------------------------------------------------------------------------------\n"
        "***  I detected an error in your input file  ***\n\n"
        "You specified ncon=%" PRIDX ", but the fmt parameter does not specify vertex weights\n"
        "Make sure that the fmt parameter is set to either 10 or 11.\n"
        "------------------------------------------------------------------------------\n", ncon);

    graph->nedges *= 2;
    ncon = graph->ncon = (ncon == 0 ? 1 : ncon);

    xadj = graph->xadj = (idx_t*)malloc((graph->nvtxs + 1) * sizeof(idx_t));
    memset((void *)xadj, 0, (graph->nvtxs + 1) * sizeof(idx_t));

    adjncy = graph->adjncy = (idx_t*)malloc((graph->nedges) * sizeof(idx_t));

    vwgt = graph->vwgt = (idx_t*)malloc((ncon*graph->nvtxs) * sizeof(idx_t));
    memset((void *)vwgt, 1, (ncon*graph->nvtxs) * sizeof(idx_t));

    adjwgt = graph->adjwgt = (idx_t*)malloc((graph->nedges) * sizeof(idx_t));
    memset((void *)adjwgt, 1, (graph->nedges) * sizeof(idx_t));

    vsize = graph->vsize = (idx_t*)malloc((graph->nvtxs) * sizeof(idx_t));
    memset((void *)vsize, 1, (graph->nvtxs) * sizeof(idx_t));

    /*----------------------------------------------------------------------
    * Read the sparse graph file
    *---------------------------------------------------------------------*/
    for (xadj[0] = 0, k = 0, i = 0; i < graph->nvtxs; i++) {
        do {
            if (gk_getline(&line, &lnlen, fpin) == -1)
                errexit("Premature end of input file while reading vertex %" PRIDX ".\n", i + 1);
        } while (line[0] == '%');

        curstr = line;
        newstr = NULL;

        /* Read vertex sizes */
        if (readvs) {
            vsize[i] = strtol(curstr, &newstr, 10);
            if (newstr == curstr)
                errexit("The line for vertex %" PRIDX " does not have vsize information\n", i + 1);
            if (vsize[i] < 0)
                errexit("The size for vertex %" PRIDX " must be >= 0\n", i + 1);
            curstr = newstr;
        }


        /* Read vertex weights */
        if (readvw) {
            for (l = 0; l < ncon; l++) {
                vwgt[i*ncon + l] = strtol(curstr, &newstr, 10);
                if (newstr == curstr)
                    errexit("The line for vertex %" PRIDX " does not have enough weights "
                    "for the %" PRIDX " constraints.\n", i + 1, ncon);
                if (vwgt[i*ncon + l] < 0)
                    errexit("The weight vertex %" PRIDX " and constraint %" PRIDX " must be >= 0\n", i + 1, l);
                curstr = newstr;
            }
        }

        while (1) {
            edge = strtol(curstr, &newstr, 10);
            if (newstr == curstr)
                break; /* End of line */
            curstr = newstr;

            if (edge < 1 || edge > graph->nvtxs)
                errexit("Edge %" PRIDX " for vertex %" PRIDX " is out of bounds\n", edge, i + 1);

            ewgt = 1;
            if (readew) {
                ewgt = strtol(curstr, &newstr, 10);
                if (newstr == curstr)
                    errexit("Premature end of line for vertex %" PRIDX "\n", i + 1);
                if (ewgt <= 0)
                    errexit("The weight (%" PRIDX ") for edge (%" PRIDX ", %" PRIDX ") must be positive.\n",
                    ewgt, i + 1, edge);
                curstr = newstr;
            }

            if (k == graph->nedges)
                errexit("There are more edges in the file than the %" PRIDX " specified.\n",
                graph->nedges / 2);

            adjncy[k] = edge - 1;
            adjwgt[k] = ewgt;
            k++;
        }
        xadj[i + 1] = k;
    }
    gk_fclose(fpin);

    if (k != graph->nedges) {
        printf("------------------------------------------------------------------------------\n");
        printf("***  I detected an error in your input file  ***\n\n");
        printf("In the first line of the file, you specified that the graph contained\n"
            "%" PRIDX " edges. However, I only found %" PRIDX " edges in the file.\n",
            graph->nedges / 2, k / 2);
        if (2 * k == graph->nedges) {
            printf("\n *> I detected that you specified twice the number of edges that you have in\n");
            printf("    the file. Remember that the number of edges specified in the first line\n");
            printf("    counts each edge between vertices v and u only once.\n\n");
        }
        printf("Please specify the correct number of edges in the first line of the file.\n");
        printf("------------------------------------------------------------------------------\n");
        exit(0);
    }

    free((void *)line);

    return graph;
}

#ifdef WIN32
// Windows "host" byte order is little endian
static inline uint64_t le64toh(uint64_t x) {
    return x;
}

#endif

/*************************************************************************/
/*! This function reads in a sparse graph */
/*************************************************************************/
graph_t *ReadGraphGR(char* filename)
{
    idx_t *xadj, *adjncy, *vwgt, *adjwgt, *vsize;
    FILE *fpin;
    graph_t *graph;

    graph = CreateGraph();

    fpin = gk_fopen(filename, "r", "ReadGraphGR: Graph");

    size_t read;
    uint64_t x[4];
    if (fread(x, sizeof(uint64_t), 4, fpin) != 4) {
        errexit("Unable to read header\n");
    }

    if (x[0] != 1) /* version */
        errexit("Unknown file version\n");

    uint64_t sizeEdgeTy = le64toh(x[1]);
    graph->nvtxs = x[2];
    graph->nedges = x[3];

    printf("%s has %lu nodes and %lu edges\n", filename, graph->nvtxs, graph->nedges);

    xadj = graph->xadj = (idx_t*)calloc((graph->nvtxs + 1), sizeof(idx_t));
    adjncy = graph->adjncy = (idx_t*)calloc((graph->nedges), sizeof(uint32_t));

    vwgt = graph->vwgt = (idx_t*)calloc((0 * graph->nvtxs), sizeof(idx_t));  // file doesn't store node weights though.
    graph->readvw = false;

    adjwgt = graph->adjwgt = (idx_t*)calloc((graph->nedges), sizeof(idx_t));
    vsize = graph->vsize = (idx_t*)calloc((graph->nvtxs), sizeof(idx_t));

    assert(xadj != NULL);
    assert(adjncy != NULL);
    //assert(vwgt != NULL);
    assert(adjwgt != NULL);

    if (sizeof(idx_t) == sizeof(uint64_t)) {
        read = fread(xadj + 1, sizeof(idx_t), graph->nvtxs, fpin); // This is little-endian data
        if (read < graph->nvtxs)
            errexit("Error: Partial read of node data\n");
        fprintf(stderr, "read %llu nodes\n", graph->nvtxs);
    }
    else {
        for (int i = 0; i < graph->nvtxs; i++) {
            uint64_t rs;
            if (fread(&rs, sizeof(uint64_t), 1, fpin) != 1) {
                errexit("Error: Unable to read node data\n");
            }
            xadj[i + 1] = rs;
        }
    }

    // edges are 32-bit

    if (sizeof(idx_t) == sizeof(uint32_t)) {
        read = fread(adjncy, sizeof(idx_t), graph->nedges, fpin); // This is little-endian data
        if (read < graph->nedges)
            errexit("Error: Partial read of edge destinations\n");

        fprintf(stderr, "read %llu edges\n", graph->nedges);
    }
    else {
        assert(false && "Not implemented"); /* need to convert sizes when reading */
    }

    if (sizeEdgeTy) {
        if (graph->nedges % 2)
            if (fseek(fpin, 4, SEEK_CUR) != 0)  // skip
                errexit("Error when seeking\n");

        if (sizeof(idx_t) == sizeof(uint32_t)) {
            read = fread(adjwgt, sizeof(idx_t), graph->nedges, fpin); // This is little-endian data
            graph->readew = true;
            if (read < graph->nedges)
                errexit("Error: Partial read of edge data\n");

            fprintf(stderr, "read data for %llu edges\n", graph->nedges);
        }
        else {
            assert(false && "Not implemented"); /* need to convert sizes when reading */
        }
    }

    return graph;
}
