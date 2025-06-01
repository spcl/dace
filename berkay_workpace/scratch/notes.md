# Preprocess notes

**1. Application of "CopyToMap" transformation for certain cases of GPU<->GPU copy that cannot be done using a single copy command:**

So I left it there because I did not in which kind of situation this will be needed, i.e.
a concrete example/situation. It seems to be interwined with the _emit_copy() function, where similar checks
are also performed (i.e. nobody cleaned this, the checks there actually make no sense after preprocessing).


**2. _compute_pool_release() Function:**

I left it because it looks useful - helps to free memory. But it seems like the actual freeing is not performed anymore
in my code, so I maybe should remove it as well? Kind of happens if 

**3. _compute_cudastreams()**

Also remove it for now? I mean, stream allocation/deallocation are not handled anyways.