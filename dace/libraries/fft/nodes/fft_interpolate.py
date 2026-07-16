# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFT-based grid-to-grid interpolation library node.

Mirrors Quantum ESPRESSO's ``fft_interpolate`` generic
(:file:`FFTXlib/src/fft_interfaces.f90`): given a function sampled on
one FFT grid, produce the same function sampled on a second FFT grid
by transforming to G-space, zero-padding / truncating to the new grid
size with the symmetric-split convention (keep low + high frequency
bins, drop the middle), and transforming back.  The lib node carries
``_inp`` / ``_out`` connectors and a ``dtype_kind`` Property
(``'real'`` -- QE's ``fft_interpolate_real`` -- vs ``'complex'`` --
QE's ``fft_interpolate_complex``).
"""
import itertools

import dace.library
import dace.properties
import dace
from dace import nodes, SDFG, SDFGState, dtypes, Memlet
from dace import transformation as xf


@dace.library.node
class FFTInterpolate(nodes.LibraryNode):
    """Fourier interpolation between two FFT grids.

    Inputs: ``_inp`` (function values on the source grid).
    Outputs: ``_out`` (function values on the target grid).

    Supports rank 1 / 2 / 3 inputs; the symmetric-split pad / truncate
    is applied independently on every axis, producing 2**N spectrum
    regions (2 for rank 1, 4 quadrants for rank 2, 8 octants for rank 3).
    """

    implementations = {}
    default_implementation = 'pure'

    dtype_kind = dace.properties.Property(dtype=str, default='complex', desc="'real' or 'complex'")

    def __init__(self, name, dtype_kind='complex', schedule=None, **kwargs):
        super().__init__(name, inputs={'_inp'}, outputs={'_out'}, schedule=schedule, **kwargs)
        self.dtype_kind = dtype_kind


def _get_input_and_output(state, node):
    """Resolve the lib node's IO connector data names."""
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn)
    out_edge = next(e for e in state.out_edges(node) if e.src_conn)
    return in_edge.data.data, out_edge.data.data


def _low_high_per_axis(indesc, outdesc):
    """Return ``(low, high)`` cut-points per axis for the symmetric-split copy.

    For each axis ``d``: ``low_d + high_d = min(nin_d, nout_d)``.  Defined
    via ``ceil`` / ``floor`` of the smaller extent so the central bin
    (Nyquist on the longer side) is dropped and the spectrum stays
    Hermitian-correct.
    """
    cuts = []
    for nin_d, nout_d in zip(indesc.shape, outdesc.shape):
        smaller = nin_d if nin_d <= nout_d else nout_d
        low_d = (smaller + 1) // 2
        high_d = smaller // 2
        cuts.append((low_d, high_d, nin_d, nout_d))
    return cuts


def _region_subset(part_per_axis, cuts, side):
    """Build the ``[start:end, ...]`` subset string for one corner.

    :param part_per_axis: tuple of ``'low'`` / ``'high'`` per axis.
    :param cuts: per-axis ``(low, high, nin, nout)`` cut-points.
    :param side: ``'in'`` (use ``nin``) or ``'out'`` (use ``nout``).
    """
    parts = []
    for part, (low_d, high_d, nin_d, nout_d) in zip(part_per_axis, cuts):
        n_d = nin_d if side == 'in' else nout_d
        if part == 'low':
            parts.append(f'0:{low_d}')
        else:
            parts.append(f'{n_d} - {high_d}:{n_d}')
    return ', '.join(parts)


def _region_index(part_per_axis, cuts, side, ivars):
    """Build the per-element index string for the copy tasklet."""
    parts = []
    for part, (low_d, high_d, nin_d, nout_d), iv in zip(part_per_axis, cuts, ivars):
        n_d = nin_d if side == 'in' else nout_d
        if part == 'low':
            parts.append(f'{iv}')
        else:
            parts.append(f'{n_d} - {high_d} + {iv}')
    return ', '.join(parts)


def _emit_fftw3_tasklet(state, sdfg, in_array, out_array, shape, complex_dtype, direction, envs):
    """Drop a self-contained ``fftw_plan_dft_*d`` Tasklet into ``state``.

    Connector names are prefixed with ``_fftw_`` so they do not collide
    with the outer SDFG's ``_inp`` / ``_out`` arrays (the lib-node-
    expansion connector name namespace overlaps with arrays).  The
    Tasklet pulls in the FFTW3 environment for the link-time
    dependency so the host expansion need not re-declare it.
    """
    from dace.codegen.targets import cpp  # avoid import loop

    if complex_dtype == dtypes.complex128:
        prefix, complex_t = 'fftw_', 'fftw_complex'
    else:
        prefix, complex_t = 'fftwf_', 'fftwf_complex'
    rank = len(shape)
    cdims = ', '.join(cpp.sym2cpp(s) for s in shape)
    code = f"""
    {{
        {prefix}plan __plan = {prefix}plan_dft_{rank}d({cdims},
            ({complex_t}*)_fftw_inp, ({complex_t}*)_fftw_out, {direction}, FFTW_ESTIMATE);
        {prefix}execute(__plan);
        {prefix}destroy_plan(__plan);
    }}
    """
    tasklet = nodes.Tasklet(f'fftw3_{direction.lower()}',
                            inputs={'_fftw_inp': dtypes.pointer(complex_dtype)},
                            outputs={'_fftw_out': dtypes.pointer(complex_dtype)},
                            code=code,
                            language=dtypes.Language.CPP)
    tasklet.environments = {e.full_class_path() for e in envs}
    state.add_node(tasklet)
    state.add_edge(state.add_read(in_array), None, tasklet, '_fftw_inp',
                   Memlet.from_array(in_array, sdfg.arrays[in_array]))
    state.add_edge(tasklet, '_fftw_out', state.add_write(out_array), None,
                   Memlet.from_array(out_array, sdfg.arrays[out_array]))


def _region_iter_ranges(part_per_axis, cuts, ivars):
    """Per-axis map iteration ranges for the copy tasklet.

    Each ``low`` side iterates ``0:low_d``; each ``high`` side iterates
    ``0:high_d``.  Returns ``{ivar: 'a:b'}`` ready for
    :meth:`add_mapped_tasklet`.
    """
    ranges = {}
    for part, (low_d, high_d, _, _), iv in zip(part_per_axis, cuts, ivars):
        size_d = low_d if part == 'low' else high_d
        ranges[iv] = f'0:{size_d}'
    return ranges


@dace.library.register_expansion(FFTInterpolate, 'pure')
class FFTInterpolatePure(xf.ExpandTransformation):
    """Backend-agnostic FFTInterpolate as compose(FFT -> pad/truncate -> IFFT).

    Forward FFT on the input grid, symmetric zero-pad in G-space (or
    truncate the high-frequency tail) to the output grid size, inverse
    FFT to produce the resampled signal.  Supports rank 1 / 2 / 3 --
    rank N applies the per-axis symmetric-split independently, producing
    2**N quadrants / octants.  Output magnitudes are normalised so the
    resampled signal matches the un-aliased continuum interpolant.
    """

    environments = []

    @staticmethod
    def expansion(node: 'FFTInterpolate', parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input_name, output_name = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input_name]
        outdesc = parent_sdfg.arrays[output_name]
        if len(indesc.shape) != len(outdesc.shape):
            raise ValueError(f'FFTInterpolate input/output rank mismatch: '
                             f'{len(indesc.shape)} vs {len(outdesc.shape)}')
        rank = len(indesc.shape)
        if rank not in (1, 2, 3):
            raise NotImplementedError(f'FFTInterpolate pure expansion supports rank 1/2/3 (got {rank})')

        from dace.libraries.fft.nodes import FFT, IFFT
        from dace.libraries.fft.environments import FFTW3 as FFTW3Env

        sdfg = SDFG(node.label + '_sdfg')
        in_inner = indesc.clone()
        in_inner.transient = False
        out_inner = outdesc.clone()
        out_inner.transient = False
        sdfg.add_datadesc('_inp', in_inner)
        sdfg.add_datadesc('_out', out_inner)

        complex_dtype = dtypes.complex128 if indesc.dtype in (dtypes.float64, dtypes.complex128) \
            else dtypes.complex64

        in_shape = list(indesc.shape)
        out_shape = list(outdesc.shape)
        in_size = 1
        for d in in_shape:
            in_size = in_size * d
        out_size = 1
        for d in out_shape:
            out_size = out_size * d

        sdfg.add_transient('__inp_c', in_shape, complex_dtype)
        sdfg.add_transient('__inp_spec', in_shape, complex_dtype)
        sdfg.add_transient('__padded_spec', out_shape, complex_dtype)
        sdfg.add_transient('__out_c', out_shape, complex_dtype)

        st_init = sdfg.add_state('s_init')
        st_fft = sdfg.add_state_after(st_init, 's_fft')
        st_ifft_state = None
        st_finalize = None

        # --- 1. Zero the padded spectrum ------------------------------------
        zero_ivars = [f'i{d}' for d in range(rank)]
        zero_ranges = {iv: f'0:{out_shape[d]}' for d, iv in enumerate(zero_ivars)}
        zero_index = ', '.join(zero_ivars)
        st_init.add_mapped_tasklet('zero_spec',
                                   zero_ranges, {},
                                   '__z = 0', {'__z': Memlet(f'__padded_spec[{zero_index}]')},
                                   external_edges=True)

        # --- 2. Cast input to complex ---------------------------------------
        cast_ivars = [f'i{d}' for d in range(rank)]
        cast_ranges = {iv: f'0:{in_shape[d]}' for d, iv in enumerate(cast_ivars)}
        cast_index = ', '.join(cast_ivars)
        st_fft.add_mapped_tasklet('cast_inp',
                                  cast_ranges, {'__x': Memlet(f'_inp[{cast_index}]')},
                                  '__y = __x', {'__y': Memlet(f'__inp_c[{cast_index}]')},
                                  external_edges=True)

        # --- 3. FFT on the input grid ---------------------------------------
        # Rank-1: use the pure DFT lib node (returns an SDFG that nests
        # cleanly).  Rank > 1: emit the FFTW3 plan-and-execute inline so
        # the resulting Tasklet's ``_inp`` / ``_out`` connectors don't
        # clash with the outer SDFG's same-named arrays (the FFTInterpolate
        # connectors).
        if rank == 1:
            fft_node = FFT('fft_inner')
            fft_node.implementation = 'pure'
            fft_node.factor = 1
            st_fft.add_node(fft_node)
            st_fft.add_edge(st_fft.add_read('__inp_c'), None, fft_node, '_inp',
                            Memlet.from_array('__inp_c', sdfg.arrays['__inp_c']))
            st_fft.add_edge(fft_node, '_out', st_fft.add_write('__inp_spec'), None,
                            Memlet.from_array('__inp_spec', sdfg.arrays['__inp_spec']))
        else:
            _emit_fftw3_tasklet(st_fft,
                                sdfg,
                                '__inp_c',
                                '__inp_spec',
                                in_shape,
                                complex_dtype,
                                direction='FFTW_FORWARD',
                                envs=[FFTW3Env])

        # --- 4. Symmetric-split spectrum copy -------------------------------
        # Per-axis (low, high) cuts.  For each combination of 'low'/'high'
        # across the ``rank`` axes (2**rank total: 2 endpoints in 1-D,
        # 4 quadrants in 2-D, 8 octants in 3-D) emit one copy tasklet that
        # transfers the matching subset.  Each axis's iteration range is
        # the (low_d, high_d) cut-point.
        cuts = _low_high_per_axis(indesc, outdesc)
        prev_state = st_fft
        for combo in itertools.product(('low', 'high'), repeat=rank):
            # Skip empty regions (a zero-width cut on any axis kills the whole
            # combination).
            if any((part == 'low' and low_d == 0) or (part == 'high' and high_d == 0)
                   for part, (low_d, high_d, _, _) in zip(combo, cuts)):
                continue
            st_copy = sdfg.add_state_after(prev_state, 's_copy_' + ''.join(p[0] for p in combo))
            ivars = [f'j{d}' for d in range(rank)]
            ranges = _region_iter_ranges(combo, cuts, ivars)
            in_idx = _region_index(combo, cuts, 'in', ivars)
            out_idx = _region_index(combo, cuts, 'out', ivars)
            st_copy.add_mapped_tasklet(f'copy_spec_{"".join(p[0] for p in combo)}',
                                       ranges, {'__x': Memlet(f'__inp_spec[{in_idx}]')},
                                       '__y = __x', {'__y': Memlet(f'__padded_spec[{out_idx}]')},
                                       external_edges=True)
            prev_state = st_copy

        # --- 5. IFFT on the output grid --------------------------------------
        # Same rank dispatch as the forward FFT: pure DFT lib node for
        # rank-1, inline FFTW3 Tasklet otherwise.  Both produce
        # un-normalised output; the 1/Nin scaling is applied as a
        # separate pass in the finalize state.
        st_ifft_state = sdfg.add_state_after(prev_state, 's_ifft')
        if rank == 1:
            ifft_node = IFFT('ifft_inner')
            ifft_node.implementation = 'pure'
            ifft_node.factor = 1
            st_ifft_state.add_node(ifft_node)
            st_ifft_state.add_edge(st_ifft_state.add_read('__padded_spec'), None, ifft_node, '_inp',
                                   Memlet.from_array('__padded_spec', sdfg.arrays['__padded_spec']))
            st_ifft_state.add_edge(ifft_node, '_out', st_ifft_state.add_write('__out_c'), None,
                                   Memlet.from_array('__out_c', sdfg.arrays['__out_c']))
        else:
            _emit_fftw3_tasklet(st_ifft_state,
                                sdfg,
                                '__padded_spec',
                                '__out_c',
                                out_shape,
                                complex_dtype,
                                direction='FFTW_BACKWARD',
                                envs=[FFTW3Env])

        # --- 6. Project to output dtype + apply 1/Nin scaling ----------------
        st_finalize = sdfg.add_state_after(st_ifft_state, 's_finalize')
        fin_ivars = [f'i{d}' for d in range(rank)]
        fin_ranges = {iv: f'0:{out_shape[d]}' for d, iv in enumerate(fin_ivars)}
        fin_index = ', '.join(fin_ivars)
        is_real = (node.dtype_kind == 'real')
        # ``inv_nin`` divides the IFFT-of-padded-spectrum by the input
        # element count.  Pre-compute as a Python float so it lowers
        # cleanly into the tasklet body even when ``in_size`` is a
        # symbolic SDFG expression.
        inv_nin_expr = f'(1.0 / ({in_size}))'
        if is_real:
            tasklet = nodes.Tasklet('cast_out_real',
                                    inputs={'__x'},
                                    outputs={'__y'},
                                    code=f'__y = (__x * {inv_nin_expr}).real();',
                                    language=dtypes.Language.CPP)
            map_entry, map_exit = st_finalize.add_map('cast_real_map', fin_ranges)
            st_finalize.add_node(tasklet)
            st_finalize.add_memlet_path(st_finalize.add_read('__out_c'),
                                        map_entry,
                                        tasklet,
                                        dst_conn='__x',
                                        memlet=Memlet(f'__out_c[{fin_index}]'))
            st_finalize.add_memlet_path(tasklet,
                                        map_exit,
                                        st_finalize.add_write('_out'),
                                        src_conn='__y',
                                        memlet=Memlet(f'_out[{fin_index}]'))
        else:
            st_finalize.add_mapped_tasklet('cast_out_complex',
                                           fin_ranges, {'__x': Memlet(f'__out_c[{fin_index}]')},
                                           f'__y = __x * {inv_nin_expr}', {'__y': Memlet(f'_out[{fin_index}]')},
                                           external_edges=True)
        return sdfg
