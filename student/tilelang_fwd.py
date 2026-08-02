import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
LOG2E = 1.4426950408889634


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_gdn_forward(H, Hg, qk_dtype, gate_dtype, accum_dtype, block_DV):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")

    G = H // Hg
    DK = HEAD_DIM_K
    DV = HEAD_DIM_V
    block_S = CHUNK_SIZE
    SCALE = DK ** -0.5
    NDV = DV // block_DV

    qk_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, block_S)
    state_shape = (batch_size, H, DK, DV)
    output_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        v: T.Tensor(v_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        A: T.Tensor(a_shape, dtype=qk_dtype),
        initial_state: T.Tensor(state_shape, dtype=accum_dtype),
        output: T.Tensor(output_shape, dtype=qk_dtype),
        final_state: T.Tensor(state_shape, dtype=accum_dtype),
        has_init_state: T.int32,
        num_chunks: T.int32,
    ):
        with T.Kernel(batch_size * H * NDV, threads=512) as (block,):
            bv = block % NDV
            bbh = block // NDV
            bb = bbh // H
            bh = bbh % H
            bhg = bh // G

            DV_start = bv * block_DV

            num_iters = T.ceildiv(num_tokens, block_S)
            num_unmasked_iters = num_tokens // block_S

            # Double-buffered shared memory
            q_shared = T.alloc_shared((2, block_S, DK), dtype=qk_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qk_dtype)
            v_shared = T.alloc_shared((2, block_S, block_DV), dtype=qk_dtype)
            a_shared = T.alloc_shared((2, block_S, block_S), dtype=qk_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            # Single-buffered intermediates
            o_shared = T.alloc_shared((2, block_S, block_DV), dtype=qk_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qk_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qk_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qk_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # Fragments
            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            # Barriers - matching FlashQLA's protocol
            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)
            bar_o = T.alloc_barrier(arrive_count=128)
            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            _bar_2 = T.alloc_barrier(arrive_count=128)
            bar_4 = T.alloc_barrier(arrive_count=128)
            bar_5 = T.alloc_barrier(arrive_count=416)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            if tx < 128:
                # === Consumer S: manages state ===
                T.set_max_nreg(160, 1)

                if has_init_state != 0:
                    T.copy(initial_state[bb, bh, 0:DK, DV_start:DV_start + block_DV], h_fragment)
                else:
                    T.clear(h_fragment)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    g_last_local[0] = g_exp_shared[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)

                    T.barrier_wait(bar_5, i_s % 2)
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        vn_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(data_is_free[i_s % 2])

                T.copy(h_fragment, final_state[bb, bh, 0:DK, DV_start:DV_start + block_DV])

            elif tx < 256:
                # === Consumer V: computes U, W, Vd, V' ===
                T.set_max_nreg(128, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    # Compute g_exp and g_rev
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[i_s % 2, j_s] * LOG2E)
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_shared[i_s % 2, block_S - 1] - g_shared[i_s % 2, j_s]) * LOG2E
                        )
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        h_shared,
                        u_fragment,
                        clear_accum=True,
                    )
                    # W = beta * (V - exp_g * U)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] = b_shared[i_s % 2, j_s] * (
                            v_shared[i_s % 2, j_s, j_v]
                            - g_exp_shared[j_s] * u_fragment[j_s, j_v]
                        )
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_shared[i_s % 2, j_s, j_v] = T.cast(u_fragment[j_s, j_v], qk_dtype)

                    # Vd = A @ W  (A from kkt_solve already includes decay+beta)
                    T.gemm(
                        a_shared[i_s % 2, :, :],
                        v_shared[i_s % 2, :, :],
                        v_fragment,
                        clear_accum=True,
                    )
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)

                    # V' = g_rev * Vd
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)

                    T.barrier_wait(bar_5, i_s % 2)
                    T.barrier_arrive(data_is_free[i_s % 2])

            elif tx < 384:
                # === Consumer O: computes P, gate, output ===
                T.set_max_nreg(128, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    # P = Q @ K^T
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # Gate matrix G = tril(exp2((g[i] - g[j]) * LOG2E))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = (
                            g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * LOG2E)
                        else:
                            g_fragment[j_s, j_t] = 0
                    # P = scale * G * P  (element-wise)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= SCALE * g_fragment[j_s, j_t]

                    T.barrier_wait(bar_1, i_s % 2)
                    # O = Q @ S
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        h_shared,
                        o_fragment,
                        clear_accum=True,
                    )
                    # O *= scale * exp_g
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        o_fragment[j_s, j_v] *= SCALE * g_exp_shared[j_s]
                    # Store P to p_shared
                    T.copy(p_fragment, p_shared)
                    T.barrier_wait(bar_4, i_s % 2)
                    # O += P @ Vd
                    T.gemm(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)

                    T.barrier_wait(bar_5, i_s % 2)
                    T.copy(o_fragment, o_shared[i_s % 2, :, :])
                    T.barrier_arrive(data_is_free[i_s % 2])

                T.barrier_arrive(bar_o)

            else:
                # === Producers (tx 384-511) ===
                T.set_max_nreg(32, 0)

                if tx < 384 + 32:
                    # Producer QK
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S

                        if right <= num_tokens:
                            T.tma_copy(
                                q[bb, left:right, bhg, 0:DK],
                                q_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            T.tma_copy(
                                k[bb, left:right, bhg, 0:DK],
                                k_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < num_tokens:
                                    q_shared[i_s % 2, j_s, j_k] = q[bb, left + j_s, bhg, j_k]
                                else:
                                    q_shared[i_s % 2, j_s, j_k] = 0
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < num_tokens:
                                    k_shared[i_s % 2, j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                                else:
                                    k_shared[i_s % 2, j_s, j_k] = 0
                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 64:
                    # Producer V+beta
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S

                        if right <= num_tokens:
                            T.tma_copy(
                                v[bb, left:right, bh, DV_start:DV_start + block_DV],
                                v_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % 2, j_s] = beta[bb, left + j_s, bh]
                        else:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                if left + j_s < num_tokens:
                                    v_shared[i_s % 2, j_s, j_v] = v[bb, left + j_s, bh, DV_start + j_v]
                                else:
                                    v_shared[i_s % 2, j_s, j_v] = 0
                            for j_s in T.Parallel(block_S):
                                if left + j_s < num_tokens:
                                    b_shared[i_s % 2, j_s] = beta[bb, left + j_s, bh]
                                else:
                                    b_shared[i_s % 2, j_s] = 0
                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 96:
                    # Producer A+g
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S

                        if right <= num_tokens:
                            T.tma_copy(
                                A[bb, left:right, bh, 0:block_S],
                                a_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % 2, j_s] = g[bb, left + j_s, bh]
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < num_tokens:
                                    a_shared[i_s % 2, j_s, j_t] = A[bb, left + j_s, bh, j_t]
                                else:
                                   a_shared[i_s % 2, j_s, j_t] = 0
                            for j_s in T.Parallel(block_S):
                                if left + j_s < num_tokens:
                                    g_shared[i_s % 2, j_s] = g[bb, left + j_s, bh]
                                else:
                                    g_shared[i_s % 2, j_s] = g[bb, num_tokens - 1, bh]
                        T.barrier_arrive(data_is_ready[i_s % 2])

                else:
                    # Output storer (tx 480-511)
                    for i_s in T.serial(num_unmasked_iters):
                        right_store = i_s * block_S
                        left_store = right_store - block_S

                        T.barrier_arrive(bar_0)
                        T.barrier_wait(bar_0, i_s % 2)
                        if i_s > 0:
                            T.copy(
                                o_shared[(i_s - 1) % 2, :, :],
                                output[bb, left_store:right_store, bh, DV_start:DV_start + block_DV],
                            )
                        T.barrier_arrive(bar_5)
                        T.barrier_wait(bar_1, i_s % 2)

                    if num_unmasked_iters < num_iters:
                        seq_split = num_unmasked_iters * block_S
                        right_store = seq_split
                        left_store = right_store - block_S

                        T.barrier_arrive(bar_0)
                        T.barrier_wait(bar_0, num_unmasked_iters % 2)
                        if num_unmasked_iters > 0:
                            T.copy(
                                o_shared[(num_unmasked_iters - 1) % 2, :, :],
                                output[bb, left_store:right_store, bh, DV_start:DV_start + block_DV],
                            )
                        T.barrier_arrive(bar_5)
                        T.barrier_wait(bar_1, num_unmasked_iters % 2)

                    # Store last chunk output
                    seq_split = (num_iters - 1) * block_S
                    T.barrier_wait(bar_o, 0)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        if seq_split + j_s < num_tokens:
                            output[bb, seq_split + j_s, bh, DV_start + j_v] = o_shared[(num_iters - 1) % 2, j_s, j_v]

    return kernel


def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T_len, Hq, _ = q.shape
    _, _, Hv, _ = v.shape

    output = torch.empty((B, T_len, Hv, HEAD_DIM_V), dtype=q.dtype, device=q.device)
    final_state = torch.empty((B, Hv, HEAD_DIM_K, HEAD_DIM_V), dtype=torch.float32, device=q.device)

    num_chunks = (T_len + CHUNK_SIZE - 1) // CHUNK_SIZE

    if initial_state is None:
        initial_state = torch.zeros((B, Hv, HEAD_DIM_K, HEAD_DIM_V), dtype=torch.float32, device=q.device)
        has_init = 0
    else:
        has_init = 1

    # Force block_DV=128 for all cases
    dv_tile = 128

    kernel = tilelang_gdn_forward(
        Hv,
        Hq,
       qk_dtype=str(q.dtype).replace("torch.", ""),
       gate_dtype=str(g_cumsum.dtype).replace("torch.", ""),
       accum_dtype="float32",
        block_DV=dv_tile,
    )
    kernel(q, k, v, g_cumsum, beta, A, initial_state, output, final_state, has_init, num_chunks)

    return output, final_state
