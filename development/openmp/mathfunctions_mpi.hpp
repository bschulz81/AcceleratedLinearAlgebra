#ifndef MATHFUNCTIONSMPI_HPP
#define MATHFUNCTIONSMPI_HPP

#include "datablock.h"
#include "datablock_mpifunctions.h"

bool Math_Functions_MPI::matrix_distribution_is_summa_compatible(
    ptrdiff_t grid_r,
    ptrdiff_t grid_c,
    ptrdiff_t Pr,
    ptrdiff_t Pc)
{
    const ptrdiff_t P = Pr * Pc;

    std::vector<ptrdiff_t> counts(P, 0);

    for(ptrdiff_t bi = 0; bi < grid_r; ++bi)
    {
        for(ptrdiff_t bj = 0; bj < grid_c; ++bj)
        {
            ptrdiff_t prow = bi % Pr;
            ptrdiff_t pcol = bj % Pc;

            ptrdiff_t rank = prow * Pc + pcol;

            counts[rank]++;
        }
    }

    ptrdiff_t min_blocks = counts[0];
    ptrdiff_t max_blocks = counts[0];

    for(ptrdiff_t c : counts)
    {
        min_blocks = std::min(min_blocks, c);
        max_blocks = std::max(max_blocks, c);
    }

    return (max_blocks <= 1) || (min_blocks > 0);
}



template<typename T>
MPI_Comm Math_Functions_MPI::create_summa_communicator(ptrdiff_t br,ptrdiff_t bc,
        const DataBlock<T>* A,const DataBlock<T>* B,const DataBlock<T>* C,
        int rootrank,MPI_Comm parent, SummaGridPolicy policy,bool printgrid)
{
    int world_rank, world_size;

    MPI_Comm_rank(parent, &world_rank);
    MPI_Comm_size(parent, &world_size);

    struct MatrixInfo
    {
        ptrdiff_t rows;
        ptrdiff_t cols;
    };

    MatrixInfo mats[3];

    if(world_rank == 0)
    {
        mats[0] = {A->dpextents[0], A->dpextents[1]};
        mats[1] = {B->dpextents[0], B->dpextents[1]};
        mats[2] = {C->dpextents[0], C->dpextents[1]};
    }

    MPI_Bcast(mats, sizeof(mats), MPI_BYTE, 0, MPI_COMM_WORLD);

    struct GridStats
    {
        ptrdiff_t min_blocks;
        ptrdiff_t max_blocks;
    };

    auto analyse_distribution =
        [&](ptrdiff_t grid_r,
            ptrdiff_t grid_c,
            int Pr,
            int Pc) -> GridStats
    {
        std::vector<ptrdiff_t> counts(Pr * Pc, 0);

        for(ptrdiff_t bi = 0; bi < grid_r; ++bi)
        {
            for(ptrdiff_t bj = 0; bj < grid_c; ++bj)
            {
                ptrdiff_t prow = bi % Pr;
                ptrdiff_t pcol = bj % Pc;

                counts[prow * Pc + pcol]++;
            }
        }

        GridStats s;

        s.min_blocks = counts[0];
        s.max_blocks = counts[0];

        for(ptrdiff_t c : counts)
        {
            s.min_blocks = std::min(s.min_blocks, c);
            s.max_blocks = std::max(s.max_blocks, c);
        }

        return s;
    };

    auto matrix_is_dense =
        [](const GridStats& s)
    {
        return s.min_blocks > 0;
    };

    auto matrix_is_compatible =
        [](const GridStats& s)
    {
        return
            (s.max_blocks <= 1) ||
            (s.min_blocks > 0);
    };

    int best_Pr = 1;
    int best_Pc = 1;
    int best_ranks = 1;

    bool found = false;

    ptrdiff_t best_score =
        std::numeric_limits<ptrdiff_t>::max();

    for(int active_ranks = world_size;
            active_ranks >= 1;
            --active_ranks)
    {
        for(int Pr = 1; Pr <= active_ranks; ++Pr)
        {
            if(active_ranks % Pr)
                continue;

            int Pc = active_ranks / Pr;

            bool valid = true;

            ptrdiff_t imbalance = 0;

            for(int m = 0; m < 3; ++m)
            {
                ptrdiff_t grid_r =
                    (mats[m].rows + br - 1) / br;

                ptrdiff_t grid_c =
                    (mats[m].cols + bc - 1) / bc;

                GridStats s =
                    analyse_distribution(
                        grid_r,
                        grid_c,
                        Pr,
                        Pc);

                if(policy == SummaGridPolicy::DenseOnly)
                {
                    if(!matrix_is_dense(s))
                    {
                        valid = false;
                        break;
                    }
                }
                else
                {
                    if(!matrix_is_compatible(s))
                    {
                        valid = false;
                        break;
                    }
                }

                imbalance =
                    std::max(
                        imbalance,
                        s.max_blocks - s.min_blocks);
            }

            if(!valid)
                continue;

            if(policy == SummaGridPolicy::DenseOnly)
            {
                if(!found || active_ranks > best_ranks)
                {
                    found = true;

                    best_Pr = Pr;
                    best_Pc = Pc;
                    best_ranks = active_ranks;
                }
            }
            else if(policy == SummaGridPolicy::Compatible)
            {
                if(!found || active_ranks > best_ranks)
                {
                    found = true;

                    best_Pr = Pr;
                    best_Pc = Pc;
                    best_ranks = active_ranks;
                }
            }
            else // LoadBalanced
            {
                if(!found ||
                        imbalance < best_score ||
                        (imbalance == best_score &&
                         active_ranks > best_ranks))
                {
                    found = true;

                    best_score = imbalance;

                    best_Pr = Pr;
                    best_Pc = Pc;
                    best_ranks = active_ranks;
                }
            }
        }
    }
    if(!found)
    {
        if(world_rank == 0)
        {
            std::cerr
                    << "No SUMMA-compatible process grid found."
                    << std::endl;
        }

        MPI_Abort(MPI_COMM_WORLD, -1);
    }


    if(world_rank == 0)
    {
        if(printgrid)
            std::cout
                    << "SUMMA grid selection:\n"
                    << "Using process grid "
                    << best_Pr
                    << " x "
                    << best_Pc
                    << " = "
                    << best_ranks
                    << " ranks\n";
    }

    int color =
        (world_rank < best_ranks)
        ? 0
        : MPI_UNDEFINED;

    MPI_Comm active_comm;

    MPI_Comm_split(
        MPI_COMM_WORLD,
        color,
        world_rank,
        &active_comm);

    if(color == MPI_UNDEFINED)
        return MPI_COMM_NULL;

    int dims[2] =
    {
        best_Pr,
        best_Pc
    };

    int periods[2] = {0,0};

    MPI_Comm cart_comm;

    MPI_Cart_create(
        active_comm,
        2,
        dims,
        periods,
        0,
        &cart_comm);

    MPI_Comm_free(&active_comm);

    return cart_comm;
}
template<typename T>
inline void Math_Functions_MPI::scale_local_blocks(
    T* cdata,
    const ptrdiff_t* coffsets,
    const ptrdiff_t* cstrides,
    const ptrdiff_t* cextents,
    const ptrdiff_t numblocks,
    const T alpha,
    const bool ongpu,
    const int devnum)
{
    if(alpha == T(1) || numblocks == 0)
        return;


    if(ongpu)
    {
        #pragma omp target teams distribute \
        device(devnum) \
        is_device_ptr(cdata,coffsets,cstrides,cextents)
        for(ptrdiff_t b = 0; b < numblocks; ++b)
        {
            const ptrdiff_t rows = cextents[2*b];
            const ptrdiff_t cols = cextents[2*b+1];

            T* Cptr = cdata + coffsets[b];

            const ptrdiff_t str0 = cstrides[2*b];
            const ptrdiff_t str1 = cstrides[2*b+1];

            #pragma omp parallel for collapse(2)
            for(ptrdiff_t i = 0; i < rows; ++i)
            {
                for(ptrdiff_t j = 0; j < cols; ++j)
                {
                    const ptrdiff_t index = i*str0 + j*str1;

                    Cptr[index] = (alpha == T(0)) ? T(0) : Cptr[index] * alpha;
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for(ptrdiff_t b = 0; b < numblocks; ++b)
        {
            const ptrdiff_t rows = cextents[2*b];
            const ptrdiff_t cols = cextents[2*b+1];

            T* Cptr = cdata + coffsets[b];

            const ptrdiff_t str0 = cstrides[2*b];
            const ptrdiff_t str1 = cstrides[2*b+1];

            for(ptrdiff_t i = 0; i < rows; ++i)
            {
                #pragma omp simd
                for(ptrdiff_t j = 0; j < cols; ++j)
                {
                    const ptrdiff_t index = i*str0 + j*str1;

                    Cptr[index] = (alpha == T(0)) ? T(0) : Cptr[index] * alpha;
                }
            }
        }
    }
}


template <typename T>
bool Math_Functions_MPI::matrix_multiply_dot_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    DistributedDataBlock<T>& C,
    const T CoefficientB,
    const T CoefficientC,
    const Math_MPI_Functions_Policy* pol)
{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : get_default_policy();

    if (A.pctx->comm == MPI_COMM_NULL) return false;
    if (B.pctx->comm == MPI_COMM_NULL) return false;
    if (C.pctx->comm == MPI_COMM_NULL) return false;
    if(A.pglobal_extents[1] != B.pglobal_extents[0])
        return false;
    if(A.pglobal_extents[0] != C.pglobal_extents[0])
        return false;
    if(B.pglobal_extents[1] != C.pglobal_extents[1])
        return false;
    if(A.pblock_extents[1] != B.pblock_extents[0])
        return false;
    if(A.pblock_extents[0] != C.pblock_extents[0])
        return false;
    if(B.pblock_extents[1] != C.pblock_extents[1])
        return false;

    if(CoefficientB == T(0)&& (CoefficientC != T(1)))
    {
        return Math_Functions_MPI::matrix_multiply_scalar_Distributed(C,CoefficientC,&policy);
    }


    MPI_Comm comma = C.pctx->comm;
    int rank;
    MPI_Comm_rank(comma, &rank);


    ptrdiff_t blocknumber=C.Dblockarray.pnumblocks;
    ptrdiff_t maxnumber,minnumber;

    MPI_Allreduce(&blocknumber, &maxnumber, 1,mpi_get_type<ptrdiff_t>(), MPI_MAX, comma);
    MPI_Allreduce(&blocknumber, &minnumber, 1, mpi_get_type<ptrdiff_t>(), MPI_MIN, comma);

    if(maxnumber<=1)
    {
        int coords[2];
        MPI_Cart_coords(comma, rank, 2, coords);
        int my_row = coords[0];
        int my_col = coords[1];
        const ptrdiff_t Pr = A.pctx->dims[0];
        const ptrdiff_t Pc = A.pctx->dims[1];
        const ptrdiff_t br = A.pblock_extents[0];
        const ptrdiff_t bk = A.pblock_extents[1];
        const ptrdiff_t bc = B.pblock_extents[1];
        const ptrdiff_t M = A.pglobal_extents[0];
        const ptrdiff_t N = B.pglobal_extents[1];
        const ptrdiff_t Ktot = A.pglobal_extents[1];
        const ptrdiff_t grid_r = (M + br - 1) / br;
        const ptrdiff_t grid_c = (N + bc - 1) / bc;
        const ptrdiff_t grid_k = (Ktot + bk - 1) / bk;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(comma, my_row, my_col, &row_comm);
        MPI_Comm_split(comma, my_col, my_row, &col_comm);
        ptrdiff_t max_A = br * bk;
        ptrdiff_t max_B = bk * bc;
        bool ongpu=policy.should_use_gpu_matrix_multiply(A,B,C);

        bool memmap=policy.memmapped_files;
        int devnum=policy.devicenum;
        if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
        if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;
        if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
        if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;
        T* A_buf;
        T* B_buf;
        if(max_A>0)
            DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_A,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_B,B_buf);

        T* adata=A.Dblockarray.pdata;
        if(ongpu)
        {
            if(A.Dblockarray.pnumblocks > 0)
            {
                if(!A.Dblockarray.pdata_is_devptr)
                {
                    adata=(T*) omp_target_alloc(sizeof(T)*A.Dblockarray.pdatalength,devnum);
                    omp_target_memcpy(adata,A.Dblockarray.pdata,sizeof(T)*A.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        T* bdata=B.Dblockarray.pdata;
        if(ongpu)
        {
            if (B.Dblockarray.pnumblocks > 0)
            {
                if(!B.Dblockarray.pdata_is_devptr)
                {
                    bdata=(T*) omp_target_alloc(sizeof(T)*B.Dblockarray.pdatalength,devnum);
                    omp_target_memcpy(bdata,B.Dblockarray.pdata,sizeof(T)*B.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        ptrdiff_t *coffsets=C.Dblockarray.pblock_offsets;
        ptrdiff_t *cstrides=C.Dblockarray.pstridesbuffer;
        T* cdata=C.Dblockarray.pdata;
        if(ongpu)
        {
            if (C.Dblockarray.pnumblocks > 0)
            {
                if(!C.Dblockarray.pdata_is_devptr)
                {
                    cdata=(T*) omp_target_alloc(sizeof(T)*C.Dblockarray.pdatalength,devnum);
                    omp_target_memcpy(cdata,C.Dblockarray.pdata,sizeof(T)*C.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }

        scale_local_blocks( cdata, coffsets, cstrides,  C.Dblockarray.pextentsbuffer,  C.Dblockarray.pnumblocks,  CoefficientC,  ongpu, devnum);



        struct BlockMeta
        {
            ptrdiff_t block_row;
            ptrdiff_t block_col;
            ptrdiff_t rows;
            ptrdiff_t cols;
            ptrdiff_t str0;
            ptrdiff_t str1;
            ptrdiff_t length;
        };

        for (ptrdiff_t k = 0; k < grid_k; k++)
        {
            BlockMeta A_meta{0,0,0,0,0};
            int root_col = k % Pc;
            T* A_ptr = A_buf;
            if (my_col == root_col)
            {
                ptrdiff_t A_lin = my_row * grid_k + k;
                auto it = A.pglobal_to_local_index.find(A_lin);
                if (it != A.pglobal_to_local_index.end())
                {
                    ptrdiff_t idx = it->second;
                    A_meta.block_row = A.pblock_grid_coords[2 * idx];
                    A_meta.block_col = A.pblock_grid_coords[2 * idx + 1];
                    A_meta.rows = A.Dblockarray.pextentsbuffer[2*idx];
                    A_meta.cols = A.Dblockarray.pextentsbuffer[2*idx+1];
                    A_meta.str0 = A.Dblockarray.pstridesbuffer[2*idx];
                    A_meta.str1 = A.Dblockarray.pstridesbuffer[2*idx+1];
                    A_meta.length = A_meta.rows * A_meta.cols;
                    ptrdiff_t offset = A.Dblockarray.pblock_offsets[idx];
                    A_ptr = adata + offset;

                }
            }
            MPI_Bcast(&A_meta, sizeof(BlockMeta), MPI_BYTE, root_col, row_comm);
            MPI_Bcast(A_ptr, A_meta.length, mpi_get_type<T>(), root_col, row_comm);
            int root_row = k % Pr;
            BlockMeta B_meta{0,0,0,0,0};

            T* B_ptr = B_buf;
            if (my_row == root_row)
            {
                ptrdiff_t B_lin = k * grid_c + my_col;

                auto it = B.pglobal_to_local_index.find(B_lin);
                if (it != B.pglobal_to_local_index.end())
                {
                    ptrdiff_t idx = it->second;
                    B_meta.block_row = B.pblock_grid_coords[2 * idx];
                    B_meta.block_col = B.pblock_grid_coords[2 * idx + 1];
                    B_meta.rows = B.Dblockarray.pextentsbuffer[2*idx];
                    B_meta.cols = B.Dblockarray.pextentsbuffer[2*idx+1];
                    B_meta.str0 = B.Dblockarray.pstridesbuffer[2*idx];
                    B_meta.str1 = B.Dblockarray.pstridesbuffer[2*idx+1];
                    B_meta.length = B_meta.rows * B_meta.cols;
                    ptrdiff_t offset = B.Dblockarray.pblock_offsets[idx];
                    B_ptr = bdata + offset;
                }
            }
            MPI_Bcast(&B_meta, sizeof(BlockMeta), MPI_BYTE, root_row, col_comm);
            MPI_Bcast(B_ptr, B_meta.length, mpi_get_type<T>(), root_row, col_comm);
            const ptrdiff_t A_block_rows=A_meta.rows;
            const ptrdiff_t A_block_cols=A_meta.cols;
            const ptrdiff_t B_block_cols=B_meta.cols;
            const ptrdiff_t A_block_str0=A_meta.str0;
            const ptrdiff_t A_block_str1=A_meta.str1;
            const ptrdiff_t B_block_str0=B_meta.str0;
            const ptrdiff_t B_block_str1=B_meta.str1;

            const bool Aconj=A.Dblockarray.pconjugate;
            const bool Bconj=B.Dblockarray.pconjugate;
            if (C.Dblockarray.pnumblocks > 0 && A_meta.length > 0 && B_meta.length > 0)
            {
                if(ongpu)
                {
                    const ptrdiff_t Cstr0=cstrides[0];
                    const ptrdiff_t Cstr1=cstrides[1];
                    T* C_ptr=cdata+coffsets[0];
                    #pragma omp target teams distribute parallel for collapse(2)device(devnum) is_device_ptr(C_ptr,A_ptr,B_ptr)
                    for (ptrdiff_t ir = 0; ir < A_block_rows; ++ir)
                    {
                        for (ptrdiff_t j = 0; j < B_block_cols; ++j)
                        {
                            T sum =T(0);
                            #pragma omp simd reduction(+:sum)
                            for (ptrdiff_t p = 0; p < A_block_cols; ++p)
                            {
                                sum += returnval(A_ptr[ir * A_block_str0 + p * A_block_str1],Aconj) *returnval(B_ptr[p  * B_block_str0 + j * B_block_str1],Bconj);
                            }
                            C_ptr[ir*Cstr0+j*Cstr1]+=CoefficientB* sum;
                        }
                    }
                }
                else
                {

                    T* C_ptr=cdata+coffsets[0];
                    const ptrdiff_t Cstr0=cstrides[0];
                    const ptrdiff_t Cstr1=cstrides[1];
                    #pragma omp parallel for collapse(2)
                    for (ptrdiff_t ir = 0; ir < A_block_rows; ++ir)
                    {
                        for (ptrdiff_t j = 0; j < B_block_cols; ++j)
                        {

                            T sum =T(0);
                            #pragma omp simd reduction(+:sum)
                            for (ptrdiff_t k = 0; k < A_block_cols; ++k)
                            {
                                sum += returnval(A_ptr[ir*A_block_str0+k*A_block_str1],Aconj) *returnval(B_ptr[k*B_block_str0+j*B_block_str1],Bconj);
                            }
                            C_ptr[ir*Cstr0+j*Cstr1]+= CoefficientB* sum;
                        }
                    }
                }
            }
        }

        if(ongpu)
        {
            if (A.Dblockarray.pnumblocks > 0)
            {
                if(!A.Dblockarray.pdata_is_devptr)
                    omp_target_free(adata,devnum);
            }
            if (B.Dblockarray.pnumblocks > 0)
            {
                if(!B.Dblockarray.pdata_is_devptr)
                    omp_target_free(bdata,devnum);
            }

            if (C.Dblockarray.pnumblocks>0)
            {
                if(!C.Dblockarray.pdata_is_devptr)
                {
                    omp_target_memcpy(C.Dblockarray.pdata,cdata,sizeof(T)*C.Dblockarray.pdatalength,0,0,omp_get_initial_device(),devnum);
                    omp_target_free(cdata,devnum);
                }
            }
        }

        if(max_A>0)
            DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_A,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_B,B_buf);

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);

        return true;
    }

    else
    {

        MPI_Comm comm;
        int Pr,Pc;
        int coords[2];
        if(minnumber==0)
            return false;

        comm=comma;
        Pr = C.pctx->dims[0];
        Pc = C.pctx->dims[1];
        MPI_Cart_coords(comm, rank, 2, coords);


        int my_row = coords[0];
        int my_col = coords[1];


        const ptrdiff_t br = A.pblock_extents[0];
        const ptrdiff_t bk = A.pblock_extents[1];
        const ptrdiff_t bc = B.pblock_extents[1];
        const ptrdiff_t M = A.pglobal_extents[0];
        const ptrdiff_t N = B.pglobal_extents[1];
        const ptrdiff_t Ktot = A.pglobal_extents[1];
        const ptrdiff_t grid_r = (M + br - 1) / br;
        const ptrdiff_t grid_c = (N + bc - 1) / bc;
        const ptrdiff_t grid_k = (Ktot + bk - 1) / bk;




        MPI_Comm row_comm, col_comm;

        MPI_Comm_split(comm, my_row, my_col, &row_comm);
        MPI_Comm_split(comm, my_col, my_row, &col_comm);
        const ptrdiff_t max_A = br * bk;
        const ptrdiff_t max_B = bk * bc;
        bool ongpu=policy.should_use_gpu_matrix_multiply(A,B,C);
        bool memmap=policy.memmapped_files;
        int devnum=policy.devicenum;
        if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
        if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;
        if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
        if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;


        struct PanelPair
        {
            ptrdiff_t a_idx;
            ptrdiff_t b_idx;
            ptrdiff_t c_idx;
        };

        ptrdiff_t num_A_panels = 0;
        ptrdiff_t num_B_panels = 0;

        ptrdiff_t* Ci_list = new ptrdiff_t[C.Dblockarray.pnumblocks];
        ptrdiff_t* Cj_list = new ptrdiff_t[C.Dblockarray.pnumblocks];


        bool* mark = new bool[grid_r > grid_c ? grid_r : grid_c];



        #pragma omp parallel for simd if(parallel:grid_r>30)
        for (ptrdiff_t i = 0; i < grid_r; i++)
            mark[i] = false;

        for (ptrdiff_t i = 0; i < C.Dblockarray.pnumblocks; i++)
        {
            ptrdiff_t Ci = C.pblock_grid_coords[2*i];

            if (!mark[Ci])
            {
                mark[Ci] = true;
                Ci_list[num_A_panels++] = Ci;
            }
        }



        #pragma omp parallel for simd if(parallel: grid_c>30)
        for (ptrdiff_t j = 0; j < grid_c; j++)
            mark[j] = false;


        for (ptrdiff_t i = 0; i < C.Dblockarray.pnumblocks; i++)
        {
            ptrdiff_t Cj = C.pblock_grid_coords[2*i+1];

            if (!mark[Cj])
            {
                mark[Cj] = true;
                Cj_list[num_B_panels++] = Cj;
            }
        }

        delete[] mark;



        T* A_buf;
        T* B_buf;
        if(max_A>0)
            DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_A*num_A_panels,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_B*num_B_panels,B_buf);

        T* adata=A.Dblockarray.pdata;
        if(ongpu)
        {
            if(A.Dblockarray.pnumblocks > 0)
            {
                if(!A.Dblockarray.pdata_is_devptr)
                {
                    adata=(T*) omp_target_alloc(sizeof(T)*A.Dblockarray.pdatalength,devnum);
                    omp_target_memcpy(adata,A.Dblockarray.pdata,sizeof(T)*A.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        T* bdata=B.Dblockarray.pdata;
        if(ongpu)
        {
            if (B.Dblockarray.pnumblocks > 0)
            {
                if(!B.Dblockarray.pdata_is_devptr)
                {
                    bdata=(T*) omp_target_alloc(sizeof(T)*B.Dblockarray.pdatalength,devnum);
                    omp_target_memcpy(bdata,B.Dblockarray.pdata,sizeof(T)*B.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }

        ptrdiff_t *coffsets=C.Dblockarray.pblock_offsets;
        ptrdiff_t *cstrides=C.Dblockarray.pstridesbuffer;
        ptrdiff_t *cblockcoords=C.pblock_grid_coords;
        T* cdata=C.Dblockarray.pdata;
        if(ongpu)
        {
            if (C.Dblockarray.pnumblocks > 0)
            {
                ptrdiff_t offsets_bytes = sizeof(ptrdiff_t) * C.Dblockarray.pnumblocks;
                ptrdiff_t pair_bytes    = sizeof(ptrdiff_t) * 2 * C.Dblockarray.pnumblocks;

                coffsets = (ptrdiff_t*) omp_target_alloc(offsets_bytes, devnum);
                omp_target_memcpy( coffsets,C.Dblockarray.pblock_offsets,  offsets_bytes, 0,0,  devnum, omp_get_initial_device());

                cstrides = (ptrdiff_t*) omp_target_alloc(pair_bytes, devnum);
                omp_target_memcpy( cstrides, C.Dblockarray.pstridesbuffer, pair_bytes, 0,0, devnum,  omp_get_initial_device());

                cblockcoords = (ptrdiff_t*) omp_target_alloc(pair_bytes, devnum);
                omp_target_memcpy( cblockcoords,  C.pblock_grid_coords,  pair_bytes,   0,0,  devnum,  omp_get_initial_device());
                if(!C.Dblockarray.pdata_is_devptr)
                {
                    cdata=(T*) omp_target_alloc(sizeof(T)*C.Dblockarray.pdatalength,devnum);
                    omp_target_memcpy(cdata,C.Dblockarray.pdata,sizeof(T)*C.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }

        struct BlockMeta
        {
            ptrdiff_t block_row;
            ptrdiff_t block_col;
            ptrdiff_t rows;
            ptrdiff_t cols;
            ptrdiff_t str0;
            ptrdiff_t str1;
            ptrdiff_t length;
        };

        BlockMeta* A_meta_arr = nullptr;
        BlockMeta* B_meta_arr = nullptr;

        T** A_panel_ptrs=nullptr;
        T** dA_panel_ptrs=nullptr;

        BlockMeta* dA_meta_arr=nullptr;
        BlockMeta* dB_meta_arr=nullptr;

        if(num_A_panels>0)
        {
            A_meta_arr = new BlockMeta[num_A_panels];
            A_panel_ptrs =new T*[num_A_panels];
            if(ongpu)
            {
                dA_panel_ptrs=(T**) omp_target_alloc(sizeof(T*)*num_A_panels,devnum);
                dA_meta_arr = (BlockMeta*)omp_target_alloc(  sizeof(BlockMeta)*num_A_panels,       devnum);
            }
        }

        T** B_panel_ptrs=nullptr;
        T** dB_panel_ptrs=nullptr;

        if( num_B_panels>0)
        {
            B_meta_arr = new BlockMeta[num_B_panels];
            B_panel_ptrs=new T*[num_B_panels];
            if(ongpu)
            {
                dB_panel_ptrs=(T**) omp_target_alloc(sizeof(T*)*num_B_panels,devnum);
                dB_meta_arr =(BlockMeta*)omp_target_alloc(sizeof(BlockMeta)*num_B_panels,devnum);
            }
        }
        for (ptrdiff_t k = 0; k < grid_k; k++)
        {
            for (ptrdiff_t p = 0; p < num_A_panels; p++)
            {
                const ptrdiff_t bi = Ci_list[p];

                BlockMeta& A_meta = A_meta_arr[p];
                A_meta = {0,0,0,0,0,0,0};

                const int root_col = k % Pc;

                T* root_ptr = nullptr;
                T* recv_ptr = A_buf + p * max_A;

                if (my_col == root_col)
                {
                    const ptrdiff_t A_lin = bi * grid_k + k;

                    auto it = A.pglobal_to_local_index.find(A_lin);

                    if (it != A.pglobal_to_local_index.end())
                    {
                        ptrdiff_t idx = it->second;

                        A_meta.block_row = A.pblock_grid_coords[2 * idx];
                        A_meta.block_col = A.pblock_grid_coords[2 * idx + 1];
                        A_meta.rows = A.Dblockarray.pextentsbuffer[2*idx];
                        A_meta.cols = A.Dblockarray.pextentsbuffer[2*idx+1];
                        A_meta.str0 = A.Dblockarray.pstridesbuffer[2*idx];
                        A_meta.str1 = A.Dblockarray.pstridesbuffer[2*idx+1];
                        A_meta.length = A_meta.rows * A_meta.cols;

                        const ptrdiff_t offset = A.Dblockarray.pblock_offsets[idx];
                        root_ptr = adata + offset;
                    }
                }

                MPI_Bcast(&A_meta, sizeof(BlockMeta), MPI_BYTE, root_col, row_comm);


                if (my_col == root_col)
                    A_panel_ptrs[p] = root_ptr;
                else
                    A_panel_ptrs[p] = recv_ptr;

                MPI_Bcast(A_panel_ptrs[p], A_meta.length, mpi_get_type<T>(), root_col, row_comm);
            }

            for (ptrdiff_t p = 0; p < num_B_panels; p++)
            {
                const ptrdiff_t bj = Cj_list[p];

                BlockMeta& B_meta = B_meta_arr[p];
                B_meta = {0,0,0,0,0,0,0};

                const int root_row = k % Pr;

                T* root_ptr = nullptr;
                T* recv_ptr = B_buf + p * max_B;

                if (my_row == root_row)
                {
                    const ptrdiff_t B_lin = k * grid_c + bj;

                    auto it = B.pglobal_to_local_index.find(B_lin);

                    if (it != B.pglobal_to_local_index.end())
                    {
                        ptrdiff_t idx = it->second;

                        B_meta.block_row = B.pblock_grid_coords[2 * idx];
                        B_meta.block_col = B.pblock_grid_coords[2 * idx + 1];
                        B_meta.rows = B.Dblockarray.pextentsbuffer[2*idx];
                        B_meta.cols = B.Dblockarray.pextentsbuffer[2*idx+1];
                        B_meta.str0 = B.Dblockarray.pstridesbuffer[2*idx];
                        B_meta.str1 = B.Dblockarray.pstridesbuffer[2*idx+1];
                        B_meta.length = B_meta.rows * B_meta.cols;

                        const ptrdiff_t offset = B.Dblockarray.pblock_offsets[idx];
                        root_ptr = bdata + offset;
                    }
                }

                MPI_Bcast(&B_meta, sizeof(BlockMeta), MPI_BYTE, root_row, col_comm);


                if (my_row == root_row)
                    B_panel_ptrs[p] = root_ptr;
                else
                    B_panel_ptrs[p] = recv_ptr;

                MPI_Bcast(B_panel_ptrs[p],B_meta.length,mpi_get_type<T>(),root_row, col_comm);
            }
            const bool Aconj=A.Dblockarray.pconjugate;
            const bool Bconj=B.Dblockarray.pconjugate;

            if(ongpu && num_A_panels>0 && num_B_panels>0)
            {
                ptrdiff_t cblocknumber=C.Dblockarray.pnumblocks;

                omp_target_memcpy(dA_panel_ptrs, A_panel_ptrs, sizeof(T*)*num_A_panels, 0,0, devnum, omp_get_initial_device());

                omp_target_memcpy(dB_panel_ptrs, B_panel_ptrs,sizeof(T*)*num_B_panels, 0,0, devnum, omp_get_initial_device());

                omp_target_memcpy(dA_meta_arr, A_meta_arr,sizeof(BlockMeta)*num_A_panels, 0,0, devnum,  omp_get_initial_device());

                omp_target_memcpy(dB_meta_arr, B_meta_arr, sizeof(BlockMeta)*num_B_panels,0,0, devnum, omp_get_initial_device());

                #pragma omp target teams distribute collapse(2) device(devnum) is_device_ptr(dA_meta_arr,dB_meta_arr,cdata,coffsets,cblockcoords,dA_panel_ptrs,dB_panel_ptrs)
                for (ptrdiff_t cpi = 0; cpi < num_A_panels; cpi++)
                {
                    for (ptrdiff_t cpj = 0; cpj < num_B_panels; cpj++)
                    {
                        const T* A_ptr = dA_panel_ptrs[cpi];
                        const T* B_ptr = dB_panel_ptrs[cpj];

                        const BlockMeta& A_meta = dA_meta_arr[cpi];
                        const BlockMeta& B_meta = dB_meta_arr[cpj];

                        #pragma omp parallel for collapse(3)
                        for (ptrdiff_t i = 0; i <cblocknumber; i++)
                        {

                            for (ptrdiff_t r = 0; r < A_meta.rows; r++)
                            {
                                for (ptrdiff_t c = 0; c < B_meta.cols; c++)
                                {

                                    if (cblockcoords[2*i] != A_meta.block_row ||cblockcoords[2*i+1] != B_meta.block_col)
                                        continue;

                                    T* C_ptr = cdata + coffsets[i];

                                    T sum = 0;
                                    #pragma omp simd reduction(+:sum)
                                    for (ptrdiff_t k = 0; k < A_meta.cols; k++)
                                    {
                                        sum += returnval(A_ptr[r*A_meta.str0 + k*A_meta.str1],Bconj) *returnval(B_ptr[k*B_meta.str0 + c*B_meta.str1],Bconj);
                                    }

                                    C_ptr[r*cstrides[2*i] + c*cstrides[2*i+1]] += CoefficientB*sum;
                                }
                            }
                        }

                    }
                }

            }
            else
            {
                #pragma omp parallel for collapse(2)
                for (ptrdiff_t cpi = 0; cpi < num_A_panels; cpi++)
                {
                    for (ptrdiff_t cpj = 0; cpj < num_B_panels; cpj++)
                    {
                        const T* A_ptr = A_panel_ptrs[cpi];
                        const T* B_ptr = B_panel_ptrs[cpj];

                        const BlockMeta& A_meta = A_meta_arr[cpi];
                        const BlockMeta& B_meta = B_meta_arr[cpj];

                        for (ptrdiff_t i = 0; i < C.Dblockarray.pnumblocks; i++)
                        {
                            for (ptrdiff_t r = 0; r <  A_meta.rows; r++)
                            {
                                for (ptrdiff_t c = 0; c < B_meta.cols; c++)
                                {
                                    if (cblockcoords[2*i] != A_meta.block_row ||
                                            cblockcoords[2*i+1] != B_meta.block_col)
                                        continue;

                                    T* C_ptr = cdata + coffsets[i];

                                    T sum = 0;
                                    #pragma omp simd reduction(+:sum)
                                    for (ptrdiff_t k = 0; k < A_meta.cols; k++)
                                    {
                                        sum += returnval(A_ptr[r*A_meta.str0 + k*A_meta.str1],Aconj) *
                                               returnval(B_ptr[k*B_meta.str0 + c*B_meta.str1],Bconj);
                                    }

                                    C_ptr[r*cstrides[2*i] + c*cstrides[2*i+1]] +=CoefficientB* sum;
                                }
                            }
                        }
                    }
                }
            }
        }
        if(num_A_panels>0)
        {
            delete[] A_meta_arr ;
            delete[] A_panel_ptrs;
            if(ongpu)
            {
                omp_target_free(dA_panel_ptrs,devnum);
                omp_target_free(dA_meta_arr,devnum);

            }
        }
        if(num_B_panels>0)
        {
            delete[] B_meta_arr;
            delete[] B_panel_ptrs;
            if(ongpu )
            {
                omp_target_free(dB_panel_ptrs,devnum);
                omp_target_free(dB_meta_arr,devnum);

            }
        }

        delete[] Ci_list ;
        delete[] Cj_list ;

        if(ongpu)
        {
            if (A.Dblockarray.pnumblocks > 0)
            {
                if(!A.Dblockarray.pdata_is_devptr)
                    omp_target_free(adata,devnum);
            }
            if (B.Dblockarray.pnumblocks > 0)
            {
                if(!B.Dblockarray.pdata_is_devptr)
                    omp_target_free(bdata,devnum);
            }

            if (C.Dblockarray.pnumblocks>0)
            {
                if(!C.Dblockarray.pdata_is_devptr)
                {
                    omp_target_memcpy(C.Dblockarray.pdata,cdata,sizeof(T)*C.Dblockarray.pdatalength,0,0,omp_get_initial_device(),devnum);
                    omp_target_free(cdata,devnum);
                }
                omp_target_free(cstrides,devnum);
                omp_target_free(coffsets,devnum);
                omp_target_free(cblockcoords,devnum);
            }
        }
        if(max_A>0)
            DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_A*num_A_panels,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},max_B*num_B_panels,B_buf);

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);

        return true;

    }
}




template<typename T>
inline bool Math_Functions_MPI::matrix_multiply_vector_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& x,
    DistributedDataBlock<T>& y,
    const T Coefficientx,
    const T Coefficienty,
    const Math_MPI_Functions_Policy*pol)
{
    const Math_MPI_Functions_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    if (A.pctx->comm == MPI_COMM_NULL)
        return false;

    int rank, size;
    MPI_Comm_rank(A.pctx->comm, &rank);
    MPI_Comm_size(A.pctx->comm, &size);

    const ptrdiff_t M = A.pglobal_extents[0];
    const ptrdiff_t K = A.pglobal_extents[1];

    const ptrdiff_t br = A.pblock_extents[0];
    const ptrdiff_t bc = A.pblock_extents[1];
    const ptrdiff_t bs = y.pblock_extents[0];

    const ptrdiff_t grid_c = (K + bc - 1) / bc;
    const ptrdiff_t grid_r = (M + bs - 1) / bs;
    bool ongpu=policy.should_use_gpu_matrix_vector(A,x,y);
    bool memmap=policy.memmapped_files;

    int devnum=policy.devicenum;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum)
        return false;
    if(x.Dblockarray.pdata_is_devptr&& x.Dblockarray.pdevnum!=devnum)
        return false;
    if(y.Dblockarray.pdata_is_devptr&& y.Dblockarray.pdevnum!=devnum)
        return false;

    if(ongpu)
    {
        if(y.Dblockarray.pdevnum!=x.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=y.Dblockarray.pdevnum)
            return false;
    }

    T* x_global=nullptr;
    if(K>0)
        DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},K,x_global);

    if (ongpu)
    {
        #pragma omp target teams distribute parallel for simd device(devnum)
        for (ptrdiff_t i=0; i<K; i++)
            x_global[i]=0;
    }
    else
    {
        #pragma omp parallel for simd
        for (ptrdiff_t i=0; i<K; i++)
            x_global[i]=0;
    }



    if(ongpu)
    {
        for (ptrdiff_t i = 0; i < x.Dblockarray.pnumblocks; i++)
        {
            ptrdiff_t b = x.pblock_linear_idx[i];
            ptrdiff_t start = b * bc;

            ptrdiff_t diff = K - start;
            ptrdiff_t len  = (bc < diff) ? bc : diff;

            ptrdiff_t off = x.Dblockarray.pblock_offsets[i];

            if(x.Dblockarray.pdata_is_devptr)
            {
                omp_target_memcpy(x_global,x.Dblockarray.pdata,len*sizeof(T),sizeof(T)*start,sizeof(T)*off,devnum,x.Dblockarray.pdevnum);
            }
            else
            {
                omp_target_memcpy(x_global,x.Dblockarray.pdata,len*sizeof(T),sizeof(T)*start,sizeof(T)*off,devnum,omp_get_initial_device());
            }
        }

    }
    else
    {
        #pragma omp parallel for if(parallel:x.Dblockarray.pnumblocks>30)
        for (ptrdiff_t i = 0; i < x.Dblockarray.pnumblocks; i++)
        {
            ptrdiff_t b = x.pblock_linear_idx[i];
            ptrdiff_t start = b * bc;

            ptrdiff_t diff = K - start;
            ptrdiff_t len  = (bc < diff) ? bc : diff;

            ptrdiff_t off = x.Dblockarray.pblock_offsets[i];

            const T* src = x.Dblockarray.pdata + off;
            T* dst = x_global + start;
            memcpy(dst, src, len * sizeof(T));
        }
    }

    MPI_Allreduce(
        MPI_IN_PLACE,
        x_global,
        K,
        mpi_get_type<T>(),
        MPI_SUM,
        A.pctx->comm);

    T* y_full=nullptr, *A_ptr=nullptr;
    ptrdiff_t* Aext=nullptr,*Ablockoff=nullptr,*Ablocklinindex=nullptr;

    if(M>0)
        DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},M,y_full);


    const bool rowm=A.Dblockarray.prowm;
    const bool aconj=A.Dblockarray.pconjugate;
    const bool xconj=x.Dblockarray.pconjugate;
    if(A.Dblockarray.pdatalength>0 &&A.Dblockarray.pnumblocks>0)
    {
        if(ongpu)
        {
            if(!A.Dblockarray.pdata_is_devptr)
            {
                A_ptr=(T*) omp_target_alloc(sizeof(T)*A.Dblockarray.pdatalength,devnum);
                omp_target_memcpy(A_ptr,A.Dblockarray.pdata,sizeof(T)*A.Dblockarray.pdatalength,0,0,devnum,omp_get_initial_device());
            }
            else
                A_ptr=A.Dblockarray.pdata;

            Ablockoff=(ptrdiff_t*) omp_target_alloc(sizeof(ptrdiff_t)*A.Dblockarray.pnumblocks,devnum);
            omp_target_memcpy(Ablockoff,A.Dblockarray.pblock_offsets, sizeof(ptrdiff_t)*A.Dblockarray.pnumblocks,0,0,devnum,omp_get_initial_device());

            Aext=(ptrdiff_t*) omp_target_alloc(sizeof(T)*A.Dblockarray.pnumblocks*2,devnum );
            omp_target_memcpy(Aext, A.Dblockarray.pextentsbuffer, sizeof(ptrdiff_t)*A.Dblockarray.pnumblocks*2,0,0,devnum,omp_get_initial_device());

            Ablocklinindex=(ptrdiff_t*) omp_target_alloc(sizeof(ptrdiff_t)*A.Dblockarray.pnumblocks,devnum);
            omp_target_memcpy(Ablocklinindex,A.pblock_linear_idx,sizeof(ptrdiff_t*)*A.Dblockarray.pnumblocks,0,0,devnum, omp_get_initial_device());


            const ptrdiff_t num=A.Dblockarray.pnumblocks;

            #pragma omp target teams distribute parallel for \
            is_device_ptr(Ablocklinindex,Aext,Ablockoff,A_ptr,y_full) \
            device(devnum)
            for (ptrdiff_t global_row = 0; global_row < M; global_row++)
            {
                T total = T(0);

                for (ptrdiff_t bi_local = 0; bi_local < num; bi_local++)
                {
                    const ptrdiff_t b = Ablocklinindex[bi_local];

                    const ptrdiff_t bi = b / grid_c;
                    const ptrdiff_t bj = b % grid_c;

                    const ptrdiff_t row0 = bi * br;
                    const ptrdiff_t col0 = bj * bc;

                    const ptrdiff_t rows = Aext[bi_local * 2 + 0];

                    if (global_row >= row0 && global_row < row0 + rows)
                    {
                        const ptrdiff_t r = global_row - row0;

                        const ptrdiff_t a_off = Ablockoff[bi_local];
                        const ptrdiff_t cols  = Aext[bi_local * 2 + 1];

                        T sum = T(0);

                        if (rowm)
                        {
                            const ptrdiff_t a_row_off = a_off + r * cols;
                            #pragma omp simd reduction(+:sum)
                            for (ptrdiff_t c = 0; c < cols; c++)
                            {
                                const ptrdiff_t indexA=a_row_off + c;
                                const ptrdiff_t indexX=col0 + c;
                                sum += returnval(A_ptr[indexA],aconj) * returnval(x_global[indexX],xconj);
                            }
                        }
                        else
                        {
                            #pragma omp simd reduction(+:sum)
                            for (ptrdiff_t c = 0; c < cols; c++)
                            {
                                const ptrdiff_t a_idx = a_off + c * rows + r;
                                const ptrdiff_t indexX =col0 + c;
                                sum += returnval(A_ptr[a_idx],aconj) * returnval(x_global[indexX],xconj);
                            }
                        }

                        total += sum;
                    }
                }
               y_full[global_row] =Coefficientx * total +Coefficienty * y_full[global_row];
            }
        }
        else
        {
            Aext= A.Dblockarray.pextentsbuffer;
            Ablockoff=A.Dblockarray.pblock_offsets;
            Ablocklinindex=A.pblock_linear_idx;
            A_ptr=A.Dblockarray.pdata;
            const ptrdiff_t num=A.Dblockarray.pnumblocks;
            #pragma omp parallel for
            for (ptrdiff_t global_row = 0; global_row < M; global_row++)
            {
                T total = T(0);

                for (ptrdiff_t bi_local = 0; bi_local < num; bi_local++)
                {
                    const ptrdiff_t b = Ablocklinindex[bi_local];

                    const ptrdiff_t bi = b / grid_c;
                    const ptrdiff_t bj = b % grid_c;

                    const ptrdiff_t row0 = bi * br;
                    const ptrdiff_t col0 = bj * bc;

                    const ptrdiff_t rows = Aext[bi_local * 2 + 0];

                    if (global_row >= row0 && global_row < row0 + rows)
                    {
                        const ptrdiff_t r = global_row - row0;

                        const ptrdiff_t a_off = Ablockoff[bi_local];
                        const  ptrdiff_t cols  = Aext[bi_local * 2 + 1];

                        T sum = T(0);

                        if (rowm)
                        {
                            const ptrdiff_t a_row_off = a_off + r * cols;

                            #pragma omp simd reduction(+:sum)
                            for (ptrdiff_t c = 0; c < cols; c++)
                            {
                                sum += returnval(A_ptr[a_row_off + c],aconj) * returnval(x_global[col0 + c],xconj);
                            }
                        }
                        else
                        {
                            #pragma omp simd reduction(+:sum)
                            for (ptrdiff_t c = 0; c < cols; c++)
                            {
                                const ptrdiff_t a_idx = a_off + c * rows + r;
                                sum += returnval(A_ptr[a_idx],aconj) * returnval(x_global[col0 + c],xconj);
                            }
                        }

                        total += sum;
                    }
                }
               y_full[global_row] =Coefficientx * total +Coefficienty * y_full[global_row];
            }
        }
    }

    if(K>0)
        DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},  K,x_global);

    int* recvcounts = new int[size];

    #pragma omp parallel for simd  if(parallel:size>30)
    for (ptrdiff_t i=0; i<size; i++)
        recvcounts[i]=0;

    int ndims;
    MPI_Cartdim_get(y.pctx->comm, &ndims);
    ptrdiff_t gridrank=(ptrdiff_t) ndims;
    ptrdiff_t *gridcoords=new ptrdiff_t [gridrank];
    int* tempcoords=new int[gridrank];

    for (ptrdiff_t b = 0; b < grid_r; b++)
    {
        ptrdiff_t diff= M - b * bs;
        ptrdiff_t len = bs<diff? bs:diff;

        ptrdiff_t bcoords[1] = { b };

        y.ppolicy->create_coords( bcoords,gridcoords,  y.Dblockarray.ptensor_rank);
        int owner = y.ppolicy->owner(gridcoords, *y.pctx, tempcoords);

        recvcounts[owner] += (int)len;
    }
    delete[] gridcoords;
    delete[] tempcoords;

    T* y_local=nullptr;

    if(recvcounts[rank]>0)
        DataBlock_MPI_Functions::alloc_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},recvcounts[rank],y_local);


    MPI_Reduce_scatter(
        y_full,
        y_local,
        recvcounts,
        mpi_get_type<T>(),
        MPI_SUM,
        y.pctx->comm);

    if(M>0)
        DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},  M,y_full);

    if(ongpu)
    {
        if(!A.Dblockarray.pdata_is_devptr)
            omp_target_free(A_ptr,devnum);

        omp_target_free(Ablockoff,devnum);
        omp_target_free(Aext,devnum);
        omp_target_free(Ablocklinindex,devnum);
    }


    ptrdiff_t offset = 0;
    #pragma omp parallel for simd reduction(+:offset)if(parallel:rank>30)
    for (int i = 0; i < rank; i++)
        offset += recvcounts[i];

    if(ongpu)
    {
        for (ptrdiff_t i = 0; i < y.Dblockarray.pnumblocks; i++)
        {
            ptrdiff_t b = y.pblock_linear_idx[i];

            ptrdiff_t start = b * bs;
            ptrdiff_t diff  = M - start;
            ptrdiff_t len   = (bs < diff) ? bs : diff;

            ptrdiff_t global_offset = start;

            ptrdiff_t local_offset = global_offset - offset;

            ptrdiff_t dst = y.Dblockarray.pblock_offsets[i];
            if(!y.Dblockarray.pdata_is_devptr)
                omp_target_memcpy( y.Dblockarray.pdata,y_local,len * sizeof(T),dst*sizeof(T),local_offset*sizeof(T), omp_get_initial_device(),devnum);
            else
                omp_target_memcpy(    y.Dblockarray.pdata,    y_local,len * sizeof(T   ),dst*sizeof(T),local_offset*sizeof(T),y.Dblockarray.pdevnum,devnum);

        }
    }

    else
    {
        for (ptrdiff_t i = 0; i < y.Dblockarray.pnumblocks; i++)
        {
            ptrdiff_t b = y.pblock_linear_idx[i];

            ptrdiff_t start = b * bs;
            ptrdiff_t diff  = M - start;
            ptrdiff_t len   = (bs < diff) ? bs : diff;

            ptrdiff_t global_offset = start;

            ptrdiff_t local_offset = global_offset - offset;

            ptrdiff_t dst = y.Dblockarray.pblock_offsets[i];
            memcpy( y.Dblockarray.pdata + dst,    y_local + local_offset,len * sizeof(T   ));

        }
    }
    if(recvcounts[rank]>0)
        DataBlock_MPI_Functions::free_helper2<T>(MPI_Sendlocation{.with_memmap=memmap,.ondevice=ongpu,.devicenum=devnum},recvcounts[rank],y_local);

    delete[] recvcounts;

    return true;
}



template<typename T>
inline bool Math_Functions_MPI::matrix_linear_combination_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    DistributedDataBlock<T>& C,
    const T CoefficientA,
    const T CoefficientB,
    const T CoefficientC,
    const Math_MPI_Functions_Policy* pol)
{



    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : Math_Functions_MPI::get_default_policy();


    bool ongpu=policy.should_use_gpu_elementwise(A, B, C);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if (!matrix_extents_equal(A,B,C)) return false;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;
    if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
    if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;

    if(A.Dblockarray.pnumblocks!=B.Dblockarray.pnumblocks || A.Dblockarray.pnumblocks!=C.Dblockarray.pnumblocks) return false;

    const ptrdiff_t cblocknum=C.Dblockarray.pnumblocks;
    if (cblocknum == 0) return true;

    const DataBlockArray Ablockarray=A.Dblockarray;
    const DataBlockArray Bblockarray=B.Dblockarray;
    DataBlockArray Cblockarray=C.Dblockarray;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadB(Bblockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>      offloadC(Cblockarray, devnum, CoefficientC==T(0), true);

        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b) =CoefficientC*Cblockarray(i,j,b)+CoefficientA* Ablockarray(i,j,b)+CoefficientB*Bblockarray(i,j,b);
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b) =CoefficientC*Cblockarray(i,j,b)+CoefficientA* Ablockarray(i,j,b)+CoefficientB*Bblockarray(i,j,b);
                }
            }
        }
    }

    return true;
}



template<typename T>
inline bool matrix_linear_combination_Distributed(
    const DistributedDataBlock<T>& A,
    DistributedDataBlock<T>& C,
    const T CoefficientA,
    const T CoefficientC,
    const Math_MPI_Functions_Policy* pol)
{



    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : Math_Functions_MPI::get_default_policy();


    bool ongpu=policy.Math_Functions_Policy::should_use_gpu_elementwise(A,  C);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if (!matrix_extents_equal(A,C)) return false;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
    if(A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;

    if(A.Dblockarray.pnumblocks!=C.Dblockarray.pnumblocks) return false;

    const ptrdiff_t cblocknum=C.Dblockarray.pnumblocks;
    if (cblocknum == 0) return true;

    const DataBlockArray Ablockarray=A.Dblockarray;
    DataBlockArray Cblockarray=C.Dblockarray;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>      offloadC(Cblockarray, devnum, CoefficientC==T(0), true);

        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b) =CoefficientC*Cblockarray(i,j,b)+CoefficientA* Ablockarray(i,j,b);
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b) =CoefficientC*Cblockarray(i,j,b)+CoefficientA* Ablockarray(i,j,b);
                }
            }
        }
    }

    return true;
}



template <typename T>
inline bool Math_Functions_MPI::matrix_multiply_scalar_Distributed(const DistributedDataBlock<T>& A,  const T B,  DistributedDataBlock<T>& C,   const Math_MPI_Functions_Policy* pol )
{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : get_default_policy();


    bool ongpu=policy.Math_Functions_Policy::should_use_gpu_elementwise(A,B,C);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if (!matrix_extents_equal(A,B,C)) return false;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;
    if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
    if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;

    if(A.Dblockarray.pnumblocks!=B.Dblockarray.pnumblocks || A.Dblockarray.pnumblocks!=C.Dblockarray.pnumblocks) return false;

    const ptrdiff_t cblocknum=C.Dblockarray.pnumblocks;
    if (cblocknum == 0) return true;

    const  DataBlockArray Ablockarray=A.Dblockarray;
    DataBlockArray Cblockarray=C.Dblockarray;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>      offloadC(Cblockarray, devnum, true, true);

        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b) = Ablockarray(i,j,b)*B;
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b)  =Ablockarray(i,j,b) *B;
                }
            }
        }
    }

    return true;
}


template <typename T>
inline bool Math_Functions_MPI::matrix_multiply_scalar_Distributed(DistributedDataBlock<T>& A,  const T B,   const Math_MPI_Functions_Policy* pol)
{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : get_default_policy();


    bool ongpu=policy.should_use_gpu_matrix(A);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;


    DataBlockArray Ablockarray=A.Dblockarray;
    const ptrdiff_t ablocknum=A.Dblockarray.pnumblocks;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>      offloadA(Ablockarray, devnum, true, true);
        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<ablocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Ablockarray(i,j,b)*=B;
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<ablocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Ablockarray(i,j,b) *=B ;
                }
            }
        }
    }

    return true;
}


template <typename T>
inline bool Math_Functions_MPI::vector_multiply_scalar_Distributed(const DistributedDataBlock<T>& A,  T B,  DistributedDataBlock<T>& C,   const Math_MPI_Functions_Policy* pol )
{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : get_default_policy();


    bool ongpu=policy.Math_Functions_Policy::should_use_gpu_elementwise(A, B, C);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if (!vector_extents_equal(A,C)) return false;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;
    if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
    if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;

    if(A.Dblockarray.pnumblocks!=B.Dblockarray.pnumblocks || A.Dblockarray.pnumblocks!=C.Dblockarray.pnumblocks) return false;

    const ptrdiff_t cblocknum=C.Dblockarray.pnumblocks;
    if (cblocknum == 0) return true;

    const  DataBlockArray Ablockarray=A.Dblockarray;
    DataBlockArray Cblockarray=C.Dblockarray;

    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>      offloadC(Cblockarray, devnum, true, true);

        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[b];
            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Cblockarray(i,b) = Ablockarray(i,b)*B;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[2 * b];
            const ptrdiff_t m = Ablockarray.pextentsbuffer[2 * b + 1];
            #pragma omp simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <m ; ++j)
                {
                    Cblockarray(i,j,b)  =Ablockarray(i,j,b) *B;
                }
            }
        }
    }

    return true;
}


template <typename T>
inline bool Math_Functions_MPI::vector_multiply_scalar_Distributed(DistributedDataBlock<T>& A,  const T B,   const Math_MPI_Functions_Policy* pol)
{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : get_default_policy();


    bool ongpu=policy.Math_Functions_Policy::should_use_gpu_elementwise(A);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;

    if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ) return false;
    if(A.Dblockarray.pnumblocks!=B.Dblockarray.pnumblocks ) return false;

    DataBlockArray Ablockarray=A.Dblockarray;
    const ptrdiff_t ablocknum=A.Dblockarray.pnumblocks;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>      offloadA(Ablockarray, devnum, true, true);
        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<ablocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[b];
            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Ablockarray(i,b)*=B;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<ablocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[b];
            #pragma omp simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Ablockarray(i,b) *=B ;
            }
        }
    }

    return true;
}


template<typename T>
inline bool Math_Functions_MPI::Math_Functions_MPI::vector_linear_combination_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    DistributedDataBlock<T>& C,
    const T CoefficientA,
    const T CoefficientB,
    const T CoefficientC,
    const Math_MPI_Functions_Policy* pol)

{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : Math_Functions_MPI::get_default_policy();


    bool ongpu=policy.should_use_gpu_vector(A,B,C);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if (!vector_extents_equal(A,B,C)) return false;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(B.Dblockarray.pdata_is_devptr&& B.Dblockarray.pdevnum!=devnum) return false;
    if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
    if(A.Dblockarray.pdevnum!=B.Dblockarray.pdevnum ||A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;

    if(A.Dblockarray.pnumblocks!=B.Dblockarray.pnumblocks || A.Dblockarray.pnumblocks!=C.Dblockarray.pnumblocks) return false;

    const ptrdiff_t cblocknum=C.Dblockarray.pnumblocks;
    if (cblocknum == 0) return true;

    const  DataBlockArray Ablockarray=A.Dblockarray;
    const  DataBlockArray Bblockarray=B.Dblockarray;
    DataBlockArray Cblockarray=C.Dblockarray;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadB(Bblockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>    offloadC(Cblockarray, devnum, true, true);

        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[ b];
            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Cblockarray(i,b) =CoefficientC*Cblockarray(i,b)+CoefficientA* Ablockarray(i,b)+CoefficientB* Bblockarray(i,b);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[ b];
            #pragma omp simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Cblockarray(i,b) =CoefficientC*Cblockarray(i,b)+CoefficientA* Ablockarray(i,b)+CoefficientB* Bblockarray(i,b);
            }
        }
    }

    return true;
}


template<typename T>
inline bool Math_Functions_MPI::vector_linear_combination_Distributed(
    const DistributedDataBlock<T>& A,
    DistributedDataBlock<T>& C,
    const T CoefficientA,
    const T CoefficientC,
    const Math_MPI_Functions_Policy* pol)

{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : Math_Functions_MPI::get_default_policy();


    bool ongpu=policy.should_use_gpu_vector(A,C);
    bool memmap=policy.memmapped_files;
    int devnum=policy.devicenum;

    if (!vector_extents_equal(A,C)) return false;

    if(A.Dblockarray.pdata_is_devptr&& A.Dblockarray.pdevnum!=devnum) return false;
    if(C.Dblockarray.pdata_is_devptr&& C.Dblockarray.pdevnum!=devnum) return false;
    if(A.Dblockarray.pdevnum!=C.Dblockarray.pdevnum) return false;

    if( A.Dblockarray.pnumblocks!=C.Dblockarray.pnumblocks) return false;

    const ptrdiff_t cblocknum=C.Dblockarray.pnumblocks;
    if (cblocknum == 0) return true;

    const  DataBlockArray Ablockarray=A.Dblockarray;
    DataBlockArray Cblockarray=C.Dblockarray;
    if (ongpu)
    {

        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>    offloadC(Cblockarray, devnum, true, true);

        #pragma omp target teams distribute
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[ b];
            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Cblockarray(i,b) =CoefficientC*Cblockarray(i,b)+CoefficientA* Ablockarray(i,b);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (ptrdiff_t b=0; b<cblocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[ b];
            #pragma omp simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Cblockarray(i,b) =CoefficientC*Cblockarray(i,b)+CoefficientA* Ablockarray(i,b);
            }
        }
    }

    return true;
}




template <typename T>
inline bool Math_Functions_MPI::dot_product_localblock(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    T* result,
    const Math_MPI_Functions_Policy* pol
)
{
    const Math_MPI_Functions_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    if (A.pctx->comm == MPI_COMM_NULL) return false;
    if (!vector_extents_equal(A, B)) return false;

    bool ongpu = policy.Math_Functions_Policy::should_use_gpu_elementwise(A, B);
    int devnum = policy.devicenum;

    if (A.Dblockarray.pdata_is_devptr && A.Dblockarray.pdevnum != devnum) return false;
    if (B.Dblockarray.pdata_is_devptr && B.Dblockarray.pdevnum != devnum) return false;
    if (A.Dblockarray.pdevnum != B.Dblockarray.pdevnum) return false;
    if (A.Dblockarray.pnumblocks != B.Dblockarray.pnumblocks) return false;

    T sum = T(0);
    const DataBlockArray Ablockarray=A.Dblockarray;
    const DataBlockArray Bblockarray=B.Dblockarray;
    const ptrdiff_t ablocknum=A.Dblockarray.pnumblocks;
    if (ongpu)
    {
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadA(Ablockarray, devnum);
        typename GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T> offloadB(Bblockarray, devnum);

        #pragma omp target data map (tofrom:sum)
        {
            #pragma omp target teams distribute reduction(+:sum)
            for (ptrdiff_t b=0; b<ablocknum; b++)
            {
                const ptrdiff_t n = Ablockarray.pextentsbuffer[b];
                #pragma omp parallel for simd reduction(+:sum)
                for (ptrdiff_t i = 0; i < n; ++i)
                {
                    sum += condconj(Ablockarray(i,b))  * Bblockarray(i,b) ;
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for reduction(+:sum)
        for (ptrdiff_t b=0; b<ablocknum; b++)
        {
            const ptrdiff_t n = Ablockarray.pextentsbuffer[b];
            #pragma omp simd reduction(+:sum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                sum += condconj(Ablockarray(i,b))  * Bblockarray(i,b) ;
            }
        }
    }

    *result=sum;

    return true;
}


template <typename T>
inline bool Math_Functions_MPI::dot_product_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    int root,
    T* result,
    const Math_MPI_Functions_Policy* pol
)
{
    T sum=0;
    //no error check... perhaps one should start to use exception handling...
    //if(!dot_product_localblock(A,B,&sum,pol)) return false;

    MPI_Reduce(&sum, result, 1, mpi_get_type<T>(), MPI_SUM, root, A.pctx->comm);

    return true;
}

template <typename T>
inline T Math_Functions_MPI::dot_product_Allreduce_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    const Math_MPI_Functions_Policy* pol
)
{
    T sum=0;
    if(!Math_Functions_MPI::dot_product_localblock(A,B,&sum,pol)) return false;

    T global_result = T(0);
    MPI_Allreduce(&sum, &global_result, 1, mpi_get_type<T>(), MPI_SUM, A.pctx->comm);
    return global_result;
}


template <typename T>
inline  void Math_Functions_MPI:: conjugate_from_root(  DistributedDataBlock<T>& A,int rootrank, MPI_Comm com)
{
    A.Dblockarray.pconjugate= !A.Dblockarray.pconjugate;
    MPI_Bcast(&A.Dblockarray.pconjugate,1,mpi_get_type<bool>(),rootrank,com);
}

template <typename T>
inline  void Math_Functions_MPI:: conjugate(  DistributedDataBlock<T>& A)
{
    A.Dblockarray.pconjugate=  !A.Dblockarray.pconjugate;
}


template <typename T>
void Math_Functions_MPI::strassen_multiply( const DataBlock<T> & A,const  DataBlock<T> & B, DataBlock<T> & C,MPI_Comm pcom,const Math_MPI_RecursiveMultiplication_Policy *pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu_winograd_start(A,B,C);
    bool separate_device_memory=false;
    if(ongpu)
{
#if !defined(Unified_Shared_Memory)
    separate_device_memory=true;
#endif
}
if(separate_device_memory)
{
    GPU_Memory_Functions::DataBlockdpdataoffloader<T> offloadA(A, policy.devicenum, false);
        GPU_Memory_Functions::DataBlockdpdataoffloader<T> offloadB(B, policy.devicenum, false);
        GPU_Memory_Functions::DataBlockdpdataoffloader<T> offloadC(C, policy.devicenum, policy.update_host);

        const DataBlock<T>  &tA=offloadA.get(),&tB=offloadB.get();
        DataBlock<T> &tC=offloadC.get_mutable();

        strassen_multiply_h(tA,tB,tC, ongpu, separate_device_memory,pcom,policy);
    }
    else
    {
        strassen_multiply_h(A,B,C,ongpu, false,pcom,policy);
    }

}


template <typename T>
void Math_Functions_MPI::strassen_multiply_h(const DataBlock<T> & A, const DataBlock<T> & B, DataBlock<T> & C,bool ongpu, bool separate_device_memory,MPI_Comm pcom, const Math_MPI_RecursiveMultiplication_Policy &policy)
{


    // Dimensions of input matrices
    ptrdiff_t n = A.dpextents[0]; // Rows in A
    ptrdiff_t m = A.dpextents[1]; // Columns in A and rows in B
    ptrdiff_t p = B.dpextents[1]; // Columns in B


    if (policy.should_use_naive_algorithm(A,B,C,ongpu))
    {
        if(ongpu)
        {
            GPU_Math_Functions::matrix_multiply_dot_g(   A,B,  C,GPUOptions{.device=policy.devicenum,.update_host=false});
            return;
        }
        else
        {
            switch (policy.mode)
            {
            case Math_Functions_Policy::GPU_ONLY:
            {
                GPU_Math_Functions::matrix_multiply_dot_g(   A,B,  C,GPUOptions{.device=policy.devicenum,.update_host=true});
                return;
            }
            case Math_Functions_Policy::AUTO:
            {
                if(policy.Math_Functions_Policy::should_use_gpu_matrix_multiply(A, B, C))
                    GPU_Math_Functions::matrix_multiply_dot_g(A,B,C,GPUOptions{.device=policy.devicenum,.update_host=true});
                else
                    In_Kernel_Mathfunctions::matrix_multiply_dot( A,B,C);
                return;
            }
            default:
            {
                In_Kernel_Mathfunctions::matrix_multiply_dot( A,B,  C);
                return;
            }
            }
        }
    }

    ptrdiff_t half_n = n / 2;
    ptrdiff_t half_m = m / 2;
    ptrdiff_t half_p = p / 2;

// Submatrices of A

    ptrdiff_t psext1[2],psext2[2],psext3[2],psext4[2],psext5[2],psext6[2],psext7[2],psext8[2];
    ptrdiff_t a11str[2],a12str[2], a21str[2], a22str[2], b11str[2], b12str[2], b21str[2], b22str[2];




// Temporary storage for intermediate results
    const ptrdiff_t s=half_n*half_p,
                    s2=half_n*half_m,
                    s3=half_m*half_p;

    ptrdiff_t ext1[2]= {half_n, half_p};
    ptrdiff_t str1[2]= {half_p, 1};


    ptrdiff_t ext2[2]= {half_n, half_m};
    ptrdiff_t str2[2]= {half_m, 1};


    ptrdiff_t ext3[2]=  {half_m, half_p};
    ptrdiff_t str3[2]= {half_p, 1};




    T* Ard1,*Ard2,*Ard3,*Ard4,*Ard5,*Brd1,*Brd2,*Brd3,*Brd4,*Brd5,*M1d,*M2d,*M3d,*M4d,*M5d,*M6d,*M7d;
    const bool aconj=A.dpconjugate,
               bconj=B.dpconjugate;

    if(separate_device_memory)
    {
        Ard1=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard2=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard3=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard4=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard5=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);

        Brd1=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd2=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd3=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd4=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd5=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);

        M1d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M2d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M3d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M4d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M5d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M6d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M7d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
    }
    else
    {
        if(policy.memmapped_files)
        {
            Ard1=Host_Memory_Functions::create_temp_mmap<T>(s2);
            Ard2=Host_Memory_Functions::create_temp_mmap<T>(s2);
            Ard3=Host_Memory_Functions::create_temp_mmap<T>(s2);
            Ard4=Host_Memory_Functions::create_temp_mmap<T>(s2);
            Ard5=Host_Memory_Functions::create_temp_mmap<T>(s2);

            Brd1=Host_Memory_Functions::create_temp_mmap<T>(s3);
            Brd2=Host_Memory_Functions::create_temp_mmap<T>(s3);
            Brd3=Host_Memory_Functions::create_temp_mmap<T>(s3);
            Brd4=Host_Memory_Functions::create_temp_mmap<T>(s3);
            Brd5=Host_Memory_Functions::create_temp_mmap<T>(s3);

            M1d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M2d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M3d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M4d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M5d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M6d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M7d=Host_Memory_Functions::create_temp_mmap<T>(s);
        }
        else
        {
            Ard1=new T[s2];
            Ard2=new T[s2];
            Ard3=new T[s2];
            Ard4=new T[s2];
            Ard5=new T[s2];

            Brd1=new T[s3];
            Brd2=new T[s3];
            Brd3=new T[s3];
            Brd4=new T[s3];
            Brd5=new T[s3];

            M1d=new T[s];
            M2d=new T[s];
            M3d=new T[s];
            M4d=new T[s];
            M5d=new T[s];
            M6d=new T[s];
            M7d=new T[s];
        }
    }


    DataBlockConfig
    aconfig=DataBlockConfig
    {
        .dprowmajor=A.dpconfig.dprowmajor,
        .pmemmap=A.dpconfig.pmemmap,
        .data_is_devptr=separate_device_memory,
        .devicenum=separate_device_memory? policy.devicenum:-INT_MAX,
    },

    bconfig=DataBlockConfig
    {
        .dprowmajor=B.dpconfig.dprowmajor,
        .pmemmap=B.dpconfig.pmemmap,
        .data_is_devptr=separate_device_memory,
        .devicenum=separate_device_memory? policy.devicenum:-INT_MAX,
    },

    mconfig=DataBlockConfig
    {
        .dprowmajor=true,
        .pmemmap=policy.memmapped_files,
        .data_is_devptr=separate_device_memory,
        .devicenum=separate_device_memory? policy.devicenum:-INT_MAX,
    };



    DataBlock<T>
    A_result1(Ard1,s2,2,ext2,str2,aconfig),
              A_result2(Ard2,s2,2,ext2,str2,aconfig),
              A_result3(Ard3,s2,2,ext2,str2,aconfig),
              A_result4(Ard4,s2,2,ext2,str2,aconfig),
              A_result5(Ard5,s2,2,ext2,str2,aconfig),

              B_result1(Brd1,s2,2,ext3,str3,bconfig),
              B_result2(Brd2,s2,2,ext3,str3,bconfig),
              B_result3(Brd3,s2,2,ext3,str3,bconfig),
              B_result4(Brd4,s2,2,ext3,str3,bconfig),
              B_result5(Brd5,s2,2,ext3,str3,bconfig),

              M1(M1d,s,2,ext1,str1,mconfig),
              M2(M2d,s,2,ext1,str1,mconfig),
              M3(M3d,s,2,ext1,str1,mconfig),
              M4(M4d,s,2,ext1,str1,mconfig),
              M5(M5d,s,2,ext1,str1,mconfig),
              M6(M6d,s,2,ext1,str1,mconfig),
              M7(M7d,s,2,ext1,str1,mconfig);


    DataBlock<T>  A11 = DataBlockUtilities::matrix_subspan(A,0, 0, half_n, half_m,psext1,a11str),
                  A12 = DataBlockUtilities::matrix_subspan(A,0, half_m, half_n, half_m,psext2,a12str),
                  A21 = DataBlockUtilities::matrix_subspan(A,half_n, 0, half_n, half_m,psext3,a21str),
                  A22 = DataBlockUtilities::matrix_subspan(A,half_n, half_m, half_n, half_m,psext4,a22str);

// Submatrices of B
    DataBlock<T>   B11 = DataBlockUtilities::matrix_subspan(B,0, 0, half_m, half_p,psext5,b11str),
                   B12 = DataBlockUtilities::matrix_subspan(B,0, half_p, half_m, half_p,psext6,b12str),
                   B21 = DataBlockUtilities::matrix_subspan(B,half_m, 0, half_m, half_p,psext7,b21str),
                   B22 = DataBlockUtilities::matrix_subspan(B,half_m, half_p, half_m, half_p,psext8,b22str);

    const ptrdiff_t str20=str2[0];
    const ptrdiff_t str21=str2[1];
    const ptrdiff_t str30=str3[0];
    const ptrdiff_t str31=str3[1];

    const ptrdiff_t a11str0=a11str[0];
    const ptrdiff_t a11str1=a11str[1];

    const ptrdiff_t a12str0=a12str[0];
    const ptrdiff_t a12str1=a12str[1];

    const ptrdiff_t a21str0=a21str[0];
    const ptrdiff_t a21str1=a21str[1];

    const ptrdiff_t a22str0=a22str[0];
    const ptrdiff_t a22str1=a22str[1];

    const ptrdiff_t b11str0=b11str[0];
    const ptrdiff_t b11str1=b11str[1];

    const ptrdiff_t b12str0=b12str[0];
    const ptrdiff_t b12str1=b12str[1];

    const ptrdiff_t b21str0=b21str[0];
    const ptrdiff_t b21str1=b21str[1];

    const ptrdiff_t b22str0=b22str[0];
    const ptrdiff_t b22str1=b22str[1];


    const T* A11d=A11.dpdata;
    const T* A12d=A12.dpdata;
    const T* A21d=A21.dpdata;
    const T* A22d=A22.dpdata;

    const T* B11d=B11.dpdata;
    const T* B12d=B12.dpdata;
    const T* B21d=B21.dpdata;
    const T* B22d=B22.dpdata;


    if (ongpu)
    {

        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(Ard1,Ard2,Ard3,Ard4,Ard5,A11d,A12d,A21d,A22d)
        for (ptrdiff_t i=0; i<half_n; i++)
        {
            for (ptrdiff_t j=0; j<half_m; j++)
            {
                const T a11dd=returnval(A11d[i*a11str0+j*a11str1],aconj);
                const T a22dd=returnval(A22d[i*a22str0+j*a22str1],aconj);
                const T a21dd=returnval(A21d[i*a21str0+j*a21str1],aconj);
                const T a12dd=returnval(A12d[i*a12str0+j*a12str1],aconj);
                const ptrdiff_t aindex=i*str20+j*str21;
                Ard1[aindex]=a11dd+a22dd;
                Ard2[aindex]=a21dd+a22dd;
                Ard3[aindex]=a11dd+a12dd;
                Ard4[aindex]=a21dd-a11dd;
                Ard5[aindex]=a12dd-a22dd;
            }
        }

        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(Brd1,Brd2,Brd3,Brd4,Brd5,B11d,B12d,B21d,B22d)
        for (ptrdiff_t i=0; i<half_m; i++)
        {
            for (ptrdiff_t j=0; j<half_p; j++)
            {
                const T b11dd=returnval(B11d[i*b11str0+j*b11str1],bconj);
                const T b21dd=returnval(B21d[i*b21str0+j*b21str1],bconj);
                const T b12dd=returnval(B12d[i*b12str0+j*b12str1],bconj);
                const T b22dd=returnval(B22d[i*b22str0+j*b22str1],bconj);
                const ptrdiff_t bindex=i*str30+j*str31;
                Brd1[bindex]=b11dd+b22dd;
                Brd2[bindex]=b12dd-b22dd;
                Brd3[bindex]=b21dd-b11dd;
                Brd4[bindex]=b11dd+b12dd;
                Brd5[bindex]=b21dd+b22dd;
            }
        }

    }
    else
    {
        #pragma omp parallel for simd collapse (2)
        for (ptrdiff_t i=0; i<half_n; i++)
        {
            for (ptrdiff_t j=0; j<half_m; j++)
            {
                const T a11dd=returnval(A11d[i*a11str0+j*a11str1],aconj);
                const T a22dd=returnval(A22d[i*a22str0+j*a22str1],aconj);
                const T a21dd=returnval(A21d[i*a21str0+j*a21str1],aconj);
                const T a12dd=returnval(A12d[i*a12str0+j*a12str1],aconj);
                const ptrdiff_t aindex=i*str20+j*str21;
                Ard1[aindex]=a11dd+a22dd;
                Ard2[aindex]=a21dd+a22dd;
                Ard3[aindex]=a11dd+a12dd;
                Ard4[aindex]=a21dd-a11dd;
                Ard5[aindex]=a12dd-a22dd;
            }
        }

        #pragma omp parallel for simd collapse (2)
        for (ptrdiff_t i=0; i<half_m; i++)
        {
            for (ptrdiff_t j=0; j<half_p; j++)
            {
                const T b11dd=returnval(B11d[i*b11str0+j*b11str1],bconj);
                const T b21dd=returnval(B21d[i*b21str0+j*b21str1],bconj);
                const T b12dd=returnval(B12d[i*b12str0+j*b12str1],bconj);
                const T b22dd=returnval(B22d[i*b22str0+j*b22str1],bconj);
                const ptrdiff_t bindex=i*str30+j*str31;
                Brd1[bindex]=b11dd+b22dd;
                Brd2[bindex]=b12dd-b22dd;
                Brd3[bindex]=b21dd-b11dd;
                Brd4[bindex]=b11dd+b12dd;
                Brd5[bindex]=b21dd+b22dd;
            }
        }
    }

    bool usempi=false;
    int mpi_init =false;
    MPI_Initialized( &mpi_init );
    if ( !mpi_init || (pcom==MPI_COMM_NULL))
    {
        usempi=false;
    }
    else
    {
    int myrank = 0;
    int mpi_size = 1;

    MPI_Comm_rank(pcom, &myrank);
    MPI_Comm_size(pcom, &mpi_size);
    usempi=policy.should_use_mpi_for_recursion(myrank, mpi_size);
    }

    if(usempi)
    {
        int myrank=0,childdest=0;

        MPI_Comm_rank(pcom, &myrank);
        childdest=myrank*7;

        int message=Math_MPI_RecursiveMultiplication_Policy::Strassen;


        MPI_Send(&message, 1, MPI_INT, childdest+1,0, pcom);
        ptrdiff_t dims[3] ={half_n,half_m,half_p};


        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+1,1, pcom);

        DataBlock_MPI_Functions::MPI_Send_DataBlock(A_result1,childdest+1,2, pcom);

        DataBlock_MPI_Functions::MPI_Send_DataBlock(B_result1,childdest+1,3, pcom);




        MPI_Send(&message, 1, MPI_INT, childdest+2, 0,  pcom);
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+2,1,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A_result2,childdest+2,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B11,childdest+2,3, pcom);



        MPI_Send(&message, 1, MPI_INT, childdest+3, 0,  pcom);
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+3,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A11,childdest+3,2, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B_result2,childdest+3,3, pcom);


        MPI_Send(&message, 1, MPI_INT, childdest+4, 0,  pcom);
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+4,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A22,childdest+4,2, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B_result3,childdest+4,3, pcom);


        MPI_Send(&message, 1, MPI_INT, childdest+5, 0,    pcom);
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+5,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A_result3,childdest+5,2,  pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B22,childdest+5,3,   pcom);


        MPI_Send(&message, 1, MPI_INT, childdest+6, 0,  pcom);
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+6,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A_result4,childdest+6,2, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B_result4,childdest+6,3, pcom);


        MPI_Send(&message, 1, MPI_INT, childdest+7, 0, pcom);
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+7,1, pcom);

        DataBlock_MPI_Functions::MPI_Send_DataBlock(A_result5,childdest+7,2,  pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B_result5,childdest+7,3,  pcom);

        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M1,childdest+1,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M2,childdest+2,4, pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M3,childdest+3,4, pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M4,childdest+4,4, pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M5,childdest+5,4, pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M6,childdest+6,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M7,childdest+7,4, pcom);

    }
    else
    {
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    strassen_multiply_h(A_result1, B_result1,   M1,ongpu, separate_device_memory,pcom, policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result2, B11,         M2,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A11, B_result2,         M3,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A22, B_result3,         M4,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result3, B22,         M5,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result4, B_result4,   M6,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result5, B_result5,   M7,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp taskwait
            }
        }
    }

    ptrdiff_t ext11a[2],ext12a[2],ext13a[2],ext14a[2];
    ptrdiff_t cstr11[2], cstr12[2], cstr21[2], cstr22[2];

// Submatrices of C

    DataBlock<T>   C11 = DataBlockUtilities::matrix_subspan(C,0, 0, half_n, half_p,ext11a,cstr11),
                   C12 = DataBlockUtilities::matrix_subspan(C,0, half_p, half_n, half_p,ext12a,cstr12),
                   C21 = DataBlockUtilities::matrix_subspan(C,half_n, 0, half_n, half_p,ext13a,cstr21),
                   C22 = DataBlockUtilities::matrix_subspan(C,half_n, half_p, half_n, half_p,ext14a,cstr22);

    const ptrdiff_t cstr110=cstr11[0];
    const ptrdiff_t cstr111=cstr11[1];

    const ptrdiff_t cstr120=cstr12[0];
    const ptrdiff_t cstr121=cstr12[1];

    const ptrdiff_t cstr210=cstr21[0];
    const ptrdiff_t cstr211=cstr21[1];

    const ptrdiff_t cstr220=cstr22[0];
    const ptrdiff_t cstr221=cstr22[1];
    T* C11d=C11.dpdata;
    T* C12d=C12.dpdata;
    T* C21d=C21.dpdata;
    T* C22d=C22.dpdata;

    const ptrdiff_t str10=str1[0];
    const ptrdiff_t str11=str1[1];
    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(C11d,C12d,C21d,C22d,M1d,M2d,M3d,M4d,M5d,M6d)
        for (ptrdiff_t i = 0; i < half_n; i++)
        {
            for (ptrdiff_t j = 0; j < half_p; j++)
            {
                const ptrdiff_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T helper1 = m1dd  +m4dd ;
                const T helper2 = -m5dd +m7dd ;

                C11d[i*cstr110+j*cstr111]  =  helper1 +helper2;
                C12d[i*cstr120+j*cstr121] = m3dd  + m5dd ;
                C21d[i*cstr210+j*cstr211] = m2dd  + m4dd ;

                T helper3 = m1dd - m2dd ;
                T helper4 = m3dd  + m6dd ;

                C22d[i*cstr220+j*cstr221]  =helper3+helper4;
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i = 0; i < half_n; i++)
        {
            for (ptrdiff_t j = 0; j < half_p; j++)
            {
                const ptrdiff_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T helper1 = m1dd  +m4dd ;
                const T helper2 = -m5dd +m7dd ;

                C11d[i*cstr110+j*cstr111]  =  helper1 +helper2;
                C12d[i*cstr120+j*cstr121] = m3dd  + m5dd ;
                C21d[i*cstr210+j*cstr211] = m2dd  + m4dd ;

                T helper3 = m1dd - m2dd ;
                T helper4 = m3dd  + m6dd ;

                C22d[i*cstr220+j*cstr221]  =helper3+helper4;
            }
        }
    }


    if(separate_device_memory)
    {
        omp_target_free(M1d,policy.devicenum);
        omp_target_free(M2d,policy.devicenum);
        omp_target_free(M3d,policy.devicenum);
        omp_target_free(M4d,policy.devicenum);
        omp_target_free(M5d,policy.devicenum);
        omp_target_free(M6d,policy.devicenum);
        omp_target_free(M7d,policy.devicenum);

        omp_target_free(Ard1,policy.devicenum);
        omp_target_free(Ard2,policy.devicenum);
        omp_target_free(Ard3,policy.devicenum);
        omp_target_free(Ard4,policy.devicenum);
        omp_target_free(Ard5,policy.devicenum);

        omp_target_free(Brd1,policy.devicenum);
        omp_target_free(Brd2,policy.devicenum);
        omp_target_free(Brd3,policy.devicenum);
        omp_target_free(Brd4,policy.devicenum);
        omp_target_free(Brd5,policy.devicenum);
    }

    else
    {
        if(policy.memmapped_files)
        {
            Host_Memory_Functions::delete_temp_mmap<T>(M1d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M2d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M3d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M4d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M5d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M6d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M7d,s);

            Host_Memory_Functions::delete_temp_mmap<T>(Ard1,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(Ard2,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(Ard3,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(Ard4,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(Ard5,s2);

            Host_Memory_Functions::delete_temp_mmap<T>(Brd1,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(Brd2,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(Brd3,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(Brd4,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(Brd5,s3);
        }
        else
        {
            delete[]M1d;
            delete[]M2d;
            delete[]M3d;
            delete[]M4d;
            delete[]M5d;
            delete[]M6d;
            delete[]M7d;
            delete[]Ard1;
            delete[]Ard2;
            delete[]Ard3;
            delete[]Ard4;
            delete[]Ard5;
            delete[]Brd1;
            delete[]Brd2;
            delete[]Brd3;
            delete[]Brd4;
            delete[]Brd5;
        }
    }

}


template <typename T>
void Math_Functions_MPI::winograd_multiply(const DataBlock<T>& A, const DataBlock<T> &B, DataBlock<T>& C,MPI_Comm pcom, const Math_MPI_RecursiveMultiplication_Policy*pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu_matrix_multiply(A,B,C);
    bool separate_device_memory=false;
    if(ongpu)
    {


#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }

    if(separate_device_memory)
    {
        GPU_Memory_Functions::DataBlockdpdataoffloader<T> offloadA(A, policy.devicenum, false);
        GPU_Memory_Functions::DataBlockdpdataoffloader<T> offloadB(B, policy.devicenum, false);
        GPU_Memory_Functions::DataBlockdpdataoffloader<T> offloadC(C, policy.devicenum, policy.update_host);

        const DataBlock<T>  &tA=offloadA.get(), &tB=offloadB.get();
        DataBlock<T>& tC=offloadC.get_mutable();
        winograd_multiply_h(tA,tB,tC,ongpu, separate_device_memory,pcom,policy);


    }
    else
    {
        winograd_multiply_h(A,B,C,ongpu,false,pcom,policy);
    }

}

template <typename T>
void Math_Functions_MPI::winograd_multiply_h(const DataBlock<T>& A,const DataBlock<T> &B, DataBlock<T>& C,bool ongpu, bool separate_device_memory,MPI_Comm pcom, const Math_MPI_RecursiveMultiplication_Policy&policy)
{
    // Dimensions of input matrices
    ptrdiff_t n = A.dpextents[0]; // Rows in A
    ptrdiff_t m = A.dpextents[1]; // Columns in A and rows in B
    ptrdiff_t p = B.dpextents[1]; // Columns in B


    if ( policy.should_use_naive_algorithm(A,B,C,ongpu))
    {
        if(ongpu)
        {
            GPU_Math_Functions::matrix_multiply_dot_g(   A,B,  C,GPUOptions{.device=policy.devicenum,.update_host=true});
            return;
        }
        else
        {
            switch (policy.mode)
            {
            case Math_Functions_Policy::GPU_ONLY:
            {
                GPU_Math_Functions::matrix_multiply_dot_g(   A,B,  C,GPUOptions{.device=policy.devicenum,.update_host=true});
                return;
            }
            case Math_Functions_Policy::AUTO:
            {
                if(policy.Math_Functions_Policy::should_use_gpu_matrix_multiply(A, B, C))
                    GPU_Math_Functions::matrix_multiply_dot_g(A,B,C,GPUOptions{.device=policy.devicenum,.update_host=true});
                else
                    In_Kernel_Mathfunctions::matrix_multiply_dot( A,B,C);
                return;
            }
            default:
            {
                In_Kernel_Mathfunctions::matrix_multiply_dot( A,B,  C);
                return;
            }
            }
        }
    }

    // Compute sizes for splitting

    ptrdiff_t half_n = n / 2;
    ptrdiff_t half_m = m / 2;
    ptrdiff_t half_p = p / 2;

    // Submatrices of A
    const bool aconj=A.dpconjugate,
               bconj=B.dpconjugate;

    ptrdiff_t psext1[2],psext2[2], psext3[2], psext4[2],psext5[2],psext6[2],psext7[2],psext8[2];
    ptrdiff_t a11str[2],a12str[2], a21str[2],  a22str[2],b11str[2],b12str[2],b21str[2],b22str[2];


    ptrdiff_t s=half_n*half_p;
    ptrdiff_t s2=half_n*half_m;
    ptrdiff_t s3=half_m*half_p;


    ptrdiff_t ext1[2]= {half_n, half_p};
    ptrdiff_t str1[2]= {half_p, 1};



    ptrdiff_t ext2[2]= {half_n, half_m};
    ptrdiff_t str2[2]= {half_m, 1};


    ptrdiff_t ext3[2]=  {half_m, half_p};
    ptrdiff_t str3[2]= {half_p, 1};



    T*S1d,*S2d,*S3d,*S4d,*S5d,*S6d,*S7d,*S8d,*M1d,*M2d,*M3d,*M4d,*M5d,*M6d,*M7d;
    if(separate_device_memory)
    {
        S1d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S2d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S3d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S4d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S5d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        S6d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        S7d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        S8d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        M1d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M2d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M3d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M4d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M5d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M6d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M7d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
    }
    else
    {
        if(policy.memmapped_files)
        {
            S1d=Host_Memory_Functions::create_temp_mmap<T>(s2);
            S2d=Host_Memory_Functions::create_temp_mmap<T>(s2);
            S3d=Host_Memory_Functions::create_temp_mmap<T>(s2);
            S4d=Host_Memory_Functions::create_temp_mmap<T>(s2);
            S5d=Host_Memory_Functions::create_temp_mmap<T>(s3);
            S6d=Host_Memory_Functions::create_temp_mmap<T>(s3);
            S7d=Host_Memory_Functions::create_temp_mmap<T>(s3);
            S8d=Host_Memory_Functions::create_temp_mmap<T>(s3);
            M1d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M2d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M3d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M4d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M5d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M6d=Host_Memory_Functions::create_temp_mmap<T>(s);
            M7d=Host_Memory_Functions::create_temp_mmap<T>(s);
        }
        else
        {
            S1d=new T[s2];
            S2d=new T[s2];
            S3d=new T[s2];
            S4d=new T[s2];
            S5d=new T[s3];
            S6d=new T[s3];
            S7d=new T[s3];
            S8d=new T[s3];
            M1d=new T[s];
            M2d=new T[s];
            M3d=new T[s];
            M4d=new T[s];
            M5d=new T[s];
            M6d=new T[s];
            M7d=new T[s];
        }

    }



    DataBlockConfig
    aconfig=DataBlockConfig
    {
        .dprowmajor=A.dpconfig.dprowmajor,
        .pmemmap=A.dpconfig.pmemmap,
        .data_is_devptr=separate_device_memory,
        .devicenum=separate_device_memory? policy.devicenum:-INT_MAX,
    },

    bconfig=DataBlockConfig
    {
        .dprowmajor=B.dpconfig.dprowmajor,
        .pmemmap=B.dpconfig.pmemmap,
        .data_is_devptr=separate_device_memory,
        .devicenum=separate_device_memory? policy.devicenum:-INT_MAX,
    },

    mconfig=DataBlockConfig
    {
        .dprowmajor=true,
        .pmemmap=policy.memmapped_files,
        .data_is_devptr=separate_device_memory,
        .devicenum=separate_device_memory? policy.devicenum:-INT_MAX,
    };

    DataBlock<T>
    S1(S1d,s2,2,ext2,str2,aconfig),
    S2(S2d,s2,2,ext2,str2,aconfig),
    S3(S3d,s2,2,ext2,str2,aconfig),
    S4(S4d,s2,2,ext2,str2,aconfig),
    S5(S5d,s3,2,ext3,str3,bconfig),
    S6(S6d,s3,2,ext3,str3,bconfig),
    S7(S7d,s3,2,ext3,str3,bconfig),
    S8(S8d,s3,2,ext3,str3,bconfig),
    M1(M1d,s,2,ext1,str1,mconfig),
    M2(M2d,s,2,ext1,str1,mconfig),
    M3(M3d,s,2,ext1,str1,mconfig),
    M4(M4d,s,2,ext1,str1,mconfig),
    M5(M5d,s,2,ext1,str1,mconfig),
    M6(M6d,s,2,ext1,str1,mconfig),
    M7(M7d,s,2,ext1,str1,mconfig);



    DataBlock<T>  A11 = DataBlockUtilities::matrix_subspan(A,0, 0, half_n, half_m,psext1,a11str),
                  A12 = DataBlockUtilities::matrix_subspan(A,0, half_m, half_n, half_m,psext2,a12str),
                  A21 = DataBlockUtilities::matrix_subspan(A,half_n, 0, half_n, half_m,psext3,a21str),
                  A22 = DataBlockUtilities::matrix_subspan(A,half_n, half_m, half_n, half_m,psext4,a22str);

    // Submatrices of B
    DataBlock<T>  B11 = DataBlockUtilities::matrix_subspan(B,0, 0, half_m, half_p,psext5,b11str),
                  B12 = DataBlockUtilities::matrix_subspan(B,0, half_p, half_m, half_p,psext6,b12str),
                  B21 = DataBlockUtilities::matrix_subspan(B,half_m, 0, half_m, half_p,psext7,b21str),
                  B22 = DataBlockUtilities::matrix_subspan(B,half_m, half_p, half_m, half_p,psext8,b22str);


    const ptrdiff_t a11str0=a11str[0];
    const ptrdiff_t a11str1=a11str[1];

    const ptrdiff_t a12str0=a12str[0];
    const ptrdiff_t a12str1=a12str[1];

    const ptrdiff_t a21str0=a21str[0];
    const ptrdiff_t a21str1=a21str[1];

    const ptrdiff_t a22str0=a22str[0];
    const ptrdiff_t a22str1=a22str[1];

    const ptrdiff_t b11str0=b11str[0];
    const ptrdiff_t b11str1=b11str[1];

    const ptrdiff_t b12str0=b12str[0];
    const ptrdiff_t b12str1=b12str[1];

    const ptrdiff_t b21str0=b21str[0];
    const ptrdiff_t b21str1=b21str[1];

    const ptrdiff_t b22str0=b22str[0];
    const ptrdiff_t b22str1=b22str[1];

    const T* A11d=A11.dpdata;
    const T* A12d=A12.dpdata;
    const T* A21d=A21.dpdata;
    const T* A22d=A22.dpdata;

    const T* B11d=B11.dpdata;
    const T* B12d=B12.dpdata;
    const T* B21d=B21.dpdata;
    const T* B22d=B22.dpdata;

    const ptrdiff_t strs0=str3[0];
    const ptrdiff_t strs1=str3[1];

    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(A11d,A12d,A21d,A22d,S1d,S2d,S3d,S4d)
        for (ptrdiff_t i=0; i<half_n; i++)
        {
            for (ptrdiff_t j=0; j<half_m; j++)
            {

                const T a11dd=returnval(A11d[a11str0*i+a11str1*j],aconj);
                const T a12dd=returnval(A12d[a12str0*i+a12str1*j],aconj);
                const T a21dd=returnval(A21d[a21str0*i+a21str1*j],aconj);
                const T a22dd=returnval(A22d[a22str0*i+a22str1*j],aconj);

                const ptrdiff_t sindex=strs0*i+strs1*j;

                const T s1=a21dd+a22dd;
                const T s2=s1-a11dd;

                S1d[sindex]=s1;
                S2d[sindex]=s2;
                S3d[sindex]=a11dd-a21dd;
                S4d[sindex]=a12dd-s2;

            }
        }
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)is_device_ptr(B11d,B12d,B21d,B22d,S5d,S6d,S7d,S8d)
        for (ptrdiff_t i=0; i<half_m; i++)
        {
            for (ptrdiff_t j=0; j<half_p; j++)
            {
                const T b11dd=returnval(B11d[b11str0*i+b11str1*j],bconj);
                const T b12dd=returnval(B12d[b12str0*i+b12str1*j],bconj);
                const T b21dd=returnval(B21d[b21str0*i+b21str1*j],bconj);
                const T b22dd=returnval(B22d[b22str0*i+b22str1*j],bconj);

                const ptrdiff_t sindex=i*strs0+j*strs1;
                const T s5=b12dd-b11dd;
                const T s6=b22dd-s5;
                S5d[sindex]=s5;
                S6d[sindex]=b22dd-s5;
                S6d[sindex]=s6;
                S7d[sindex]=b22dd-b12dd;
                S8d[sindex]=s6-b21dd;
            }
        }
    }
    else
    {
        #pragma omp  parallel for simd collapse(2)
        for (ptrdiff_t i=0; i<half_n; i++)
        {
            for (ptrdiff_t j=0; j<half_m; j++)
            {
                const T a11dd=returnval(A11d[a11str0*i+a11str1*j],aconj);
                const T a12dd=returnval(A12d[a12str0*i+a12str1*j],aconj);
                const T a21dd=returnval(A21d[a21str0*i+a21str1*j],aconj);
                const T a22dd=returnval(A22d[a22str0*i+a22str1*j],aconj);

                const ptrdiff_t sindex=strs0*i+strs1*j;

                const T s1=a21dd+a22dd;
                const T s2=s1-a11dd;

                S1d[sindex]=s1;

                S2d[sindex]=s2;
                S3d[sindex]=a11dd-a21dd;
                S4d[sindex]=a12dd-s2;

            }
        }
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i=0; i<half_m; i++)
        {
            for (ptrdiff_t j=0; j<half_p; j++)
            {
                const T b11dd=returnval(B11d[b11str0*i+b11str1*j],bconj);
                const T b12dd=returnval(B12d[b12str0*i+b12str1*j],bconj);
                const T b21dd=returnval(B21d[b21str0*i+b21str1*j],bconj);
                const T b22dd=returnval(B22d[b22str0*i+b22str1*j],bconj);

                const ptrdiff_t sindex=i*strs0+j*strs1;
                const T s5=b12dd-b11dd;
                const T s6=b22dd-s5;
                S5d[sindex]=s5;
                S6d[sindex]=b22dd-s5;
                S6d[sindex]=s6;
                S7d[sindex]=b22dd-b12dd;
                S8d[sindex]=s6-b21dd;
            }
        }
    }

    bool usempi=false;
    int mpi_init =false;
    int myrank = 0;
    int mpi_size = 1;

    MPI_Initialized( &mpi_init );
    if ( !mpi_init || (pcom==MPI_COMM_NULL))
    {
        usempi=false;
    }
    else
    {


    MPI_Comm_rank(pcom, &myrank);
    MPI_Comm_size(pcom, &mpi_size);
    usempi=policy.should_use_mpi_for_recursion(myrank, mpi_size);
    }

    if(usempi)
    {

       int childdest=myrank*7;


        int message=Math_MPI_RecursiveMultiplication_Policy::WinogradVariant;

        MPI_Send(&message, 1, MPI_INT, childdest+1,0, pcom);
        ptrdiff_t dims[3] ={half_n,half_m,half_p};
        MPI_Send(dims, 3, mpi_get_type<ptrdiff_t>(), childdest+1,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S2,childdest+1,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S6,childdest+1,3,pcom);


        MPI_Send(&message, 1, MPI_INT, childdest+2,0, pcom);
        MPI_Send(dims,3, mpi_get_type<ptrdiff_t>(), childdest+2,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A11,childdest+2,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B11,childdest+2,3,pcom);



        MPI_Send(&message, 1, MPI_INT, childdest+3,0, pcom);
        MPI_Send(dims,3, mpi_get_type<ptrdiff_t>(), childdest+3,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A12,childdest+3,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B21,childdest+3,3,pcom);

        MPI_Send(&message, 1, MPI_INT, childdest+4,0, pcom);
        MPI_Send(dims,3, mpi_get_type<ptrdiff_t>(), childdest+4,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S3,childdest+4,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S7,childdest+4,3,pcom);

        MPI_Send(&message, 1, MPI_INT, childdest+5,0, pcom);
        MPI_Send(dims,3, mpi_get_type<ptrdiff_t>(), childdest+5,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S1,childdest+5,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S5,childdest+5,3,pcom);

        MPI_Send(&message, 1, MPI_INT, childdest+6,0, pcom);
        MPI_Send(dims,3, mpi_get_type<ptrdiff_t>(), childdest+6,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S4,childdest+6,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(B22,childdest+6,3,pcom);

        MPI_Send(&message, 1, MPI_INT, childdest+7,0, pcom);
        MPI_Send(dims,3, mpi_get_type<ptrdiff_t>(), childdest+7,1, pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(A22,childdest+7,2,pcom);
        DataBlock_MPI_Functions::MPI_Send_DataBlock(S8,childdest+7,3,pcom);


        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M1,childdest+1,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M2,childdest+2,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M3,childdest+3,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M4,childdest+4,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M5,childdest+5,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M6,childdest+6,4,pcom);
        DataBlock_MPI_Functions::MPI_Recv_DataBlock_pdata(M7,childdest+7,4,pcom);

    }
    else
    {

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    winograd_multiply_h(S2,S6,M1, ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(A11,B11,M2, ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(A12,B21,M3, ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(S3,S7,M4,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(S1,S5,M5,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(S4,B22,M6,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(A22,S8,M7,ongpu,  separate_device_memory,pcom,policy);
                }
                #pragma omp taskwait
            }

        }

    }


    ptrdiff_t pext10a[2],pext11a[2],pext12a[2],pext13a[2];
    ptrdiff_t cstr11[2],cstr12[2],cstr21[2],cstr22[2];

    DataBlock<T>  C11 = DataBlockUtilities::matrix_subspan(C,0, 0, half_n, half_p,pext10a,cstr11),
                  C12 = DataBlockUtilities::matrix_subspan(C,0, half_p, half_n, half_p,pext11a,cstr12),
                  C21 = DataBlockUtilities::matrix_subspan(C,half_n, 0, half_n, half_p,pext12a,cstr21),
                  C22 = DataBlockUtilities::matrix_subspan(C,half_n, half_p, half_n, half_p,pext13a,cstr22);

    const ptrdiff_t cstr110=cstr11[0];
    const ptrdiff_t cstr111=cstr11[1];

    const ptrdiff_t cstr120=cstr12[0];
    const ptrdiff_t cstr121=cstr12[1];

    const ptrdiff_t cstr210=cstr21[0];
    const ptrdiff_t cstr211=cstr21[1];

    const ptrdiff_t cstr220=cstr22[0];
    const ptrdiff_t cstr221=cstr22[1];

    T* C11d=C11.dpdata;
    T* C12d=C12.dpdata;
    T* C21d=C21.dpdata;
    T* C22d=C22.dpdata;


    const ptrdiff_t str10=str1[0];
    const ptrdiff_t str11=str1[1];

    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(M1d,M2d,M3d,M4d,M5d,M6d,M7d,C11d,C12d,C21d,C22d)
        for (ptrdiff_t i = 0; i < half_n; ++i)
        {
            for (ptrdiff_t j = 0; j < half_p; ++j)
            {
                const ptrdiff_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T T1=m1dd+m2dd;
                const T T2=T1+m4dd;
                const T helper=m5dd+m6dd;
                C11d[cstr110*i+cstr111*j] = m2dd + m3dd;
                C12d[cstr120*i+cstr121*j] = T1 +helper ;
                C21d[cstr210*i+cstr211*j] = T2 - m7dd;
                C22d[cstr220*i+cstr221*j] = T2 + m5dd;
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i = 0; i < half_n; ++i)
        {
            for (ptrdiff_t j = 0; j < half_p; ++j)
            {
                const ptrdiff_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T T1=m1dd+m2dd;
                const T T2=T1+m4dd;
                const T helper=m5dd+m6dd;
                C11d[cstr110*i+cstr111*j] = m2dd + m3dd;
                C12d[cstr120*i+cstr121*j] = T1 +helper ;
                C21d[cstr210*i+cstr211*j] = T2 - m7dd;
                C22d[cstr220*i+cstr221*j] = T2 + m5dd;
            }
        }

    }


    if(separate_device_memory)
    {
        omp_target_free(M1d,policy.devicenum);
        omp_target_free(M2d,policy.devicenum);
        omp_target_free(M3d,policy.devicenum);
        omp_target_free(M4d,policy.devicenum);
        omp_target_free(M5d,policy.devicenum);
        omp_target_free(M6d,policy.devicenum);
        omp_target_free(M7d,policy.devicenum);

        omp_target_free(S1d,policy.devicenum);
        omp_target_free(S2d,policy.devicenum);
        omp_target_free(S3d,policy.devicenum);
        omp_target_free(S4d,policy.devicenum);
        omp_target_free(S5d,policy.devicenum);

        omp_target_free(S6d,policy.devicenum);
        omp_target_free(S7d,policy.devicenum);
        omp_target_free(S8d,policy.devicenum);

    }
    else
    {

        if(policy.memmapped_files)
        {
            Host_Memory_Functions::delete_temp_mmap<T>(M1d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M2d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M3d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M4d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M5d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M6d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(M7d,s);
            Host_Memory_Functions::delete_temp_mmap<T>(S1d,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(S2d,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(S3d,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(S4d,s2);
            Host_Memory_Functions::delete_temp_mmap<T>(S5d,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(S6d,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(S7d,s3);
            Host_Memory_Functions::delete_temp_mmap<T>(S8d,s3);
        }
        else
        {
            delete[]M1d;
            delete[]M2d;
            delete[]M3d;
            delete[]M4d;
            delete[]M5d;
            delete[]M6d;
            delete[]M7d;
            delete[]S1d;
            delete[]S2d;
            delete[]S3d;
            delete[]S4d;
            delete[]S5d;
            delete[]S6d;
            delete[]S7d;
            delete[]S8d;
        }
    }
}

template <typename T>
void Math_Functions_MPI::cholesky_decomposition(const DataBlock<T> & A, DataBlock<T> &L,MPI_Comm pcomm, Math_MPI_Decomposition_Policy *pol)
{

    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=pol->should_use_gpu_decomposition(A);
    bool separate_device_memory=false;
    if(ongpu)
    {
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }

    L.dpconjugate=false;

    bool aconj=A.dpconjugate;


    const ptrdiff_t n = A.dpextents[0];

    ptrdiff_t step_size=policy.step_size;

    if(step_size==0)
        step_size=(ptrdiff_t)pow(n,0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    ptrdiff_t tempsize=(n-step_size)*(n-step_size);


    if(ongpu)
    {
        T * sdata;
        T* tempad;

        if(separate_device_memory)
        {
            sdata= (T*) omp_target_alloc(sizeof(T)*tempsize, policy.devicenum);
            tempad= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                sdata=Host_Memory_Functions::create_temp_mmap<T>(tempsize);
                tempad=Host_Memory_Functions::create_temp_mmap<T>(A.dpdatalength);
            }
            else
            {
                sdata=new T[tempsize];
                tempad=new T[A.dpdatalength];
            }
        }
        ptrdiff_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        ptrdiff_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};


        DataBlockConfig tempAconf({.dprowmajor=A.dpconfig.dprowmajor,
                                   .pmemmap=policy.memmapped_files,
                                   .data_is_devptr=separate_device_memory,
                                   .devicenum=policy.devicenum,
                                  });
        DataBlock<T> tempA(tempad,A.dpdatalength,2,aext,astr,tempAconf);

        DataBlock<T> tA=A,tL=L;


        if(separate_device_memory)
        {
            GPU_Memory_Functions::create_in(A,policy.devicenum);
            GPU_Memory_Functions::create_out(L,policy.devicenum);

            if(!A.dpconfig.data_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);


            if(!L.dpconfig.data_is_devptr)
                tL.dpdata=(T*) omp_get_mapped_ptr(L.dpdata,policy.devicenum);


            tA.dpconfig.data_is_devptr=true;
            tL.dpconfig.data_is_devptr=true;
            tA.dpconfig.devicenum=policy.devicenum;
            tL.dpconfig.devicenum=policy.devicenum;
        }

        const ptrdiff_t Lstr0=tL.dpstrides[0];
        const ptrdiff_t Lstr1=tL.dpstrides[1];
        const ptrdiff_t Astr0=tA.dpstrides[0];
        const ptrdiff_t Astr1=tA.dpstrides[1];
        T* tempAptr=tempA.dpdata;
        T* tempLptr=tL.dpdata;
        const T* tAdptr=tA.dpdata;

        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for simd collapse(2)is_device_ptr(tempLptr,tempAptr,tAdptr) device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    tempLptr[i*Lstr0+j*Lstr1]=0;
                    tempAptr[i*Astr0+j*Astr1]=returnval(tAdptr[i*Astr0+j*Astr1],aconj);
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(tempLptr,tempAptr,tAdptr)  device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    tempAptr[i*Astr0+j*Astr1]=returnval(tAdptr[i*Astr0+j*Astr1],aconj);
                }
            }
        }

        ptrdiff_t z=0;

        DataBlockConfig sconf({.dprowmajor=true,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=true,
                               .devicenum=tA.dpconfig.devicenum,
                              });
        for (ptrdiff_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                ptrdiff_t u=n-c;
                ptrdiff_t v=c-z;
                ptrdiff_t sub_ext[2];
                ptrdiff_t sub_str[2];
                DataBlock<T> R = DataBlockUtilities::matrix_subspan(tL,c, z,u,v,sub_ext,sub_str);

                ptrdiff_t sextt[2]= {u,u};
                ptrdiff_t sstrt[2]= {u,1};

                DataBlock<T>  S(sdata,u*u,2,sextt,sstrt,sconf);


                ptrdiff_t rtext[2];
                ptrdiff_t strtext[2];


                DataBlock<T> RT=DataBlockUtilities::matrix_hermitian_transpose(R,rtext,strtext);

                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    GPU_Math_Functions::matrix_multiply_dot_g(R,RT,S,GPUOptions{.device=policy.devicenum,.update_host=false});
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(R,RT,S,ongpu,separate_device_memory,pcomm,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(R,RT,S,ongpu,separate_device_memory,pcomm,policy);
                    break;
                }

                const T*Sptr=S.dpdata;
                #pragma omp target teams distribute parallel for simd collapse(2)is_device_ptr(tempAptr,Sptr) device(policy.devicenum)
                for (ptrdiff_t i = c; i < n; ++i)
                {
                    for (ptrdiff_t j = c; j < n; ++j)
                    {
                        tempAptr[i*Astr0+j*Astr1] -= Sptr[(i - c)*sstrt[0]+ (j - c)*sstrt[1]];
                    }
                }

                z = c;
            }


            T tmp=T(0);
            #pragma omp target teams distribute parallel for simd map(tofrom:tmp) reduction(+:tmp)  is_device_ptr(tempLptr) device(policy.devicenum)
            for (ptrdiff_t k = z; k < c; ++k)
            {
                const T tmp3=tempLptr[c*Lstr0+k*Lstr1];
                tmp+= tmp3 *  cond_conj(tmp3);
            }

            T tmp42=T(0);
            omp_target_memcpy(&tmp42,tempAptr,sizeof(T),0,sizeof(T)*(Astr0*c+Astr1*c),omp_get_initial_device(),policy.devicenum);
            tmp=tmp42-tmp;

            const T temp4=sqrt(tmp);

            omp_target_memcpy(tempLptr,&temp4,sizeof(T),sizeof(T)*(Lstr0*c+Lstr1*c),0,policy.devicenum,omp_get_initial_device());

            #pragma omp target teams distribute parallel for map(to:temp4) is_device_ptr(tempLptr,tempAptr) device(policy.devicenum)
            for (ptrdiff_t i = c + 1; i < n; ++i)
            {
                T tmp2 = T(0);
                #pragma omp simd reduction(+:tmp2)
                for (ptrdiff_t k = z; k < c; ++k)
                {
                    tmp2 += tempLptr[i*Lstr0+k*Lstr1] *  cond_conj(tempLptr[c*Lstr0+k*Lstr1]);
                }
                tmp2=tempAptr[i*Astr0+c*Astr1]-tmp2;
                tempLptr[i*Lstr0+ c*Lstr1]=tmp2/temp4;
            }
        }


        if(separate_device_memory)
        {
            if(policy.update_host)
                GPU_Memory_Functions::update_host(L,policy.devicenum);
            GPU_Memory_Functions::release(L,policy.devicenum);
            GPU_Memory_Functions::release(A,policy.devicenum);

            omp_target_free(sdata,  policy.devicenum);
            omp_target_free(tempad, policy.devicenum);

        }
        else
        {
            if(policy.memmapped_files)
            {
                Host_Memory_Functions::delete_temp_mmap(sdata,tempsize);
                Host_Memory_Functions::delete_temp_mmap(tempad,A.dpdatalength);
            }
            else
            {
                delete[] sdata;
                delete[] tempad;
            }
        }

    }
    else
    {

        T * sdata= Host_Memory_Functions::alloc_data_ptr<T>(tempsize,policy.memmapped_files);

        DataBlock<T>  tempA=Host_Memory_Functions::alloc_data_copy_strides_extents<T>(A.dpdatalength,A.dprank,A.dpextents,A.dpstrides,
                            DataBlockConfig({.dprowmajor=A.dpconfig.dprowmajor,
                                             .pmemmap=policy.memmapped_files,
                                             .data_is_devptr=false,
                                             .devicenum=-INT_MAX
                                            }));

        if (policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=0;
                    tempA(i,j)=A(i,j);
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=A(i,j);
                }
            }
        }


        ptrdiff_t z=0;
        DataBlockConfig sconf({.dprowmajor=true,
                               .pmemmap= policy.memmapped_files,
                               .data_is_devptr=false,
                               .devicenum=-INT_MAX
                              });
        for (ptrdiff_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                ptrdiff_t u=n-c;
                ptrdiff_t v=c-z;
                ptrdiff_t sub_ext[2];
                ptrdiff_t sub_str[2];
                DataBlock<T> R = DataBlockUtilities::matrix_subspan(L,c, z,u,v,sub_ext,sub_str);

                ptrdiff_t sextt[2]= {u,u};
                ptrdiff_t sstrt[2]= {u,1};

                DataBlock<T>  S(sdata,u*u,2,sextt,sstrt,sconf);


                ptrdiff_t rtext[2];
                ptrdiff_t strtext[2];


                DataBlock<T> RT=DataBlockUtilities::matrix_hermitian_transpose(R,rtext,strtext);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    if(policy.should_use_gpu_matrix_multiply(R,RT,S))
                        GPU_Math_Functions::matrix_multiply_dot_g(R,RT,S,GPUOptions{.device=policy.devicenum,.update_host=true});
                    else
                    {
                        In_Kernel_Mathfunctions::matrix_multiply_dot(R,RT,S);
                        break;
                    }
                case Math_MPI_Decomposition_Policy::Strassen:
                {


                    strassen_multiply_h(R,RT,S,ongpu,separate_device_memory,pcomm,policy);
                    break;
                }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                {


                    winograd_multiply_h(R,RT,S,ongpu,separate_device_memory,pcomm,policy);
                    break;
                }
                }



                #pragma omp parallel for simd collapse(2)
                for (ptrdiff_t i = c; i < n; ++i)
                {
                    for (ptrdiff_t j = c; j < n; ++j)
                    {
                        tempA(i, j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }
            T tmp=T(0);
            #pragma omp parallel for simd  reduction(+: tmp)
            for (ptrdiff_t k = z; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 * cond_conj( tmp3);
            }

            tmp=tempA(c, c)-tmp;
            T tmp4=sqrt(tmp);
            L(c, c)=tmp4;

            #pragma omp parallel for
            for (ptrdiff_t i = c + 1; i < n; ++i)
            {
                T tmp2 = T(0);
                for (ptrdiff_t k = z; k < c; ++k)
                {
                    tmp2 += L(i, k) *cond_conj(  L(c, k));
                }

                tmp2=tempA(i, c)-tmp2;

                L(i, c)=tmp2/tmp4;
            }
        }
        Host_Memory_Functions::free_copy<T>(tempA);
        Host_Memory_Functions::free_data_ptr<T>(sdata,tempsize,policy.memmapped_files);
    }
}



template <typename T>
void Math_Functions_MPI::lu_decomposition(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U, MPI_Comm pcomm, Math_MPI_Decomposition_Policy* pol)
{
    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    bool ongpu=policy.should_use_gpu_decomposition(A);

    ptrdiff_t n = A.dpextents[0];
    int step_size=policy.step_size;

    if(step_size==0)
        step_size=(ptrdiff_t)pow(n,0.8385);
    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    ptrdiff_t tempsize=(n-step_size)*(n-step_size);
    L.dpconjugate=false;
    U.dpconjugate=false;
    bool aconj=A.dpconjugate;
    if(ongpu)
    {
        bool separate_device_memory=false;
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif

        T * sdata;
        T* tempad;

        if(separate_device_memory)
        {
            sdata= (T*) omp_target_alloc(sizeof(T)*tempsize, policy.devicenum);
            tempad= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                sdata=Host_Memory_Functions::create_temp_mmap<T>(tempsize);
                tempad=Host_Memory_Functions::create_temp_mmap<T>(A.dpdatalength);
            }
            else
            {
                sdata=new T[tempsize];
                tempad=new T[A.dpdatalength];
            }
        }

        ptrdiff_t taext[2]= {A.dpextents[0],A.dpextents[1]};
        ptrdiff_t tastr[2]= {A.dpstrides[0],A.dpstrides[1]};
        DataBlockConfig tempAconf({.dprowmajor=A.dpconfig.dprowmajor,
                                   .pmemmap=policy.memmapped_files,
                                   .data_is_devptr=separate_device_memory,
                                   .devicenum=policy.devicenum
                                  });
        DataBlock<T> tempA(tempad,A.dpdatalength,2,taext,tastr,tempAconf);


        DataBlock<T> tA=A,tL=L,tU=U;


        if(separate_device_memory)
        {
            GPU_Memory_Functions::create_in(A,policy.devicenum);
            GPU_Memory_Functions::create_out(L,policy.devicenum);
            GPU_Memory_Functions::create_out(U,policy.devicenum);

            if(!A.dpconfig.data_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
            if(!L.dpconfig.data_is_devptr)
                tL.dpdata=(T*) omp_get_mapped_ptr(L.dpdata,policy.devicenum);
            if(!U.dpconfig.data_is_devptr)
                tU.dpdata=(T*) omp_get_mapped_ptr(U.dpdata,policy.devicenum);

            tA.dpconfig.data_is_devptr=true;
            tL.dpconfig.data_is_devptr=true;
            tU.dpconfig.data_is_devptr=true;
            tA.dpconfig.devicenum=policy.devicenum;
            tL.dpconfig.devicenum=policy.devicenum;
            tU.dpconfig.devicenum=policy.devicenum;

        }

        const ptrdiff_t Astr0=tA.dpstrides[0];
        const ptrdiff_t Astr1=tA.dpstrides[1];
        const ptrdiff_t Lstr0=tL.dpstrides[0];
        const ptrdiff_t Lstr1=tL.dpstrides[1];
        const ptrdiff_t Ustr0=tU.dpstrides[0];
        const ptrdiff_t Ustr1=tU.dpstrides[1];
        T* tempAdptr=tempA.dpdata;
        T* tadptr=tA.dpdata;
        T* tLdptr=tL.dpdata;
        T* tUdptr=tU.dpdata;
        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for simd collapse(2)  is_device_ptr(tLdptr,tUdptr,tempAdptr)device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    tLdptr[i*Lstr0+j*Lstr1]=0;
                    tUdptr[i*Ustr0+j*Ustr1]=0;
                    tempAdptr[i*Astr0+j*Astr1]=returnval(tadptr[i*Astr0+j*Astr1],aconj);
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(tempAdptr,tadptr)  device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    tempAdptr[i*Astr0+j*Astr1]=returnval(tadptr[i*Astr0+j*Astr1],aconj);
                }
            }
        }

        ptrdiff_t z=0;
        DataBlockConfig sconf({.dprowmajor=true,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=separate_device_memory,
                               .devicenum=policy.devicenum
                              });
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                ptrdiff_t u=n-c;
                ptrdiff_t v=c-z;

                ptrdiff_t sub_ext[2];
                ptrdiff_t sub_str[2];
                DataBlock<T> RL = DataBlockUtilities::matrix_subspan(tL,c, z,u, v,sub_ext,sub_str);
                ptrdiff_t sub_ext2[2];
                ptrdiff_t sub_str2[2];
                DataBlock<T> RU = DataBlockUtilities::matrix_subspan(tU,z, c,v, u,sub_ext2,sub_str2);

                ptrdiff_t sextt[2]= {u,u};
                ptrdiff_t sstrt[2]= {u,1};

                DataBlock<T>  S(sdata,u*u,2,sextt,sstrt,sconf);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                {
                    GPU_Math_Functions::matrix_multiply_dot_g(RL,RU,S,GPUOptions{.device=policy.devicenum,.update_host=false});
                    break;
                }
                case Math_MPI_Decomposition_Policy::Strassen:
                {
                    strassen_multiply_h(RL,RU,S,ongpu,separate_device_memory,pcomm, policy);
                    break;
                }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                {
                    winograd_multiply_h(RL,RU,S,ongpu, separate_device_memory,pcomm,policy);
                    break;
                }
                }
                T*Sdptr=S.dpdata;
                #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(tUdptr,Sdptr)device(policy.devicenum)
                for (ptrdiff_t i = c; i < n; ++i)
                {
                    for (ptrdiff_t j = c; j < n; ++j)
                    {
                        tempAdptr[i*Astr0+j*Astr1] -= Sdptr[(i - c)*sstrt[0]+(j - c)*sstrt[1]];
                    }
                }

                z = c;
            }


            #pragma omp target teams distribute is_device_ptr(tUdptr,tLdptr,tempAdptr) device(policy.devicenum)
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp parallel for simd reduction(+:temp)
                for (ptrdiff_t k = z; k < c; ++k)
                {
                    temp += tUdptr[ k*Ustr0+i*Ustr1] *tLdptr[ c*Lstr0+k*Lstr1];
                }
                temp=tempAdptr[c*Astr0+i*Astr1]-temp;
                tUdptr[c*Ustr0+i*Ustr1]=temp;
            }

            T temp4=T(0);
            omp_target_memcpy(&temp4,tUdptr,sizeof(T),0,sizeof(T)*(Ustr0*c+Ustr1*c),omp_get_initial_device(),policy.devicenum);


            #pragma omp target teams distribute is_device_ptr(tUdptr,tLdptr,tempAdptr)  device(policy.devicenum)
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp = T(0);
                #pragma omp parallel for simd reduction(+:temp)
                for (ptrdiff_t k = z; k < c; ++k)
                {
                    temp += tUdptr[k*Ustr0+c*Ustr1] * tLdptr[i*Lstr0+k*Lstr1];
                }
                temp=tempAdptr[i*Astr0+c*Astr1]-temp;
                tLdptr[i*Lstr0+c*Lstr1]=temp/temp4;
            }
        }


        if(separate_device_memory)
        {
            GPU_Memory_Functions::release(A,policy.devicenum);
            if(policy.update_host)
            {
                GPU_Memory_Functions::update_host(L,policy.devicenum);
                GPU_Memory_Functions::update_host(U,policy.devicenum);
            }

            omp_target_free(sdata,  policy.devicenum);
            omp_target_free(tempad, policy.devicenum);

            GPU_Memory_Functions::release(L,policy.devicenum);
            GPU_Memory_Functions::release(U,policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                Host_Memory_Functions::delete_temp_mmap<T>(sdata,tempsize);
                Host_Memory_Functions::delete_temp_mmap<T>(tempad,A.dpdatalength);
            }
            else
            {
                delete[] sdata;
                delete[] tempad;
            }
        }

    }
    else
    {

        T * sdata= Host_Memory_Functions::alloc_data_ptr<T>(tempsize,policy.memmapped_files);

        DataBlock<T>  tempA=Host_Memory_Functions::alloc_data_copy_strides_extents<T>(A.dpdatalength, A.dprank,A.dpextents,A.dpstrides,
                            DataBlockConfig({.dprowmajor=A.dpconfig.dprowmajor,
                                             .pmemmap=policy.memmapped_files,
                                             .data_is_devptr=false,
                                             .devicenum=-INT_MAX
                                            }));

        if (policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=0;
                    U(i,j)=0;
                    tempA(i,j)=A(i,j);
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=A(i,j);
                }
            }

        }
        DataBlockConfig sconf({.dprowmajor=true,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=false,
                               .devicenum=-INT_MAX
                              });
        ptrdiff_t z=0;
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                ptrdiff_t u=n-c;
                ptrdiff_t v=c-z;

                ptrdiff_t sub_ext[2];
                ptrdiff_t sub_str[2];
                DataBlock<T> RL = DataBlockUtilities::matrix_subspan(L,c, z,u, v,sub_ext,sub_str);
                ptrdiff_t sub_ext2[2];
                ptrdiff_t sub_str2[2];
                DataBlock<T> RU = DataBlockUtilities::matrix_subspan(U,z, c,v, u,sub_ext2,sub_str2);

                ptrdiff_t sextt[2]= {u,u};
                ptrdiff_t sstrt[2]= {u,1};

                DataBlock<T>  S(sdata,u*u,2,sextt,sstrt,sconf);



                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                {
                    if(policy.should_use_gpu_matrix_multiply(RL,RU,S))
                    {
                        GPU_Math_Functions::matrix_multiply_dot_g(RL,RU,S,GPUOptions{.device=policy.devicenum,.update_host=true});
                    }
                    else
                    {
                        In_Kernel_Mathfunctions::matrix_multiply_dot(RL,RU,S);
                    }
                    break;
                }
                case Math_MPI_Decomposition_Policy::Strassen:
                {
                    strassen_multiply_h(RL,RU,S,false,false,pcomm,policy);
                    break;
                }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                {
                    winograd_multiply_h(RL,RU,S,false,false,pcomm,policy);
                    break;
                }
                }

                #pragma omp parallel for simd collapse(2)
                for (ptrdiff_t i = c; i < n; ++i)
                {
                    for (ptrdiff_t j = c; j < n; ++j)
                    {
                        tempA(i,j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }

            #pragma omp parallel for
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp=T(0);
                for (ptrdiff_t k = z; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=tempA(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);

            #pragma omp parallel for
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp = 0;
                for (ptrdiff_t k = z; k < c; ++k)
                {
                    temp += U(k,c) * L( i,k);
                }
                temp=tempA(i,c)-temp;
                L(i,c)=temp/temp4;
            }
        }

        Host_Memory_Functions::free_copy<T>(tempA);
        Host_Memory_Functions::free_data_ptr<T>(sdata,tempsize,policy.memmapped_files);
    }

}


template <typename T>
void Math_Functions_MPI::qr_decomposition(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R,MPI_Comm pcomm, Math_MPI_Decomposition_Policy *pol)
{
    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu_decomposition(A);

    int step_size=policy.step_size;

    if(step_size==0)
        step_size=(ptrdiff_t)pow(A.dpextents[0],0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    ptrdiff_t n = A.dpextents[0];
    ptrdiff_t m = A.dpextents[1];
    bool aconj=A.dpconjugate;
    Q.dpconjugate=false;
    R.dpconjugate=false;

    ptrdiff_t nm=n*m, mm=m*m;
    if(ongpu)
    {


        bool separate_device_memory=false;
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif

        T * tempC;
        T * tempS;
        T*  tempM;
        if(separate_device_memory)
        {
            tempS= (T*) omp_target_alloc(sizeof(T)*nm, policy.devicenum);
            tempC= (T*) omp_target_alloc(sizeof(T)*mm, policy.devicenum);
            tempM= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                tempS=Host_Memory_Functions::create_temp_mmap<T>(nm);
                tempC=Host_Memory_Functions::create_temp_mmap<T>(mm);
                tempM= Host_Memory_Functions::create_temp_mmap<T>(A.dpdatalength);
            }
            else
            {
                tempS= (T*)omp_alloc(sizeof(T)*nm,omp_default_mem_alloc);
                tempC= (T*) omp_alloc(sizeof(T)*mm, omp_default_mem_alloc);
                tempM= (T*) omp_alloc(sizeof(T)*A.dpdatalength, omp_default_mem_alloc);
            }
        }
        ptrdiff_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        ptrdiff_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};
        DataBlockConfig mconf({.dprowmajor=A.dpconfig.dprowmajor,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=separate_device_memory,
                               .devicenum=policy.devicenum,
                              });
        DataBlock<T> M(tempM,A.dpdatalength,2,aext,astr,mconf);


        DataBlock<T> tA=A,tQ=Q,tR=R;

        T* Mdptr=M.dpdata;

        if(separate_device_memory)
        {
            GPU_Memory_Functions::create_in(A,policy.devicenum);
            GPU_Memory_Functions::create_out(Q,policy.devicenum);
            GPU_Memory_Functions::create_out(R,policy.devicenum);


            if(!A.dpconfig.data_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
            if(!Q.dpconfig.data_is_devptr)
                tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,policy.devicenum);
            if(!R.dpconfig.data_is_devptr)
                tR.dpdata=(T*) omp_get_mapped_ptr(R.dpdata,policy.devicenum);

            tA.dpconfig.data_is_devptr=true;
            tQ.dpconfig.data_is_devptr=true;
            tR.dpconfig.data_is_devptr=true;
            tA.dpconfig.devicenum=policy.devicenum;
            tQ.dpconfig.devicenum=policy.devicenum;
            tR.dpconfig.devicenum=policy.devicenum;
        }

        const ptrdiff_t Qstr0=Q.dpstrides[0];
        const ptrdiff_t Qstr1=Q.dpstrides[1];
        const ptrdiff_t Rstr0=R.dpstrides[0];
        const ptrdiff_t Rstr1=R.dpstrides[1];
        const ptrdiff_t Astr0=A.dpstrides[0];
        const ptrdiff_t Astr1=A.dpstrides[1];
        T* tQdptr=tQ.dpdata;
        T* tRdptr=tR.dpdata;
        const T* tAdptr=tA.dpdata;
        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for simd collapse(2)is_device_ptr(tQdptr) device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j < n; ++j)
                {
                    tQdptr[i*Qstr0 + j*Qstr1] = T(0);
                }
            }

            #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(tAdptr,tRdptr,Mdptr)device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    Mdptr[i*Astr0 + j*Astr1] =returnval(tAdptr[i*Astr0 + j*Astr1],aconj);
                    tRdptr[i*Rstr0 + j*Rstr1] = T(0);
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2)  is_device_ptr(tAdptr,tRdptr,Mdptr) device(policy.devicenum)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    Mdptr[i*Astr0+j*Astr1]=returnval(tAdptr[i*Astr0+j*Astr1],aconj);
                }
            }
        }

        ptrdiff_t z = 0;
        DataBlockConfig cconf({.dprowmajor=true,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=separate_device_memory,
                               .devicenum=policy.devicenum
                              });
        for (ptrdiff_t c = 0; c < m; ++c)
        {

            if (c == z +step_size)
            {

                ptrdiff_t cz=c-z;
                ptrdiff_t mc=m-c;
                // Extract submatrices

                ptrdiff_t extBQ[2];
                ptrdiff_t strBQ[2];

                ptrdiff_t extBM[2];
                ptrdiff_t strBM[2];

                DataBlock<T> BQ = DataBlockUtilities::matrix_subspan(tQ,0, z, n, cz,extBQ,strBQ);
                DataBlock<T> BM = DataBlockUtilities::matrix_subspan(M,0, c, n,mc,extBM,strBM);

                ptrdiff_t tempCextt[2]= {cz,mc};
                ptrdiff_t tempCstrt[2]= {mc,1};

                DataBlock<T>  C(tempC,cz*mc,2,tempCextt,tempCstrt,cconf);


                ptrdiff_t extBQT[2];
                ptrdiff_t strBQT[2];

                DataBlock<T> BQT=DataBlockUtilities::matrix_hermitian_transpose(BQ,extBQT,strBQT);

                GPU_Math_Functions::matrix_multiply_dot_g(BQT,BM,C,GPUOptions{.device=policy.devicenum,.update_host=false});



                ptrdiff_t sextt[2]= {n,mc};
                ptrdiff_t sstrt[2]= {mc,1};

                DataBlock<T>  S(tempS,n*mc,2,sextt,sstrt,cconf);



                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    GPU_Math_Functions::matrix_multiply_dot_g(BQ,C,S,GPUOptions{.device=policy.devicenum,.update_host=false});
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(BQ,C,S,ongpu,separate_device_memory,pcomm,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(BQ,C,S,ongpu,separate_device_memory,pcomm,policy);
                    break;
                }

                T* Sdptr=S.dpdata;
                #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(Sdptr,Mdptr) device(policy.devicenum)
                for (ptrdiff_t i = 0; i < n; ++i)
                {
                    for (ptrdiff_t j = c; j < n; ++j)
                    {
                        Mdptr[i*Astr0+j*Astr1] -= Sdptr[i*sstrt[0]+(j-c)*sstrt[1]];
                    }
                }
                z = c;
            }
//            // Extract column c of M

            ptrdiff_t vext[1];
            ptrdiff_t vstr[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,vext,vstr);
            const ptrdiff_t pextv0=vext[0];
            T* vdptr=v.dpdata;
            for (ptrdiff_t j = z; j < c; ++j)
            {
                ptrdiff_t uext[1];
                ptrdiff_t ustr[1];
                DataBlock<T>  u =DataBlockUtilities::matrix_column(tQ,j,uext,ustr);
                T*udptr=u.dpdata;
                T dot_pr=T(0);

                #pragma omp target teams distribute parallel for simd  map(tofrom: dot_pr) is_device_ptr(tQdptr,vdptr) reduction(+:dot_pr) device(policy.devicenum)
                for (ptrdiff_t i = 0; i < pextv0; ++i)
                {
                    dot_pr +=cond_conj( udptr[i*ustr[0]]) * vdptr[i*vstr[0]];
                }

                const T cdot_pr = dot_pr;
                #pragma omp target teams distribute parallel for simd is_device_ptr(udptr,vdptr)device(policy.devicenum)
                for (ptrdiff_t i = 0; i < pextv0; ++i)
                {
                    vdptr[i*vstr[0]] -= cdot_pr * udptr[i*ustr[0]];
                }

            }

            T norm = T(0);
            #pragma omp target  teams distribute parallel for simd map(tofrom:norm) is_device_ptr(vdptr)reduction(+:norm)device(policy.devicenum)
            for (ptrdiff_t i = 0; i < pextv0; ++i)
            {
                T val=vdptr[i*vstr[0]] ;
                norm += cond_conj(val) *vdptr[i*vstr[0]];
            }

            const T normc = sqrt(norm);

            #pragma omp target teams distribute parallel for simd is_device_ptr(tQdptr,vdptr) device(policy.devicenum)
            for (ptrdiff_t i = 0; i < pextv0; ++i)
            {
                tQdptr[i*Qstr0+c*Qstr1] = vdptr[i*vstr[0]]/normc;
            }

        }
        // Compute R = Q^T * A for real values and Q^\dagger for complex values... i have no algorithm for conjugate transpose multiplication...
        // the conjugate is done at best on the fly instead of making a separate copy... so make the conjugate transpose multiplication explicitely here.

        ptrdiff_t extQT[2];
        ptrdiff_t strQT[2];
        DataBlock<T> QT=DataBlockUtilities::matrix_hermitian_transpose(tQ,extQT,strQT);

        switch (policy.algorithm_version)
        {
        case Math_MPI_Decomposition_Policy::Naive:
            GPU_Math_Functions::matrix_multiply_dot_g(QT,tA,tR,GPUOptions{.device=policy.devicenum,.update_host=false});
            break;
        case Math_MPI_Decomposition_Policy::Strassen:
            strassen_multiply_h(QT,tA,tR,ongpu,separate_device_memory,pcomm,policy);
            break;
        case Math_MPI_Decomposition_Policy::WinogradVariant:
            winograd_multiply_h(QT,tA,tR,ongpu,separate_device_memory,pcomm,policy);
            break;
        }




        if(separate_device_memory)
        {
            if(policy.update_host)
            {
                GPU_Memory_Functions::update_host(Q,policy.devicenum);
                GPU_Memory_Functions::update_host(R,policy.devicenum);
            }
            GPU_Memory_Functions::release(A,policy.devicenum);
            GPU_Memory_Functions::release(Q,policy.devicenum);
            GPU_Memory_Functions::release(R,policy.devicenum);

            omp_target_free(tempS, policy.devicenum);
            omp_target_free(tempC, policy.devicenum);
            omp_target_free(tempM, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                Host_Memory_Functions::delete_temp_mmap<T>(tempS,nm);
                Host_Memory_Functions::delete_temp_mmap<T>(tempM,A.dpdatalength);
                Host_Memory_Functions::delete_temp_mmap<T>(tempC,mm);
            }
            else
            {
                omp_free(tempS, omp_default_mem_alloc);
                omp_free(tempC, omp_default_mem_alloc);
                omp_free(tempM, omp_default_mem_alloc);
            }
        }


    }
    else
    {

        DataBlockConfig mconf({.dprowmajor=A.dpconfig.dprowmajor,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=false,
                               .devicenum=false,
                              });
        DataBlock<T> M= Host_Memory_Functions::alloc_data_copy_strides_extents<T>(A.dpdatalength,
                        A.dprank,A.dpextents,A.dpstrides,
                        mconf);

        T * tempC= Host_Memory_Functions::alloc_data_ptr<T>(mm,policy.memmapped_files);
        T * tempS= Host_Memory_Functions::alloc_data_ptr<T>(nm,policy.memmapped_files);


        if(policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (ptrdiff_t j = 0; j < n; ++j)
                    Q(i,j) = 0;

                #pragma omp simd
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = T(0);
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }


        ptrdiff_t z = 0;
        DataBlockConfig cconf({.dprowmajor=true,
                               .pmemmap=policy.memmapped_files,
                               .data_is_devptr=false,
                               .devicenum=-INT_MAX
                              });
        for (ptrdiff_t c = 0; c < m; ++c)
        {
            if (c == z +step_size)
            {
                ptrdiff_t cz=c-z;
                ptrdiff_t mc=m-c;
                // Extract submatrices

                ptrdiff_t extBQ[2];
                ptrdiff_t strBQ[2];

                ptrdiff_t extBM[2];
                ptrdiff_t strBM[2];

                DataBlock<T> BQ = DataBlockUtilities::matrix_subspan(Q,0, z, n, cz,extBQ,strBQ);
                DataBlock<T> BM = DataBlockUtilities::matrix_subspan(M,0, c, n,mc,extBM,strBM);

                ptrdiff_t Cextt[2]= {cz,mc};
                ptrdiff_t Cstrt[2]= {mc,1};

                DataBlock<T>  C(tempC,cz*mc,2,Cextt,Cstrt,cconf);


                ptrdiff_t extBQT[2];
                ptrdiff_t strBQT[2];
                DataBlock<T> BQT=DataBlockUtilities::matrix_hermitian_transpose(BQ,extBQT,strBQT);

                if(policy.should_use_gpu_matrix_multiply(BQT,BM,C))
                {
                    GPU_Math_Functions::matrix_multiply_dot_g(BQT,BM,C,GPUOptions{.device=policy.devicenum,.update_host=false});
                }
                else
                {
                    In_Kernel_Mathfunctions::matrix_multiply_dot(BQT,BM,C);
                }
                    ptrdiff_t sexttt[2]= {n,mc};
                    ptrdiff_t sstrtt[2]= {mc,1};

                    DataBlock<T>  S(tempS,n*mc,2,sexttt,sstrtt,cconf);




                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    if(policy.should_use_gpu_matrix_multiply(BQ,C,S))
                        GPU_Math_Functions::matrix_multiply_dot_g(BQ,C,S,GPUOptions{.device=policy.devicenum,.update_host=true});
                    else
                        In_Kernel_Mathfunctions::matrix_multiply_dot(BQ,C,S);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(BQ,C,S,false,false,pcomm,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(BQ,C,S,false,false,pcomm,policy);
                }


                #pragma omp parallel for simd collapse(2)
                for (ptrdiff_t i = 0; i < n; ++i)
                {
                    for (ptrdiff_t j = c; j < n; ++j)
                    {
                        M(i, j) -= S(i, j-c);
                    }
                }
                z = c;
            }

            ptrdiff_t vext[1];
            ptrdiff_t vstr[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,vext,vstr);

            for (ptrdiff_t j = z; j < c; ++j)
            {
                ptrdiff_t uext[1];
                ptrdiff_t ustr[1];
                DataBlock<T>  u = DataBlockUtilities::matrix_column(Q,j,uext,ustr);
                const T dot_pr =Math_Functions::dot_product(u,v,&policy);

                #pragma omp parallel for simd
                for (ptrdiff_t i = 0; i < n; ++i)
                {
                    v(i) -= dot_pr * u(i);
                }
            }

            // Normalize v
            const T norm = sqrt(Math_Functions::dot_product(v,v,&policy));

            // Set column c of Q

            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                Q(i,c) = v(i)/norm;
            }
        }


        // Compute R = Q^T * A for real values and Q^\dagger for complex values... i have no algorithm for conjugate transpose multiplication...
        // the conjugate is done at best on the fly instead of making a separate copy... so make the conjugate transpose multiplication explicitely here.

        ptrdiff_t extQT[2];
        ptrdiff_t strQT[2];
        DataBlock<T> QT=DataBlockUtilities::matrix_hermitian_transpose(Q,extQT,strQT);

        switch (policy.algorithm_version)
        {
        case Math_MPI_Decomposition_Policy::Naive:
            In_Kernel_Mathfunctions::matrix_multiply_dot(QT,A,R);
            break;
        case Math_MPI_Decomposition_Policy::Strassen:
            strassen_multiply_h(QT,A,R,false,false,pcomm,policy);
            break;
        case Math_MPI_Decomposition_Policy::WinogradVariant:
            winograd_multiply_h(QT,A,R,false,false,pcomm,policy);
        }

        Host_Memory_Functions::free_data_ptr<T>(tempC,mm,policy.memmapped_files);
        Host_Memory_Functions::free_data_ptr<T>(tempS,nm,policy.memmapped_files);
        Host_Memory_Functions::free_copy<T>(M);

    }
}





template<typename T>
void Math_Functions_MPI::MPI_recursive_multiplication_helper(MPI_Comm pcom,const Math_MPI_RecursiveMultiplication_Policy *pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    MPI_Status status;
    int message;
    for(;;)
    {
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, pcom, &status);



        bool strassen=false;
        switch (message)
        {
        case Math_MPI_RecursiveMultiplication_Policy::Strassen:
            strassen=true;
        case Math_MPI_RecursiveMultiplication_Policy::WinogradVariant:
        {

            ptrdiff_t dims[3];
            MPI_Recv(dims, 3, mpi_get_type<ptrdiff_t>(), MPI_ANY_SOURCE, 1, pcom, &status);



            bool ongpu=policy.should_use_gpu_matrix_multiply<T>(dims[0],dims[1],dims[2]);

            bool separate_device_memory=false;
            if(ongpu)
            {
#if !defined(Unified_Shared_Memory)
                separate_device_memory=true;
#endif
            }
            DataBlock<T> A,B;
            A=DataBlock_MPI_Functions::MPI_Recv_alloc_DataBlock<T>(MPI_Sendlocation{.with_memmap=policy.memmapped_files,.ondevice=separate_device_memory,.devicenum=policy.devicenum},status.MPI_SOURCE, 2, pcom);
            B=DataBlock_MPI_Functions::MPI_Recv_alloc_DataBlock<T>(MPI_Sendlocation{.with_memmap=policy.memmapped_files,.ondevice=separate_device_memory,.devicenum=policy.devicenum},status.MPI_SOURCE, 3, pcom);


            bool crowm=true;
            ptrdiff_t rowsC=A.dpextents[0],
                      colsC=B.dpextents[1];

            ptrdiff_t extC[2];
            ptrdiff_t strC[2];

            extC[0]=(crowm==true)?rowsC:colsC;
            extC[1]=(crowm==true)?colsC:rowsC;

            strC[0]=(crowm==true)? colsC:1;
            strC[1]=(crowm==true)?1: rowsC;

            T* C_data;
            ptrdiff_t length=rowsC*colsC;
            if(separate_device_memory)
            {
                C_data=GPU_Memory_Functions::alloc_data_device_ptr<T>(length,policy.memmapped_files,policy.devicenum);
            }
            else
            {
                C_data=Host_Memory_Functions::alloc_data_ptr<T>(length,policy.memmapped_files);
            }
            DataBlockConfig cconf({.dprowmajor=crowm,
                                   .pmemmap=policy.memmapped_files,
                                   .data_is_devptr=separate_device_memory,
                                   .devicenum=policy.devicenum});
            DataBlock<T> C(C_data,length,2,extC,strC,cconf);


            if(policy.size_to_stop_recursion>=length)
            {
                if(ongpu)
                {
                    GPU_Math_Functions::matrix_multiply_dot_g(A, B, C,GPUOptions{.device=policy.devicenum,.update_host=false});
                }
                else
                    In_Kernel_Mathfunctions::matrix_multiply_dot(A, B, C);
            }
            else
            {
                if(strassen)
                    strassen_multiply_h(A,B,C,ongpu,separate_device_memory,pcom,policy);
                else
                    winograd_multiply_h(A,B,C,ongpu,separate_device_memory,pcom,policy);
            }

            DataBlock_MPI_Functions::MPI_Send_DataBlock_pdata(C,status.MPI_SOURCE,4,pcom);

            DataBlock_MPI_Functions::MPI_Free_DataBlock(A);
            DataBlock_MPI_Functions::MPI_Free_DataBlock(B);
            if(separate_device_memory)
            {
                GPU_Memory_Functions::free_data_device_ptr(C.dpdata,C.dpdatalength,policy.memmapped_files,policy.devicenum);
            }
            else
            {
                Host_Memory_Functions::free_data_ptr<T>(C.dpdata,C.dpdatalength,policy.memmapped_files);
            }


            break;
        }

        case Math_MPI_RecursiveMultiplication_Policy::End_Listener:
            goto endloop;
        }
    }

endloop:
    return;

}
template <typename T>
void Math_Functions_MPI::MPI_recursion_helper_end(MPI_Comm pcomm)
{
    int commsize=0;
    MPI_Comm_size(pcomm, &commsize);
    int message=Math_MPI_RecursiveMultiplication_Policy::End_Listener;
    for (int i=0; i<commsize; i++)
    {
        MPI_Send(&message,1,MPI_INT,i,0,pcomm);
    }
}
#endif
