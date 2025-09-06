#ifndef MATHFUNCTIONS_MPI
#define MATHFUNCTIONS_MPI

#include "mpi.h"
#include "datastruct.h"
#include "mdspan_omp.h"
#include "mdspan_data.h"
#include <math.h>
#include "datastruct_host_memory_functions.h"
#include "datastruct_gpu_memory_functions.h"
#include "datastruct_mpifunctions.h"
#include "inkernel_mathfunctions.h"




struct Math_MPI_Functions_Policy : public Math_Functions_Policy
{
public:
    MPI_Comm comm = MPI_COMM_WORLD;
    bool mpi_enabled = true;

    int mpi_rank = 0;
    int mpi_size = 1;
    bool allow_gpu_sharing = true;

    Math_MPI_Functions_Policy(Mode m = AUTO,bool gpu_sharing=false,bool mpi=true)
        : Math_Functions_Policy(m), mpi_enabled(mpi),allow_gpu_sharing(gpu_sharing)
    {
        if (mpi_enabled)
        {
            int init;
            MPI_Initialized(&init);
            if(init)
            {
                MPI_Comm_rank(comm, &mpi_rank);
                MPI_Comm_size(comm, &mpi_size);
            }
            else
                mpi_enabled=false;
        }

        // Use cached num_gpus from base class
        if(mpi_enabled)
        {
            if (num_gpus > 0)
            {
                allow_gpu_sharing=gpu_sharing;
                if (allow_gpu_sharing)
                    devicenum = mpi_rank % num_gpus;  // shared mode
                else if (mpi_rank < num_gpus)
                    devicenum = mpi_rank;             // exclusive mode
                else
                    devicenum = -1;                   // CPU fallback
            }
            else
            {
                devicenum = -1; // no GPU available
            }
        }
    }

    bool rank_can_use_gpu() const
    {
        return ((mpi_enabled) && (devicenum >= 0));
    }

    bool should_use_gpu(const size_t problem_size,
                        const size_t threshold,
                        const bool any_input_output_on_device)const
    {
        if (!Math_Functions_Policy::should_use_gpu(problem_size, threshold,any_input_output_on_device))
            return false;
        else
        {
            if(mpi_enabled)
                return rank_can_use_gpu();
            else
            {
                return true;
            }

        }
    }

    template <typename T>
    bool should_use_gpu(const datastruct<T>& A,
                        const datastruct<T>& B,
                        const  datastruct<T>& C,
                        const size_t threshold)const
    {
        const size_t problem_size = A.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(A, devicenum);
            const bool B_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(B, devicenum);
            const bool C_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(C, devicenum);
            if(A_on_dev|| C_on_dev|| B_on_dev) return true;
            return this->should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev);
        }
        return false;
    }

    template <typename T>
    bool should_use_gpu(const datastruct<T>& v1,
                        const datastruct<T>& v2,
                        const size_t threshold)const
    {
        const size_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
            const bool C_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(v2, devicenum);
            if(A_on_dev||C_on_dev) return true;

            return this->should_use_gpu(problem_size, threshold, A_on_dev  || C_on_dev);

        }
    }

    template <typename T>
    bool should_use_gpu(const datastruct<T>& v1,
                        const size_t threshold)const
    {
        const size_t problem_size = v1.datalength();
        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
            if(A_on_dev) return true;
            return this->should_use_gpu(problem_size, threshold, A_on_dev );

        }
    }

};


struct Math_MPI_RecursiveMultiplication_Policy : public Math_MPI_Functions_Policy
{
public:

    enum Listener_Commands
    {
        Strassen=1,
        WinogradVariant=2,
        End_Listener=3
    };

    size_t size_to_stop_recursion = 64;  // below this size: stop recursion

    using Math_MPI_Functions_Policy::Math_MPI_Functions_Policy;

    bool should_use_mpi_for_recursion(size_t num_subcalls) const
    {
        if (!mpi_enabled)
            return false;
        int myrank=0;
        MPI_Comm_rank(comm, &myrank);
        return std::abs(mpi_size) >= pow(num_subcalls,myrank+1);
    }


    bool should_use_recursion(size_t problem_size) const
    {
        if (problem_size <= size_to_stop_recursion)
            return false; // base case → naive CPU multiply
        else
            return true;
    }


    bool should_use_gpu(const size_t problem_size,
                        const size_t threshold,
                        const bool any_input_output_on_device,
                        const size_t num_subcalls)const
    {
        if (!should_use_mpi_for_recursion(num_subcalls))
        {
            // Not enough ranks to distribute → maybe still use GPU locally
            return Math_Functions_Policy::should_use_gpu(problem_size, threshold, any_input_output_on_device);
        }

        // Enough ranks → allow GPU if mapping allows it
        return Math_MPI_Functions_Policy::should_use_gpu(problem_size, threshold, any_input_output_on_device);
    }

    template <typename T>
    bool should_use_gpu(const datastruct<T>& A,
                        const datastruct<T>& B,
                        const datastruct<T>& C,
                        const size_t threshold,
                        const size_t num_subcalls)const
    {
        size_t problem_size = A.datalength();

        bool A_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(A, devicenum);
        bool B_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(B, devicenum);
        bool C_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(C, devicenum);

        if(A_on_dev||B_on_dev||C_on_dev) return true;

        return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev, num_subcalls);
    }

    template <typename T>
    bool should_use_gpu(const datastruct<T>& v1,
                        const datastruct<T>& v2,
                        const size_t threshold,
                        const size_t num_subcalls)const
    {
        const size_t problem_size = v1.datalength();

        bool v1_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
        bool v2_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(v2, devicenum);
        if(v1_on_dev||v1_on_dev) return true;

        return should_use_gpu(problem_size, threshold, v1_on_dev || v2_on_dev,num_subcalls);

    }

    template <typename T>
    bool should_use_gpu(const datastruct<T>& v1,
                        size_t threshold,size_t num_subcalls)
    {
        const size_t problem_size = v1.datalength();

        const bool v1_on_dev = Datastruct_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
        if(v1_on_dev) return true;
        return should_use_gpu(problem_size, threshold, v1_on_dev,num_subcalls);

    }
};


struct Math_MPI_Decomposition_Policy : public Math_MPI_RecursiveMultiplication_Policy
{
public:
    enum Matrix_Multiplication_Algorithm
    {
        Naive=0,
        Strassen=1,
        WinogradVariant=2
    } algorithm_version=Naive;


    size_t step_size=0;

    using Math_MPI_RecursiveMultiplication_Policy::Math_MPI_RecursiveMultiplication_Policy;

    // New constructor that also sets the algorithm
    Math_MPI_Decomposition_Policy(
        Mode m,
        bool mpi,
        bool sharing,
        Matrix_Multiplication_Algorithm algo,
        size_t step = 0)
        : Math_MPI_RecursiveMultiplication_Policy(m, mpi, sharing),
          algorithm_version(algo),
          step_size(step)
    {}

};







using namespace std;


template <typename T>
class Math_Functions_MPI: public Math_Functions<T>
{
public:


    inline static void strassen_multiply( datastruct<T> &aA, datastruct<T> &aB,datastruct<T>& aC, const Math_MPI_RecursiveMultiplication_Policy *par=nullptr);

    inline static void winograd_multiply( datastruct<T> &aA, datastruct<T> &aB,datastruct<T>& aC, const Math_MPI_RecursiveMultiplication_Policy *par=nullptr);

    inline static void cholesky_decomposition(datastruct<T>& aA, datastruct<T> & aL,  Math_MPI_Decomposition_Policy *par=nullptr);

    inline static void lu_decomposition(datastruct<T> &aA, datastruct<T> & aL,datastruct<T> & aU, Math_MPI_Decomposition_Policy *par=nullptr);

    inline static void qr_decomposition(datastruct<T> &aA,datastruct<T>& aQ, datastruct<T> & aR,    Math_MPI_Decomposition_Policy *par=nullptr);

    inline static void MPI_recursive_multiplication_helper( const Math_MPI_RecursiveMultiplication_Policy*par=nullptr);
    inline static void MPI_recursion_helper_end(MPI_Comm pcomm);
protected:
    inline static void strassen_multiply_h( datastruct<T> &aA, datastruct<T> &aB,datastruct<T>& aC,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy &par);

    inline static void winograd_multiply_h( datastruct<T> &aA, datastruct<T> &aB,datastruct<T>& aC,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy &par);

    inline static void cholesky_decomposition_h(datastruct<T>& aA, datastruct<T> & aL,  Math_MPI_Decomposition_Policy &par);

    inline static void lu_decomposition_h(datastruct<T> &aA, datastruct<T> & aL,datastruct<T> & aU, Math_MPI_Decomposition_Policy &par);

    inline static void qr_decomposition_h(datastruct<T> &aA,datastruct<T>& aQ, datastruct<T> & aR,    Math_MPI_Decomposition_Policy &par);


    // optional default policy (initially empty = not constructed)
    inline static std::optional<Math_MPI_Decomposition_Policy> default_policy;

    // helper to access it with lazy init
    static const Math_MPI_Decomposition_Policy& get_default_policy()
    {
        if (!default_policy.has_value())
        {
            // only construct when needed
            default_policy.emplace(Math_Functions_Policy::AUTO);
        }
        return *default_policy;
    }

    static void set_default_policy(const Math_MPI_Decomposition_Policy& p)
    {
        default_policy = p; // assigns, overwrites if already constructed
    }

    static void reset_default_policy()
    {
        default_policy.reset(); // clear back to "uninitialized"
    }
};



template <typename T>
void Math_Functions_MPI<T>::strassen_multiply( datastruct<T> & A,  datastruct<T> & B, datastruct<T> & C,const Math_MPI_RecursiveMultiplication_Policy *pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,7);
    bool separate_device_memory=false;
    if(ongpu)
    {
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }

    if(separate_device_memory)
    {
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadA(A, policy.devicenum, false, false);
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadB(B,  policy.devicenum, false, false);
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadC(C,  policy.devicenum, true, policy.update_host);
        datastruct<T> tA=A,tB=B,tC=C;

        if(!tA.dpdata_is_devptr)
            tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
        if(!tB.dpdata_is_devptr)
            tB.dpdata=(T*) omp_get_mapped_ptr(B.dpdata,policy.devicenum);
        if(!tC.dpdata_is_devptr)
            tC.dpdata=(T*) omp_get_mapped_ptr(C.dpdata,policy.devicenum);

        tA.dpdata_is_devptr=true;
        tB.dpdata_is_devptr=true;
        tC.dpdata_is_devptr=true;

        strassen_multiply_h(tA,tB,tC,ongpu, separate_device_memory,policy);
    }
    else
    {
        strassen_multiply_h(A,B,C,ongpu, false,policy);
    }

}

template <typename T>
void Math_Functions_MPI<T>::strassen_multiply_h( datastruct<T> & A,  datastruct<T> & B, datastruct<T> & C,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy &policy)
{


    // Dimensions of input matrices
    size_t n = A.dpextents[0]; // Rows in A
    size_t m = A.dpextents[1]; // Columns in A and rows in B
    size_t p = A.dpextents[1]; // Columns in B


    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || !policy.should_use_recursion(n*p))
    {
        if(ongpu)
        {
            GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,false);
            return;
        }
        else
        {
            switch (policy.mode)
            {
            case Math_Functions_Policy::GPU_ONLY:
            {
                GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,false);
                return;
                break;
            }
            case Math_Functions_Policy::AUTO:
            {
                if(policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,1))
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(A,B,C,policy.devicenum,false);
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,C);
                return;
                break;
            }
            default:
            {
                In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,  C);
                return;
                break;
            }
            }
        }
        return;
    }


    size_t half_n = n / 2;
    size_t half_m = m / 2;
    size_t half_p = p / 2;

// Submatrices of A

    size_t psext1[2],psstr1[2],psext2[2],psstr2[2],psext3[2],psstr3[2],psext4[2],psstr4[2],
           psext5[2],psstr5[2],psext6[2],psstr6[2],psext7[2],psstr7[2],psext8[2],psstr8[2];




// Temporary storage for intermediate results
    size_t s=half_n*half_p,
           s2=half_n*half_m,
           s3=half_m*half_p;

    size_t ext1[2]= {half_n, half_p};
    size_t str1[2]= {half_p, 1};


    size_t ext2[2]= {half_n, half_m};
    size_t str2[2]= {half_m, 1};


    size_t ext3[2]=  {half_m, half_p};
    size_t str3[2]= {half_p, 1};




    T* Ard1,*Ard2,*Ard3,*Ard4,*Ard5,*Brd1,*Brd2,*Brd3,*Brd4,*Brd5,*M1d,*M2d,*M3d,*M4d,*M5d,*M6d,*M7d;

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
            Ard1=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard2=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard3=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard4=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard5=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);

            Brd1=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd2=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd3=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd4=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd5=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);

            M1d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M2d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M3d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M4d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M5d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M6d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M7d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
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


    datastruct<T>
    A_result1(Ard1,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result2(Ard2,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result3(Ard3,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result4(Ard4,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result5(Ard5,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),

              B_result1(Brd1,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result2(Brd2,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result3(Brd3,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result4(Brd4,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result5(Brd5,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),

              M1(M1d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M2(M2d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M3(M3d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M4(M4d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M5(M5d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M6(M6d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M7(M7d,s,true,2,ext1,str1,false,false,separate_device_memory);





    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(A_result1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(A_result2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(A_result3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(A_result4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(A_result5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(B_result1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(B_result2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(B_result3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(B_result4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(B_result5,policy.devicenum);
    }

    datastruct<T>  A11 = A.subspanmatrix(0, 0, half_n, half_m,psext1,psstr1),
                   A12 = A.subspanmatrix(0, half_m, half_n, half_m,psext2,psstr2),
                   A21 = A.subspanmatrix(half_n, 0, half_n, half_m,psext3,psstr3),
                   A22 = A.subspanmatrix(half_n, half_m, half_n, half_m,psext4,psstr4);





// Submatrices of B
    datastruct<T>  B11 = B.subspanmatrix(0, 0, half_m, half_p,psext5,psstr5),
                   B12 = B.subspanmatrix(0, half_p, half_m, half_p,psext6,psstr6),
                   B21 = B.subspanmatrix(half_m, 0, half_m, half_p,psext7,psstr7),
                   B22 = B.subspanmatrix(half_m, half_p, half_m, half_p,psext8,psstr8);


    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A22,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B22,policy.devicenum);
    }

    if (ongpu)
    {


        #pragma omp target teams distribute parallel for simd collapse (2) shared(A_result1,A_result2,A_result3,A_result4,A_result5,A11,A22,A12,A21) device(policy.devicenum)
        for (size_t i=0; i<half_n; i++)
            for (size_t j=0; j<half_m; j++)
            {
                A_result1(i,j)=A11(i,j)+A22(i,j);
                A_result2(i,j)=A21(i,j)+A22(i,j);
                A_result3(i,j)=A11(i,j)+A12(i,j);
                A_result4(i,j)=A21(i,j)-A11(i,j);
                A_result5(i,j)=A12(i,j)-A22(i,j);
            }


        #pragma omp target teams distribute parallel for simd collapse (2) shared(B_result1,B_result2,B_result3,B_result4,B_result5,B11,B22,B12,B21) device(policy.devicenum)
        for (size_t i=0; i<half_m; i++)
            for (size_t j=0; j<half_p; j++)
            {
                B_result1(i,j)=B11(i,j)+B22(i,j);
                B_result2(i,j)=B12(i,j)-B22(i,j);
                B_result3(i,j)=B21(i,j)-B11(i,j);
                B_result4(i,j)=B11(i,j)+B12(i,j);
                B_result5(i,j)=B21(i,j)+B22(i,j);
            }

    }
    else
    {
        #pragma omp parallel for simd collapse (2)shared(A_result1,A_result2,A_result3,A_result4,A_result5,A11,A22,A12,A21)
        for (size_t i=0; i<half_n; i++)
            for (size_t j=0; j<half_m; j++)
            {
                A_result1(i,j)=A11(i,j)+A22(i,j);
                A_result2(i,j)=A21(i,j)+A22(i,j);
                A_result3(i,j)=A11(i,j)+A12(i,j);
                A_result4(i,j)=A21(i,j)-A11(i,j);
                A_result5(i,j)=A12(i,j)-A22(i,j);
            }

        #pragma omp parallel for simd collapse (2)shared(B_result1,B_result2,B_result3,B_result4,B_result5,B11,B22,B12,B21)
        for (size_t i=0; i<half_m; i++)
            for (size_t j=0; j<half_p; j++)
            {
                B_result1(i,j)=B11(i,j)+B22(i,j);
                B_result2(i,j)=B12(i,j)-B22(i,j);
                B_result3(i,j)=B21(i,j)-B11(i,j);
                B_result4(i,j)=B11(i,j)+B12(i,j);
                B_result5(i,j)=B21(i,j)+B22(i,j);
            }
    }

    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M6,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M7,policy.devicenum);
    }
    if (policy.should_use_mpi_for_recursion(7))
    {
        int myrank=0,childdest=0;

        MPI_Comm_rank(policy.comm, &myrank);
        childdest=myrank*7;

        int message=Math_MPI_RecursiveMultiplication_Policy::Strassen;


        MPI_Send(&message, 1, MPI_INT, childdest+1,0, policy.comm);
        size_t problemsize=s2;

        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+1,1, policy.comm);

        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A_result1,childdest+1,2, policy.comm);

        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B_result1,childdest+1,3, policy.comm);




        MPI_Send(&message, 1, MPI_INT, childdest+2, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+2,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A_result2,childdest+2,2, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B11,childdest+2,3, policy.comm);



        MPI_Send(&message, 1, MPI_INT, childdest+3, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+3,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A11,childdest+3,2, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B_result2,childdest+3,3, policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+4, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+4,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A22,childdest+4,2, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B_result3,childdest+4,3, policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+5, 0,    policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+5,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A_result3,childdest+5,2,   policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B22,childdest+5,3,   policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+6, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+6,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A_result4,childdest+6,2, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B_result4,childdest+6,3, policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+7, 0, policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+7,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A_result5,childdest+7,2,   policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B_result5,childdest+7,3,   policy.comm);

        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M1,childdest+1,4, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M2,childdest+2,4, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M3,childdest+3,4, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M4,childdest+4,4, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M5,childdest+5,4, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M6,childdest+6,4, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M7,childdest+7,4, policy.comm);

    }
    else
    {
        #pragma omp parallel shared(A_result1,A_result2,A_result3,A_result4,A_result5,  B_result1,B_result2,B_result3,B_result4,B_result5,   A11,A22,B22,B11,M1,M2,M3,M4,M5,M6,M7)
        {
            strassen_multiply_h(A_result1, B_result1, M1, ongpu,  separate_device_memory, policy);
            strassen_multiply_h(A_result2, B11, M2,ongpu,  separate_device_memory, policy);
            strassen_multiply_h(A11, B_result2, M3,ongpu,  separate_device_memory, policy);
            strassen_multiply_h(A22, B_result3, M4,ongpu,  separate_device_memory, policy);
            strassen_multiply_h(A_result3, B22, M5,ongpu,  separate_device_memory,policy);
            strassen_multiply_h(A_result4, B_result4, M6,ongpu,  separate_device_memory,policy);
            strassen_multiply_h(A_result5, B_result5, M7,ongpu,  separate_device_memory, policy);
        }

    }

    size_t ext11a[2],str11a[2], ext12a[2],str12a[2], ext13a[2],str13a[2], ext14a[2],str14a[2];

// Submatrices of C
    datastruct<T>   C11 = C.subspanmatrix(0, 0, half_n, half_p,ext11a,str11a),
                    C12 = C.subspanmatrix(0, half_p, half_n, half_p,ext12a,str12a),
                    C21 = C.subspanmatrix(half_n, 0, half_n, half_p,ext13a,str13a),
                    C22 = C.subspanmatrix(half_n, half_p, half_n, half_p,ext14a,str14a);



    if(ongpu)
    {
        if(separate_device_memory)
        {
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(C11,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(C12,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(C21,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(C22,policy.devicenum);
        }

        #pragma omp target teams distribute parallel for simd collapse(2) shared(M1,M2,M3,M4,M5,M6,M7,C11,C12,C21,C22)      device(policy.devicenum)
        for (size_t i = 0; i < half_n; i++)
            for (size_t j = 0; j < half_p; j++)
            {
                T helper1 = M1(i,j)  +M4(i,j);
                T helper2 = -M5(i,j) +M7(i,j);
                C11(i,j)  =  helper1 +helper2;
                C12(i, j) = M3(i, j) + M5(i, j);
                C21(i, j) = M2(i, j) + M4(i, j);
                T helper3 = M1(i, j) - M2(i, j) ;
                T helper4 = M3(i, j) + M6(i, j);
                C22(i,j)  =helper3+helper4;
            }
    }
    else
    {
        #pragma omp parallel for simd shared(M1,M2,M3,M4,M5,M6,M7,C11,C12,C21,C22)   collapse(2)
        for (size_t i = 0; i < half_n; i++)
            for (size_t j = 0; j < half_p; j++)
            {
                T helper1 = M1(i,j)+M4(i,j);
                T helper2 =-M5(i,j)+M7(i,j);
                C11(i,j)  = helper1+helper2;
                C12(i, j) = M3(i, j) + M5(i, j);
                C21(i, j) = M2(i, j) + M4(i, j);
                T helper3 = M1(i, j) - M2(i, j) ;
                T helper4 = M3(i, j) + M6(i, j);
                C22(i,j)=helper3+helper4;
            }
    }


    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::release_struct(M1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M6,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M7,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(A_result1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A_result2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A_result3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A_result4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A_result5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B_result1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B_result2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B_result3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B_result4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B_result5,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(C11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(C12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(C21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(C22,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(B11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B22,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(A11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A22,policy.devicenum);

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
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M1d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M2d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M3d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M4d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M5d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M6d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M7d,s);

            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Ard1,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Ard2,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Ard3,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Ard4,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Ard5,s2);

            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Brd1,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Brd2,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Brd3,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Brd4,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(Brd5,s3);
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
void Math_Functions_MPI<T>::winograd_multiply(datastruct<T>& A, datastruct<T> &B, datastruct<T>& C,const Math_MPI_RecursiveMultiplication_Policy*pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,7);
    bool separate_device_memory=false;
    if(ongpu)
    {


#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }

    if(separate_device_memory)
    {
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadA(A, policy.devicenum, false, false);
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadB(B,  policy.devicenum, false, false);
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadC(C,  policy.devicenum, true, policy.update_host);
        datastruct<T> tA=A,tB=B,tC=C;

        if(!tA.dpdata_is_devptr)
            tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
        if(!tB.dpdata_is_devptr)
            tB.dpdata=(T*) omp_get_mapped_ptr(B.dpdata,policy.devicenum);
        if(!tC.dpdata_is_devptr)
            tC.dpdata=(T*) omp_get_mapped_ptr(C.dpdata,policy.devicenum);

        tA.dpdata_is_devptr=true;
        tB.dpdata_is_devptr=true;
        tC.dpdata_is_devptr=true;

        winograd_multiply_h(tA,tB,tC,ongpu, separate_device_memory,policy);
    }
    else
    {
        winograd_multiply_h(A,B,C,ongpu,false,policy);
    }

}

template <typename T>
void Math_Functions_MPI<T>::winograd_multiply_h(datastruct<T>& A, datastruct<T> &B, datastruct<T>& C,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy&policy)
{
    // Dimensions of input matrices
    size_t n = A.dpextents[0]; // Rows in A
    size_t m = A.dpextents[1]; // Columns in A and rows in B
    size_t p = A.dpextents[1]; // Columns in B


    // Base case: if no dimension is divisible by 2, use standard multiplication

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || !policy.should_use_recursion(n*p))
    {
        if(ongpu)
        {
            GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,false);
            return;
        }
        else
        {
            switch (policy.mode)
            {
            case Math_Functions_Policy::GPU_ONLY:
            {
                GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,false);
                return;
                break;
            }
            case Math_Functions_Policy::AUTO:
            {
                if(policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,1))
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(A,B,C,policy.devicenum,true);
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,C);
                return;
                break;
            }
            default:
            {
                In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,  C);
                return;
                break;
            }
            }
        }
        return;
    }

    // Compute sizes for splitting

    size_t half_n = n / 2;
    size_t half_m = m / 2;
    size_t half_p = p / 2;

    // Submatrices of A

    size_t psext1[2],psstr1[2],psext2[2],psstr2[2],psext3[2],psstr3[2],psext4[2],psstr4[2],
           psext5[2],psstr5[2],psext6[2],psstr6[2],psext7[2],psstr7[2],psext8[2],psstr8[2];


    // Temporary storage for intermediate results
    size_t s=half_n*half_p;
    size_t s2=half_n*half_m;
    size_t s3=half_m*half_p;


    size_t ext1[2]= {half_n, half_p};
    size_t str1[2]= {half_p, 1};



    size_t ext2[2]= {half_n, half_m};
    size_t str2[2]= {half_m, 1};


    size_t ext3[2]=  {half_m, half_p};
    size_t str3[2]= {half_p, 1};



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
            S1d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S2d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S3d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S4d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S5d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            S6d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            S7d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            S8d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s3);
            M1d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M2d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M3d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M4d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M5d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M6d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
            M7d=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(s);
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


    datastruct<T>
    S1(S1d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
    S2(S2d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
    S3(S3d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
    S4(S4d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),

    S5(S5d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
    S6(S6d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
    S7(S7d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
    S8(S8d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),


    M1(M1d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M2(M2d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M3(M3d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M4(M4d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M5(M5d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M6(M6d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M7(M7d,s,true,2,ext1,str1,false,false,separate_device_memory);



    datastruct<T>  A11 = A.subspanmatrix(0, 0, half_n, half_m,psext1,psstr1),
                   A12 = A.subspanmatrix(0, half_m, half_n, half_m,psext2,psstr2),
                   A21 = A.subspanmatrix(half_n, 0, half_n, half_m,psext3,psstr3),
                   A22 = A.subspanmatrix(half_n, half_m, half_n, half_m,psext4,psstr4);

    // Submatrices of B
    datastruct<T>  B11 = B.subspanmatrix(0, 0, half_m, half_p,psext5,psstr5),
                   B12 = B.subspanmatrix(0, half_p, half_m, half_p,psext6,psstr6),
                   B21 = B.subspanmatrix(half_m, 0, half_m, half_p,psext7,psstr7),
                   B22 = B.subspanmatrix(half_m, half_p, half_m, half_p,psext8,psstr8);


    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(A22,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(B22,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S6,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S7,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(S8,policy.devicenum);
    }

    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) shared(A22,A11,A21,S1,S2,S3,S4) device(policy.devicenum)
        for (size_t i=0; i<half_n; i++)
            for (size_t j=0; j<half_m; j++)
            {
                S1(i,j)=A21(i,j)+A22(i,j);
                S2(i,j)=S1(i,j)-A11(i,j);
                S3(i,j)=A11(i,j)-A21(i,j);
                S4(i,j)=A12(i,j)-S2(i,j);

            }

        #pragma omp target teams distribute parallel for simd collapse(2) shared(B12,B11,B22,B21,S5,S6,S7,S8) device(policy.devicenum)
        for (size_t i=0; i<half_m; i++)
            for (size_t j=0; j<half_p; j++)
            {
                S5(i,j)=B12(i,j)-B11(i,j);
                S6(i,j)=B22(i,j)-S5(i,j);
                S7(i,j)=B22(i,j)-B12(i,j);
                S8(i,j)=S6(i,j)-B21(i,j);
            }
    }
    else
    {
        #pragma omp  parallel for simd  shared(A22,A11,A21,S1,S2,S3,S4)collapse(2)
        for (size_t i=0; i<half_n; i++)
            for (size_t j=0; j<half_m; j++)
            {
                S1(i,j)=A21(i,j)+A22(i,j);
                S2(i,j)=S1(i,j)-A11(i,j);
                S3(i,j)=A11(i,j)-A21(i,j);
                S4(i,j)=A12(i,j)-S2(i,j);

            }
        #pragma omp parallel for simd shared(B12,B11,B22,B21,S5,S6,S7,S8)  collapse(2)
        for (size_t i=0; i<half_m; i++)
            for (size_t j=0; j<half_p; j++)
            {

                S5(i,j)=B12(i,j)-B11(i,j);
                S6(i,j)=B22(i,j)-S5(i,j);
                S7(i,j)=B22(i,j)-B12(i,j);
                S8(i,j)=S6(i,j)-B21(i,j);
            }
    }

    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M6,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_in_struct(M7,policy.devicenum);
    }


    if (policy.should_use_mpi_for_recursion(7))
    {
        int myrank=0,childdest=0;

        MPI_Comm_rank(policy.comm, &myrank);
        childdest=myrank*7;


        int message=Math_MPI_RecursiveMultiplication_Policy::WinogradVariant;

        MPI_Send(&message, 1, MPI_INT, childdest+1,0, policy.comm);
        size_t problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+1,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S2,childdest+1,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S6,childdest+1,3,policy.comm);


        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+2,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+2,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A11,childdest+2,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B11,childdest+2,3,policy.comm);



        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+3,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+3,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A12,childdest+3,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B21,childdest+3,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+4,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+4,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S3,childdest+4,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S7,childdest+4,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+5,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+5,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S1,childdest+5,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S5,childdest+5,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+6,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+6,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S4,childdest+6,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(B22,childdest+6,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+7,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+7,1, policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(A22,childdest+7,2,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Send_datastruct(S8,childdest+7,3,policy.comm);


        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M1,childdest+1,4,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M2,childdest+2,4,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M3,childdest+3,4,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M4,childdest+4,4,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M5,childdest+5,4,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M6,childdest+6,4,policy.comm);
        Datastruct_MPI_Functions<T>::MPI_Recv_datastruct_pdata(M7,childdest+7,4,policy.comm);

    }
    else
    {

        #pragma omp parallel shared(M1,M2,M3,M4,M5,M6,M7,S1,S2,S3,S4,S5,S6,S7,S8,A11,B11,A12,B21,B22,A22)
        {
            winograd_multiply_h(S2,S6,M1, ongpu,  separate_device_memory,policy);
            winograd_multiply_h(A11,B11,M2, ongpu,  separate_device_memory,policy);
            winograd_multiply_h(A12,B21,M3, ongpu,  separate_device_memory,policy);
            winograd_multiply_h(S3,S7,M4,ongpu,  separate_device_memory,policy);
            winograd_multiply_h(S1,S5,M5,ongpu,  separate_device_memory,policy);
            winograd_multiply_h(S4,B22,M6,ongpu,  separate_device_memory,policy);
            winograd_multiply_h(A22,S8,M7,ongpu,  separate_device_memory,policy);
        }

    }


    size_t pext10a[2],pstr10a[2],pext11a[2],pstr11a[2],pext12a[2],pstr12a[2],pext13a[2],pstr13a[2];

    datastruct<T>  C11 = C.subspanmatrix(0, 0, half_n, half_p,pext10a,pstr10a),
                   C12 = C.subspanmatrix(0, half_p, half_n, half_p,pext11a,pstr11a),
                   C21 = C.subspanmatrix(half_n, 0, half_n, half_p,pext12a,pstr12a),
                   C22 = C.subspanmatrix(half_n, half_p, half_n, half_p,pext13a,pstr13a);


    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(C11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(C12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(C21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::create_out_struct(C22,policy.devicenum);
    }



    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) shared(M1,M2,M3,M4,M5,M6,M7,C11,C12,C21,C22) device(policy.devicenum)
        for (size_t i = 0; i < half_n; ++i)
            for (size_t j = 0; j < half_p; ++j)
            {
                T T1=M1(i,j)+M2(i,j);
                T T2=T1+M4(i,j);
                C11(i, j) = M2(i, j) + M3(i,j);
                T helper=M5(i,j)+M6(i,j);
                C12(i, j) = T1 +helper ;
                C21(i, j) = T2 - M7(i, j);
                C22(i, j) = T2 + M5(i, j);
            }
    }
    else
    {
        #pragma omp parallel for simd collapse(2) shared(M1,M2,M3,M4,M5,M6,M7,C11,C12,C21,C22)
        for (size_t i = 0; i < half_n; ++i)
            for (size_t j = 0; j < half_p; ++j)
            {
                T T1=M1(i,j)+M2(i,j);
                T T2=T1+M4(i,j);
                C11(i, j) = M2(i, j) + M3(i,j);
                T helper= M5(i,j)+M6(i,j);
                C12(i, j) = T1 +helper;
                C21(i, j) = T2 - M7(i, j);
                C22(i, j) = T2 + M5(i, j);
            }

    }


    if(separate_device_memory)
    {
        Datastruct_GPU_Memory_Functions<T>::release_struct(C11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(C12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(C21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(C22,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(M1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M6,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(M7,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(S1,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S2,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S3,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S4,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S5,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S6,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S7,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(S8,policy.devicenum);

        Datastruct_GPU_Memory_Functions<T>::release_struct(A11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(A22,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B11,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B12,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B21,policy.devicenum);
        Datastruct_GPU_Memory_Functions<T>::release_struct(B22,policy.devicenum);

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
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M1d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M2d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M3d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M4d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M5d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M6d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(M7d,s);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S1d,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S2d,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S3d,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S4d,s2);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S5d,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S6d,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S7d,s3);
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(S8d,s3);
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
void Math_Functions_MPI<T>::cholesky_decomposition(datastruct<T> & A,datastruct<T> & L, Math_MPI_Decomposition_Policy *pol)
{
    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    Math_Functions_MPI<T>::cholesky_decomposition_h(A,L,policy);
}

template <typename T>
void Math_Functions_MPI<T>::cholesky_decomposition_h(datastruct<T> & A,datastruct<T> & L, Math_MPI_Decomposition_Policy &policy)
{


    bool ongpu=policy.should_use_gpu(A,L,Math_Functions_Policy::default_cubic_treshold,1);
    bool separate_device_memory=false;
    if(ongpu)
    {
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }



    const size_t n = A.dpextents[0];

    size_t step_size=policy.step_size;

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t tempsize=(n-step_size)*(n-step_size);



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
                sdata=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(tempsize);
                tempad=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
            }
            else
            {
                sdata=new T[tempsize];
                tempad=new T[A.dpdatalength];
            }
        }
        size_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};

        datastruct<T> tempA(tempad,A.dpdatalength,A.dprowmajor,2,aext,astr,false,false,separate_device_memory);

        datastruct<T> tA=A,tL=L;

        if(separate_device_memory)
        {
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(A,policy.devicenum);

            Datastruct_GPU_Memory_Functions<T>::create_out_struct(tempA,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_out_struct(L,policy.devicenum);
        }
        if(!tA.dpdata_is_devptr)
            tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);


        if(!tL.dpdata_is_devptr)
            tL.dpdata=(T*) omp_get_mapped_ptr(L.dpdata,policy.devicenum);


        tA.dpdata_is_devptr=true;
        tL.dpdata_is_devptr=true;





        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for simd collapse(2) shared(tL,tempA) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j <n; ++j)
                {
                    tL(i,j)=0;
                    tempA(i,j)=tA(i,j);
                }
        }
        else
        {

            #pragma omp target teams distribute parallel for simd collapse(2) shared(tempA,tA) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=tA(i,j);
                }
        }

        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;

                // Extract submatrix R = L[c:n, z:c-1]
                size_t sub_ext[2];
                size_t sub_str[2];
                datastruct<T> R = tL.subspanmatrix(c, z,u, v,sub_ext,sub_str);

                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadRL(R, policy.devicenum, false, false);

                size_t rtext[2];
                size_t strtext[2];
                datastruct<T> RT=R.transpose(rtext,strtext);

                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadRT(RT, policy.devicenum, false, false);


                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};

                datastruct<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,separate_device_memory);

                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadS(S,  policy.devicenum, true, false);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(R,RT,S,policy.devicenum,false);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                    break;
                }

                #pragma omp target teams distribute parallel for simd collapse(2)  device(policy.devicenum)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i, j) -= S(i - c, j - c);
                    }
                }
                z = c;

            }


            T tmp=0,temp4=0;
            #pragma omp target map(tofrom:tmp)map(to:c) device(policy.devicenum)
            {
            tmp=tempA(c,c);
            }

            #pragma omp target data map(tofrom:tmp)map(to:c) device(policy.devicenum)
            #pragma omp target teams distribute parallel for simd shared(tL,tmp)  device(policy.devicenum)
            for (size_t k = 0; k < c; ++k)
            {
                 T tmp3=tL(c,k);
                #pragma omp atomic
                tmp-= tmp3 * tmp3;
            }
            temp4=sqrt(tmp);
            #pragma omp target map(tofrom:temp4) map(to:c)device(policy.devicenum)
            {
                tL(c,c)=temp4;
            }


            #pragma omp target  data map(to:temp4, c)device(policy.devicenum)
            #pragma omp target teams distribute parallel for shared(tempA,tL) device(policy.devicenum)
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = tempA(i, c);

                for (size_t k = 0; k < c; ++k)
                {
                    tmp2 -= tL(i, k) * tL(c, k);
                }
                tL(i, c)=tmp2/temp4;
            }
        }

        if(separate_device_memory)
        {
            if(policy.update_host)
                Datastruct_GPU_Memory_Functions<T>::update_host(L,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::release_struct(L,policy.devicenum);


            Datastruct_GPU_Memory_Functions<T>::release_struct(A,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::release_struct(tempA,policy.devicenum);
            omp_target_free(sdata,  policy.devicenum);
            omp_target_free(tempad, policy.devicenum);


        }
        else
        {
            if(policy.memmapped_files)
            {
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(sdata,tempsize);
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(tempad,A.dpdatalength);
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

        T * sdata= Datastruct_Host_Memory_Functions<T>::alloc_data_ptr(tempsize,policy.memmapped_files);
        datastruct<T>  tempA=Datastruct_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides
                             ,policy.memmapped_files);

        if (policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for simd shared(L,tempA,A) collapse(2)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=0;
                    tempA(i,j)=A(i,j);
                }
        }
        else
        {
            #pragma omp parallel for simd shared(tempA,A) collapse(2)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=A(i,j);
                }
        }


        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                size_t sub_ext[2];
                size_t sub_str[2];
                datastruct<T> R = L.subspanmatrix(c, z,u,v,sub_ext,sub_str);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};
                datastruct<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,false);

                size_t rtext[2],strtext[2];

                datastruct<T> RT=R.transpose(rtext,strtext);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    if(policy.should_use_gpu(R,RT,S,Math_Functions_Policy::default_cubic_treshold,1))
                        GPU_Math_Functions<T>::matrix_multiply_dot_g(R,RT,S,policy.devicenum,true);
                    else
                        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(R,RT,S);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                }


                #pragma omp parallel for simd  shared(tempA,S) collapse (2)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i, j) -= S(i - c, j - c);
                    }
                }

                z = c;
            }


            T tmp=tempA(c, c);

            #pragma omp parallel for simd shared(L) reduction(-: tmp)
            for (size_t k = z; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp-= tmp3 * tmp3;
            }


            T tmp4=sqrt(tmp);
            L(c, c)=tmp4;

            #pragma omp parallel for shared(tempA)
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = tempA(i, c);

                #pragma omp simd reduction(-:tmp2)
                for (size_t k = z; k < c; ++k)
                {
                    tmp2 -= L(i, k) * L(c, k);
                }
                L(i, c)=tmp2/tmp4;
            }
        }
        Datastruct_Host_Memory_Functions<T>::free_copy(tempA,policy.memmapped_files);
        Datastruct_Host_Memory_Functions<T>::free_data_ptr(sdata,tempsize,policy.memmapped_files);
    }
}


template <typename T>
void Math_Functions_MPI<T>::lu_decomposition(datastruct<T>& A, datastruct<T> &L,datastruct<T>& U,  Math_MPI_Decomposition_Policy* pol)
{
    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    Math_Functions_MPI<T>::lu_decomposition_h(A,L,U,policy);


}
template <typename T>
void Math_Functions_MPI<T>::lu_decomposition_h(datastruct<T>& A, datastruct<T> &L,datastruct<T>& U,  Math_MPI_Decomposition_Policy& policy)
{



    bool ongpu=policy.should_use_gpu(A,L,U,Math_Functions_Policy::default_cubic_treshold,1);



    size_t n = A.dpextents[0];
    int step_size=policy.step_size;

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);
  if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t tempsize=(n-step_size)*(n-step_size);





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
                sdata=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(tempsize);
                tempad=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
            }
            else
            {
                sdata=new T[tempsize];
                tempad=new T[A.dpdatalength];
            }
        }

        size_t taext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t tastr[2]= {A.dpstrides[0],A.dpstrides[1]};
        datastruct<T> tempA(tempad,A.dpdatalength,A.dprowmajor,2,taext,tastr,false,false,separate_device_memory);

        datastruct<T> tA=A,tL=L,tU=U;

        if(separate_device_memory)
        {
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(A,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_out_struct(L,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_out_struct(U,policy.devicenum);

            Datastruct_GPU_Memory_Functions<T>::create_out_struct(tempA,policy.devicenum);

            if(!tA.dpdata_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
            if(!tL.dpdata_is_devptr)
                tL.dpdata=(T*) omp_get_mapped_ptr(L.dpdata,policy.devicenum);
            if(!tU.dpdata_is_devptr)
                tU.dpdata=(T*) omp_get_mapped_ptr(U.dpdata,policy.devicenum);

            tA.dpdata_is_devptr=true;
            tL.dpdata_is_devptr=true;
            tU.dpdata_is_devptr=true;
        }


        if(policy.initialize_output_to_zeros)
        {
            #pragma omp target teams distribute parallel for simd shared(tL,tU,tempA,tA) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
                #pragma omp simd
                for (size_t j = 0; j <n; ++j)
                {
                    tL(i,j)=0;
                    tU(i,j)=0;
                    tempA(i,j)=tA(i,j);
                }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd shared(tempA,tA) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
                #pragma omp simd
                for (size_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=tA(i,j);
                }
        }

        size_t z=0;

        #pragma omp ordered
        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;

                size_t sub_ext[2];
                size_t sub_str[2];
                datastruct<T> RL = tL.subspanmatrix(c, z,u, v,sub_ext,sub_str);
                size_t sub_ext2[2];
                size_t sub_str2[2];
                datastruct<T> RU = tU.subspanmatrix(z, c,v, u,sub_ext2,sub_str2);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};


                datastruct<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,separate_device_memory);

                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadRL(RL, policy.devicenum, false, false);
                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadRT(RL, policy.devicenum, false, false);
                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadS(S,  policy.devicenum, true, false);

                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    {
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(RL,RU,S,policy.devicenum,false);
                    break;
                    }
                case Math_MPI_Decomposition_Policy::Strassen:
                    {
                    strassen_multiply_h(RL,RU,S,ongpu,separate_device_memory, policy);
                    break;
                    }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    {
                    winograd_multiply_h(RL,RU,S,ongpu, separate_device_memory,policy);
                    break;
                    }
                }



                #pragma omp target teams distribute parallel for simd collapse(2) shared(tempA,S) device(policy.devicenum)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i,j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }


            #pragma omp target teams distribute shared(tempA,tU,tL) device(policy.devicenum)
            for (size_t i = c; i < n; ++i)
            {
                T temp=tempA(c,i);
                #pragma omp parallel for simd reduction(-:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= tU( k,i) * tL( c,k);
                }
                tU(c,i)=temp;
            }



            #pragma omp target teams distribute shared(tU,tempA,tL) device(policy.devicenum)
            for (size_t i = c; i < n; ++i)
            {
                const T temp4=tU(c,c);
                T temp = tempA(i,c);
                #pragma omp parallel for simd reduction(-:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= tU(k,c) * tL( i,k);
                }
                tL(i,c)=temp/temp4;
            }
        }


        if(separate_device_memory)
        {
            Datastruct_GPU_Memory_Functions<T>::release_struct(A,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::release_struct(tempA,policy.devicenum);
            omp_target_free(sdata,  policy.devicenum);
            omp_target_free(tempad, policy.devicenum);
            if(policy.update_host)
            {
                Datastruct_GPU_Memory_Functions<T>::update_host(L,policy.devicenum);
                Datastruct_GPU_Memory_Functions<T>::update_host(U,policy.devicenum);
            }
            Datastruct_GPU_Memory_Functions<T>::release_struct(L,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::release_struct(U,policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(sdata,tempsize);
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(tempad,A.dpdatalength);
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

        T * sdata= Datastruct_Host_Memory_Functions<T>::alloc_data_ptr(tempsize,policy.memmapped_files);

        datastruct<T>  tempA=Datastruct_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides
                             ,policy.memmapped_files);

        if (policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for shared(L,U,tempA,A)
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=0;
                    U(i,j)=0;
                    tempA(i,j)=A(i,j);
                }
            }
        }
        else
        {
            #pragma omp parallel for shared(tempA,A)
            for (size_t i = 0; i < n; ++i)
            {
                # pragma omp simd
                for (size_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=A(i,j);
                }
            }
        }
//
        size_t z=0;
        #pragma omp ordered
        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;

                size_t sub_ext[2];
                size_t sub_str[2];
                datastruct<T> RL = L.subspanmatrix(c, z,u, v,sub_ext,sub_str);
                size_t sub_ext2[2];
                size_t sub_str2[2];
                datastruct<T> RU = U.subspanmatrix(z, c,v, u,sub_ext2,sub_str2);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};
                datastruct<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,false);




                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                {
                    if(policy.should_use_gpu(RL,RU,S,Math_Functions_Policy::default_cubic_treshold,1))
                    {
                        GPU_Math_Functions<T>::matrix_multiply_dot_g(RL,RU,S,policy.devicenum,true);
                    }
                    else
                    {
                        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(RL,RU,S);
                    }
                    break;
                }
                case Math_MPI_Decomposition_Policy::Strassen:
                    {
                    strassen_multiply_h(RL,RU,S,false,false,policy);
                    break;
                    }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    {
                    winograd_multiply_h(RL,RU,S,false,false,policy);
                    break;
                    }
                }

                #pragma omp parallel for shared(tempA,S) collapse(2)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i,j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }

            #pragma omp parallel for shared(U,L,tempA)
            for (size_t i = c; i < n; ++i)
            {
                T temp=tempA(c,i);
                #pragma omp simd reduction(-:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U( k,i) * L( c,k);
                }
                U(c,i)=temp;
            }

            const T temp4=U(c,c);

            #pragma omp parallel for shared(tempA,U,L)
            for (size_t i = c; i < n; ++i)
            {
                T temp = tempA(i,c);
                #pragma omp simd reduction(-:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U(k,c) * L( i,k);
                }
                L(i,c)=temp/temp4;
            }
        }

        Datastruct_Host_Memory_Functions<T>::free_copy(tempA,policy.memmapped_files);
        Datastruct_Host_Memory_Functions<T>::free_data_ptr(sdata,tempsize,policy.memmapped_files);
    }

}

template <typename T>
void Math_Functions_MPI<T>::qr_decomposition(datastruct<T>& A, datastruct<T>& Q, datastruct<T>& R, Math_MPI_Decomposition_Policy *pol)
{

    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    Math_Functions_MPI<T>::qr_decomposition_h(A,Q,R,policy);

}
template <typename T>
void Math_Functions_MPI<T>::qr_decomposition_h(datastruct<T>& A, datastruct<T>& Q, datastruct<T>& R, Math_MPI_Decomposition_Policy &policy)
{

    bool ongpu=policy.should_use_gpu(A,Q,R,Math_Functions_Policy::default_cubic_treshold,1);

    int step_size=policy.step_size;

    if(step_size==0)
        step_size=(size_t)pow(A.dpextents[0],0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t n = A.dpextents[0]; // Number of rows (assuming 2D matrix)
    size_t m = A.dpextents[1]; // Number of columns

    // Initialize Q and R matrices
    size_t nm=n*m, mm=m*m;
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
                tempS=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(nm);
                tempC=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(mm);
                tempM= Datastruct_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
            }
            else
            {
                tempS=new T[nm];
                tempC=new T[mm];
                tempM=new T[A.dpdatalength];
            }
        }
        size_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};
        datastruct<T> M(tempM,A.dpdatalength,A.dprowmajor,2,aext,astr,false,false,separate_device_memory);

        datastruct<T> tA=A,tQ=Q,tR=R;

        if(separate_device_memory)
        {
            Datastruct_GPU_Memory_Functions<T>::create_in_struct(A,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_out_struct(Q,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::create_out_struct(R,policy.devicenum);

            Datastruct_GPU_Memory_Functions<T>::create_out_struct(M,policy.devicenum);

            if(!tA.dpdata_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
            if(!tQ.dpdata_is_devptr)
                tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,policy.devicenum);
            if(!tR.dpdata_is_devptr)
                tR.dpdata=(T*) omp_get_mapped_ptr(R.dpdata,policy.devicenum);

            tA.dpdata_is_devptr=true;
            tQ.dpdata_is_devptr=true;
            tR.dpdata_is_devptr=true;
        }



        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for shared(Q,M,tA,tR) device(policy.devicenum)

            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    tQ(i,j) = 0;
                #pragma omp simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=tA(i,j);
                    tR(i,j) = 0;
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for shared(tA,M) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=tA(i,j);
                }
            }
        }


        size_t z = 0;

        for (size_t c = 0; c < m; ++c)
        {

            if (c == z +step_size)
            {

                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                size_t extBQ[2],strBQ[2];

                size_t extBM[2],strBM[2];

                datastruct<T> BQ = tQ.subspanmatrix(0, z, n, cz,extBQ,strBQ);
                datastruct<T> BM = M.subspanmatrix(0, c, n,mc,extBM,strBM);




                size_t tempCextt[2]= {cz,mc};
                size_t tempCstrt[2]= {mc,1};

                datastruct<T>  C(tempC,cz*mc,true,2,tempCextt,tempCstrt,false,false,separate_device_memory);


                size_t extBQT[2],strBQT[2];
                datastruct<T> BQT=BQ.transpose(extBQT,strBQT);

                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadBQ(BQ, policy.devicenum, false, false);
                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadBQT(BQT, policy.devicenum, false, false);
                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadBM(BM, policy.devicenum, false, false);
                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadC(C,  policy.devicenum, true, false);


                GPU_Math_Functions<T>::matrix_multiply_dot_g(BQT,BM,C,policy.devicenum,false);



                size_t sextt[2]= {n,mc};
                size_t sstrt[2]= {mc,1};
                datastruct<T>  S(tempS,n*mc,true,2,sextt,sstrt,false,false,separate_device_memory);


                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadS(S,  policy.devicenum, true, false);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(BQ,C,S,policy.devicenum,false);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(BQ,C,S,ongpu,separate_device_memory,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(BQ,C,S,ongpu,separate_device_memory,policy);
                    break;
                }


                #pragma omp target teams distribute parallel for simd shared(M,S) device(policy.devicenum)
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        M(i, j) -= S(i, j-c);
                    }
                }
                z = c;
            }
            // Extract column c of M

            size_t vext[2],vstr[2];
            datastruct<T> v = M.column(c,vext,vstr);
            typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadv(v,  policy.devicenum, false, false);
            for (size_t j = z; j < c; ++j)
            {
                size_t uext[2],ustr[2];
                datastruct<T>  u = tQ.column(j,uext,ustr);
                typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadu(u,  policy.devicenum, false, false);
                const T dot_pr =GPU_Math_Functions<T>::dot_product_g(u,v,policy.devicenum);

                #pragma omp target teams distribute parallel for simd shared(v,u) device(policy.devicenum)
                for (size_t i = 0; i < n; ++i)
                {
                    v(i) -= dot_pr * u(i);
                }
            }

            // Normalize v
            const T norm = sqrt(GPU_Math_Functions<T>::dot_product_g(v,v,policy.devicenum));
            #pragma omp target teams distribute parallel for simd shared(v) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                v(i) /= norm;
            }

            // Set column c of Q

            #pragma omp target teams distribute parallel for simd shared(tQ,v) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                tQ(i,c) = v(i);
            }
        }
        // Compute R = Q^T * A
        size_t extQT[2],strQT[2];

        datastruct<T> QT=tQ.transpose(extQT,strQT);

        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadQT(QT, policy.devicenum, false, false);

        switch (policy.algorithm_version)
        {
        case Math_MPI_Decomposition_Policy::Naive:
            GPU_Math_Functions<T>::matrix_multiply_dot_g(QT,tA,tR,policy.devicenum,false);
            break;
        case Math_MPI_Decomposition_Policy::Strassen:
            strassen_multiply_h(QT,tA,tR,ongpu,separate_device_memory,policy);
            break;
        case Math_MPI_Decomposition_Policy::WinogradVariant:
            winograd_multiply_h(QT,tA,tR,ongpu,separate_device_memory,policy);
            break;
        }

        if(separate_device_memory)
        {
            Datastruct_GPU_Memory_Functions<T>::release_struct(A,policy.devicenum);
            if(policy.update_host)
            {
                Datastruct_GPU_Memory_Functions<T>::update_host(Q,policy.devicenum);
                Datastruct_GPU_Memory_Functions<T>::update_host(R,policy.devicenum);
            }
            Datastruct_GPU_Memory_Functions<T>::release_struct(Q,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::release_struct(R,policy.devicenum);
            Datastruct_GPU_Memory_Functions<T>::release_struct(M,policy.devicenum);

            omp_target_free (tempS, policy.devicenum);
            omp_target_free(tempC, policy.devicenum);
            omp_target_free(tempM, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(tempS,nm);
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(tempM,A.dpdatalength);
                Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(tempC,mm);
            }
            else
            {
                delete[] tempS;
                delete[] tempM;
                delete[] tempC;
            }
        }


    }
    else
    {


        datastruct<T> M= Datastruct_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(A.dpdatalength,A.dprowmajor,
                         A.dprank,A.dpextents,A.dpstrides,
                         policy.memmapped_files);

        T * tempC= Datastruct_Host_Memory_Functions<T>::alloc_data_ptr(mm,policy.memmapped_files);
        T * tempS= Datastruct_Host_Memory_Functions<T>::alloc_data_ptr(nm,policy.memmapped_files);


        if(policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for shared(Q,M,R,A)
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = 0;
                }
            }
        }
        else
        {
            #pragma omp parallel for shared(M,A)
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }


        size_t z = 0;

        for (size_t c = 0; c < m; ++c)
        {
            if (c == z +step_size)
            {
                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                size_t extBQ[2],strBQ[2];

                size_t extBM[2],strBM[2];

                datastruct<T> BQ = Q.subspanmatrix(0, z, n, cz,extBQ,strBQ);
                datastruct<T> BM = M.subspanmatrix(0, c, n,mc,extBM,strBM);

                size_t Cextt[2]= {cz,mc};
                size_t Cstrt[2]= {mc,1};

                datastruct<T>  C(tempC,cz*mc,true,2,Cextt,Cstrt,false,false,false);


                size_t extBQT[2],strBQT[2];
                datastruct<T> BQT=BQ.transpose(extBQT,strBQT);


                if(policy.should_use_gpu(BQT,BM,C,Math_Functions_Policy::default_cubic_treshold,1))
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(BQT,BM,C,policy.devicenum,true);
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(BQT,BM,C);

                size_t sexttt[2]= {n,mc};
                size_t sstrtt[2]= {mc,1};

                datastruct<T>  S(tempS,n*mc,true,2,sexttt,sstrtt,false,false,false);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    if(policy.should_use_gpu(BQ,C,S,Math_Functions_Policy::default_cubic_treshold,1))
                        GPU_Math_Functions<T>::matrix_multiply_dot_g(BQ,C,S,policy.devicenum,true);
                    else
                        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(BQ,C,S);

                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(BQ,C,S,false,false,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(BQ,C,S,false,false,policy);
                }


                #pragma omp parallel for shared(M,S)
                for (size_t i = 0; i < n; ++i)
                {
                    #pragma omp simd
                    for (size_t j = c; j < n; ++j)
                    {
                        M(i, j) -= S(i, j-c);
                    }
                }
                z = c;
            }

            size_t vext[2],vstr[2];
            datastruct<T> v = M.column(c,vext,vstr);

            for (size_t j = z; j < c; ++j)
            {
                size_t uext[2],ustr[2];
                datastruct<T>  u = Q.column(j,uext,ustr);
                const T dot_pr =Math_Functions<T>::dot_product(u,v,&policy);
                #pragma omp parallel for simd shared(v,u)
                for (size_t i = 0; i < n; ++i)
                {
                    v(i) -= dot_pr * u(i);
                }
            }

            // Normalize v
            const T norm = sqrt(Math_Functions<T>::dot_product(v,v,&policy));
            #pragma omp parallel for simd shared(v)
            for (size_t i = 0; i < n; ++i)
            {
                v(i) /= norm;
            }

            // Set column c of Q

            #pragma omp parallel for simd shared(Q,v)
            for (size_t i = 0; i < n; ++i)
            {
                Q(i,c) = v(i);
            }
        }


        // Compute R = Q^T * A
        size_t extQT[2],strQT[2];

        datastruct<T> QT=Q.transpose(extQT,strQT);

        switch (policy.algorithm_version)
        {
        case Math_MPI_Decomposition_Policy::Naive:
            In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(QT,A,R);
            break;
        case Math_MPI_Decomposition_Policy::Strassen:
            strassen_multiply_h(QT,A,R,false,false,policy);
            break;
        case Math_MPI_Decomposition_Policy::WinogradVariant:
            winograd_multiply_h(QT,A,R,false,false,policy);
        }

        Datastruct_Host_Memory_Functions<T>::free_data_ptr(tempC,mm,policy.memmapped_files);
        Datastruct_Host_Memory_Functions<T>::free_data_ptr(tempS,nm,policy.memmapped_files);
        Datastruct_Host_Memory_Functions<T>::free_copy(M,policy.memmapped_files);

    }
}






template <typename T>
void Math_Functions_MPI<T>::MPI_recursive_multiplication_helper(const Math_MPI_RecursiveMultiplication_Policy *pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    MPI_Status status;
    int message;
    for(;;)
    {
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, policy.comm, &status);



        bool strassen=false;
        switch (message)
        {
        case Math_MPI_RecursiveMultiplication_Policy::Strassen:
            strassen=true;
        case Math_MPI_RecursiveMultiplication_Policy::WinogradVariant:
        {
            size_t problemsize;
            MPI_Recv(&problemsize, 1, mpi_get_type<size_t>(), MPI_ANY_SOURCE, 1, policy.comm, &status);
            datastruct<T> A,B;
            bool ongpu=policy.should_use_gpu(problemsize,Math_Functions_Policy::default_cubic_treshold,false,7);
            bool separate_device_memory=false;
            if(ongpu)
            {
#if !defined(Unified_Shared_Memory)
                separate_device_memory=true;
#endif
            }
            if(separate_device_memory)
            {
                A=Datastruct_MPI_Functions<T>::MPI_Recv_device_alloc_datastruct(policy.memmapped_files,policy.devicenum,status.MPI_SOURCE, 2, policy.comm);
                B=Datastruct_MPI_Functions<T>::MPI_Recv_device_alloc_datastruct(policy.memmapped_files,policy.devicenum,status.MPI_SOURCE, 3, policy.comm);
            }
            else
            {
                A=Datastruct_MPI_Functions<T>::MPI_Recv_alloc_datastruct(policy.memmapped_files,status.MPI_SOURCE, 2, policy.comm);
                B=Datastruct_MPI_Functions<T>::MPI_Recv_alloc_datastruct(policy.memmapped_files,status.MPI_SOURCE, 3, policy.comm);
            }

            bool crowm=true;
            size_t rowsC=A.dpextents[0],
                   colsC=B.dpextents[1];

            size_t extC[2];
            size_t strC[2];

            extC[0]=(crowm==true)?rowsC:colsC;
            extC[1]=(crowm==true)?colsC:rowsC;

            strC[0]=(crowm==true)? colsC:1;
            strC[1]=(crowm==true)?1: rowsC;

            T* C_data;
            size_t length=rowsC*colsC;
            if(separate_device_memory)
            {
                C_data=Datastruct_GPU_Memory_Functions<T>::alloc_data_device_ptr(length,policy.memmapped_files,policy.devicenum);
            }
            else
            {
                C_data=Datastruct_Host_Memory_Functions<T>::alloc_data_ptr(length,policy.memmapped_files);
            }

            datastruct<T> C(C_data,length,crowm,2,extC,strC,false,false,separate_device_memory);


            if(policy.size_to_stop_recursion>=problemsize)
            {
                if(ongpu)
                {
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(A, B, C,policy.devicenum,false);
                }
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(A, B, C);
            }
            else
            {
                if(strassen)
                    strassen_multiply_h(A,B,C,ongpu,separate_device_memory,policy);
                else
                    winograd_multiply_h(A,B,C,ongpu,separate_device_memory,policy);
            }

            Datastruct_MPI_Functions<T>::MPI_Send_datastruct_pdata(C,status.MPI_SOURCE,4,policy.comm);
            if(separate_device_memory)
            {
                Datastruct_MPI_Functions<T>::MPI_Free_device_datastruct(A,policy.devicenum);
                Datastruct_MPI_Functions<T>::MPI_Free_device_datastruct(B,policy.devicenum);
                Datastruct_GPU_Memory_Functions<T>::free_data_device_ptr(C.dpdata,C.dpdatalength,policy.memmapped_files,policy.devicenum);
            }
            else
            {

                Datastruct_MPI_Functions<T>::MPI_Free_datastruct(A);
                Datastruct_MPI_Functions<T>::MPI_Free_datastruct(B);
                Datastruct_Host_Memory_Functions<T>::free_data_ptr(C.dpdata,C.dpdatalength,policy.memmapped_files);
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
void Math_Functions_MPI<T>::MPI_recursion_helper_end(MPI_Comm pcomm)
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

