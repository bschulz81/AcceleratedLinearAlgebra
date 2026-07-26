#ifndef DATABLOCK
#define DATABLOCK
#include <complex>
#include <omp.h>
#include <stdio.h>
#include <print>
#include<iostream>
#include <limits.h>
#include "indiceshelperfunctions.h"


#if defined(Unified_Shared_Memory)
#pragma omp requires unified_shared_memory
#else
#pragma omp requires unified_address
#endif


#pragma omp begin declare target
inline void fill_strides(const ptrdiff_t*    extents,ptrdiff_t*    strides, const ptrdiff_t rank, const bool rowmajor)
{
    if (rank==0)
        return;

    if (rowmajor)
    {

        strides[rank - 1] = 1;
        #pragma omp unroll partial
        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        strides[0] = 1;
        #pragma omp unroll partial
        for (ptrdiff_t i = 1; i < rank; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
    }
}

#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
struct is_complex : std::false_type {};
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline constexpr auto cond_conj(const T& val)
{
    if constexpr (is_complex<T>::value)
    {
        return std::conj(val);
    }
    else
    {
        return val;
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline constexpr auto returnval(const T& val,bool conj)
{
    if constexpr (is_complex<T>::value)
    {
        return conj? std::conj(val):val;
    }
    else
    {
        return val;
    }
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T, typename = std::void_t<>>
struct has_print : std::false_type {};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
struct has_print<T, std::void_t<decltype(std::declval<T>().print())>> : std::true_type {};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T, typename = std::void_t<>>
struct has_buffer_print : std::false_type {};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
struct has_buffer_print<T, std::void_t<
decltype(std::declval<T>().print_to_buffer(std::declval<char*>(), std::declval<ptrdiff_t>())),
decltype(std::declval<T>().required_buffer_size())
>> : std::true_type {};
#pragma omp end declare target





#pragma omp declare reduction(+: std::complex<double>: \
omp_out += omp_in) \
initializer(omp_priv(0.0, 0.0))

#pragma omp declare reduction(+: std::complex<float>: \
omp_out += omp_in) \
initializer(omp_priv(0.0f, 0.0f))

#pragma omp declare reduction(+: std::complex<long double>: \
omp_out += omp_in) \
initializer(omp_priv(0, 0))


#pragma omp begin declare target
template <typename T>
void print_variable(const T& var,bool conjugate)
{

    if constexpr (is_complex<T>::value)
    {
        double real_part = static_cast<double>(var.real());
        double imag_part = static_cast<double>(var.imag());
        if(conjugate)
            printf("(%g, %g)", real_part, -imag_part);
        else
            printf("(%g, %g)", real_part, imag_part);

    }

    else if constexpr (std::is_floating_point_v<T>)
    {
        printf("%g", static_cast<double>(var));
    }

    else if constexpr (std::is_integral_v<T>)
    {
        printf("%lld", static_cast<long long>(var));
    }

    else if constexpr (has_print<T>::value)
    {
        var.print();
    }
    else
    {
        printf("[Unknown Object]");
    }
}
#pragma omp end declare target



class GPU_Memory_Functions;
    class Host_Memory_Functions;

        class DataBlock_MPI_Functions;

            class In_Kernel_Mathfunctions;

                class Math_Functions_MPI;

                    class GPU_Math_Functions;

                        template <typename T>
                        class BlockedDataView;

                            template <typename T, typename Container>
                            class mdspan;

                                template <typename T, typename Container>
                                class mdspan_data;

                                    template <typename T>
                                    class DistributedDataBlock;

                                        template <typename T>
                                        class DataBlockArray;


                                            class DataBlockUtilities;
                                                class mdspan_utilities;

                                                    class Math_Functions_Policy;

                                                        #pragma omp begin declare target
                                                        enum class DataBlockObject
                                                        {
                                                            Scalar,
                                                            Vector,
                                                            Matrix,
                                                            Tensor
                                                        };
#pragma omp end declare target


#pragma omp begin declare target
struct ComputeMetadata
{
    bool ComputeStrides=true;
    bool ComputeLength=true;
};
#pragma omp end declare target

#pragma omp begin declare target
struct DataBlockConfig
{
    bool dprowmajor=true;
    bool dpconjugate=false;
    bool pmemmap=        false;
    bool data_is_devptr= false;
    int devicenum =     -INT_MAX;

};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
class DataBlock
{
public:
    friend class GPU_Memory_Functions;
    friend class Host_Memory_Functions;
    friend class DataBlock_MPI_Functions;
    friend class In_Kernel_Mathfunctions;
    friend class GPU_Math_Functions;
    friend class Math_Functions;
    friend class Math_Functions_MPI;
    friend class BlockedDataView<T>;
    friend class DistributedDataBlock<T>;
    friend class DataBlockArray<T>;
    friend class DataBlockUtilities;
    friend class mdspan_utilities;

    template <typename U, typename Containerr>
    friend class ::mdspan;

    template <typename U, typename Container>
    friend class ::mdspan_data;


    DataBlock() {};

    DataBlock(T*  data,    ptrdiff_t datalength,   ptrdiff_t   rank,
              ptrdiff_t*   extents,      ptrdiff_t*    strides,
              DataBlockConfig config, ComputeMetadata method);

    DataBlock(T*  data,    ptrdiff_t datalength,   ptrdiff_t   rank,
              ptrdiff_t*   extents,      ptrdiff_t*    strides,
              DataBlockConfig config);


    inline ptrdiff_t datalength() const
    {
        return dpdatalength;
    }

    inline ptrdiff_t rank() const
    {
        return dprank;
    }

    inline bool rowmajor() const
    {
        return dpconfig.dprowmajor;
    }


    inline int devptr_num()const
    {
        return dpconfig.devicenum;
    }

    inline bool data_is_devptr()const
    {
        return dpconfig.data_is_devptr;
    }

    inline T* former_hostptr()const
    {
        return devptr_former_hostptr;
    }

    inline T& data(ptrdiff_t i)
    {
        return dpdata[i];
    }

    inline  T data(ptrdiff_t i)const
    {
        return dpdata[i];
    }

    inline ptrdiff_t& extent(ptrdiff_t i)
    {
        return dpextents[i];
    }

    inline  ptrdiff_t extent(ptrdiff_t i) const
    {
        return dpextents[i];
    }

    inline ptrdiff_t& stride(ptrdiff_t i)
    {
        return dpstrides[i];
    }

    inline ptrdiff_t stride(ptrdiff_t i) const
    {
        return dpstrides[i];
    }


    inline T* data()
    {
        return dpdata;
    }

    inline const T* data() const
    {
        return dpdata;
    }

    inline ptrdiff_t* extents()
    {
        return dpextents;
    }

    inline const ptrdiff_t* extents() const
    {
        return dpextents;
    }

    inline ptrdiff_t* strides()
    {
        return dpstrides;
    }

    inline const ptrdiff_t* strides() const
    {
        return dpstrides;
    }



    inline T& operator()(const ptrdiff_t* indices)
    {
        return dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)];
    };


    inline T& operator()(const ptrdiff_t row,  const ptrdiff_t col)
    {
        return dpdata[row*dpstrides[0]+col*dpstrides[1]];
    };

    inline T& operator()(const ptrdiff_t i)
    {
        return dpdata[i*dpstrides[0]];
    };

    inline T operator()(const ptrdiff_t row, const ptrdiff_t col) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (dpconfig.dpconjugate)
            {
                return std::conj(dpdata[row * dpstrides[0] + col * dpstrides[1]]);
            }
        }

        return dpdata[row * dpstrides[0] + col * dpstrides[1]];
    }


    inline T operator()(const ptrdiff_t i) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (dpconfig.dpconjugate)
            {
                return std::conj(dpdata[i * dpstrides[0]]);
            }
        }

        return  dpdata[i * dpstrides[0]];;
    }

    inline T operator()(const ptrdiff_t* indices) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (dpconfig.dpconjugate)
            {
                return std::conj( dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)]);
            }
        }

        return  dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)];
    }

    inline void print()const;
    ptrdiff_t print_to_buffer( char* buffer,   ptrdiff_t capacity) const;
    ptrdiff_t print_required_size() const;


    template <typename Expr>
    requires requires(Expr e, DataBlock<T>& self, const Math_Functions_Policy* pol)
    {
        e.assign_to(self, pol);
    }
    DataBlock& operator=(const Expr& expr)
    {
        expr.assign_to(*this, nullptr);
        return *this;
    }

    template <typename Expr>
    requires requires(Expr e, DataBlock<T>& self, const Math_Functions_Policy* pol)
    {
        e.assign_to(self, pol);
    }
    DataBlock& assign(const Expr& expr, const Math_Functions_Policy* policy)
    {
        expr.assign_to(*this, policy);
        return *this;
    }

    enum Type
    {
        Scalar,
        Vector,
        Matrix,
        Tensor
    };

    inline Type ObjectType() const;

    inline bool is_scalar() const
    {
        return ObjectType() == Type::Scalar;
    }
    inline bool is_vector() const
    {
        return ObjectType() == Type::Vector;
    }
    inline bool is_matrix() const
    {
        return ObjectType() == Type::Matrix;
    }
    inline bool is_tensor() const
    {
        return ObjectType() == Type::Tensor;
    }

    bool is_conjugate() const
    {
        return dpconfig.dpconjugate;
    }

    inline bool is_contiguous()const;
protected:
    void printtensor_recursive(ptrdiff_t* indices, ptrdiff_t depth,bool ondevice) const;
    void printtensor_recursive_buffer( char*& cur, char* end,ptrdiff_t* indices,ptrdiff_t depth, bool ondevice) const;
    void printtensor_required_size_recursive(ptrdiff_t& count, ptrdiff_t* indices, ptrdiff_t depth,  bool ondevice) const;

    T*          dpdata = nullptr;
    ptrdiff_t      dpdatalength = 0;
    ptrdiff_t*     dpextents = nullptr;
    ptrdiff_t*     dpstrides = nullptr;

    ptrdiff_t      dprank = 0;

    DataBlockConfig dpconfig;
    T*          devptr_former_hostptr=nullptr;

};
#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
DataBlock<T>::DataBlock(
    T* data,
    ptrdiff_t datalength,
    ptrdiff_t rank,
    ptrdiff_t* extents,
    ptrdiff_t* strides,
    DataBlockConfig config,
    ComputeMetadata method
) : dpdata(data),
    dpdatalength(abs(datalength)),
    dpextents(extents),
    dpstrides(strides),
    dprank(abs(rank)),
    dpconfig(config)
{
#if defined(Unified_Shared_Memory)
    dpconfig.data_is_devptr =false;
#endif
    if(extents != nullptr)
    {
        #pragma omp unroll partial
        for(ptrdiff_t i=0; i<dprank; i++)
            dpextents[i]=abs(dpextents[i]);

        if( strides != nullptr)
        {
            if(method.ComputeStrides)
                fill_strides(dpextents, dpstrides, rank, dpconfig.dprowmajor);
            else
            {
                switch (dprank)
                {
                case 0:
                    dpconfig.dprowmajor=true;
                    break;
                case 1:
                    dpconfig.dprowmajor = true;
                    break;
                case 2:
                    dpconfig.dprowmajor = (abs(dpstrides[1]) < abs(dpstrides[0])) ? true : false;
                    break;
                default:
                    dpconfig.dprowmajor = is_row_major(extents, strides, dprank) ? true : false;
                    break;
                }
            }
            if(method.ComputeLength)
                dpdatalength=abs(compute_data_length<OpenMPVariant::Sequential>(extents, strides, abs(rank)));
            else
                dpdatalength =abs(datalength);
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
DataBlock<T>::DataBlock(
    T* data,
    ptrdiff_t datalength,
    ptrdiff_t rank,
    ptrdiff_t* extents,
    ptrdiff_t* strides,
    DataBlockConfig config
) : dpdata(data),
    dpdatalength(abs(datalength)),
    dpextents(extents),
    dpstrides(strides),
    dprank(abs(rank)),
    dpconfig(config)
{
#if defined(Unified_Shared_Memory)
    dpconfig.data_is_devptr =false;
#endif
    if(dpextents!=nullptr)
    {
        #pragma omp unroll partial
        for(ptrdiff_t i=0; i<dprank; i++)
            dpextents[i]=abs(dpextents[i]);
    }
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
DataBlock<T>::Type  DataBlock<T>::ObjectType() const
{
    if (abs(dprank) == 1)
    {
        if (abs(dpextents[0]) == 1) return Type::Scalar;
        return Type::Vector;
    }
    if (abs(dprank) == 2)
    {
        if (abs(dpextents[0]) == 1 && abs(dpextents[1]) == 1) return Type::Scalar;
        if (abs(dpextents[0]) == 1 || abs(dpextents[1]) == 1) return Type::Vector;
        return Type::Matrix;
    }
    if (abs(dprank) > 2) return Type::Tensor;

    // fallback
    return Type::Scalar;
}
#pragma omp end declare target






#pragma omp begin declare target
template<typename T>
bool DataBlock<T>::is_contiguous() const
{
    if (dprank == 0)
    {
        return dpdatalength == 1;
    }
    ptrdiff_t expected_stride = 1;

    if (dpconfig.dprowmajor)
    {

        for (int i = (int)abs(dprank) - 1; i >= 0; --i)
        {
            if (abs(dpstrides[i]) != expected_stride)return false;
            expected_stride *= abs(dpextents[i]);
        }
    }
    else
    {

        for (ptrdiff_t i = 0; i < dprank; ++i)
        {
            if (abs(dpstrides[i]) != expected_stride)return false;
            expected_stride *= abs(dpextents[i]);
        }
    }

    return expected_stride == dpdatalength;
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
ptrdiff_t DataBlock<T>::print_to_buffer(
    char* buffer,
    ptrdiff_t capacity) const
{
    if(capacity == 0)
        return 0;

    char* cur = buffer;
    char* end = buffer + capacity - 1; // reserve space for '\0'

    if(dpdata == nullptr ||
            dpextents == nullptr ||
            dpstrides == nullptr ||
            dpdatalength == 0)
    {
        int n = snprintf(cur,end-cur+1,"\n[]\n");

        if(n > 0)
            cur += (n < (end-cur+1)) ? n : (end-cur);

        *cur = '\0';

        return (ptrdiff_t)(cur-buffer);
    }

    int n = snprintf(cur,end-cur+1,"\n");

    if(n > 0)
        cur += (n < (end-cur+1)) ? n : (end-cur);

    ptrdiff_t* indices = new ptrdiff_t[dprank];

    #pragma omp unroll partial
    for(ptrdiff_t i=0; i<dprank; i++)
        indices[i]=0;

    bool ondevice =
        omp_is_initial_device() &&
        dpconfig.data_is_devptr;

    printtensor_recursive_buffer(
        cur,
        end,
        indices,
        0,
        ondevice);

    delete[] indices;

    if(cur < end)
        *cur++ = '\n';

    *cur = '\0';

    return (ptrdiff_t)(cur-buffer);
}

#pragma omp end declare target


#pragma omp begin declare target

template<typename T>
void DataBlock<T>::printtensor_recursive_buffer(
    char*& cur,
    char* end,
    ptrdiff_t* indices,
    ptrdiff_t depth,
    bool ondevice) const
{
    if(cur >= end)
        return;

    if(depth == dprank)
    {
        ptrdiff_t offset =
            compute_offset<OpenMPVariant::Sequential>(
                indices,
                dpstrides,
                dprank);

        T value;

        if(ondevice)
        {
            omp_target_memcpy(
                &value,
                dpdata,
                sizeof(T),
                0,
                sizeof(T)*offset,
                omp_get_initial_device(),
                dpconfig.devicenum);
        }
        else
        {
            value = dpdata[offset];
        }

        int n = 0;
        ptrdiff_t max_avail = (end - cur) + 1;

        if constexpr (is_complex<T>::value)
        {
            double r = static_cast<double>(value.real());
            double i = static_cast<double>(value.imag());
            if(this->dpconfig.dpconjugate)
                n = snprintf(cur, max_avail, "(%g, %g)", r, -i);
            else
                n = snprintf(cur, max_avail, "(%g, %g)", r, i);
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            n = snprintf(cur, max_avail, "%g", static_cast<double>(value));
        }
        else if constexpr (std::is_integral_v<T>)
        {
            n = snprintf(cur, max_avail, "%lld", static_cast<long long>(value));
        }
        else if constexpr (has_buffer_print<T>::value)
        {

            ptrdiff_t written = value.print_to_buffer(cur, max_avail);
            cur += (written < max_avail) ? written : (max_avail - 1);
            return;
        }
        else
        {
            n = snprintf(cur, max_avail, "[Unknown Object]");
        }

        if(n > 0)
        {
            ptrdiff_t avail = end-cur;

            cur += ((ptrdiff_t)n < avail)
                   ? n
                   : avail;
        }

        return;
    }

    if(cur < end)
        *cur++ = '[';

    for(ptrdiff_t i=0; i<dpextents[depth]; i++)
    {
        indices[depth] = i;

        printtensor_recursive_buffer(
            cur,
            end,
            indices,
            depth+1,
            ondevice);

        if(i+1 < dpextents[depth])
        {
            int n =
                snprintf(
                    cur,
                    end-cur+1,
                    ", ");

            if(n > 0)
            {
                ptrdiff_t avail = end-cur;

                cur += ((ptrdiff_t)n < avail)
                       ? n
                       : avail;
            }

            if(depth < dprank-1)
            {
                if(cur < end)
                    *cur++ = '\n';

                for(ptrdiff_t k=0; k<depth+1; k++)
                {
                    if(cur < end)
                        *cur++ = ' ';
                }
            }
        }

        if(cur >= end)
            break;
    }

    if(cur < end)
        *cur++ = ']';
}

#pragma omp end declare target




#pragma omp begin declare target

template<typename T>
void DataBlock<T>::printtensor_required_size_recursive(
    ptrdiff_t& count,
    ptrdiff_t* indices,
    ptrdiff_t depth,
    bool ondevice) const
{
    if(depth == dprank)
    {
        ptrdiff_t offset =
            compute_offset<OpenMPVariant::Sequential>(
                indices,
                dpstrides,
                dprank);

        T value;

        if(ondevice)
        {
            omp_target_memcpy(
                &value,
                dpdata,
                sizeof(T),
                0,
                sizeof(T)*offset,
                omp_get_initial_device(),
                dpconfig.devicenum);
        }
        else
        {
            value = dpdata[offset];
        }


        if constexpr (is_complex<T>::value)
        {
            double r = static_cast<double>(value.real());
            double i = static_cast<double>(value.imag());
            int n=0;
            if(this->dpconfig.dpconjugate)
                n= snprintf(nullptr, 0, "(%g, %g)", r, -i);
            else
                n = snprintf(nullptr, 0, "(%g, %g)", r, i);
            if(n > 0) count += (ptrdiff_t)n;
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            int n = snprintf(nullptr, 0, "%g", static_cast<double>(value));
            if(n > 0) count += (ptrdiff_t)n;
        }
        else if constexpr (std::is_integral_v<T>)
        {
            int n = snprintf(nullptr, 0, "%lld", static_cast<long long>(value));
            if(n > 0) count += (ptrdiff_t)n;
        }
        else if constexpr (has_buffer_print<T>::value)
        {
            count += value.required_buffer_size();
        }
        else
        {
            count += 16; // "[Unknown Object]"
        }
        return;
    }

    count += 1; // '['

    for(ptrdiff_t i=0; i<dpextents[depth]; i++)
    {
        indices[depth] = i;

        printtensor_required_size_recursive(
            count,
            indices,
            depth+1,
            ondevice);

        if(i + 1 < dpextents[depth])
        {
            count += 2; // ", "

            if(depth < dprank - 1)
            {
                count += 1;           // '\n'
                count += depth + 1;   // indentation spaces
            }
        }
    }

    count += 1; // ']'
}

#pragma omp end declare target



#pragma omp begin declare target
template<typename T>
ptrdiff_t DataBlock<T>::print_required_size() const
{
    if(dpdata == nullptr ||
            dpextents == nullptr ||
            dpstrides == nullptr ||
            dpdatalength == 0)
    {
        return 4; // "\n[]\n"
    }

    ptrdiff_t count = 2; // leading and trailing '\n'

    ptrdiff_t* indices = new ptrdiff_t[dprank];

    #pragma omp unroll partial
    for(ptrdiff_t i=0; i<dprank; i++)
        indices[i] = 0;

    bool ondevice =
        omp_is_initial_device() &&
        dpconfig.data_is_devptr;

    printtensor_required_size_recursive(
        count,
        indices,
        0,
        ondevice);

    delete[] indices;

    return count;
}

#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void DataBlock<T>::print() const
{
    if(dpdata==nullptr || dpextents==nullptr|| dpstrides==nullptr ||dpdatalength==0)
    {
        printf("\n[]\n");
        return;
    }

    printf("\n");

    ptrdiff_t* indices= new ptrdiff_t[dprank];
    #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < dprank; ++i)
        indices[i] = 0;

    bool ondevice=omp_is_initial_device()&&dpconfig.data_is_devptr;
    printtensor_recursive(indices, 0,ondevice);
    delete []indices;

    printf("\n");
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void DataBlock<T>::printtensor_recursive(ptrdiff_t* indices, ptrdiff_t depth,bool ondevice) const
{
    if (depth == dprank)
    {
        ptrdiff_t offset=compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank);
        T d;
        if(ondevice)
            omp_target_memcpy(&d,dpdata,sizeof(T),0,sizeof(T)*offset,omp_get_initial_device(),this->dpconfig.devicenum);
        else
            d= dpdata[offset];

        print_variable(d,dpconfig.dpconjugate);
        return;
    }

    printf("[");

    for (ptrdiff_t i = 0; i < (ptrdiff_t) dpextents[depth]; ++i)
    {
        indices[depth] = i;
        printtensor_recursive(indices, depth + 1,ondevice);

        if (i + 1 < (ptrdiff_t) dpextents[depth])
        {
            printf(", ");
            if (depth < dprank - 1)
            {
                printf("\n");
                for (ptrdiff_t k = 0; k < depth + 1; ++k)
                    printf(" ");
            }
        }
    }
    printf("]");
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
class DataBlockArray
{
public:
    T* pdata=nullptr;
    ptrdiff_t pdatalength=0;
    bool prowm=true;
    ptrdiff_t ptensor_rank=0;
    ptrdiff_t *pblock_offsets=nullptr;
    ptrdiff_t* pextentsbuffer=nullptr;
    ptrdiff_t* pstridesbuffer=nullptr;
    ptrdiff_t pnumblocks=0;
    bool pdata_is_devptr=false;
    int pdevnum=-INT_MAX;
    bool pconjugate=false;

    inline T& operator()(const ptrdiff_t* indices,const ptrdiff_t blocknumber)
    {
        return pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)];
    };

    inline T operator()(const ptrdiff_t* indices,const ptrdiff_t blocknumber) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj( pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)]);
            }
        }
        return  pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)];
    }


    inline T& operator()(const ptrdiff_t row,  const ptrdiff_t col,const ptrdiff_t blocknumber)
    {
        T* const data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[2*blocknumber];
        const ptrdiff_t stride1=pstridesbuffer[2*blocknumber+1];

        return data_ptr[row*stride0+col*stride1];
    };

    inline T operator()(const ptrdiff_t row, const ptrdiff_t col, const ptrdiff_t blocknumber) const
    {
        const T* data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[2*blocknumber];
        const ptrdiff_t stride1=pstridesbuffer[2*blocknumber+1];

        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(data_ptr[row*stride0+col*stride1]);
            }
        }

        return data_ptr[row*stride0+col*stride1];
    }

    inline T& operator()(const ptrdiff_t i,const ptrdiff_t blocknumber)
    {
        T* const data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[blocknumber];
        return data_ptr[i*stride0];
    };

    inline T operator()(const ptrdiff_t i,const ptrdiff_t blocknumber) const
    {
        const T* data_ptr=pdata+pblock_offsets[blocknumber];
        const ptrdiff_t stride0=pstridesbuffer[blocknumber];
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(data_ptr[i*stride0]);
            }
        }

        return  data_ptr[i*stride0];
    }

};


#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline DataBlock<T>get_datablock_from_arrays(const ptrdiff_t i, const DataBlockArray<T> &arr)
{

    ptrdiff_t len =(i + 1 <arr.pnumblocks)? arr.pblock_offsets[i+1] - arr.pblock_offsets[i]: arr.pdatalength - arr.pblock_offsets[i];
    return DataBlock<T>(arr.pdata + arr.pblock_offsets[i],
                        len,  arr.ptensor_rank,
                        arr.pextentsbuffer + i*arr.ptensor_rank,
                        arr.pstridesbuffer + i*arr.ptensor_rank,
                        DataBlockConfig{.dprowmajor=arr.prowm,
                                        .dpconjugate=arr.pconjugate,
                                        .data_is_devptr=arr.pdata_is_devptr,
                                        .devicenum=arr.pdata_is_devptr? arr.pdevnum:-INT_MAX,
                                       }  );

}
#pragma omp end declare target


#endif
