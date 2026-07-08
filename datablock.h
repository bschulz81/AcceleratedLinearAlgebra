#ifndef DATABLOCK
#define DATABLOCK
#include <complex>
#include <omp.h>
#include <stdio.h>
#include <print>
#include<iostream>

#include "indiceshelperfunctions.h"


#if defined(Unified_Shared_Memory)
#pragma omp requires unified_shared_memory
#else
#pragma omp requires unified_address
#endif


#pragma omp begin declare target
inline void fill_strides(const size_t*    extents,size_t*    strides, const size_t rank, const bool rowmajor)
{
    if (rank==0)
        return;

    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[rank - 1] = 1;
        #pragma omp unroll partial
        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        #pragma omp unroll partial
        for (size_t i = 1; i < rank; ++i)
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
decltype(std::declval<T>().print_to_buffer(std::declval<char*>(), std::declval<size_t>())),
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


template <typename T>
class BlockedDataView;

class In_Kernel_Mathfunctions;


class Math_Functions_MPI;
class GPU_Math_Functions;
template<typename U, typename Container>
class mdspan;

template<typename U, typename Container>
class mdspan_data;

template <typename T>
class DistributedDataBlock;

template <typename T>
class DataBlockArray;


class DataBlockUtilities;
class mdspan_utilities;

class Math_Functions_Policy;

#pragma omp begin declare target
enum class DataBlockObject{
    Scalar,
    Vector,
    Matrix,
    Tensor
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

    template<typename U, typename Container>
    friend class ::mdspan;

    template<typename U, typename Container>
    friend class ::mdspan_data;


    DataBlock() {};


    //vector constructor
   DataBlock(T*  data,   size_t* extents,   size_t* strides,  bool data_is_devptr,     int       devicenum,    bool conjugateflag);

    //matrix constructor
   DataBlock(T*  data,    bool   rowm,       size_t  rows,     size_t   cols,
                 size_t*   extents,      size_t*   strides,  bool compute_strides_from_extents,
                 bool data_is_devptr,              int devicenum,      bool conjugateflag);
   //raw copy constructor
   DataBlock(T*  data,    size_t datalength, bool    rowm,     size_t   rank,
                size_t*   extents,      size_t*    strides,
                    bool data_is_devptr,                int devicenum,                    bool conjugateflag );
   //constructor for tensors that computes datalenght and strides from extents
   DataBlock(T*  data,    size_t datalength, bool    rowm,     size_t   rank,
                 size_t*   extents,      size_t*   strides,
                bool compute_datalength,            bool compute_strides_from_extents,bool data_is_devptr,int devicenum, bool conjugateflag );


    inline size_t datalength() const
    {
        return dpdatalength;
    }

    inline size_t rank() const
    {
        return dprank;
    }

    inline bool rowmajor() const
    {
        return dprowmajor;
    }


    inline int devptr_num()const
    {
        return devptr_devicenum;
    }
    inline bool is_dev_ptr()const
    {
        return dpdata_is_devptr;
    }
    inline T* former_hostptr()const
    {
        return devptr_former_hostptr;
    }



    inline bool data_is_devptr() const
    {
        return dpdata_is_devptr;
    }

    inline T& data(size_t i)
    {
        return dpdata[i];
    }

    inline  T data(size_t i)const
    {
        return dpdata[i];
    }

    inline size_t& extent(size_t i)
    {
        return dpextents[i];
    }

    inline  size_t extent(size_t i) const
    {
        return dpextents[i];
    }

    inline size_t& stride(size_t i)
    {
        return dpstrides[i];
    }

    inline size_t stride(size_t i) const
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

    inline size_t* extents()
    {
        return dpextents;
    }

    inline const size_t* extents() const
    {
        return dpextents;
    }

    inline size_t* strides()
    {
        return dpstrides;
    }

    inline const size_t* strides() const
    {
        return dpstrides;
    }



    inline T& operator()(const size_t* indices)
    {
        return dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)];
    };


    inline T& operator()(const size_t row,  const size_t col)
    {
        return dpdata[row*dpstrides[0]+col*dpstrides[1]];
    };

    inline T& operator()(const size_t i)
    {
        return dpdata[i*dpstrides[0]];
    };

    inline T operator()(const size_t row, const size_t col) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(dpdata[row * dpstrides[0] + col * dpstrides[1]]);
            }
        }

        return dpdata[row * dpstrides[0] + col * dpstrides[1]];
    }


    inline T operator()(const size_t i) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(dpdata[i * dpstrides[0]]);
            }
        }

        return  dpdata[i * dpstrides[0]];;
    }

    inline T operator()(const size_t* indices) const
    {
        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj( dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)]);
            }
        }

        return  dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)];
    }

    inline void print()const;
    size_t print_to_buffer( char* buffer,   size_t capacity) const;
    size_t print_required_size() const;


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
        return this->pconjugate;
    }

    inline bool is_contiguous()const;
protected:
    void printtensor_recursive(size_t* indices, size_t depth,bool ondevice) const;
    void printtensor_recursive_buffer( char*& cur, char* end,size_t* indices,size_t depth, bool ondevice) const;
    void printtensor_required_size_recursive(size_t& count, size_t* indices, size_t depth,  bool ondevice) const;

    T*          dpdata = nullptr;
    size_t*     dpextents = nullptr;
    size_t*     dpstrides = nullptr;
    size_t      dpdatalength = 0;
    size_t      dprank = 0;
    bool        dprowmajor = true;
    int         devptr_devicenum=-1;
    bool        dpdata_is_devptr=false;
    T*          devptr_former_hostptr=nullptr;
    bool        pconjugate=false;
};
#pragma omp end declare target






#pragma omp begin declare target
template<typename T>
DataBlock<T>::DataBlock(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t*    extents,
    size_t*    strides,
    bool compute_datalength,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    int devicenum,
    bool conj_flag):
    dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dprank(rank),
    dprowmajor(rowm),
    devptr_devicenum( devicenum),

#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false),
#else
    dpdata_is_devptr(data_is_devptr),
#endif
    pconjugate(conj_flag)

{
    if(compute_strides_from_extents==true && extents!=nullptr && strides!=nullptr && rank !=0)
    {
        fill_strides(dpextents,dpstrides,rank,rowm);
    }

    if(compute_strides_from_extents==false && extents!=nullptr && strides!=nullptr && rank !=0)
    {
        switch(dprank)
        {
        case 0:
            dprowmajor=true;
            break;
        case 1:
            dprowmajor=true;
            break;
        case 2:
            dprowmajor=dpstrides[1]<dpstrides[0];
            break;
        default:
            dprowmajor=is_row_major(extents, strides,dprank);
        }
    }

    if(compute_datalength==true && extents!=nullptr && strides!=nullptr && rank !=0)
    {
        dpdatalength=compute_data_length<OpenMPVariant::Sequential>(extents,strides,rank);
    }
    else
        dpdatalength=datalength;
}
#pragma omp end declare target





#pragma omp begin declare target
template<typename T>
DataBlock<T>::DataBlock(
    T*    data,
    size_t datalength,
    bool rowm,
    size_t rank,
    size_t*    extents,
    size_t*    strides,
    bool data_is_devptr,
    int devicenum,
    bool    conj_flag
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dpdatalength(datalength),
    dprank(rank),
    dprowmajor(rowm),
    devptr_devicenum( devicenum),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false),
#else
    dpdata_is_devptr(data_is_devptr),
#endif
    pconjugate(conj_flag)
{}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T>
DataBlock<T>::DataBlock(T*  data,
                        size_t* extents,
                        size_t*   strides,
                        bool data_is_devptr,
                        int devicenum,
                        bool conjugateflag):
    dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dprank(1),
    dprowmajor(true),
    devptr_devicenum( devicenum),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false),
#else
    dpdata_is_devptr(data_is_devptr),
#endif
    pconjugate(conjugateflag)
{dpdatalength=(extents[0]-1) * strides[0]+1;}
#pragma omp end declare target




#pragma omp begin declare target
template<typename T> DataBlock<T>::DataBlock(
    T*    data,
    bool rowm,
    size_t rows,
    size_t cols,
    size_t*    extents,
    size_t*    strides,
    bool compute_strides_from_extents,
    bool data_is_devptr,
    int devicenum,
    bool    conj_flag
) : dpdata(data),
    dpextents(extents),
    dpstrides(strides),
    dprowmajor(rowm),
    devptr_devicenum( devicenum),
#if defined(Unified_Shared_Memory)
    dpdata_is_devptr(false),
#else
    dpdata_is_devptr(data_is_devptr),
#endif
    pconjugate(conj_flag)
{
    if((rows>1) && (cols>1))
    {
        dprank=2;
        if(extents!=nullptr)
        {
            dpextents[0]=rows;
            dpextents[1]=cols;
        }
        if(strides!=nullptr)
        {
            if(compute_strides_from_extents)
            {
                dpstrides[0]=(rowm==true)? cols:1;
                dpstrides[1]=(rowm==true)?1: rows;
                dprowmajor=rowm;
            }
            else
            {
                dprowmajor=dpstrides[1]<dpstrides[0];
            }
            if(extents!=nullptr)
            {
                dpdatalength=(rows-1) * strides[0]+(cols-1)*strides[1]+1;
            }
        }
    }
    else
    {
        if(rows>1)
        {
            dprank=1;
            if(compute_strides_from_extents)
                dpstrides[0]=1;
            dpdatalength=rows;
            dpextents[0]=rows;
            dpdatalength=(extents[0]-1) * strides[0]+1;
        }
        if(cols>1)
        {
            dprank=1;
            if(compute_strides_from_extents)
                dpstrides[0]=1;
            dpextents[0]=cols;
            dpdatalength=(extents[0]-1) * strides[0]+1;
        }
        if(rows==0 && cols==0)
        {
            dprank=0;
            dpdatalength=0;
        }
        dprowmajor=true;
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
DataBlock<T>::Type  DataBlock<T>::ObjectType() const
{
    if (dprank == 1)
    {
        if (dpextents[0] == 1) return Type::Scalar;
        return Type::Vector;
    }
    if (dprank == 2)
    {
        if (dpextents[0] == 1 && dpextents[1] == 1) return Type::Scalar;
        if (dpextents[0] == 1 || dpextents[1] == 1) return Type::Vector;
        return Type::Matrix;
    }
    if (dprank > 2) return Type::Tensor;

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
    size_t expected_stride = 1;

    if (dprowmajor)
    {

        for (int i = (int)dprank - 1; i >= 0; --i)
        {
            if (dpstrides[i] != expected_stride)return false;
            expected_stride *= dpextents[i];
        }
    }
    else
    {

        for (size_t i = 0; i < dprank; ++i)
        {
            if (dpstrides[i] != expected_stride)return false;
            expected_stride *= dpextents[i];
        }
    }

    return expected_stride == dpdatalength;
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
size_t DataBlock<T>::print_to_buffer(
    char* buffer,
    size_t capacity) const
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

        return (size_t)(cur-buffer);
    }

    int n = snprintf(cur,end-cur+1,"\n");

    if(n > 0)
        cur += (n < (end-cur+1)) ? n : (end-cur);

    size_t* indices = new size_t[dprank];

    #pragma omp unroll partial
    for(size_t i=0; i<dprank; i++)
        indices[i]=0;

    bool ondevice =
        omp_is_initial_device() &&
        dpdata_is_devptr;

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

    return (size_t)(cur-buffer);
}

#pragma omp end declare target


#pragma omp begin declare target

template<typename T>
void DataBlock<T>::printtensor_recursive_buffer(
    char*& cur,
    char* end,
    size_t* indices,
    size_t depth,
    bool ondevice) const
{
    if(cur >= end)
        return;

    if(depth == dprank)
    {
        size_t offset =
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
                devptr_devicenum);
        }
        else
        {
            value = dpdata[offset];
        }

        int n = 0;
        size_t max_avail = (end - cur) + 1;

        if constexpr (is_complex<T>::value)
        {
            double r = static_cast<double>(value.real());
            double i = static_cast<double>(value.imag());
            if(this->pconjugate)
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

            size_t written = value.print_to_buffer(cur, max_avail);
            cur += (written < max_avail) ? written : (max_avail - 1);
            return;
        }
        else
        {
            n = snprintf(cur, max_avail, "[Unknown Object]");
        }

        if(n > 0)
        {
            size_t avail = end-cur;

            cur += ((size_t)n < avail)
                   ? n
                   : avail;
        }

        return;
    }

    if(cur < end)
        *cur++ = '[';

    for(size_t i=0; i<dpextents[depth]; i++)
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
                size_t avail = end-cur;

                cur += ((size_t)n < avail)
                       ? n
                       : avail;
            }

            if(depth < dprank-1)
            {
                if(cur < end)
                    *cur++ = '\n';

                for(size_t k=0; k<depth+1; k++)
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
    size_t& count,
    size_t* indices,
    size_t depth,
    bool ondevice) const
{
    if(depth == dprank)
    {
        size_t offset =
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
                devptr_devicenum);
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
            if(this->pconjugate)
                n= snprintf(nullptr, 0, "(%g, %g)", r, -i);
            else
                n = snprintf(nullptr, 0, "(%g, %g)", r, i);
            if(n > 0) count += (size_t)n;
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            int n = snprintf(nullptr, 0, "%g", static_cast<double>(value));
            if(n > 0) count += (size_t)n;
        }
        else if constexpr (std::is_integral_v<T>)
        {
            int n = snprintf(nullptr, 0, "%lld", static_cast<long long>(value));
            if(n > 0) count += (size_t)n;
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

    for(size_t i=0; i<dpextents[depth]; i++)
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
size_t DataBlock<T>::print_required_size() const
{
    if(dpdata == nullptr ||
            dpextents == nullptr ||
            dpstrides == nullptr ||
            dpdatalength == 0)
    {
        return 4; // "\n[]\n"
    }

    size_t count = 2; // leading and trailing '\n'

    size_t* indices = new size_t[dprank];

    #pragma omp unroll partial
    for(size_t i=0; i<dprank; i++)
        indices[i] = 0;

    bool ondevice =
        omp_is_initial_device() &&
        dpdata_is_devptr;

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

    size_t* indices= new size_t[dprank];
    #pragma omp unroll partial
    for (size_t i = 0; i < dprank; ++i)
        indices[i] = 0;

    bool ondevice=omp_is_initial_device()&&dpdata_is_devptr;
    printtensor_recursive(indices, 0,ondevice);
    delete []indices;

    printf("\n");
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void DataBlock<T>::printtensor_recursive(size_t* indices, size_t depth,bool ondevice) const
{
    if (depth == dprank)
    {
        size_t offset=compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank);
        T d;
        if(ondevice)
            omp_target_memcpy(&d,dpdata,sizeof(T),0,sizeof(T)*offset,omp_get_initial_device(),this->devptr_devicenum);
        else
            d= dpdata[offset];

        print_variable(d,pconjugate);
        return;
    }

    printf("[");

    for (size_t i = 0; i < dpextents[depth]; ++i)
    {
        indices[depth] = i;
        printtensor_recursive(indices, depth + 1,ondevice);

        if (i + 1 < dpextents[depth])
        {
            printf(", ");
            if (depth < dprank - 1)
            {
                printf("\n");
                for (size_t k = 0; k < depth + 1; ++k)
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
    size_t pdatalength=0;
    bool prowm=true;
    size_t ptensor_rank=0;
    size_t *pblock_offsets=nullptr;
    size_t* pextentsbuffer=nullptr;
    size_t* pstridesbuffer=nullptr;
    size_t pnumblocks=0;
    bool pdata_is_devptr=false;
    int pdevnum=-1;
    bool pconjugate=false;

    inline T& operator()(const size_t* indices,const size_t blocknumber)
    {
        return pdata[compute_offset<OpenMPVariant::Sequential>(indices, pstridesbuffer, ptensor_rank,blocknumber)];
    };

    inline T operator()(const size_t* indices,const size_t blocknumber) const
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


    inline T& operator()(const size_t row,  const size_t col,const size_t blocknumber)
    {
        T* const data_ptr=pdata+pblock_offsets[blocknumber];
        const size_t stride0=pstridesbuffer[2*blocknumber];
        const size_t stride1=pstridesbuffer[2*blocknumber+1];

        return data_ptr[row*stride0+col*stride1];
    };

    inline T operator()(const size_t row, const size_t col, const size_t blocknumber) const
    {
        const T* data_ptr=pdata+pblock_offsets[blocknumber];
        const size_t stride0=pstridesbuffer[2*blocknumber];
        const size_t stride1=pstridesbuffer[2*blocknumber+1];

        if constexpr (is_complex<T>::value)
        {
            if (pconjugate)
            {
                return std::conj(data_ptr[row*stride0+col*stride1]);
            }
        }

        return data_ptr[row*stride0+col*stride1];
    }

    inline T& operator()(const size_t i,const size_t blocknumber)
    {
        T* const data_ptr=pdata+pblock_offsets[blocknumber];
        const size_t stride0=pstridesbuffer[blocknumber];
        return data_ptr[i*stride0];
    };

    inline T operator()(const size_t i,const size_t blocknumber) const
    {
        const T* data_ptr=pdata+pblock_offsets[blocknumber];
        const size_t stride0=pstridesbuffer[blocknumber];
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
inline DataBlock<T>get_datablock_from_arrays(const size_t i, const DataBlockArray<T> &arr)
{

    size_t len =(i + 1 <arr.pnumblocks)? arr.pblock_offsets[i+1] - arr.pblock_offsets[i]: arr.pdatalength - arr.pblock_offsets[i];
    return DataBlock<T>(arr.pdata + arr.pblock_offsets[i],
                        len, arr.prowm, arr.ptensor_rank,
                        arr.pextentsbuffer + i*arr.ptensor_rank,
                        arr.pstridesbuffer + i*arr.ptensor_rank,
                        false, false,arr.pdata_is_devptr,arr.pdevnum,arr.pconjugate);

}
#pragma omp end declare target


#endif
