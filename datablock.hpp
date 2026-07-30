#ifndef DATABLOCKIMPL
#define DATABLOCKIMPL

#include "omp.h"

#include <stdio.h>

#include "datablock.h"
#include "expression_templates.h"

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
    dpconfig(config),
    dpconjugate(false)
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
    dpconfig(config),
    dpconjugate(false)
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
template<typename T>
inline ptrdiff_t DataBlock<T>::  datalength() const
{
    return dpdatalength;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t DataBlock<T>::  rank() const
{
    return dprank;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>::   rowmajor() const
{
    return dpconfig.dprowmajor;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline int DataBlock<T>::  devptr_num()const
{
    return dpconfig.devicenum;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>::  data_is_devptr()const
{
    return dpconfig.data_is_devptr;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T* DataBlock<T>:: former_hostptr()const
{
    return devptr_former_hostptr;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T& DataBlock<T>:: data(ptrdiff_t i)
{
    return dpdata[i];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T DataBlock<T>:: data(ptrdiff_t i)const
{
    return dpdata[i];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t& DataBlock<T>::extent(ptrdiff_t i)
{
    return dpextents[i];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t DataBlock<T>::extent(ptrdiff_t i) const
{
    return dpextents[i];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t& DataBlock<T>::stride(ptrdiff_t i)
{
    return dpstrides[i];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t DataBlock<T>::stride(ptrdiff_t i) const
{
    return dpstrides[i];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T* DataBlock<T>::data()
{
    return dpdata;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline const T* DataBlock<T>::data() const
{
    return dpdata;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t* DataBlock<T>:: extents()
{
    return dpextents;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline const ptrdiff_t* DataBlock<T>:: extents() const
{
    return dpextents;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline ptrdiff_t* DataBlock<T>:: strides()
{
    return dpstrides;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline const ptrdiff_t* DataBlock<T>:: strides() const
{
    return dpstrides;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T& DataBlock<T>:: operator()(const ptrdiff_t* indices)
{
    return dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)];
};
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T& DataBlock<T>:: operator()(const ptrdiff_t row,  const ptrdiff_t col)
{
    return dpdata[row*dpstrides[0]+col*dpstrides[1]];
};
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T& DataBlock<T>:: operator()(const ptrdiff_t i)
{
    return dpdata[i*dpstrides[0]];
};
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T DataBlock<T>:: operator()(const ptrdiff_t row, const ptrdiff_t col) const
{
    if constexpr (is_complex<T>::value)
    {
        if (this->dpconjugate)
        {
            return std::conj(dpdata[row * dpstrides[0] + col * dpstrides[1]]);
        }
    }

    return dpdata[row * dpstrides[0] + col * dpstrides[1]];
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T DataBlock<T>:: operator()(const ptrdiff_t i) const
{
    if constexpr (is_complex<T>::value)
    {
        if (this->dpconjugate)
        {
            return std::conj(dpdata[i * dpstrides[0]]);
        }
    }

    return  dpdata[i * dpstrides[0]];;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline T DataBlock<T>:: operator()(const ptrdiff_t* indices) const
{
    if constexpr (is_complex<T>::value)
    {
        if (this->dpconjugate)
        {
            return std::conj( dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)]);
        }
    }

    return  dpdata[compute_offset<OpenMPVariant::Sequential>(indices, dpstrides, dprank)];
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>::is_tensor() const
{
    return DataShape() == DataBlockObject::Tensor;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>::is_conjugate() const
{
    return this->dpconjugate;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>:: is_scalar() const
{
    return DataShape() == DataBlockObject::Scalar;
}
#pragma omp end declare target

#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>::  is_vector() const
{
    return DataShape() == DataBlockObject::Vector;
}
#pragma omp end declare target


#pragma omp begin declare target
template<typename T>
inline bool DataBlock<T>::  is_matrix() const
{
    return DataShape() == DataBlockObject::Matrix;
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
DataBlockObject DataBlock<T>::DataShape() const
{
    if (abs(dprank) == 1)
    {
        if (abs(dpextents[0]) == 1) return DataBlockObject::Scalar;
        return DataBlockObject::Vector;
    }
    if (abs(dprank) == 2)
    {
        if (abs(dpextents[0]) == 1 && abs(dpextents[1]) == 1) return DataBlockObject::Scalar;
        if (abs(dpextents[0]) == 1 || abs(dpextents[1]) == 1) return DataBlockObject::Vector;
        return DataBlockObject::Matrix;
    }
    if (abs(dprank) > 2) return DataBlockObject::Tensor;

    // fallback
    return DataBlockObject::Scalar;
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
            if(this->dpconjugate)
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
            if(this->dpconjugate)
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

        print_variable(d,dpconjugate);
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





#endif // DATABLOCKIMPL
