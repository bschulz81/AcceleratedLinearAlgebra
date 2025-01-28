#ifndef MDSPANH
#define MDSPANH

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>



#include <iostream>
#include <array>
#include <vector>

#include <numeric>
#include <cmath>
#include <numbers>

#include <omp.h>
#include "openacc.h"

#include <iostream>
#include <thread>
#include <mpi.h>


enum Matrix_Multiplication_Algorithm
{
    Naive=0,
    Strassen=1,
    WinogradVariant=2
};

enum MessageType
{
    COMMAND_STRASSEN,
    COMMAND_WINOGRAD,
    COMMAND_SENDMATRIX
};

struct matrix_multiplication_parameters
{
    size_t algorithm_version{Matrix_Multiplication_Algorithm::Naive};
    size_t size_for_naive_algorithm=2;
    bool memmapped_files=true;
    bool gpu_offload=true;
    bool omp=true;
    bool mpi=false;
    MPI_Comm comm=MPI_COMM_NULL;
    MPI_Status status;
    bool size_for_mpi=2;
};


template <typename T>struct datastruct
{
    T* pdata = nullptr;
    size_t* pextents = nullptr;
    size_t* pstrides = nullptr;
    size_t pdatalength = 0;
    size_t prank = 0;
    bool prowmajor=true;
    datastruct(
        T* data,
        size_t pdatalength,
        bool rowm,
        size_t rank,
        size_t* extents=nullptr,
        size_t* strides = nullptr,
        bool pcompute_datalength=false,
        bool compute_strides_from_extents=false
    );

    datastruct(
        T* data,
        size_t datalength,
        bool rowm,
        size_t rows,
        size_t cols,
        size_t* extents=nullptr,
        size_t* strides = nullptr,
        bool compute_datalength=true,
        bool compute_strides_from_extents=false
    );

    datastruct(
        T* data,
        size_t datalength,
        bool rowm,
        bool rowvector,
        size_t length,
        size_t* extents=nullptr,
        size_t* strides =nullptr,
        bool compute_datalength=true,
        bool compute_strides_from_extents=true
    );

    ~datastruct();
    T&operator()(const size_t* indices);
    T operator()(const size_t* indices)const;
    T&operator()(const size_t row, const size_t col);
    T operator()(const size_t row, const size_t col)const;
    T&operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1);
    T operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)const;
    T&operator()(const size_t row);
    T operator()(const size_t row)const;


    datastruct<T> substruct(size_t *poffsets, size_t *psub_extents, size_t*psub_strides=nullptr, T* sub_data=nullptr) ;
    datastruct<T> subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,
                                 size_t*sub_extents, size_t *sub_strides=nullptr,  T*sub_data=nullptr) ;

    datastruct<T> transpose(size_t*newextents, size_t *newstrides);
    datastruct<T> row(const size_t row_index,size_t *newextents);
    datastruct<T> column(const size_t col_index,size_t *newextents);
};




#pragma acc routine vector
inline size_t  compute_offset(const size_t* indices,  size_t* strides,const size_t r, bool rowmajor=true)
{
    size_t offset = 0;

    if (rowmajor)
    {
        // Row-major layout: iterate outermost to innermost
#pragma acc loop reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        // Column-major layout: iterate innermost to outermost
#pragma acc loop reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[r - 1 - i] * strides[r - 1 - i];
        }
    }

    return offset;
}

#pragma acc routine seq
inline size_t compute_offset(const size_t row, const size_t col,  size_t* strides)
{
    return row * strides[0]+col*strides[1];
}

#pragma acc routine seq
inline size_t compute_offset(const size_t row, const size_t col, const size_t matrixstride0, const size_t matrixstride1)
{
    return row * matrixstride0+col*matrixstride1;
}


#pragma acc routine vector
size_t inline compute_offset(const size_t *indices, const size_t* strides,const size_t r, bool rowmajor=true)
{
    size_t offset = 0;

    if (rowmajor)
    {
        // Row-major layout: iterate outermost to innermost
#pragma acc loop reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        // Column-major layout: iterate innermost to outermost
#pragma acc loop reduction(+ : offset)
        for (size_t i = 0; i < r; ++i)
        {
            offset += indices[r - 1 - i] * strides[r - 1 - i];
        }
    }

    return offset;
}


#pragma acc routine seq
inline size_t compute_data_length(const size_t* extents, const size_t* strides,const size_t rank)
{
    size_t offset=0;
//#pragma acc loop vector reduction(+:offset)
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}

#pragma acc routine seq
template<typename T>inline T& datastruct<T>::operator()(const size_t row)
{

    return pdata[row * pstrides[0]];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t row)const
{

    return pdata[row * pstrides[0]];
}


#pragma acc routine seq
template<typename T>inline T& datastruct<T>::operator()(const size_t row, const size_t col)
{
    return pdata[row * pstrides[0] + col * pstrides[1]];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t row, const size_t col)const
{
    return pdata[row * pstrides[0] + col * pstrides[1]];
}


#pragma acc routine seq
template<typename T>inline T& datastruct<T>::operator()(const size_t* indices)
{
    return pdata[compute_offset(indices, this->pstrides, this->prank)];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t* indices)const
{
    return pdata[compute_offset(indices, this->pstrides, this->prank)];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)const
{
    return pdata[row * strides0 + col *strides1];
}

#pragma acc routine seq
template<typename T>inline T& datastruct<T>::operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)
{
    return pdata[row * strides0 + col * strides1];
}



#pragma acc routine seq
template<typename T>inline datastruct<T> datastruct<T>::transpose(size_t*newextents, size_t *newstrides)
{
    newextents[0]=pextents[1];
    newextents[1]=pextents[0];

    newstrides[0]=pstrides[1];
    newstrides[1]=pstrides[0];

    return datastruct(pdata,pdatalength,prowmajor,prank,newextents,newstrides,false,false);

}


#pragma acc routine seq
inline void fill_strides(const size_t* extents,size_t* strides, const size_t rank, const bool rowmajor)
{
    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        for (size_t i = 1; i < rank; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];

        }
    }
}


#pragma acc routine seq
template<typename T> datastruct<T>::datastruct(
    T* data,
    size_t pdatalength,
    bool rowm,
    size_t rank,
    size_t* extents,
    size_t* strides,
    bool compute_datalength,
    bool compute_strides_from_extents
) : pdata(data),
    pextents(extents),
    pstrides(strides),
    pdatalength(pdatalength),
    prank(rank),
    prowmajor(rowm)

{
    if(compute_strides_from_extents==true && pextents!=nullptr && pstrides!=nullptr && rank !=0)
    {
        fill_strides(pextents,pstrides,rank,rowm);
    }
    if(compute_datalength==true && pextents!=nullptr && pstrides!=nullptr && rank !=0)
    {
        pdatalength=compute_data_length(pextents,pstrides,rank);
    }

}


#pragma acc routine seq
template<typename T> datastruct<T>::datastruct(
    T* data,
    size_t datalength,
    bool rowm,
    size_t rows,
    size_t cols,
    size_t* extents,
    size_t* strides,
    bool compute_datalength,
    bool compute_strides_from_extents
) : pdata(data),
    pextents(extents),
    pstrides(strides),
    pdatalength(datalength),
    prank(2),
    prowmajor(rowm)
{
    if(extents!=nullptr)
    {
        pextents[0]=(rowm==true)?rows:cols;
        pextents[1]=(rowm==true)?cols:rows;
    }
    if(pstrides!=nullptr && compute_strides_from_extents)
    {
        pstrides[0]=(rowm==true)? cols:1;
        pstrides[1]=(rowm==true)?1: rows;
    }
    if(compute_datalength==true && extents!=nullptr && strides!=nullptr)
    {
        pdatalength=(rows-1) * strides[0]+(cols-1)*strides[1]+1;
    }

}


#pragma acc routine seq
template<typename T> datastruct<T>::datastruct(
    T* data,
    size_t datalength,
    bool rowm,
    bool rowvector,
    size_t noelements,
    size_t* extents,
    size_t* strides,
    bool compute_datalength,
    bool compute_strides_from_extents
) : pdata(data),
    pextents(extents),
    pstrides(strides),
    pdatalength(datalength),
    prank(1),
    prowmajor(true)
{
    if(extents!=nullptr)
    {
        pextents[0]=noelements;
    }
    if(pstrides!=nullptr && compute_strides_from_extents)
    {
        if(rowvector)
            pstrides[0]=(rowm==true)? 1:noelements;
        else
            pstrides[0]=(rowm==true)? noelements:1;
    }
    if(compute_datalength==true && strides!=nullptr)
    {
        pdatalength=(noelements-1) * strides[0]+1;
    }

}

template<typename T> datastruct<T>::~datastruct()
{

}


#pragma acc routine seq
template<typename T>datastruct<T> datastruct<T>::substruct(size_t *poffsets, size_t *psub_extents, size_t*psub_strides, T* sub_data)
{
    size_t offset_index = 0;
    const size_t r=this->prank;
    if(sub_data==nullptr)
    {


#pragma acc loop auto reduction( + : offset_index )
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += poffsets[i] * pstrides[i];

        }
        return datastruct(pdata + offset_index,0,this->prowmajor, r,this->prowmajor, psub_extents,pstrides, true,false );
    }
    else
    {
        // Compute the new strides for the subspan
        size_t *indices;
        size_t *global_indices;

        indices=new size_t[r];
        global_indices= new size_t[r];

#pragma acc loop auto
        for (size_t i=0; i<r; i++)
        {
            indices[i]=0;
        }

        size_t largest_buffer_index=0;
        // Fill the supplied buffer with subspan data
        while (true)
        {
            // Compute the current global indices
#pragma acc loop auto
            for (size_t i = 0; i < r; ++i)
            {
                global_indices[i] = poffsets[i] + indices[i];
            }

            // Compute the offsets for the original data and the new buffer
            size_t original_index = compute_offset(global_indices, pstrides, prowmajor);
            size_t buffer_index = compute_offset(indices,psub_strides, prowmajor);

            // Copy the data from the original tensor to the sub-buffer
            sub_data[buffer_index] = pdata[original_index];

            if(buffer_index>largest_buffer_index)
                largest_buffer_index=buffer_index;

            // Increment the indices for the Cartesian product
            size_t dim = r;
            while (dim-- > 0)
            {
                if (++indices[dim] < psub_extents[dim])
                {
                    break; // If no overflow, stop carrying
                }
                indices[dim] = 0; // Reset the current dimension and carry to the next
            }

            // If all dimensions have overflowed, we're done
            if (dim == size_t(-1))
            {
                break;
            }

        }

        // Create and return a new mdspan with the updated pointer, extents, and strides
        datastruct pd(sub_data,0,prowmajor,psub_extents, psub_strides,true,true);
        if(omp_is_initial_device()!=true)
        {
            omp_free(global_indices,omp_default_mem_alloc);
            omp_free(indices,omp_default_mem_alloc);
        }
        else
        {
            delete[] global_indices;
            delete[] indices;
        }


        return pd;
    }

}


#pragma acc routine seq
template<typename T>datastruct<T>  datastruct<T>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,  size_t tile_cols,  size_t *sub_extents,  size_t *sub_strides,  T*sub_data)
{
    if(sub_data==nullptr)
    {
        return datastruct(pdata +row * pstrides[0]+col * pstrides[1],0,prowmajor,tile_rows,tile_cols,sub_extents,pstrides,true,false);
    }
    else
    {
        if (prowmajor)
        {
            const size_t s0=pstrides[0];
            const size_t s1=pstrides[1];
            const T* pd=this->pdata;
            // Row-major layout: fill row by row
//#pragma acc loop auto collapse (2)
            for (size_t i = 0; i < tile_rows; ++i)
            {
                for (size_t j = 0; j < tile_cols; ++j)
                {
                    sub_data[i * tile_cols + j] = pd[compute_offset(row + i, col + j, s0, s1)];
                }
            }
        }
        else
        {
            const size_t s0=pstrides[0];
            const size_t s1=pstrides[1];
            const T* pd=this->pdata;
            // Column-major layout: fill column by column
//#pragma acc loop auto collapse (2)
            for (size_t j = 0; j < tile_cols; ++j)
            {
                for (size_t i = 0; i < tile_rows; ++i)
                {
                    sub_data[j * tile_rows + i] = pd[compute_offset(row + i, col + j, s0, s1)];
                }
            }
        }

        return datastruct(sub_data,0,prowmajor,tile_rows, tile_cols,sub_extents,sub_strides,true,true);
    }
}













#pragma acc routine seq
template <typename T>
datastruct<T> datastruct<T>::row(const size_t row_index, size_t* extents)
{

    // Offset the data pointer to the start of the row
    T* row_data = pdata + row_index * pstrides[0];

    // Fill the extents array with the appropriate values for the row
    extents[0] = pextents[1]; // Extent for a row is the number of columns

    return datastruct<T>(
               row_data,                   // Adjusted data pointer
               pstrides[1] * extents[0],   // Data length (stride[1] * number of columns)
               true,                       // Row-major layout for 1D
               1,                          // Rank is now 1
               &extents[0],                    // Updated extents
               pstrides                    // Reuse parent's stride pointer
           );
}




#pragma acc routine seq
template <typename T>
datastruct<T> datastruct<T>::column(const size_t col_index, size_t* extents)
{
    // Offset the data pointer to the start of the column
    T* col_data = pdata + col_index * pstrides[1];

    // Fill the extents array with the appropriate values for the column
    extents[0] = pextents[0]; // Extent for a column is the number of rows

    return datastruct(col_data, pstrides[0] * extents[0], false,  1,  &extents[0],   pstrides  );
}

#pragma acc routine seq
template <typename T>
void printmatrix(datastruct<T>&span)
{
    const size_t rows= span.pextents[0];
    const size_t cols=span.pextents[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            printf("%f ",span(i, j));
        }
        printf("%s \n","");
    }
}

#pragma acc routine seq
template <typename T>
void printvector(datastruct<T>&span)
{
    const size_t rows= span.pextents[0];

    for (size_t i = 0; i <rows; ++i)
    {
        printf("%f\n",span(i));
    }

}

#define STRINGIFY(x) #x

#define CREATE_IN_STRUCT(dA)                                           \
  enter data copyin(dA)\
  copyin(dA.pdata[0:dA.pdatalength])\
  copyin(dA.pextents[0:dA.prank])\
  copyin(dA.pstrides[0:dA.prank])

#define CREATE_OUT_STRUCT(dL)\
 enter data copyin(dL)\
 create(dL.pdata[0:dL.pdatalength])\
 copyin(dL.pextents[0:dL.prank])\
 copyin(dL.pstrides[0:dL.prank])

#define EXIT_STRUCT(dL)\
exit data delete(dL.pdata[0:dL.pdatalength])\
 delete(dL.pextents[0:dL.prank])\
 delete(dL.pstrides[0:dL.prank])\
 delete(dL)

#define UPDATE_HOST(dL) update self(dL.pdata[0:dL.pdatalength])

#define UPDATE_DEVICE(dA) update device(dL.pdata[0:dL.pdatalength])


template<typename T>
T* create_temp_mmap(const size_t array_size)
{
    size_t file_size = array_size * sizeof(double);

    // Create a temporary file using std::tmpfile()
    FILE* tmpf = tmpfile();
    if (!tmpf)
    {
        perror("tmpfile");
        return NULL;
    }

    // Get the file descriptor from the FILE*
    int fd = fileno(tmpf);
    if (fd == -1)
    {
        perror("fileno");
        fclose(tmpf);
        return NULL;
    }

    // Resize the file to the required size
    if (ftruncate(fd, file_size) == -1)
    {
        perror("ftruncate");
        fclose(tmpf);
        return NULL;
    }

    // Memory map the file
    T* mmap_ptr = (T*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED)
    {
        perror("mmap");
        fclose(tmpf);
        return NULL;
    }

    // Close the FILE* but keep the memory mapping valid
    fclose(tmpf);

    // Return the pointer to the mapped memory
    return mmap_ptr;
}

// Function to unmap the memory-mapped file
void delete_temp_mmap(double* mmap_ptr,const size_t array_size)
{
    size_t file_size = array_size * sizeof(double);
    if (munmap(mmap_ptr, file_size) == -1)
    {
        perror("munmap");
    }
}





using namespace std;







// Concept definitions
template <typename Container>
concept StaticContainer =
    requires(Container c, size_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<size_t>;
    {
        c[i]
    }
    -> std::convertible_to<typename Container::value_type>;
    (!requires(Container c, size_t i)
    {
        c.reserve(i);
    });
};

template <typename Container>
concept DynamicContainer =
    requires(Container c, size_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<size_t>;
    {
        c[i]
    }
    -> std::convertible_to<typename Container::value_type>;
    c.reserve(i);  // Require reserve() for dynamic containers
};




// Concept to check if two containers are of the same type and have matching size
template <typename ExtentsContainer>
concept Container =
    (StaticContainer<ExtentsContainer>   ||  // Same size for static containers
     (DynamicContainer<ExtentsContainer>));  // Same size for dynamic containers
// Class template for mdspan
template <typename T, typename Container>
class mdspan
{
public:

    // Constructors
    // Simplified constructors
    mdspan(T* data, size_t datalength, bool rowm, const Container& extents, const Container& strides);
    mdspan(T* data, bool rowm, const Container& extents, const Container& strides);
    mdspan(T* data, bool rowm, const Container& extents);
    mdspan(T* data, bool rowm, size_t rows, size_t cols);

    mdspan(size_t datalength, bool rowm, bool memmap, const Container& extents, const Container& strides);
    mdspan(bool rowm, bool memmap, const Container& extents, const Container& strides);
    mdspan(bool rowm, bool memmap, const Container& extents);
    mdspan(bool rowm, bool memmap, size_t rows, size_t cols);

    mdspan(mdspan<T,Container>&& other) noexcept;
    mdspan& operator=(mdspan&& other) noexcept;
    ~mdspan();


    // Deleted copy constructor and copy assignment
    mdspan(const mdspan<T,Container>&) = delete;
    mdspan& operator=(const mdspan<T,Container>&) = delete;
    // Access operators
    inline T& operator()(const Container& extents);
    inline T& operator()(const size_t row, const size_t col);
    inline T& operator()(const size_t i);
    inline T operator()(const Container& extents)const;
    inline T operator()(const size_t row, const size_t col)const;
    inline T operator()(const size_t i)const;

    // Subspan methods
    mdspan<T, Container> subspan(const Container& offsets, const Container& sub_extents, T* sub_data=nullptr) const;
    mdspan<T, Container> subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data=nullptr)const;
    mdspan<T, Container> row(const size_t row_index);
    mdspan<T, Container> column(size_t col_index);
    mdspan<T, Container> transpose() ;


    // Other utility methods
    size_t extent(const size_t dim) const;
    size_t rank() const;
    size_t stride(const size_t dim) const;

    // Member function declarations
    Container& extents()const;
    Container& strides()const;

    size_t datalength() const;
    // Data structure for parallel device allocation (assumed type)
    datastruct<T> pdatastruct;



private:
void initialize_extents_and_strides(const Container&extents,const Container & strides);
void initialize_extents(const Container&extents);
void allocate_data(bool memmap, size_t datalength);

    // Private member variables
    Container pextents;  // Use the ExtentContainer type
    Container pstrides;  // Use the StrideContainer type
    bool pis_associated=false;
    bool pdatac_copied_to_device=false;
    bool pownsdata=false;
    bool pwith_memmap=false;
};




// Access operator for multidimensional indices
template <typename T, typename Container>
T& mdspan<T, Container>::operator()(const Container& indices)
{


    size_t offset = 0;
    #pragma omp parallel for reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * pdatastruct.pstrides[i];
    }

    return pdatastruct.pdata[offset];
}

template <typename T, typename Container>
T& mdspan<T, Container>::operator()(const size_t row,const size_t col)
{


    return pdatastruct.pdata[row * pdatastruct.pstrides[0] + col * pdatastruct.pstrides[1]];
}

template <typename T, typename Container>
T& mdspan<T, Container>::operator()(size_t i)
{

    return pdatastruct.pdata[i * pdatastruct.pstrides[0]];
}


// Access operator for multidimensional indices
template <typename T, typename Container>
T mdspan<T, Container>::operator()(const Container& indices)const
{

    size_t offset = 0;
    #pragma omp parallel for reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * pdatastruct.pstrides[i];
    }

    return pdatastruct.pdata[offset];
}

template <typename T, typename Container>
T mdspan<T, Container>::operator()(const size_t row,const size_t col)const
{


    return pdatastruct.pdata[row * pdatastruct.pstrides[0] + col * pdatastruct.pstrides[1]];
}

template <typename T, typename Container>
T mdspan<T, Container>::operator()(const size_t i)const
{

    return pdatastruct.pdata[i * pdatastruct.pstrides[0]];
}







template <typename Container>
void compute_strides(const Container& extents, Container& strides,const bool rowmajor)
{
    const size_t n = extents.size();

    if constexpr (StaticContainer<Container>)
    {
        strides = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        strides.resize(n); // Resize dynamic container
    }

    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        for (size_t i = 1; i < n; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
    }
}

template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents_and_strides(const Container& extents, const Container& strides) {
    const size_t r = extents.size();

    if constexpr (StaticContainer<Container>) {
        pextents = {};
        pstrides = {};
    }

    if constexpr (DynamicContainer<Container>) {
        pextents.resize(r);
        pstrides.resize(r);
    }
   #pragma omp simd
    for (size_t i = 0; i < r; ++i) {
        pextents[i] = extents[i];
        pstrides[i] = strides[i];
    }
    // Assign to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
}
template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents(const Container& extents) {
    const size_t r = extents.size();
    if constexpr (StaticContainer<Container>) {
        pextents = {};
    }

    if constexpr (DynamicContainer<Container>) {
        pextents.resize(r);

    }

   #pragma omp simd
    for (size_t i = 0; i < r; ++i) {
        pextents[i] = extents[i];
    }
    // Assign to datastruct
    pdatastruct.pextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,   size_t datalength,  bool rowm, const Container& extents, const Container& strides)
    :pdatastruct(data,datalength,rowm,extents.size(),nullptr,nullptr,false,false),
     pis_associated(false),
     pownsdata(false),
     pwith_memmap(false)
{
    initialize_extents_and_strides(extents,strides);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm, const Container& extents, const Container& strides )
    : pdatastruct(data, 0,rowm,extents.size(),nullptr,nullptr,false,false),
      pis_associated(false),
      pownsdata(false),
      pwith_memmap(false)
      // Initialize pdatastruct with placeholders
{
    initialize_extents_and_strides(extents,strides);
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, bool rowm,const  Container& extents)
    :  pdatastruct(data,0,rowm,extents.size(),nullptr,nullptr,false,false),
       pis_associated(false),
       pownsdata(false),
       pwith_memmap(false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}







template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  const size_t rows, const size_t cols)
    :  pdatastruct(data,0,rowm,2,nullptr,nullptr,false,false),
       pis_associated(false),
       pownsdata(false),
       pwith_memmap(false)

{
    const size_t r=2;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }
    // Resize and copy extents from container


    pextents[0]=(rowm==true)?rows:cols;
    pextents[1]=(rowm==true)?cols:rows;
    compute_strides(pextents,pstrides,rowm);

    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}





template <typename T, typename Container>
mdspan<T, Container>::mdspan(  size_t datalength,  bool rowm,bool memmap, const Container& extents, const Container& strides)
    :pdatastruct(nullptr,
                 datalength,rowm,extents.size(),nullptr,nullptr,false,false),
     pis_associated(false)
{
   initialize_extents_and_strides(extents,strides,rowm);
    allocate_data(memmap,pdatastruct.pdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan( bool rowm,bool memmap, const Container& extents, const Container& strides )
    : pdatastruct(nullptr, 0,rowm,extents.size(),nullptr,nullptr,false,false),
      pis_associated(false)
      // Initialize pdatastruct with placeholders
{
   initialize_extents_and_strides(extents,strides,rowm);
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
    allocate_data(memmap,pdatastruct.pdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan( bool rowm,bool memmap,const  Container& extents)
    :  pdatastruct(nullptr,0,rowm,extents.size(),nullptr,nullptr,false,false),
       pis_associated(false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    // Assign actual pointers to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();

    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
    pownsdata=true;
    allocate_data(memmap,pdatastruct.pdatalength);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const bool rowm, bool memmap, const size_t rows, const size_t cols)
    :  pdatastruct(nullptr,0,rowm,2,nullptr,nullptr,false,false),
       pis_associated(false)

{
    const size_t r=2;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }
    // Resize and copy extents from container


    pextents[0]=(rowm==true)?rows:cols;
    pextents[1]=(rowm==true)?cols:rows;
    compute_strides(pextents,pstrides,rowm);

    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
    allocate_data(memmap,pdatastruct.pdatalength);

}

template <typename T, typename Container>
mdspan<T, Container>::~mdspan()
{
    if(pownsdata==true)
    {
        if (pwith_memmap==true)
            delete_temp_mmap(pdatastruct.pdata,sizeof(T)*pdatastruct.pdatalength);
        else
            delete[] pdatastruct.pdata;
    }
}

template <typename T, typename Container>
void mdspan<T, Container>::allocate_data(bool memmap, size_t datalength) {
    pownsdata = true;
    if (memmap) {
        const size_t s=sizeof(T) * datalength;
        pdatastruct.pdata = create_temp_mmap<T>(s);
        pwith_memmap = true;
    } else {
        pdatastruct.pdata = new T[datalength];
        pwith_memmap = false;
    }
}

template <typename T, typename Container>
mdspan<T, Container>::mdspan(mdspan<T,Container>&& other) noexcept
    : pstrides(std::move(other.pstrides)),
      pextents(std::move(other.pextents)),
      pdatastruct(other.pdatastruct.pdata,other.pdatastruct.pdatalength,other.pdatastruct.rowmajor,nullptr,nullptr,false,false)
{
    pownsdata=other.pownsdata;
    pwith_memmap=other.pwith_memmap;
    pis_associated=other.pis_associated;
    // Update pointers in datastruct to the new strides and extents
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pextents = pextents.data();

    // Null out the other's pointers to avoid double delete
    other.pdatastruct.pdata = nullptr;
    other.pdatastruct.pstrides = nullptr;
    other.pdatastruct.pextents = nullptr;
    other.pdatastruct.pdata = nullptr;

}


// Move assignment operator

template <typename T, typename Container>
mdspan<T, Container> &  mdspan<T, Container>::operator=(mdspan<T, Container> && other) noexcept
{
    if (this != &other)
    {
        // Free existing resources
        if(pownsdata==true)
        {
            if (pwith_memmap==true)
                delete_temp_mmap(pdatastruct.pdata,sizeof(T)*pdatastruct.pdatalength);
            else
                delete[] pdatastruct.pdata;
        }
        pownsdata=other.pownsdata;
        pwith_memmap=other.pwith_memmap;
        pis_associated=other.pis_associated;
        // Move data members

        pdatastruct.pdata = other.pdatastruct.pdata;
        pdatastruct.pdatalength=other.pdatastruct.pdatalength;
        pdatastruct.prank=other.pdatastruct.prank;
        pdatastruct.prowmajor=other.pdatastruct.prowmajor;
        pstrides = std::move(other.pstrides);
        pextents = std::move(other.pextents);
        // Update pointers in datastruct to the new strides and extents
        pdatastruct.pstrides = pstrides.data();
        pdatastruct.pextents = pextents.data();
        // Null out the other's pointers to avoid double delete
        other.pdatastruct.pdata = nullptr;
        other.pdatastruct.pstrides = nullptr;
        other.pdatastruct.pextents = nullptr;
        other.pdatastruct.pdata = nullptr;
    }
    return *this;
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan(const Container&offsets, const Container &sub_extents, T*sub_data)const
{
    const size_t r=pdatastruct.prank;

    if (sub_data==nullptr)
    {
        // Compute the offset to the starting point
        size_t offset_index = 0;

        //  #pragma omp parallel for reduction( + : offset_index )
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += offsets[i] * pdatastruct.pstrides[i];
        }

        // Create a new mdspan_dynamic with the updated pointer, extents, and the same strides
        return mdspan(pdatastruct.pdata + offset_index,pdatastruct.prowmajor, sub_extents, pstrides);

    }
    else
    {
        // Compute the new strides for the subspan
        Container sub_strides;
        compute_strides(sub_extents, sub_strides, pdatastruct.prowmajor);
        vector<size_t> indices(r,0);
        vector<size_t> global_indices(r,0);
        while (true)
        {
            // Compute the current global indices
            #pragma omp parallel for
            for (size_t i = 0; i < r; ++i)
            {
                global_indices[i] = offsets[i] + indices[i];
            }

            // Compute the offsets for the original data and the new buffer
            size_t original_index = compute_offset(global_indices.data(), pdatastruct.pstrides,global_indices.size(), pdatastruct.prowmajor);
            size_t buffer_index = compute_offset(indices.data(),sub_strides.data(),indices.size(), pdatastruct.prowmajor);

            // Copy the data from the original tensor to the sub-buffer
            sub_data[buffer_index] = pdatastruct.pdata[original_index];

            // Increment the indices for the Cartesian product
            size_t dim = r;
            while (dim-- > 0)
            {
                if (++indices[dim] < sub_extents[dim])
                {
                    break; // If no overflow, stop carrying
                }
                indices[dim] = 0; // Reset the current dimension and carry to the next
            }

            // If all dimensions have overflowed, we're done
            if (dim == size_t(-1))
            {
                break;
            }
        }
        size_t size=1;
        #pragma omp simd reduction(* : size)
        for (size_t i = 0; i < r; ++i)
        {
            size*=sub_extents[i];
        }
        // Create and return a new mdspan with the updated pointer, extents, and strides
        return mdspan(sub_data, pdatastruct.prowmajor, sub_extents, sub_strides );
    }
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data)const
{

    if(sub_data==nullptr)
    {

        size_t offset=row * pdatastruct.pstrides[0]+col * pdatastruct.pstrides[1];
        const Container ext= {tile_rows,tile_cols};
        return mdspan(pdatastruct.pdata +offset,pdatastruct.prowmajor,ext,pstrides);
    }
    else
    {
        if (pdatastruct.prowmajor)
        {
            // Row-major layout: fill row by row
            #pragma omp parallel for collapse (2)
            for (size_t i = 0; i < tile_rows; ++i)
            {
                for (size_t j = 0; j < tile_cols; ++j)
                {
                    sub_data[i * tile_cols + j] = pdatastruct.pdata[
                                                      compute_offset(row + i, col + j, pdatastruct.pstrides[0], pdatastruct.pstrides[1])
                                                  ];
                }
            }
        }
        else
        {
            // Column-major layout: fill column by column
            #pragma omp parallel for collapse (2)
            for (size_t j = 0; j < tile_cols; ++j)
            {
                for (size_t i = 0; i < tile_rows; ++i)
                {
                    sub_data[j * tile_rows + i] = pdatastruct.pdata[
                                                      compute_offset(row + i, col + j, pdatastruct.pstrides[0], pdatastruct.pstrides[1])
                                                  ];
                }
            }
        }
        // Create and return a new mdspan with the updated pointer and extents
        Container sub_extents = {tile_rows, tile_cols};

        Container sub_strides = (pdatastruct.prowmajor==true)? Container{tile_cols, 1} :
                                Container{1,tile_rows}; // Contiguous row-major layout

        return mdspan(sub_data,pdatastruct.prowmajor, sub_extents, sub_strides );
    }
}



// Function to glue mdspans into the target, using offsets for placement
template<typename TargetSpan, typename SourceSpan>
bool glue_matrices(TargetSpan target, const vector<SourceSpan>& spans,
                   const vector<pair<size_t, size_t>>& offsets)
{
    // Ensure we have the same number of spans and offsets





    // Iterate over spans and their corresponding offsets
    #pragma omp parallel for
    for (size_t idx = 0; idx < spans.size(); ++idx)
    {
        const auto& span = spans[idx];
        size_t row_offset = offsets[idx].first;
        size_t col_offset = offsets[idx].second;


        // Copy the current span into the target at the given offset
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < span.extent(0); ++i)    // Rows of the span
        {
            for (size_t j = 0; j < span.extent(1); ++j)    // Columns of the span
            {
                target(row_offset + i, col_offset + j) = span(i, j);
            }
        }
    }

    return true;
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::column(const size_t col_index)
{
    assert(pdatastruct.prank == 2 && "Column extraction is only valid for 2D matrices.");
    assert(col_index < pdatastruct.pextents[1] && "Column index out of bounds.");

    size_t num_rows = pdatastruct.pextents[0];
    Container column_extents = {num_rows};
    Container column_strides = {pdatastruct.pstrides[0]};

    return mdspan<T, Container>(
               pdatastruct.pdata + col_index * pdatastruct.pstrides[1], // Offset to the column
               pdatastruct.prowmajor,                                  // Column-major layout
               column_extents,                                         // Updated extents
               column_strides                                          // Updated strides
           );
}

template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::row(const size_t row_index)
{
    assert(pdatastruct.prank == 2 && "Row extraction is only valid for 2D matrices.");
    assert(row_index < pdatastruct.pextents[0] && "Row index out of bounds.");

    size_t num_cols = pdatastruct.pextents[1];
    Container row_extents = {num_cols};
    Container row_strides = {pdatastruct.pstrides[1]};

    return mdspan<T, Container>(
               pdatastruct.pdata + row_index * pdatastruct.pstrides[0], // Offset to the row
               pdatastruct.prowmajor,                                  // Row-major layout
               row_extents,                                            // Updated extents
               row_strides                                             // Updated strides
           );
}



template <typename T>
mdspan<T,vector<size_t>> & MPI_recv_mdspan(bool memmap, int source, int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    // Receive metadata
    size_t pdatalength, prank;
    bool prowmajor;
    MPI_Recv(&pdatalength, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prank, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(&prowmajor, 1, MPI_C_BOOL, source, tag, pcomm, &status);

    vector<size_t> extents(prank,0);
    vector<size_t> strides(prank,0);
    // Allocate memory for extents and strides

    MPI_Recv(extents.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);
    MPI_Recv(strides.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &status);

    mdspan<T,vector<size_t>> md(pdatalength,prowmajor,memmap,extents,strides);

    MPI_Recv(md.pdatastruct.pdata,sizeof(T)* md.pdatastruct.pdata, MPI_BYTE, source, tag, pcomm, &status);

    return md;
}

template <typename T>
mdspan<T,vector<size_t>> & MPI_Irecv_mdspan(bool memmap, int source, int tag, MPI_Comm pcomm)
{
    MPI_Request request;
    // Receive metadata
    size_t pdatalength, prank;
    bool prowmajor;
    MPI_Irecv(&pdatalength, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);
    MPI_Irecv(&prank, 1, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);
    MPI_Irecv(&prowmajor, 1, MPI_C_BOOL, source, tag, pcomm, &request);

    vector<size_t> extents(prank,0);
    vector<size_t> strides(prank,0);
    // Allocate memory for extents and strides
    MPI_Irecv(extents.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);
    MPI_Irecv(strides.data(),prank, MPI_UNSIGNED_LONG, source, tag, pcomm, &request);

    mdspan<T,vector<size_t>> md(pdatalength,prowmajor,memmap,extents,strides);

    MPI_Irecv(md.pdatastruct.pdata,sizeof(T)* md.pdatastruct.pdata, MPI_BYTE, source, tag, pcomm, &request);

    return md;
}


template <typename T>
void MPI_recv_mdspan_pdata(mdspan<T,vector<size_t>> & mds, int source, int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Recv(mds.pdatastruct.pdata,sizeof(T)* mds.pdatastruct.pdatalength, MPI_BYTE, source, tag, pcomm, &status);
}

template <typename T,typename Container>
void MPI_send_mdspan_pdata(mdspan<T,Container> & m, int dest, int tag,MPI_Comm pcomm)
{
    if (m.pdatastruct.pdata != nullptr)
    {
        MPI_Send(m.pdatastruct.pdata,sizeof(T)* m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);
    }
}

template <typename T>
void MPI_Irecv_mdspan_pdata(mdspan<T,vector<size_t>> & mds, int source, int tag, MPI_Comm pcomm)
{
    MPI_Status status;
    MPI_Irecv(mds.pdatastruct.pdata,sizeof(T)* mds.pdatastruct.pdatalength, MPI_BYTE, source, tag, pcomm, &status);
}

template <typename T,typename Container>
void MPI_Isend_mdspan_pdata(mdspan<T,Container> & m, int dest, int tag,MPI_Comm pcomm)
{
    if (m.pdatastruct.pdata != nullptr)
    {
        MPI_Isend(m.pdatastruct.pdata,sizeof(T)* m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);
    }
}


template <typename T,typename Container>
void MPI_Isend_mdspan(mdspan<T,Container> & m, int dest, int tag, MPI_Comm pcomm)
{
    // Send metadata
    MPI_Isend(&m.pdatastruct.pdatalength, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Isend(&m.pdatastruct.prank, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Isend(&m.pdatastruct.prowmajor, 1, MPI_C_BOOL, dest, tag, pcomm);

    // Send extents and strides
    if (m.pdatastruct.pextents != nullptr)
    {
        MPI_Isend(m.pdatastruct.pextents, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    }
    if (m.pdatastruct.pstrides != nullptr)
    {
        MPI_Isend(m.pdatastruct.pstrides, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    }
    // Send data
    if (m.pdatastruct.pdata != nullptr)
    {
        MPI_Isend(m.pdatastruct.pdata, m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);
    }
}



template <typename T,typename Container>
void MPI_send_mdspan(mdspan<T,Container> & m, int dest, int tag, MPI_Comm pcomm)
{
    // Send metadata
    MPI_Send(&m.pdatastruct.pdatalength, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.pdatastruct.prank, 1, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    MPI_Send(&m.pdatastruct.prowmajor, 1, MPI_C_BOOL, dest, tag, pcomm);

    // Send extents and strides
    if (m.pdatastruct.pextents != nullptr)
    {
        MPI_Send(m.pdatastruct.pextents, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    }
    if (m.pdatastruct.pstrides != nullptr)
    {
        MPI_Send(m.pdatastruct.pstrides, m.pdatastruct.prank, MPI_UNSIGNED_LONG, dest, tag, pcomm);
    }
    // Send data
    if (m.pdatastruct.pdata != nullptr)
    {
        MPI_Send(m.pdatastruct.pdata, m.pdatastruct.pdatalength, MPI_BYTE, dest, tag, pcomm);
    }
}

template <typename T>
void MPI_listener(MPI_Comm pcomm)
{
    while (true)
    {
        MPI_Status status;
        int message_type;

        MPI_Recv(&message_type, 1, MPI_INT, MPI_ANY_SOURCE, 0, pcomm, &status);

        switch (message_type)
        {
        case COMMAND_STRASSEN:
        {

            mdspan<T,vector<size_t>> A=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 1, pcomm);

            mdspan<T,vector<size_t>> B=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 2, pcomm);

            size_t rowsC=A.pdatastruct.pextents0,
                   colsC=B.pdatastruct.pextents1;

            mdspan<T,std::vector<size_t>> C(A.prowmajor,true, {rowsC, colsC});

            matrix_multiplication_parameters algorithm;

            algorithm.mpi=true;
            algorithm.memmapped_files=true;
            algorithm.gpu_offload=true;
            algorithm.comm=pcomm;
            algorithm.status=status;
            strassen_multiply(A,B,C,algorithm,true);
            MPI_send_mdspan_pdata(C,status.MPI_SOURCE,3,pcomm);
            break;
        }
        case COMMAND_WINOGRAD:
        {

            mdspan<T,vector<size_t>> A=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 1, pcomm);
            mdspan<T,vector<size_t>> B=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 2, pcomm);
            size_t rowsC=A.pdatastruct.pextents0,colsC=B.pdatastruct.pextents1;
            mdspan<T,std::vector<size_t>> C(A.prowmajor,true, {rowsC, colsC});
            matrix_multiplication_parameters algorithm;
            algorithm.mpi=true;
            algorithm.memmapped_files=true;
            algorithm.gpu_offload=true;
            algorithm.comm=pcomm;
            algorithm.status=status;
            winograd_multiply(A,B,C,algorithm,true);
            MPI_send_mdspan_pdata(C,status.MPI_SOURCE,3,pcomm);
            break;
        }
        case COMMAND_SENDMATRIX:
        {
            mdspan<T,vector<size_t>> A=MPI_recv_mdspan<T>(true,status.MPI_SOURCE, 1, pcomm);
            if(A.pdatastruct.prank==2)
                printmatrix(A);
            break;
        }

        }
    }
}

template <typename T, typename Container>
size_t mdspan<T, Container> ::extent(const size_t dim) const
{
    return pdatastruct.pextents[dim];
}



template <typename T, typename Container>
size_t mdspan<T, Container>  ::rank() const
{
    return pdatastruct.prank;
}


template <typename T, typename Container>
size_t mdspan<T, Container> ::stride(const size_t dim)const
{
    return pstrides[dim];
}


template <typename T, typename Container>
Container & mdspan<T, Container> ::extents()const
{
    return pextents;
}
template <typename T, typename Container>
Container & mdspan<T, Container> ::strides()const
{
    return pstrides;
}


template <typename T, typename Container>
size_t mdspan<T, Container> ::datalength()const
{
    return pdatastruct.pdatalength;
}





template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::transpose()
{
    assert(pdatastruct.prank == 2 && "Transpose is only valid for 2D matrices");

    // Swap extents (rows <-> cols)
    Container transposed_extents = {pdatastruct.pextents[1], pdatastruct.pextents[0]};
    // Swap strides (row-major -> column-major)
    Container transposed_strides = {pdatastruct.pstrides[1], pdatastruct.pstrides[0]};

    // Create a new mdspan with swapped extents and strides
    return mdspan(pdatastruct.pdata,pdatastruct.pdatalength, pdatastruct.prowmajor,  transposed_extents,   transposed_strides);
}

#pragma acc routine worker
template <typename T>
inline void gpu_matrix_multiply_dot_w( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
{

    const size_t rows=A.pextents[0];
     const size_t cols=B.pextents[1];
     const size_t inner_dim=A.pextents[1];
#pragma acc loop worker collapse(2)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
#pragma acc loop vector reduction(+: sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}

#pragma acc routine vector
template <typename T>
inline void gpu_matrix_multiply_dot_v( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
{
     const size_t rows=A.pextents[0];
     const size_t cols=B.pextents[1];
     const size_t inner_dim=A.pextents[1];
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
#pragma acc loop vector reduction(+: sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}


#pragma acc routine seq
template <typename T>
inline void gpu_matrix_multiply_dot_s( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
{
     const size_t rows=A.pextents[0];
     const size_t cols=B.pextents[1];
     const size_t inner_dim=A.pextents[1];
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
#pragma acc loop reduction(+: sum)
            for (size_t k = 0; k <inner_dim ; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}



#pragma acc routine worker
template <typename T>
inline void gpu_cholesky_decomposition( datastruct<T>& A, datastruct<T>& L, T*buffer=nullptr, size_t step_size=0)
{

    const size_t n = A.pextents[0];
    size_t z = 0; // Zero-based indexing, starts at the first column



    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    const size_t tempsize = (n - step_size) * (n - step_size);

    size_t pext3[2];
    size_t pstrides3[2];

    const size_t nn=n*n;
    // Allocate memory for S on the device
    T* sdata;
    T* adata;

    if (buffer==(T*) nullptr)
       // sdata=(T*) acc_malloc(sizeof(T)*tempsize);
       {
        sdata=new T[tempsize];
        adata=new T[nn];
       }
    else
    {
        sdata=buffer;
        adata=buffer+tempsize;
    }

#pragma acc loop vector
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=A.pdata[i];
        L.pdata[i]=0;
    }

    datastruct<T> tempA(adata, 0,A.prowmajor,n, n,pext3, pstrides3,true,true);

#pragma acc loop seq
    for (size_t c = 0; c < n; ++c)
    {
        if (c == z + step_size)
        {
            size_t u=n-c;
            size_t v=c-z;
            size_t rtext[2];
            size_t rtstrides[2];
            size_t pstrides2[2];
            size_t pext2[2];
            size_t pext[2];

            datastruct<T> R = L.subspanmatrix(c, z, u, v,pext,nullptr);
            datastruct<T> RT=R.transpose(rtext,rtstrides);
            datastruct<T> S(sdata, 0, A.prowmajor,u, u, pext2, pstrides2,true,true);

            gpu_matrix_multiply_dot_w(R,RT,S);

#pragma acc loop worker collapse(2)
            for (size_t i = c; i < n; ++i)
            {
                for (size_t j = c; j < n; ++j)
                {
                    tempA(i,j) -=S(i-c,j-c);
                }
            }

            z = c;
        }

        // Compute L[c, c]
        T temp = 0;
#pragma acc loop worker reduction(+:temp)
        for (size_t k = z; k < c; ++k)
        {
            T tmp3=L(c,k);
            temp +=  tmp3*tmp3;
        }
        temp=tempA(c,c)-temp;
        T temp4=sqrt(temp);
        L(c,c) = temp4;

        // Compute L[i, c]
#pragma acc loop worker
        for (size_t i = c + 1; i < n; ++i)
        {
            T temp2 =0;
#pragma acc loop vector reduction(+:temp2)
            for (size_t k = z; k < c; ++k)
            {
                temp2 += L(i,k)*L(c,k);
            }
            temp2= tempA(i,c)-temp2;
            L(i,c) = temp2 / temp4;
        }
    }

    // Deallocate memory for S on the device
    if(buffer==nullptr)
    {
       // acc_free(sdata);
        delete[] sdata;
        delete[] adata;
    }


}

#pragma acc routine worker
template <typename T>
inline void gpu_lu_decomposition( datastruct<T>& dA, datastruct<T>& dL, datastruct<T>& dU, T* buffer=nullptr, size_t step_size=0)
{

    const size_t n = dA.pextents[0];
    size_t z = 0; // Zero-based indexing, starts at the first column


    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    const size_t tempsize = (n - step_size) * (n - step_size);
    size_t pext3[2];
    size_t pstrides3[2];
    const size_t nn=n*n;


    T* sdata;
    T* adata;

    if (buffer==nullptr)
     {
     sdata=new T[tempsize];
     adata=new T[nn];
     }
    else
    {
        sdata=buffer;
        adata=buffer+tempsize;
    }

#pragma acc loop vector
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=dA.pdata[i];
        dL.pdata[i]=0;
        dU.pdata[i]=0;
    }
    datastruct<T> tempA(adata,  0, dA.prowmajor,n, n,pext3, pstrides3,true,true);
#pragma acc loop seq
    for (size_t c = 0; c < n; ++c)
    {
        if (c == z + step_size)
        {
            size_t u=n-c;
            size_t v=c-z;
            size_t pext5[2];
            size_t pext6[2];
            size_t pstrides2[2];
            size_t pext2[2];
            datastruct<T> RL = dL.subspanmatrix(c, z, u, v,pext5);
            datastruct<T> RU = dU.subspanmatrix(z, c, v, u,pext6);

            datastruct<T> S(sdata,  0, dA.prowmajor,u, u,pext2, pstrides2,true,true);
            gpu_matrix_multiply_dot_w(RL,RU,S);


#pragma acc loop worker collapse(2)
            for (size_t i = c; i < n; ++i)
            {
                for (size_t j = c; j < n; ++j)
                {
                    tempA( i,j) -= S(i - c, j - c);
                }
            }
            z = c;
        }

#pragma acc loop worker
        for (size_t i = c; i < n; ++i)
        {
            T temp=0;
#pragma acc loop vector reduction(+:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp+= dU( k,i) * dL( c,k);
            }
            dU(c,i)=tempA(c,i)-temp;
        }

#pragma acc loop worker
        for (size_t i = c; i < n; ++i)
        {
            T temp= 0;
#pragma acc loop vector reduction(+:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp += dU(k,c) * dL( i,k);
            }
            temp=tempA(i,c)-temp;
            dL(i,c)=temp/dU(c,c);
        }
    }

    if(buffer==nullptr)
    {
        delete[] sdata;
        delete[] adata;
    }



}

#pragma acc routine worker
template <typename T >
void gpu_qr_decomposition(datastruct<T>&A, datastruct<T> Q, datastruct<T> &R, T* buffer=nullptr, size_t step_size=0)
{
    const size_t n = A.pextents[0]; // Number of rows (assuming 2D matrix)
    const size_t m = A.pextents[1]; // Number of columns

    if(step_size==0)
        step_size=(size_t)pow(A.pextents[0],0.8385);

    const size_t nm=n*m;
    T* tempC;
    T* tempS;
    T* tempM;

    if(buffer==nullptr)
      {
      tempC=new T[m*m];
      tempS=new T[nm];
      tempM=new T[nm];
      }
    else
    {
        tempC=buffer;
        tempS=buffer+m*m;
        tempM=tempS+nm;
    }

    size_t mext[2];
    mext[0]= A.pextents[1];
    mext[1]=A.pextents[0];
    size_t mstrides[2];
    mstrides[0]= A.pstrides[0];
    mstrides[1]=A.pstrides[1];



#pragma acc loop vector
    for (size_t i=0; i<nm; i++)
    {
        tempM[i]=A.pdata[i];
    }

#pragma acc loop vector
    for (size_t i=0; i<Q.pdatalength; i++)
    {
        Q.pdata[i]=0;
    }

#pragma acc loop vector
    for (size_t i=0; i<R.pdatalength; i++)
    {
        R.pdata[i]=0;
    }


    datastruct<T> M(tempM,A.pdatalength,A.prowmajor,A.prank,mext,mstrides,false,false); //Copy of A
    size_t z = 0;

#pragma acc loop seq
    for (size_t c = 0; c < m; ++c)
    {
        if (c == z +step_size)
        {
            // Extract submatrices
            size_t cz=c-z;
            size_t mc=m-c;

            size_t exts[2];
            size_t strs[2];
            size_t extbq[2];
            size_t extbm[2];
            size_t extc[2];
            size_t strc[2];
            size_t extbqt[2];
            size_t strbqt[2];

            datastruct<T> BQ = Q.subspanmatrix(0, z, n, cz,extbq);
            datastruct<T> BM = M.subspanmatrix(0, c, n, mc,extbm);

            // Compute C = BQ^T * BM
            datastruct<T> C(tempC,0, BM.prowmajor,cz, mc,extc,strc,true,true);
            datastruct<T> BQT=BQ.transpose(extbqt,strbqt);

            gpu_matrix_multiply_dot_w(BQT,BM,C);

            // Compute S = BQ * C
            datastruct<T>S(tempS, 0,BQ.prowmajor,n, mc,exts,strs,true,true);

            gpu_matrix_multiply_dot_w(BQ,C,S);


            // Update M: M[:, c:] -= S
#pragma acc loop worker collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = c; j <n; ++j)
                {
                    M(i,  j) -= S(i, j-c);
                }
            }
            z = c;
        }
        // Extract column c of M
        size_t pextv[1];
        datastruct<T> v = M.column(c,pextv);

#pragma acc loop worker
        for (size_t j = z; j < c; ++j)
        {
            size_t pext2v[1];
            datastruct<T> u = Q.column(j,pext2v);

            T dot_pr =gpu_dot_product_s(u,v);

#pragma acc loop vector
            for (size_t i = 0; i < n; ++i)
            {
                v(i) -= dot_pr * u(i);
            }
        }

        // Normalize v
        T norm = sqrt(gpu_dot_product_s(v,v));

#pragma acc loop vector
        for (size_t i = 0; i < n; ++i)
        {
            v(i) /= norm;
        }

        // Set column c of Q
#pragma acc loop vector
        for (size_t i = 0; i < n; ++i)
        {
            T tmp=v(i);
            Q(i, c) = tmp;
        }
    }

    // Compute R = Q^T * A
    size_t qtext[2];
    size_t qtstrides[2];


    datastruct<T> QT=Q.transpose(qtext,qtstrides);
    gpu_matrix_multiply_dot_w(QT,A,R);
    if(buffer==nullptr)
      {
    delete[] tempC;
    delete[] tempS;
    delete[] tempM;
      }


}

#pragma acc routine worker
template <typename T>
inline bool gpu_matrix_add_w(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
#pragma acc loop worker collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }

    return true;
}



#pragma acc routine worker
template <typename T>
inline bool gpu_matrix_subtract_w(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
#pragma acc loop worker collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }
    return true;
}

#pragma acc routine worker
template <typename T>
inline bool gpu_matrix_multiply_vector_w( const datastruct<T>&M,const  datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
#pragma acc loop worker collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i, j) * V(j);  // This works because i, k, j are row/col indices
        }
    }
    return true;
}

#pragma acc routine worker
template <typename T>
inline bool gpu_matrix_multiply_vector_w( const datastruct<T>M, const T*V, datastruct<T> & C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=M.pextents[1];
#pragma acc loop worker collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V[i];  // This works because i, k, j are row/col indices
        }
    }

    return true;
}



#pragma acc routine worker
template <typename T>
inline bool gpu_matrix_multiply_scalar_w(  const datastruct<T>& M, const T& V, datastruct<T>& C)
{
    // Perform matrix multiplication: C = A * B

    const size_t n=C.pextents[0];
    const size_t m= C.pextents[1];

#pragma acc loop worker collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i,j)*V;
        }

    }

    return true;
}


#pragma acc routine worker
template <typename T>
inline void gpu_vector_scalar_multiply_w( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
{
    const size_t n=vec.pextents[0];
#pragma acc loop worker
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma acc routine seq
template <typename T>
inline void gpu_cross_product( const datastruct<T>& vec1, const  datastruct<T>& vec2, datastruct<T>& res)
{
    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}

#pragma acc routine worker
template <typename T>
inline void gpu_vector_add_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
#pragma acc loop worker
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}

#pragma acc routine worker
template <typename T>
inline void gpu_vector_subtract_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
#pragma acc loop worker
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}



#pragma acc routine vector
template <typename T>
inline bool gpu_matrix_add_v( const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
#pragma acc loop vector collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }

    return true;
}



#pragma acc routine vector
template <typename T>
inline bool gpu_matrix_subtract_v( const datastruct<T>& A, const datastruct<T>& B, datastruct<T>& C)
{
    const size_t n=A.pextents[0];
    const size_t m=A.pextents[1];
#pragma acc loop vector collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }
    return true;
}

#pragma acc routine vector
template <typename T>
inline bool gpu_matrix_multiply_vector_v( const datastruct<T>&M, const datastruct<T> V, datastruct<T> C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=V.pextents[0];
#pragma acc loop vector collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i, j) * V(j);  // This works because i, k, j are row/col indices
        }
    }
    return true;
}

#pragma acc routine vector
template <typename T>
inline bool gpu_matrix_multiply_vector_v(const datastruct<T>M, const T*V, datastruct<T> & C)
{

    // Perform matrix multiplication: C = A * B
    const size_t n= M.pextents[0];
    const size_t m=M.pextents[1];
#pragma acc loop vector collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V[i];  // This works because i, k, j are row/col indices
        }
    }

    return true;
}



#pragma acc routine vector
template <typename T>
inline bool gpu_matrix_multiply_scalar_v( const datastruct<T>& M,const  T& V, datastruct<T>& C)
{
    // Perform matrix multiplication: C = A * B

    const size_t n=C.pextents[0];
    const size_t m= C.pextents[1];

#pragma acc loop vector collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j)= M(i,j)*V;
        }

    }

    return true;
}

#pragma acc routine vector
template <typename T>
inline T gpu_dot_product_v( const datastruct<T> vec1, const datastruct<T> vec2)
{
    const size_t n=vec1.pextents[0];
    T result = 0;
    const size_t v1s=vec1.pstrides[0];
    const size_t v2s=vec2.pstrides[0];
//#pragma acc loop vector reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1.pdata[i *v1s]  * vec2.pdata[i *v2s];
    }
    return result;
}

#pragma acc routine worker
template <typename T>
inline T gpu_dot_product_w(const  datastruct<T> vec1, const datastruct<T> vec2)
{
    const size_t n=vec1.pextents[0];
    T result=0;
#pragma acc loop worker reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}


#pragma acc routine seq
template <typename T>
inline T gpu_dot_product_s(  const datastruct<T> vec1,const  datastruct<T> vec2)
{
    const size_t n=vec1.pextents[0];
    T result = 0;
    const size_t v1s=vec1.pstrides[0];
    const size_t v2s=vec2.pstrides[0];
//#pragma acc loop  reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1.pdata[i *v1s]  * vec2.pdata[i *v2s];
    }
    return result;
}

#pragma acc routine vector
template <typename T>
inline void gpu_vector_scalar_multiply_v( datastruct<T>& vec, T scalar,datastruct<T>& res)
{
    const size_t n=vec.pextents[0];
#pragma acc loop vector
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}



#pragma acc routine vector
template <typename T>
inline void gpu_vector_add_v(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
#pragma acc loop vector
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}

#pragma acc routine vector
template <typename T>
inline void gpu_vector_subtract_v(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
#pragma acc loop vector
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}









template <typename T, typename CA,typename CB,typename CC>
bool strassen_multiply(const  mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C, const matrix_multiplication_parameters & algorithm)
{
    // Dimensions of input matrices
    const  size_t n = A.extent(0); // Rows in A
    const  size_t m = A.extent(1); // Columns in A and rows in B
    const size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (n*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,algorithm.gpu_offload);
        return true;
    }
    // Compute sizes for splitting
    const size_t half_n = n / 2;
    const size_t half_m = m / 2;
    const size_t half_p = p / 2;

    // Submatrices of A
    auto A11 = A.subspan({0, 0}, {half_n, half_m});
    auto A12 = A.subspan({0, half_m}, {half_n, half_m});
    auto A21 = A.subspan({half_n, 0}, {half_n, half_m});
    auto A22 = A.subspan({half_n, half_m}, {half_n, half_m});


    // Submatrices of B
    auto B11 = B.subspan({0, 0}, {half_m, half_p});
    auto B12 = B.subspan({0, half_p}, {half_m, half_p});
    auto B21 = B.subspan({half_m, 0}, {half_m, half_p});
    auto B22 = B.subspan({half_m, half_p}, {half_m, half_p});

    // Temporary storage for intermediate results
    size_t s=half_n*half_p,
           s2=half_n*half_m,
           s3=half_m*half_p;
    T* M1_storage,*M2_storage,*M3_storage,*M4_storage,*M5_storage,*M6_storage,*M7_storage,
    * A_result1_storage,*B_result1_storage,
    * A_result2_storage,*B_result2_storage,
    * A_result3_storage,*B_result3_storage,
    * A_result4_storage,*B_result4_storage,
    * A_result5_storage,*B_result5_storage;


    if (algorithm.memmapped_files)
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
        B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
        {
            M1_storage=create_temp_mmap<T>(s);
            M2_storage=create_temp_mmap<T>(s);
            M3_storage=create_temp_mmap<T>(s);
            M4_storage=create_temp_mmap<T>(s);
            M5_storage=create_temp_mmap<T>(s);
            M6_storage=create_temp_mmap<T>(s);
            M7_storage=create_temp_mmap<T>(s);
            A_result1_storage=create_temp_mmap<T>(s2);
            A_result2_storage=create_temp_mmap<T>(s2);
            A_result2_storage=create_temp_mmap<T>(s2);
            A_result3_storage=create_temp_mmap<T>(s2);
            A_result4_storage=create_temp_mmap<T>(s2);
            A_result5_storage=create_temp_mmap<T>(s2);
            B_result1_storage=create_temp_mmap<T>(s3);
            B_result2_storage=create_temp_mmap<T>(s3);
            B_result3_storage=create_temp_mmap<T>(s3);
            B_result4_storage=create_temp_mmap<T>(s3);
            B_result5_storage=create_temp_mmap<T>(s3);
        }
    }

    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
        B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
        {
            M1_storage =new T[s],
            M2_storage =new T[s],
            M3_storage =new T[s],
            M4_storage =new T[s],
            M5_storage =new T[s],
            M6_storage =new T[s],
            M7_storage =new T[s],
            A_result1_storage =new T[s2],
            A_result2_storage =new T[s2],
            A_result3_storage =new T[s2],
            A_result4_storage =new T[s2],
            A_result5_storage =new T[s2],
            B_result1_storage =new T[s3];
            B_result2_storage =new T[s3];
            B_result3_storage =new T[s3];
            B_result4_storage =new T[s3];
            B_result5_storage =new T[s3];
        }
    }

    mdspan<T, CC>   M1(M1_storage, s, {half_n, half_p}, {half_p, 1}),
           M2(M2_storage, s, {half_n, half_p}, {half_p, 1}),
           M3(M3_storage, s, {half_n, half_p}, {half_p, 1}),
           M4(M4_storage, s, {half_n, half_p}, {half_p, 1}),
           M5(M5_storage, s, {half_n, half_p}, {half_p, 1}),
           M6(M6_storage, s, {half_n, half_p}, {half_p, 1}),
           M7(M7_storage,s, {half_n, half_p}, {half_p, 1});

    mdspan<T, CA> A_result1(A_result1_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result2(A_result2_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result3(A_result3_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result4(A_result4_storage, s2, {half_n, half_m}, {half_m, 1});
    mdspan<T, CA> A_result5(A_result5_storage, s2, {half_n, half_m}, {half_m, 1});

    mdspan<T, CB> B_result1(B_result1_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result2(B_result2_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result3(B_result3_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result4(B_result4_storage, s3, {half_m, half_p}, {half_p, 1});
    mdspan<T, CB> B_result5(B_result5_storage, s3, {half_m, half_p}, {half_p, 1});

    #pragma omp parallel shared (A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
    A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
    {
        matrix_add(A11, A22, A_result1);
        matrix_add(B11, B22, B_result1);
        matrix_add(A21, A22, A_result2);
        matrix_subtract(B12, B22, B_result2);
        matrix_subtract(B21, B11, B_result3);
        matrix_add(A11, A12, A_result3);
        matrix_subtract(A21, A11, A_result4);
        matrix_add(B11, B12, B_result4);
        matrix_subtract(A12, A22, A_result5);
        matrix_add(B21, B22, B_result5);
    }

    if (algorithm.mpi==true && n*p>=algorithm.size_for_mpi)
{

    int childdest=algorithm.status.MPI_SOURCE*7;
    int commsize;
    MPI_Comm_size(algorithm.comm, &commsize);
        if (childdest+7<commsize)
        {

                    MPI_Send(&m, 1, MPI_INT, childdest+1, 0, algorithm.comm);
                    MPI_send_mdspan(A_result1,childdest+1,1,algorithm.comm);
                    MPI_send_mdspan(B_result1,childdest+1,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+2, 0, algorithm.comm);
                    MPI_send_mdspan(A_result2,childdest+2,1,algorithm.comm);
                    MPI_send_mdspan(B11,childdest+2,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+3, 0, algorithm.comm);
                    MPI_send_mdspan(A11,childdest+3,1,algorithm.comm);
                    MPI_send_mdspan(B_result2,childdest+3,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+4, 0, algorithm.comm);
                    MPI_send_mdspan(A22,childdest+4,1,algorithm.comm);
                    MPI_send_mdspan(B_result3,childdest+4,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+5, 0, algorithm.comm);
                    MPI_send_mdspan(A_result3,childdest+5,1,algorithm.comm);
                    MPI_send_mdspan(B22,childdest+5,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+6, 0, algorithm.comm);
                    MPI_send_mdspan(A_result4,childdest+6,1,algorithm.comm);
                    MPI_send_mdspan(B_result4,childdest+6,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+7, 0, algorithm.comm);
                    MPI_send_mdspan(A_result5,childdest+7,1,algorithm.comm);
                    MPI_send_mdspan(B_result5,childdest+7,2,algorithm.comm);

                    MPI_recv_mdspan_pdata(M1,childdest+1,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M2,childdest+2,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M3,childdest+3,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M4,childdest+4,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M5,childdest+5,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M6,childdest+6,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M7,childdest+7,3,algorithm.comm);
        }
        else
        {
            if(algorithm.omp)
            {
                #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
                A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
                {
                    strassen_multiply(A_result1, B_result1, M1, algorithm);
                    strassen_multiply(A_result2, B11, M2, algorithm);
                    strassen_multiply(A11, B_result2, M3, algorithm);
                    strassen_multiply(A22, B_result3, M4, algorithm);
                    strassen_multiply(A_result3, B22, M5,algorithm);
                    strassen_multiply(A_result4, B_result4, M6,algorithm);
                    strassen_multiply(A_result5, B_result5, M7, algorithm);
                }
            }
            else
            {
                strassen_multiply(A_result1, B_result1, M1, algorithm);
                strassen_multiply(A_result2, B11, M2, algorithm);
                strassen_multiply(A11, B_result2, M3, algorithm);
                strassen_multiply(A22, B_result3, M4, algorithm);
                strassen_multiply(A_result3, B22, M5,algorithm);
                strassen_multiply(A_result4, B_result4, M6,algorithm);
                strassen_multiply(A_result5, B_result5, M7, algorithm);
            }
        }

    }
    else
    {
        if(algorithm.omp)
        {
            #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
            A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
            {
                strassen_multiply(A_result1, B_result1, M1, algorithm);
                strassen_multiply(A_result2, B11, M2, algorithm);
                strassen_multiply(A11, B_result2, M3, algorithm);
                strassen_multiply(A22, B_result3, M4, algorithm);
                strassen_multiply(A_result3, B22, M5,algorithm);
                strassen_multiply(A_result4, B_result4, M6,algorithm);
                strassen_multiply(A_result5, B_result5, M7, algorithm);
            }
        }
        else
        {
            strassen_multiply(A_result1, B_result1, M1, algorithm);
            strassen_multiply(A_result2, B11, M2, algorithm);
            strassen_multiply(A11, B_result2, M3, algorithm);
            strassen_multiply(A22, B_result3, M4, algorithm);
            strassen_multiply(A_result3, B22, M5,algorithm);
            strassen_multiply(A_result4, B_result4, M6,algorithm);
            strassen_multiply(A_result5, B_result5, M7, algorithm);
        }
    }

    // Submatrices of C
    auto C11 = C.subspan({0, 0}, {half_n, half_p});
    auto C12 = C.subspan({0, half_p}, {half_n, half_p});
    auto C21 = C.subspan({half_n, 0}, {half_n, half_p});
    auto C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    #pragma omp parallel for collapse(2) shared(M2,M3,M5,M6,M7,C11,C12,C21,C22)
    for (size_t i = 0; i < half_n; ++i)
{
    for (size_t j = 0; j < half_p; ++j)
        {
            C11(i, j) = M1(i, j) + M4(i, j) - M5(i, j) + M7(i, j);
            C12(i, j) = M3(i, j) + M5(i, j);
            C21(i, j) = M2(i, j) + M4(i, j);
            C22(i, j) = M1(i, j) - M2(i, j) + M3(i, j) + M6(i, j);
        }
    }

    if (algorithm.memmapped_files)
{
    #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
    A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
    B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
    {
        delete_temp_mmap(M1_storage, s);
            delete_temp_mmap(M2_storage, s);
            delete_temp_mmap(M3_storage, s);
            delete_temp_mmap(M4_storage, s);
            delete_temp_mmap(M5_storage, s);
            delete_temp_mmap(M6_storage, s);
            delete_temp_mmap(M7_storage, s);

            delete_temp_mmap(A_result1_storage, s2);
            delete_temp_mmap(A_result2_storage, s2);
            delete_temp_mmap(A_result3_storage, s2);
            delete_temp_mmap(A_result4_storage, s2);
            delete_temp_mmap(A_result5_storage, s2);

            delete_temp_mmap(B_result1_storage, s3);
            delete_temp_mmap(B_result2_storage, s3);
            delete_temp_mmap(B_result3_storage, s3);
            delete_temp_mmap(B_result4_storage, s3);
            delete_temp_mmap(B_result5_storage, s3);
        }
    }
    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        A_result1_storage, A_result2_storage, A_result3_storage, A_result4_storage, A_result5_storage,\
        B_result1_storage, B_result2_storage, B_result3_storage, B_result4_storage, B_result5_storage)
        {
            delete []M1_storage;
            delete []M2_storage;
            delete []M3_storage;
            delete []M4_storage;
            delete []M5_storage;
            delete []M6_storage;
            delete []M7_storage;
            delete[]A_result1_storage;
            delete[]A_result2_storage;
            delete[]A_result3_storage;
            delete[]A_result4_storage;
            delete[]A_result5_storage;
            delete[]B_result1_storage;
            delete[]B_result2_storage;
            delete[]B_result3_storage;
            delete[]B_result4_storage;
            delete[]B_result5_storage;
        }
    }

    return true;
}

template <typename T, typename CA,typename CB,typename CC>
bool winograd_multiply(const  mdspan<T, CA>& A, const mdspan<T, CB>& B, mdspan<T, CC>& C,const matrix_multiplication_parameters& algorithm)
{
    // Dimensions of input matrices
    size_t n = A.extent(0); // Rows in A
    size_t m = A.extent(1); // Columns in A and rows in B
    size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (n*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,algorithm.gpu_offload);
        return true;
    }
    // Compute sizes for splitting
    size_t half_n = n / 2;
    size_t half_m = m / 2;
    size_t half_p = p / 2;

    // Submatrices of A
    auto A11 = A.subspan({0, 0}, {half_n, half_m});
    auto A12 = A.subspan({0, half_m}, {half_n, half_m});
    auto A21 = A.subspan({half_n, 0}, {half_n, half_m});
    auto A22 = A.subspan({half_n, half_m}, {half_n, half_m});

    // Submatrices of B
    auto B11 = B.subspan({0, 0}, {half_m, half_p});
    auto B12 = B.subspan({0, half_p}, {half_m, half_p});
    auto B21 = B.subspan({half_m, 0}, {half_m, half_p});
    auto B22 = B.subspan({half_m, half_p}, {half_m, half_p});

    // Temporary storage for intermediate results
    size_t s=half_n*half_p;
    size_t s2=half_n*half_m;
    size_t s3=half_m*half_p;

    T* M1_storage,*M2_storage,*M3_storage,*M4_storage,*M5_storage,*M6_storage,*M7_storage,
    *S1_result_storage,*S2_result_storage,*S3_result_storage,*S4_result_storage,\
    *S5_result_storage,*S6_result_storage,*S7_result_storage,*S8_result_storage,
    *T1_result_storage,*T2_result_storage;
    if (algorithm.memmapped_files)
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
        {

            M1_storage=create_temp_mmap<T>(s);
            M2_storage=create_temp_mmap<T>(s);
            M3_storage=create_temp_mmap<T>(s);
            M4_storage=create_temp_mmap<T>(s);
            M5_storage=create_temp_mmap<T>(s);
            M6_storage=create_temp_mmap<T>(s);
            M7_storage=create_temp_mmap<T>(s);
            S1_result_storage=create_temp_mmap<T>(s2);
            S2_result_storage=create_temp_mmap<T>(s2);
            S3_result_storage=create_temp_mmap<T>(s2);
            S4_result_storage=create_temp_mmap<T>(s2);
            S5_result_storage=create_temp_mmap<T>(s3);
            S6_result_storage=create_temp_mmap<T>(s3);
            S7_result_storage=create_temp_mmap<T>(s3);
            S8_result_storage=create_temp_mmap<T>(s3);
            T1_result_storage=create_temp_mmap<T>(s);
            T2_result_storage=create_temp_mmap<T>(s);
        }

    }
    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
        {
            M1_storage=new T[s];
            M2_storage=new T[s];
            M3_storage=new T[s];
            M4_storage=new T[s];
            M5_storage=new T[s];
            M6_storage=new T[s];
            M7_storage=new T[s];
            S1_result_storage=new T[s2];
            S2_result_storage=new T[s2];
            S3_result_storage=new T[s2];
            S4_result_storage=new T[s2];
            S5_result_storage=new T[s3];
            S6_result_storage=new T[s3];
            S7_result_storage=new T[s3];
            S8_result_storage=new T[s3];
            T1_result_storage=new T[s];
            T2_result_storage=new T[s];
        }
    }
    mdspan<T, CC> M1(M1_storage, s, {half_n, half_p}, {half_p, 1}),
           M2(M2_storage, s, {half_n, half_p}, {half_p, 1}),
           M3(M3_storage, s, {half_n, half_p}, {half_p, 1}),
           M4(M4_storage, s, {half_n, half_p}, {half_p, 1}),
           M5(M5_storage, s, {half_n, half_p}, {half_p, 1}),
           M6(M6_storage, s, {half_n, half_p}, {half_p, 1}),
           M7(M7_storage, s, {half_n, half_p}, {half_p, 1});

    mdspan<T, CA> S1(S1_result_storage, s2, {half_n, half_m}, {half_m, 1}),
           S2(S2_result_storage, s2, {half_n, half_m}, {half_m, 1}),
           S3(S3_result_storage,s2, {half_n, half_m}, {half_m, 1}),
           S4(S4_result_storage, s2, {half_n, half_m}, {half_m, 1});

    mdspan<T, CB> S5(S5_result_storage, s3, {half_m, half_p}, {half_p, 1}),
           S6(S6_result_storage,s3, {half_m, half_p}, {half_p, 1}),
           S7(S7_result_storage, s3, {half_m, half_p}, {half_p, 1}),
           S8(S8_result_storage, s3, {half_m, half_p}, {half_p, 1});

    #pragma omp parallel shared(A11,A21,A12,A22,B11,B12,B22,B21,S1,S2,S3,S4,S5,S6,S7,S8)
    {
        #pragma omp single
        {
            matrix_add(A21, A22, S1);
            matrix_subtract(S1, A11, S2);
            matrix_subtract(A12, S2, S4);
        }
        #pragma omp single
        {
            matrix_subtract(A11, A21, S3);
            matrix_subtract(B22, B12, S7);
        }
        #pragma omp single
        {
            matrix_subtract(B12, B11, S5);
            matrix_subtract(B22, S5, S6);
            matrix_subtract(S6, B21, S8);
        }
    }

    if (algorithm.mpi==true && n*p>=algorithm.size_for_mpi)
    {
        int source=algorithm.status.MPI_SOURCE;
        int childdest=source*7;
        int commsize;
        MPI_Comm_size(algorithm.comm, &commsize);

        if (childdest+7<commsize )
        {

                    int m=COMMAND_WINOGRAD;
                    MPI_Send(&m, 1, MPI_INT, childdest+1, 0, algorithm.comm);
                    MPI_send_mdspan(S2,childdest+1,1,algorithm.comm);
                    MPI_send_mdspan(S6,childdest+1,2,algorithm.comm);


                    MPI_Send(&m, 1, MPI_INT, childdest+2, 0, algorithm.comm);
                    MPI_send_mdspan(A11,childdest+2,1,algorithm.comm);
                    MPI_send_mdspan(B11,childdest+2,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+3, 0, algorithm.comm);
                    MPI_send_mdspan(A12,childdest+3,1,algorithm.comm);
                    MPI_send_mdspan(B21,childdest+3,2,algorithm.comm);
                    MPI_recv_mdspan_pdata(M3,childdest+3,3,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+4, 0, algorithm.comm);
                    MPI_send_mdspan(S3,childdest+4,1,algorithm.comm);
                    MPI_send_mdspan(S7,childdest+4,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+5, 0, algorithm.comm);
                    MPI_send_mdspan(S1,childdest+5,1,algorithm.comm);
                    MPI_send_mdspan(S5,childdest+5,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+6, 0, algorithm.comm);
                    MPI_send_mdspan(S4,childdest+6,1,algorithm.comm);
                    MPI_send_mdspan(B22,childdest+6,2,algorithm.comm);

                    MPI_Send(&m, 1, MPI_INT, childdest+7, 0, algorithm.comm);
                    MPI_send_mdspan(A22,childdest+7,1,algorithm.comm);
                    MPI_send_mdspan(S8,childdest+7,2,algorithm.comm);

                    MPI_recv_mdspan_pdata(M1,childdest+1,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M2,childdest+2,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M4,childdest+4,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M5,childdest+5,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M6,childdest+6,3,algorithm.comm);
                    MPI_recv_mdspan_pdata(M7,childdest+7,3,algorithm.comm);

        }
        else
        {
            if(algorithm.omp)
            {
                #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,algorithm)
                {
                    winograd_multiply(S2,S6,M1,algorithm);
                    winograd_multiply(A11,B11,M2,algorithm);
                    winograd_multiply(A12,B21,M3,algorithm);
                    winograd_multiply(S3,S7,M4,algorithm);
                    winograd_multiply(S1,S5,M5,algorithm);
                    winograd_multiply(S4,B22,M6,algorithm);
                    winograd_multiply(A22,S8,M7,algorithm);
                }
            }
            else
            {
                winograd_multiply(S2,S6,M1,algorithm);
                winograd_multiply(A11,B11,M2,algorithm);
                winograd_multiply(A12,B21,M3,algorithm);
                winograd_multiply(S3,S7,M4,algorithm);
                winograd_multiply(S1,S5,M5,algorithm);
                winograd_multiply(S4,B22,M6,algorithm);
                winograd_multiply(A22,S8,M7,algorithm);
            }

        }
    }
    else
    {
        if(algorithm.omp)
        {
            #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,algorithm)
            {
                winograd_multiply(S2,S6,M1,algorithm);
                winograd_multiply(A11,B11,M2,algorithm);
                winograd_multiply(A12,B21,M3,algorithm);
                winograd_multiply(S3,S7,M4,algorithm);
                winograd_multiply(S1,S5,M5,algorithm);
                winograd_multiply(S4,B22,M6,algorithm);
                winograd_multiply(A22,S8,M7,algorithm);
            }
        }
        else
        {
            winograd_multiply(S2,S6,M1,algorithm);
            winograd_multiply(A11,B11,M2,algorithm);
            winograd_multiply(A12,B21,M3,algorithm);
            winograd_multiply(S3,S7,M4,algorithm);
            winograd_multiply(S1,S5,M5,algorithm);
            winograd_multiply(S4,B22,M6,algorithm);
            winograd_multiply(A22,S8,M7,algorithm);
        }
    }


    mdspan<T, CB> T1(T1_result_storage, s, {half_n, half_p}, {half_p, 1});
    mdspan<T, CB> T2(T2_result_storage, s, {half_n, half_p}, {half_p, 1});

    matrix_add(M1, M2, T1);
    matrix_add(T1, M4, T2);

    auto C11 = C.subspan({0, 0}, {half_n, half_p});
    auto C12 = C.subspan({0, half_p}, {half_n, half_p});
    auto C21 = C.subspan({half_n, 0}, {half_n, half_p});
    auto C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    #pragma omp parallel for collapse(2) shared(M2,M3,M5,M6,M7,T1,T2)
    for (size_t i = 0; i < half_n; ++i)
    {
        for (size_t j = 0; j < half_p; ++j)
        {
            C11(i, j) = M2(i, j) + M3(i,j);
            C12(i, j) = T1(i, j) + M5(i,j)+M6(i,j);
            C21(i, j) = T2(i, j) - M7(i, j);
            C22(i, j) = T2(i, j) + M5(i, j);
        }
    }


    if (algorithm.memmapped_files)
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
        {
            delete_temp_mmap(M1_storage, s);
            delete_temp_mmap(M2_storage, s);
            delete_temp_mmap(M3_storage, s);
            delete_temp_mmap(M4_storage, s);
            delete_temp_mmap(M5_storage, s);
            delete_temp_mmap(M6_storage, s);
            delete_temp_mmap(M7_storage, s);
            delete_temp_mmap(S1_result_storage, s2);
            delete_temp_mmap(S2_result_storage, s2);
            delete_temp_mmap(S3_result_storage, s2);
            delete_temp_mmap(S4_result_storage, s2);
            delete_temp_mmap(S5_result_storage, s3);
            delete_temp_mmap(S6_result_storage, s3);
            delete_temp_mmap(S7_result_storage, s3);
            delete_temp_mmap(S8_result_storage, s3);
            delete_temp_mmap(T1_result_storage, s);
            delete_temp_mmap(T2_result_storage, s);
        }
    }
    else
    {
        #pragma omp parallel shared (M1_storage,M2_storage,M3_storage,M4_storage,M5_storage,M6_storage,M7_storage, \
        S1_result_storage, S2_result_storage, S3_result_storage, S4_result_storage, S5_result_storage,\
        S6_result_storage, S7_result_storage, S8_result_storage, T1_result_storage, T2_result_storage)
        {
            delete []M1_storage;
            delete []M2_storage;
            delete []M3_storage;
            delete []M4_storage;
            delete []M5_storage;
            delete []M6_storage;
            delete []M7_storage;
            delete[]S3_result_storage;
            delete[]S7_result_storage;
            delete[]S2_result_storage;
            delete[]S6_result_storage;
            delete[]S1_result_storage;
            delete[]S5_result_storage;
            delete[]S4_result_storage;
            delete[]S8_result_storage;
            delete[]T1_result_storage;
            delete[]T2_result_storage;
        }
    }
    return true;
}



template <typename T, typename CA>
void cholesky_decomposition(mdspan<T, CA>& A, mdspan<T, CA>& L,matrix_multiplication_parameters algorithm, size_t step_size=0,  bool gpu_offload=false)
{
    assert(A.extent(0) == A.extent(1) && "Matrix A must be square");

    if (gpu_offload==true)
    {
        datastruct<T> dA=A.pdatastruct,dL=L.pdatastruct;
        T*buffer=(T*) acc_malloc(2*A.pdatastruct.pdatalength);
        #pragma acc CREATE_IN_STRUCT(dA)
        #pragma acc CREATE_OUT_STRUCT(dL)

#pragma acc enter data copyin(step_size)

#pragma acc parallel present(dA,dL, step_size)deviceptr(buffer)
{
     gpu_cholesky_decomposition(dA,dL,buffer,step_size);
}

#pragma acc UPDATE_HOST(dL)

#pragma acc EXIT_STRUCT(dA)
#pragma acc EXIT_STRUCT(dL)

#pragma acc exit data delete(step_size)

acc_free(buffer);

    }
    else
    {
        if(step_size==0)
            step_size=(size_t)pow(A.extent(0),0.8385);
        size_t n = A.extent(0);
        size_t nn=n*n;
        size_t tempsize=(n-step_size)*(n-step_size);

        T *sdata,*adata;

        if(algorithm.memmapped_files)
        {
            sdata=create_temp_mmap<T>(tempsize);
            adata=create_temp_mmap<T>(nn);
        }
        else
        {
            sdata=new T[tempsize];
            adata=new T[nn];
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<nn; i++)
        {
            adata[i]=A.pdatastruct.pdata[i];
            L.pdatastruct.pdata[i]=0;
        }


        mdspan<T, CA> tempA(adata,A.pdatastruct.prowmajor, n,n);

        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                // Extract submatrix R = L[c:n, z:c-1]

                auto R = L.subspanmatrix(c, z,u, c - z);

                // Compute S = RR^T using a fast matrix multiplication algorithm
                mdspan<T, CA> S(sdata,R.pdatastruct.prowmajor, u,u);
                mdspan<T,CA> RT=R.transpose();

                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(R,RT,S,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(R,RT,S,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(R,RT,S,algorithm);
                }


                #pragma omp parallel for
                for (size_t i = c; i < n; ++i)
                {
                    #pragma omp parallel for simd
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i, j) -= S(i - c, j - c);
                    }
                }

                // Update the block boundary
                z = c;
            }

            // Update the diagonal element L[c, c]
            T tmp=tempA(c, c);
            #pragma omp parallel for simd reduction(-: tmp)
            for (size_t k = z; k < c; ++k)
            {
                T tmp3=L(c,k);
                tmp-= tmp3 * tmp3;
            }

            T temp4= sqrt(tmp);
            L(c, c) =temp4;

            #pragma omp parallel for
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = tempA(i, c);
                #pragma omp parallel for simd reduction(-:tmp2)
                for (size_t k = z; k < c; ++k)
                {
                    tmp2 -= L(i, k) * L(c, k);
                }
                L(i, c)=tmp2/temp4;
            }
        }
        if(algorithm.memmapped_files)
        {
            delete_temp_mmap(sdata,tempsize);
            delete_temp_mmap(adata,nn);
        }
        else
        {
            delete[] sdata;
            delete[] adata;
        }
    }
}

template <typename T, typename CA>
inline void lu_decomposition( mdspan<T, CA>& A, mdspan<T, CA>& L, mdspan<T, CA>& U,  matrix_multiplication_parameters &algorithm,  size_t step_size=0,
                              bool gpu_offload=false)
{
    assert(A.extent(0) == A.extent(1) && "Matrix must be square");


    if (gpu_offload==true)
    {

        datastruct<T>dA=A.pdatastruct, dL=L.pdatastruct, dU=U.pdatastruct;
        T*buffer=(T*) acc_malloc(2*A.pdatastruct.pdatalength);
        #pragma acc CREATE_IN_STRUCT(dA)
        #pragma acc CREATE_OUT_STRUCT(dL)
        #pragma acc CREATE_OUT_STRUCT(dU)

#pragma acc enter data copyin (step_size)

#pragma acc parallel present(dA,dL,dU,step_size)deviceptr(buffer)
{
        gpu_lu_decomposition( dA,  dL, dU, buffer,step_size);
}

#pragma acc UPDATE_HOST(dL)
#pragma acc UPDATE_HOST(dU)

#pragma acc EXIT_STRUCT(dA)
#pragma acc EXIT_STRUCT(dL)
#pragma acc EXIT_STRUCT(dU)

acc_free(buffer);

    }
    else
    {
        if(step_size==0) step_size=(size_t)pow(A.extent(0),0.8385);
        size_t n = A.extent(0);
        size_t tempsize=(n-step_size)*(n-step_size);
        size_t nn=n*n;
        T *sdata,*adata;
        if(algorithm.memmapped_files)
        {
            sdata=create_temp_mmap<T>(tempsize);
            adata=create_temp_mmap<T>(nn);
        }
        else
        {
            sdata=new T[tempsize];
            adata=new T[nn];
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<nn; i++)
        {
            adata[i]=A.pdatastruct.pdata[i];
            L.pdatastruct.pdata[i]=0;
            U.pdatastruct.pdata[i]=0;
        }
        mdspan<T, CA> tempA(adata,nn,A.pdatastruct.prowmajor, {A.pdatastruct.pextents[0],A.pdatastruct.pextents[1]}, {A.pdatastruct.pstrides[0], A.pdatastruct.pstrides[1]});

        size_t z=0;
        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                auto RL = L.subspanmatrix(c, z, u,v);
                auto RU = U.subspanmatrix(z, c, v, u);
                mdspan<T, CA> S(sdata,RU.pdatastruct.prowmajor, u,u);
                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(RL,RU,S,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(RL,RU,S,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(RL,RU,S,algorithm);
                }


                #pragma omp parallel for
                for (size_t i = c; i < n; ++i)
                {
                    #pragma omp  parallel for simd
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i,j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }

            #pragma omp parallel for
            for (size_t i = c; i < n; ++i)
            {
                T temp=tempA(c,i);
                #pragma omp parallel for simd reduction(-:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U( k,i) * L( c,k);
                }
                U(c,i)=temp;
            }

            #pragma omp parallel for
            for (size_t i = c; i < n; ++i)
            {
                T temp = tempA(i,c);
                #pragma omp parallel for simd reduction(-:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp -= U(k,c) * L( i,k);
                }
                L(i,c)=temp/U(c,c);
            }
        }


        if(algorithm.memmapped_files)
        {
            delete_temp_mmap(sdata,tempsize);
            delete_temp_mmap(adata,nn);
        }
        else
        {
            delete[] sdata;
            delete[] adata;
        }
    }
}
// Fast QR Decomposition Algorithm for mdspan
template <typename T, typename CA>
inline void qr_decomposition(mdspan<T, CA>& A, mdspan<T, CA>& Q, mdspan<T, CA>& R,  matrix_multiplication_parameters algorithm,  size_t step_size=0,
                             bool gpu_offload=false)
{

    if (gpu_offload==true)
    {
        datastruct<T> dA= A.pdatastruct;
        datastruct<T> dQ=Q.pdatastruct;
        datastruct<T> dR=R.pdatastruct;
        T* buffer=(T*) acc_malloc(dA.pextents[1]*dA.pextents[1]+dA.pextents[0]* dA.pextents[1]*dA.pextents[0]* dA.pextents[1]);

        #pragma acc CREATE_IN_STRUCT(dA)
        #pragma acc CREATE_OUT_STRUCT(dQ)
        #pragma acc CREATE_OUT_STRUCT(dR)

#pragma acc enter data copyin(step_size)

#pragma acc parallel present(dA,dQ,dR, step_size) deviceptr(buffer)
{
    gpu_qr_decomposition(dA,dQ,dR,buffer,step_size);
}
#pragma acc UPDATE_HOST(dQ)
#pragma acc UPDATE_HOST(dR)

#pragma acc EXIT_STRUCT(dA)
#pragma acc EXIT_STRUCT(dQ)
#pragma acc EXIT_STRUCT(dR)

#pragma acc exit data delete   (step_size)
acc_free(buffer);
    }
    else
    {

        if(step_size==0)
            step_size=(size_t)pow(A.extent(0),0.8385);
        size_t n = A.extent(0); // Number of rows (assuming 2D matrix)
        size_t m = A.extent(1); // Number of columns

        // Initialize Q and R matrices
        size_t nm=n*m, mm=m*m;

        T* tempC,*tempS,*tempM;

        if(algorithm.memmapped_files)
        {
            tempC=create_temp_mmap<T>(mm);
            tempS=create_temp_mmap<T>(nm);
            tempM=create_temp_mmap<T>(nm);
        }
        else
        {
            tempC=new T[m*m];
            tempS=new T[nm];
            tempM=new T[nm];
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<nm; i++)
        {
            tempM[i]=A.pdatastruct.pdata[i];
        }
        #pragma omp parallel for simd
        for (size_t i=0; i<Q.pdatastruct.pdatalength; i++)
        {
            Q.pdatastruct.pdata[i]=0;
        }

        #pragma omp parallel for simd
        for (size_t i=0; i<R.pdatastruct.pdatalength; i++)
        {
            R.pdatastruct.pdata[i]=0;
        }
        mdspan<T, CA> M(tempM,A.pdatastruct.pdatalength,A.pdatastruct.prowmajor, {A.pdatastruct.pextents[0],A.pdatastruct.pextents[1]}, {A.pdatastruct.pstrides[0],A.pdatastruct.pstrides[1]}); // Copy of A

        size_t z = 0;

        for (size_t c = 0; c < m; ++c)
        {
            if (c == z +step_size)
            {
                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                auto BQ = Q.subspanmatrix(0, z, n, cz);
                auto BM = M.subspanmatrix(0, c, n,mc);

                // Compute C = BQ^T * BM
                auto C = mdspan<T, CA>(tempC, BM.pdatastruct.prowmajor,cz, mc);

                auto BQT=BQ.transpose();
                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(BQT,BM,C,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(BQT,BM,C,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(BQT,BM,C,algorithm);
                }


                // Compute S = BQ * C
                auto S = mdspan<T, CA>(tempS, BQ.pdatastruct.prowmajor, n, mc);

                switch (algorithm.algorithm_version)
                {
                case Matrix_Multiplication_Algorithm::Naive:
                    matrix_multiply_dot(BQ,C,S,algorithm.gpu_offload);
                    break;
                case Matrix_Multiplication_Algorithm::Strassen:
                    strassen_multiply(BQ,C,S,algorithm);
                    break;
                case Matrix_Multiplication_Algorithm::WinogradVariant:
                    winograd_multiply(BQ,C,S,algorithm);
                }


                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i)
                {
                    #pragma omp parallel for simd
                    for (size_t j = c; j < n; ++j)
                    {
                        M(i, j) -= S(i, j-c);
                    }
                }
                z = c;
            }
            // Extract column c of M
            auto v = M.column(c);

            // #pragma omp parallel for
            for (size_t j = z; j < c; ++j)
            {
                auto u = Q.column(j);

                T dot_pr =dot_product(u,v);

                //      #pragma omp parallel for simd
                for (size_t i = 0; i < n; ++i)
                {
                    v(i) -= dot_pr * u(i);
                }
            }

            // Normalize v
            T norm = sqrt(dot_product(v,v));
            //  #pragma omp parallel for simd
            for (size_t i = 0; i < n; ++i)
            {
                v(i) /= norm;
            }

            // Set column c of Q
            //   #pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                Q( i,c) = v(i);
            }
        }

        // Compute R = Q^T * A
        auto QT=Q.transpose();
        switch (algorithm.algorithm_version)
        {
        case Matrix_Multiplication_Algorithm::Naive:
            matrix_multiply_dot(QT,A,R,algorithm.gpu_offload);
            break;
        case Matrix_Multiplication_Algorithm::Strassen:
            strassen_multiply(QT,A,R,algorithm);
            break;
        case Matrix_Multiplication_Algorithm::WinogradVariant:
            winograd_multiply(QT,A,R,algorithm);
        }


        if(algorithm.memmapped_files)
        {
            delete_temp_mmap(tempC,mm);
            delete_temp_mmap(tempS,nm);
            delete_temp_mmap(tempM,nm);
        }
        else
        {
            delete[] tempC;
            delete[] tempS;
            delete[] tempM;
        }
    }

}

template <typename T, typename CA,typename CB,typename CC>
bool matrix_multiply_dot(const mdspan<T,CA>& A, const  mdspan<T,CB>& B, mdspan<T,CC>& C,  bool gpu_upload=false)
{


    datastruct<T> dA=A.pdatastruct;
    datastruct<T> dB=B.pdatastruct;
    datastruct<T> dC=C.pdatastruct;

    const size_t rows = dA.pextents[0]; // Number of rows in A and C
    const size_t cols = dB.pextents[1]; // Number of columns in B and C
    const  size_t inner_dim = dA.pextents[1]; // Number of columns in A and rows in B



    if (gpu_upload)
    {
          #pragma acc CREATE_IN_STRUCT(dA)
          #pragma acc CREATE_IN_STRUCT(dB)
          #pragma acc CREATE_OUT_STRUCT(dC)

#pragma acc enter data copyin(inner_dim, rows, cols)
        // Parallel computation
#pragma acc parallel loop gang collapse(2) present(dA, dB, dC,inner_dim,rows,cols)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum=0;
#pragma acc loop reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum+=dA(i,k)*dB(k,j);
                }
                dC(i,j)=sum;
            }
        }

#pragma acc UPDATE_HOST(dC)
#pragma acc EXIT_STRUCT(dA)
#pragma acc EXIT_STRUCT(dB)
#pragma acc EXIT_STRUCT(dC)

#pragma acc exit data delete(inner_dim, rows, cols)
    }
    else
    {
        #pragma omp parallel for collapse(2) shared(dC,dA,dB,rows,cols,inner_dim)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp parallel for simd reduction (+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += dA(i, k) * dB(k, j);
                }
                dC(i, j) = sum;
            }
        }
    }

    return true;


}


template <typename T, typename CA,typename CB,typename CC>
bool matrix_add( const mdspan<T, CA>& A,const   mdspan<T, CB>& B, mdspan<T, CC>& C)
{

    const size_t rows = C.extent(0);
    const size_t cols = C.extent(1);


    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j)=A(i,j)+B(i,j);
        }
    }

    return true;
}
template <typename T, typename CA,typename CB,typename CC>
bool matrix_subtract( const mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C)
{


    const size_t rows = C.extent(0);
    const size_t cols = C.extent(1);
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j)=A(i,j)-B(i,j);
        }
    }

    return true;
}


template <typename T, typename CA,typename CB,typename CC>
bool matrix_multiply_vector(const mdspan<T, CA>& M, const mdspan<T, CB>& V, mdspan<T, CC>& C)
{



    const size_t rows = M.extent(0); // Number of rows in A and C
    const size_t cols = V.extent(0); // Number of columns in B and C

    // Perform matrix multiplication: C = A * B
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j)= M(i, j) * V(j);  // This works because i, k, j are row/col indices
        }
    }

    return true;
}

template <typename T, typename CA,typename CB,typename CC>
bool matrix_multiply_vector(const mdspan<T, CA>& M,const  T*V, mdspan<T, CC>& C)
{


    const size_t rows = M.extent(0); // Number of rows in A and C
    const size_t cols = M.extent(1); // Number of columns in B and C

    // Perform matrix multiplication: C = A * B
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for simd
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j)= M(i, j) * V[i];  // This works because i, k, j are row/col indices
        }
    }

    return true;
}

template <typename T, typename CA,typename CC>
bool matrix_multiply_scalar(const mdspan<T, CA>& M, const T& V, mdspan<T, CC>& C)
{



    const size_t rows = C.extent(0); // Number of rows in A and C
    const size_t cols = C.extent(1); // Number of columns in B and C

    // Perform matrix multiplication: C = A * B
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
    {
        T sum=0;
        #pragma omp parallel for simd
        for (size_t j = 0; j < cols; ++j)
        {
            C(i,j)= M(i,j)*V;
        }

    }

    return true;
}

template <typename T, typename Container>
T dot_product(const  mdspan<T, Container>& vec1,const   mdspan<T, Container>& vec2)
{


    T result = 0;
    size_t n = vec1.extent(0);
    #pragma omp parallel for simd reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}

template <typename T, typename Container>
void vector_scalar_multiply( const mdspan<T, Container>& vec, const T scalar,mdspan<T, Container>& res)
{
    size_t n = vec.extent(0);

    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}


template <typename T, typename Container>
void cross_product(const mdspan<T, Container>& vec1,const  mdspan<T, Container>& vec2, mdspan<T, Container>& res)
{

    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}
template <typename T, typename Container>
void vector_add( const mdspan<T, Container>& vec1, const  mdspan<T, Container>& vec2, mdspan<T, Container>& vec3)
{


    #pragma omp parallel for simd
    for(size_t i=0; i<vec1.extent(0); i++)
    {
        vec3(i)=vec1(i)+vec2(i);
    }
}

template <typename T, typename Container>
void vector_subtract( const mdspan<T, Container>& vec1, const mdspan<T, Container>& vec2, mdspan<T, Container>& vec3)
{

    #pragma omp parallel for simd
    for(size_t i=0; i<vec1.extent(0); i++)
    {
        vec3(i)=vec1(i)-vec2(i);
    }
}

template <typename T, typename Container>
void printmatrix(const mdspan<T,Container>& span)
{
    const size_t rows= span.extent(0);
    const size_t cols=span.extent(1);
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << span(i, j) << " ";
        }
        std::cout << "\n";
    }
    cout <<endl;
}
template <typename T, typename Container>
void printvector(const mdspan<T,Container>& span)
{
    const size_t rows= span.extent(0);

    for (size_t i = 0; i <rows; ++i)
    {
        std::cout << span(i) << " ";
    }
    cout <<endl;
}



#endif
