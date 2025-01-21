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
//#include <openacc.h>

enum Matrix_Multiplication_Algorithm
{
    Naive=0,
    Strassen=1,
    WinogradVariant=2
};


struct matrix_multiplication_parameters
{
    size_t algorithm_version{Matrix_Multiplication_Algorithm::Naive};
    size_t size_for_naive_algorithm=2;
    bool memmapped_files=true;
};


template <typename T>struct datastruct
{
    T* pdata = nullptr;
    size_t* pextents = nullptr;
    size_t* pstrides = nullptr;
    size_t pdatalength = 0;
    size_t prank = 0;
    int prowmayor=1;
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
    T&operator()(const size_t row, const size_t col);
    T operator()(const size_t row, const size_t col)const;
    T&operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1);
    T operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)const;
    T&operator()(const size_t row);
    T operator()(const size_t* indices)const;
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
size_t compute_data_length(const size_t* extents, const size_t* strides,const size_t rank)
{
    size_t offset=0;
#pragma acc loop reduction(+:offset)
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
template<typename T>inline T& datastruct<T>::operator()(const size_t row, const size_t col)
{
    return pdata[row * pstrides[0] + col * pstrides[1]];
}

#pragma acc routine seq
template<typename T>inline T& datastruct<T>::operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)
{
    return pdata[row * strides0 + col * strides1];
}

#pragma acc routine seq
template<typename T>inline T& datastruct<T>::operator()(const size_t* indices)
{
    return pdata[compute_offset(indices, this->pstrides, this->prank)];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t row)const
{

    return pdata[row * pstrides[0]];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t row, const size_t col)const
{
    return pdata[row * pstrides[0] + col * pstrides[1]];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t row, const size_t col, const size_t strides0, const size_t strides1)const
{
    return pdata[row * strides0 + col *strides1];
}

#pragma acc routine seq
template<typename T>inline T datastruct<T>::operator()(const size_t* indices)const
{
    return pdata[compute_offset(indices, this->pstrides, this->prank)];
}


#pragma acc routine seq
template<typename T>inline datastruct<T> datastruct<T>::transpose(size_t*newextents, size_t *newstrides)
{
    newextents[0]=pextents[1];
    newextents[1]=pextents[0];

    newstrides[0]=pstrides[1];
    newstrides[1]=pstrides[0];

    return datastruct(pdata,pdatalength,prowmayor,prank,newextents,newstrides,false,false);

}


#pragma acc routine seq
void fill_strides(const size_t* extents,size_t* strides, const size_t rank, const bool rowmajor)
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
    prowmayor((int)rowm)

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
    prowmayor((int) rowm)
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
    prowmayor((int)true)
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
    size_t size=1;
    const size_t r=this->prank;
    if(sub_data==nullptr)
    {


#pragma acc loop auto reduction( + : offset_index ) reduction(* : size)
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += poffsets[i] * pstrides[i];
            size*=psub_extents[i];
        }
        return datastruct(pdata + offset_index,0,this->prowmayor, r,this->prowmayor, psub_extents,pstrides, true,false );
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
            size_t original_index = compute_offset(global_indices, pstrides, prowmayor);
            size_t buffer_index = compute_offset(indices,psub_strides, prowmayor);

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
        datastruct pd(sub_data,0,prowmayor,psub_extents, psub_strides,true,true);
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
        return datastruct(pdata +row * pstrides[0]+col * pstrides[1],0,prowmayor,tile_rows,tile_cols,sub_extents,pstrides,true,false);
    }
    else
    {
        if (prowmayor)
        {
            // Row-major layout: fill row by row
#pragma acc loop auto collapse (2)
            for (size_t i = 0; i < tile_rows; ++i)
            {
                for (size_t j = 0; j < tile_cols; ++j)
                {
                    sub_data[i * tile_cols + j] = pdata[
                                                      compute_offset(row + i, col + j, pstrides[0], pstrides[1])
                                                  ];
                }
            }
        }
        else
        {
            // Column-major layout: fill column by column
#pragma acc loop auto collapse (2)
            for (size_t j = 0; j < tile_cols; ++j)
            {
                for (size_t i = 0; i < tile_rows; ++i)
                {
                    sub_data[j * tile_rows + i] = pdata[
                                                      compute_offset(row + i, col + j, pstrides[0], pstrides[1])
                                                  ];
                }
            }
        }

        return datastruct(sub_data,0,prowmayor,tile_rows, tile_cols,sub_extents,sub_strides,true,true);
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
    mdspan(T* data, size_t datalength, bool rowm, const Container&  extents,const  Container& strides);
    mdspan(T* data, bool rowm,                    const Container&  extents,const  Container& strides);
    mdspan(T* data, bool rowm,                    const Container&  extents);

    mdspan(T* data, size_t datalength, bool rowm,      Container& extents,      Container& strides);
    mdspan(T* data, bool rowm,                         Container& extents,      Container& strides);
    mdspan(T* data, bool rowm,                         Container& extents);

    mdspan(T* data, bool rowm,  size_t rows,  size_t cols);

    explicit mdspan(datastruct<T> &pd);
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

    mdspan<T, Container> subspan( Container& offsets, Container& sub_extents, T* sub_data=nullptr);
    mdspan<T, Container> subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data=nullptr);

    mdspan<T, Container> row(const size_t row_index);
    mdspan<T, Container> column(size_t col_index);
    datastruct<T> substructmatrix( const size_t row,const size_t col, const size_t tile_rows,const  size_t tile_cols) ;
    datastruct<T> substruct( const Container&offsets, const Container &sub_extents) ;
    mdspan<T, Container> transpose() ;
    // Other utility methods
    size_t extent(const size_t dim) const;
    size_t rank() const;
    size_t stride(const size_t dim) const;

    // Member function declarations
    Container& extents();
    Container& strides();

    size_t datalength() const;
    // Data structure for parallel device allocation (assumed type)
    datastruct<T> pdatastruct;

private:
    // Private member variables
    Container pextents;  // Use the ExtentContainer type
    Container pstrides;  // Use the StrideContainer type
    bool pis_associated=false;
    bool pdatac_copied_to_device=false;
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
mdspan<T, Container>::mdspan(T* data,   size_t datalength,  bool rowm, const Container& extents, const Container& strides)
    :
    pdatastruct(data,
                datalength,rowm,extents.size(),nullptr,nullptr,false,false),
    pis_associated(false)
{
    // Resize and copy extents from container
    const size_t r=extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
        pstrides.resize(r); // Resize dynamic container
    }

    // Resize and copy extents from container
    #pragma omp simd
    for (size_t i=0; i<r; i++)
    {
        pextents[i]=extents[i];
        pstrides[i]=strides[i];
    }

    // Assign actual pointers to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm, const Container& extents, const Container& strides )
    : pdatastruct(data, 0,rowm,extents.size(),nullptr,nullptr,false,false),
      pis_associated(false)
      // Initialize pdatastruct with placeholders
{
    // Resize and copy extents and strides from containers
    const size_t r=extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents= {}; // Default-initialize static container
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
        pstrides.resize(r); // Resize dynamic container
    }

    size_t s=1;
    #pragma omp simd reduction(*:s)
    for (size_t i=0; i<r; i++)
    {
        s*=extents[i];
        pextents[i]=extents[i];
        pstrides[i]=strides[i];
    }
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, bool rowm,const  Container& extents)
    :  pdatastruct(data,0,rowm,extents.size(),nullptr,nullptr,false,false),
       pis_associated(false)
{
    const size_t r=extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }

    // Resize and copy extents from container
    #pragma omp simd
    for (size_t i=0; i<r; i++)
    {
        pextents[i]=extents[i];
    }
    compute_strides(pextents,pstrides,rowm);
    // Assign actual pointers to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();

    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,   size_t datalength,  bool rowm,  Container& extents,  Container& strides)
    : pdatastruct(data,
                  datalength,rowm,extents.size(),nullptr,nullptr,false,false),
      pis_associated(false)
{
    // Resize and copy extents from container
    const size_t r=extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
        pstrides.resize(r); // Resize dynamic container
    }

    // Resize and copy extents from container
    #pragma omp simd
    for (size_t i=0; i<r; i++)
    {
        pextents[i]=extents[i];
        pstrides[i]=strides[i];
    }

    // Assign actual pointers to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  Container& extents,  Container& strides )
    : pdatastruct(data, 0,rowm,extents.size(),nullptr,nullptr,false,false),
      pis_associated(false)
      // Initialize pdatastruct with placeholders
{
    // Resize and copy extents and strides from containers
    const size_t r=extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents= {}; // Default-initialize static container
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
        pstrides.resize(r); // Resize dynamic container
    }

    size_t s=1;
    #pragma omp simd reduction(*:s)
    for (size_t i=0; i<r; i++)
    {
        s*=extents[i];
        pextents[i]=extents[i];
        pstrides[i]=strides[i];
    }
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, bool rowm,  Container& extents)
    :  pdatastruct(data,0,rowm,extents.size(),nullptr,nullptr,false,false),
       pis_associated(false)
{
    const size_t r=extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }

    // Resize and copy extents from container
    #pragma omp simd
    for (size_t i=0; i<r; i++)
    {
        pextents[i]=extents[i];
    }
    compute_strides(pextents,pstrides,rowm);
    // Assign actual pointers to datastruct
    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();
    pdatastruct.pdatalength=compute_data_length(pdatastruct.pextents,pdatastruct.pstrides,pdatastruct.prank);

}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  const size_t rows, const size_t cols)
    :  pdatastruct(data,0,rowm,2,nullptr,nullptr,false,false),
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

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(datastruct<T> &pd):
    pis_associated(false),
    pdatastruct(data,
                pd.pdatalength,pd.rowmayor,nullptr,nullptr,false,false)
{
    if constexpr (StaticContainer<Container>)
    {
        pextents= {}; // Default-initialize static container
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(pd.prank); // Resize dynamic container
        pstrides.resize(pd.prank); // Resize dynamic container
    }
    const size_t size=pextents.size();
    // Resize and copy extents from container
    #pragma omp simd
    for (size_t i=0; i<size; i++)
    {
        pextents[i]=pd.pextents[i];
        pstrides[i]=pd.pstrides[i];
    }

    pdatastruct.pextents = pextents.data();
    pdatastruct.pstrides = pstrides.data();

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
        return mdspan(pdatastruct.pdata + offset_index,pdatastruct.prowmayor, sub_extents, pstrides);

    }
    else
    {
        // Compute the new strides for the subspan
        Container sub_strides;
        compute_strides(sub_extents, sub_strides, pdatastruct.prowmayor);
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
            size_t original_index = compute_offset(global_indices.data(), pdatastruct.pstrides,global_indices.size(), pdatastruct.prowmayor);
            size_t buffer_index = compute_offset(indices.data(),sub_strides.data(),indices.size(), pdatastruct.prowmayor);

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
        return mdspan(sub_data, pdatastruct.prowmayor, sub_extents, sub_strides );
    }
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data)const
{

    if(sub_data==nullptr)
    {

        size_t offset=row * pdatastruct.pstrides[0]+col * pdatastruct.pstrides[1];
        const Container ext= {tile_rows,tile_cols};
        return mdspan(pdatastruct.pdata +offset,pdatastruct.prowmayor,ext,pstrides);
    }
    else
    {
        if (pdatastruct.prowmayor)
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

        Container sub_strides = (pdatastruct.prowmayor==true)? Container{tile_cols, 1} :
                                Container{1,tile_rows}; // Contiguous row-major layout

        return mdspan(sub_data,pdatastruct.prowmayor, sub_extents, sub_strides );
    }
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan( Container&offsets,  Container &sub_extents, T*sub_data)
{
    const size_t r=pdatastruct.prank;




    if (sub_data==nullptr)
    {
        // Compute the offset to the starting point
        size_t offset_index = 0;

        #pragma omp  simd reduction( + : offset_index )
        for (size_t i = 0; i < r; ++i)
        {
            offset_index += offsets[i] * pdatastruct.pstrides[i];
        }

        // Create a new mdspan_dynamic with the updated pointer, extents, and the same strides
        return mdspan(pdatastruct.pdata + offset_index,pdatastruct.prowmayor, sub_extents, pstrides);

    }
    else
    {
        // Compute the new strides for the subspan
        Container sub_strides;
        compute_strides(sub_extents, sub_strides, pdatastruct.prowmayor);
        vector<size_t> indices(r,0);
        vector<size_t> global_indices(r,0);
        while (true)
        {
            // Compute the current global indices
            #pragma omp parallel for simd
            for (size_t i = 0; i < r; ++i)
            {
                global_indices[i] = offsets[i] + indices[i];
            }

            // Compute the offsets for the original data and the new buffer
            size_t original_index = compute_offset(global_indices.data(), pdatastruct.pstrides,global_indices.size(), pdatastruct.prowmayor);
            size_t buffer_index = compute_offset(indices.data(),sub_strides.data(),indices.size(), pdatastruct.prowmayor);

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
        #pragma omp parallel for reduction(* : size)
        for (size_t i = 0; i < r; ++i)
        {
            size*=sub_extents[i];
        }
        // Create and return a new mdspan with the updated pointer, extents, and strides
        return mdspan(sub_data, pdatastruct.prowmayor, sub_extents, sub_strides );
    }
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix( const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*sub_data)
{

    if(sub_data==nullptr)
    {

        size_t offset=row * pdatastruct.pstrides[0]+col * pdatastruct.pstrides[1];
        const Container ext= {tile_rows,tile_cols};
        return mdspan(pdatastruct.pdata +offset,pdatastruct.prowmayor,ext,pstrides);
    }
    else
    {
        if (pdatastruct.prowmayor)
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

        Container sub_strides = (pdatastruct.prowmayor==true)? Container{tile_cols, 1} :
                                Container{1,tile_rows}; // Contiguous row-major layout

        return mdspan(sub_data,pdatastruct.prowmayor, sub_extents, sub_strides );
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
               pdatastruct.prowmayor,                                  // Column-major layout
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
               pdatastruct.prowmayor,                                  // Row-major layout
               row_extents,                                            // Updated extents
               row_strides                                             // Updated strides
           );
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
Container & mdspan<T, Container> ::extents()
{
    return pextents;
}
template <typename T, typename Container>
Container & mdspan<T, Container> ::strides()
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
    return mdspan(pdatastruct.pdata,pdatastruct.pdatalength, pdatastruct.prowmayor,  transposed_extents,   transposed_strides);
}

#pragma acc routine worker
template <typename T>
void gpu_matrix_multiply_dot_w( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
{

#pragma acc loop worker collapse(2)
    for (size_t i = 0; i < A.pextents[0]; ++i)
    {
        for (size_t j = 0; j < B.pextents[1]; ++j)
        {
            T sum = 0;
#pragma acc loop vector reduction(+: sum)
            for (size_t k = 0; k < A.pextents[1]; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}

#pragma acc routine vector
template <typename T>
void gpu_matrix_multiply_dot_v( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
{

    for (size_t i = 0; i < A.pextents[0]; ++i)
    {
        for (size_t j = 0; j < B.pextents[1]; ++j)
        {
            T sum = 0;
#pragma acc loop vector reduction(+: sum)
            for (size_t k = 0; k < A.pextents[1]; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}


#pragma acc routine seq
template <typename T>
void gpu_matrix_multiply_dot_s( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
{

    for (size_t i = 0; i < A.pextents[0]; ++i)
    {
        for (size_t j = 0; j < B.pextents[1]; ++j)
        {
            T sum = 0;
#pragma acc loop reduction(+: sum)
            for (size_t k = 0; k < A.pextents[1]; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}



#pragma acc routine worker
template <typename T>
inline void gpu_cholesky_decomposition( datastruct<T>& A, datastruct<T>& L, T*buffer1=nullptr, T*buffer2=nullptr,size_t step_size=0)
{

    const size_t n = A.pextents[0];
    size_t z = 0; // Zero-based indexing, starts at the first column

    const size_t tempsize = (n - step_size) * (n - step_size);

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    size_t pext3[2];
    size_t pstrides3[2];

    const size_t nn=n*n;

    // Allocate memory for S on the device
    T* sdata;
    T* adata;
    sdata=new T[tempsize];
    adata=new T[nn];

    if (buffer1==(T*) nullptr)
        sdata=new T[tempsize];
    else
        sdata=buffer1;

    if (buffer2==(T*) nullptr)
        adata=new T[nn];
    else
        adata=buffer2;


#pragma acc loop vector
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=A.pdata[i];
        L.pdata[i]=0;
    }

    datastruct<T> tempA(adata, 0,A.prowmayor,n, n,pext3, pstrides3,true,true);
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
            datastruct<T> S(sdata, 0, A.prowmayor,u, u, pext2, pstrides2,true,true);

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
    if(buffer1==nullptr)
        delete[] sdata;
    if(buffer2==nullptr)
        delete[] adata;

}

#pragma acc routine worker
template <typename T>
inline void gpu_lu_decomposition( datastruct<T>& dA, datastruct<T>& dL, datastruct<T>& dU, T* buffer1=nullptr, T*buffer2=nullptr, size_t step_size=0)
{

    const size_t n = dA.pextents[0];
    size_t z = 0; // Zero-based indexing, starts at the first column


    const size_t tempsize = (n - step_size) * (n - step_size);

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    size_t pext3[2];
    size_t pstrides3[2];
    const size_t nn=n*n;


    T* sdata;
    T* adata;

    if (buffer1==nullptr)
        sdata=new T[tempsize];
    else
        sdata=buffer1;

    if (buffer2==nullptr)
        adata=new T[nn];
    else
        adata=buffer2;


#pragma acc loop vector
    for (size_t i=0; i<nn; i++)
    {
        adata[i]=dA.pdata[i];
        dL.pdata[i]=0;
        dU.pdata[i]=0;
    }
    datastruct<T> tempA(adata,  0, dA.prowmayor,n, n,pext3, pstrides3,true,true);
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

            datastruct<T> S(sdata,  0, dA.prowmayor,u, u,pext2, pstrides2,true,true);
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

    if(buffer1==nullptr)
        delete[] sdata;
    if(buffer2==nullptr)
        delete[] adata;

}

#pragma acc routine worker
template <typename T >
void gpu_qr_decomposition(datastruct<T>&A, datastruct<T> Q, datastruct<T> &R, T* buffer1=nullptr, T*buffer2=nullptr,T*buffer3=nullptr, size_t step_size=0)
{

    const size_t n = A.pextents[0]; // Number of rows (assuming 2D matrix)
    const size_t m = A.pextents[1]; // Number of columns

    if(step_size==0)
        step_size=(size_t)pow(A.pextents[0],0.8385);

    const size_t nm=n*m;
    T* tempC;
    T* tempS;
    T* tempM;

    if(buffer1==nullptr)
        tempC=new T[m*m];
    else
        tempC=buffer1;

    if(buffer2==nullptr)
        tempS=new T[nm];
    else
        tempS=buffer2;

    if(buffer3==nullptr)
        tempM=new T[nm];
    else
        tempM=buffer3;

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


    datastruct<T> M(tempM,A.pdatalength,A.prowmayor,A.prank,mext,mstrides,false,false); //Copy of A
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
            datastruct<T> C(tempC,0, BM.prowmayor,cz, mc,extc,strc,true,true);
            datastruct<T> BQT=BQ.transpose(extbqt,strbqt);

            gpu_matrix_multiply_dot_w(BQT,BM,C);

            // Compute S = BQ * C
            datastruct<T>S(tempS, 0,BQ.prowmayor,n, mc,exts,strs,true,true);

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
    if(buffer1==nullptr)
        delete[] tempC;
    if(buffer2==nullptr)
        delete[] tempS;
    if(buffer3==nullptr)
        delete[] tempM;

}

#pragma acc routine worker
template <typename T>
bool gpu_matrix_add_w( datastruct<T>& A, datastruct<T>& B, datastruct<T>& C)
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
bool gpu_matrix_subtract_w( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
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
bool gpu_matrix_multiply_vector_w(  datastruct<T>&M,  datastruct<T> V, datastruct<T> C)
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
bool gpu_matrix_multiply_vector_w( datastruct<T>M,  T*V, datastruct<T> & C)
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
bool gpu_matrix_multiply_scalar_w(  datastruct<T>& M,  T& V, datastruct<T>& C)
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
T gpu_dot_product_w(  datastruct<T> vec1,  datastruct<T> vec2)
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

#pragma acc routine worker
template <typename T>
void gpu_vector_scalar_multiply_w( datastruct<T>& vec, T scalar,datastruct<T>& res)
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
void gpu_cross_product( datastruct<T>& vec1,   datastruct<T>& vec2, datastruct<T>& res)
{
    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}

#pragma acc routine worker
template <typename T>
void gpu_vector_add_w(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
#pragma acc loop vector
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}

#pragma acc routine worker
template <typename T>
void gpu_vector_subtract_w(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res)
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
bool gpu_matrix_add_v( datastruct<T>& A, datastruct<T>& B, datastruct<T>& C)
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
bool gpu_matrix_subtract_v( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C)
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
bool gpu_matrix_multiply_vector_v(  datastruct<T>&M,  datastruct<T> V, datastruct<T> C)
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
bool gpu_matrix_multiply_vector_v( datastruct<T>M,  T*V, datastruct<T> & C)
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
bool gpu_matrix_multiply_scalar_v(  datastruct<T>& M,  T& V, datastruct<T>& C)
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
T gpu_dot_product_v(  datastruct<T> vec1,  datastruct<T> vec2)
{
    const size_t n=vec1.pextents[0];
    T result = 0;
#pragma acc loop vector reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}

#pragma acc routine seq
template <typename T>
T gpu_dot_product_s(  datastruct<T> vec1,  datastruct<T> vec2)
{
    const size_t n=vec1.pextents[0];
    T result = 0;
#pragma acc loop  reduction(+:result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}

#pragma acc routine vector
template <typename T>
void gpu_vector_scalar_multiply_v( datastruct<T>& vec, T scalar,datastruct<T>& res)
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
void gpu_vector_add_v(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res)
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
void gpu_vector_subtract_v(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res)
{
    const size_t n=vec1.pextents[0];
#pragma acc loop vector
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}









template <typename T, typename CA,typename CB,typename CC>
bool strassen_multiply(const  mdspan<T, CA>& A,  const mdspan<T, CB>& B, mdspan<T, CC>& C, const matrix_multiplication_parameters algorithm, bool gpu_upload=false)
{
    // Dimensions of input matrices
    const  size_t n = A.extent(0); // Rows in A
    const  size_t m = A.extent(1); // Columns in A and rows in B
    const size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (m*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,gpu_upload);
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
        #pragma omp parallel
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
        #pragma omp parallel
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

    mdspan<T, CC> M1(M1_storage, s, {half_n, half_p}, {half_p, 1}),
           M2(M2_storage, s, {half_n, half_p}, {half_p, 1}),
           M3(M3_storage, s, {half_n, half_p}, {half_p, 1}),
           M4(M4_storage, s, {half_n, half_p}, {half_p, 1}),
           M5(M5_storage, s, {half_n, half_p}, {half_p, 1}),
           M6(M6_storage, s, {half_n, half_p}, {half_p, 1}),
           M7(M7_storage,s, {half_n, half_p}, {half_p, 1});


    // Task 1: A_result = A11 + A22


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

    #pragma omp parallel shared(A11,A22,A21,A12,B12,B21,B11,B22,M1,M2,M3,M4,M5,M6,M7,\
    A_result1,A_result2,A_result3,A_result4,A_result5,B_result1,B_result2,B_result3,B_result4,B_result5)
    {
        #pragma omp single
        {
            matrix_add(A11, A22, A_result1);
            matrix_add(B11, B22, B_result1);
            strassen_multiply(A_result1, B_result1, M1, algorithm,gpu_upload);
        }
        #pragma omp single
        {
            matrix_add(A21, A22, A_result2);
            strassen_multiply(A_result2, B11, M2, algorithm,gpu_upload);
        }
        #pragma omp single
        {
            matrix_subtract(B12, B22, B_result2);
            strassen_multiply(A11, B_result2, M3, algorithm,gpu_upload);
        }
        #pragma omp single
        {
            matrix_subtract(B21, B11, B_result3);
            strassen_multiply(A22, B_result3, M4, algorithm,gpu_upload);
        }
        #pragma omp single
        {
            matrix_add(A11, A12, A_result3);
            strassen_multiply(A_result3, B22, M5,algorithm,gpu_upload);
        }
        #pragma omp single
        {
            matrix_subtract(A21, A11, A_result4);
            matrix_add(B11, B12, B_result4);
            strassen_multiply(A_result4, B_result4, M6,algorithm,gpu_upload);
        }
        #pragma omp single
        {
            matrix_subtract(A12, A22, A_result5);
            matrix_add(B21, B22, B_result5);
            strassen_multiply(A_result5, B_result5, M7, algorithm,gpu_upload);
        }
    }

    // Submatrices of C
    auto C11 = C.subspan({0, 0}, {half_n, half_p});
    auto C12 = C.subspan({0, half_p}, {half_n, half_p});
    auto C21 = C.subspan({half_n, 0}, {half_n, half_p});
    auto C22 = C.subspan({half_n, half_p}, {half_n, half_p});

    #pragma omp parallel for collapse(2) shared(M2,M3,M5,M6,M7)
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
    else
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

    return true;
}

template <typename T, typename CA,typename CB,typename CC>
bool winograd_multiply(const  mdspan<T, CA>& A, const mdspan<T, CB>& B, mdspan<T, CC>& C, matrix_multiplication_parameters algorithm, bool gpu_upload=false)
{
    // Dimensions of input matrices
    size_t n = A.extent(0); // Rows in A
    size_t m = A.extent(1); // Columns in A and rows in B
    size_t p = B.extent(1); // Columns in B

    // Base case: if no dimension is divisible by 2, use standard multiplication
    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || (m*p<=algorithm.size_for_naive_algorithm))
    {
        matrix_multiply_dot(A, B, C,gpu_upload);
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
    else
    {
        #pragma omp parallel
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

    #pragma omp parallel shared(S1,S2,S3,S4,S5,S6,S7,S8,A11,A12,B11,B21,A22,B22,M1,M2,M3,M4,M5,M6,algorithm,gpu_upload)
    {
        winograd_multiply(S2,S6,M1,algorithm,gpu_upload);
        winograd_multiply(A11,B11,M2,algorithm,gpu_upload);
        winograd_multiply(A12,B21,M3,algorithm,gpu_upload);
        winograd_multiply(S3,S7,M4,algorithm,gpu_upload);
        winograd_multiply(S1,S5,M5,algorithm,gpu_upload);
        winograd_multiply(S4,B22,M6,algorithm,gpu_upload);
        winograd_multiply(A22,S8,M7,algorithm,gpu_upload);
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
    else
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
    return true;
}

template <typename T, typename CA>
void cholesky_decomposition(mdspan<T, CA>& A, mdspan<T, CA>& L, matrix_multiplication_parameters algorithm, size_t step_size=0,  bool gpu_offload=false,
                            bool gpu_multiplication=false)
{
    assert(A.extent(0) == A.extent(1) && "Matrix A must be square");

    if (gpu_offload==true)
    {
        datastruct<T> dA=A.pdatastruct,dL=L.pdatastruct;
#pragma acc enter data copyin(dA)
#pragma acc enter data copyin(dA.pdata[0:dA.pdatalength])
#pragma acc enter data copyin(dA.pextents[0:dA.prank])
#pragma acc enter data copyin(dA.pstrides[0:dA.prank])

#pragma acc enter data copyin(dL)
#pragma acc enter data create(dL.pdata[0:dL.pdatalength])
#pragma acc enter data copyin(dL.pextents[0:dL.prank])
#pragma acc enter data copyin(dL.pstrides[0:dL.prank])

#pragma acc enter data copyin(step_size)

#pragma acc kernels present(dA,dL,step_size)
        do
        {
             gpu_cholesky_decomposition(dA,dL,(T*)nullptr,(T*)nullptr,step_size);
        }
        while(false);
#pragma acc update self(dL.pdata[0:dL.pdatalength])

#pragma acc exit data delete(dA.pdata[0:dA.pdatalength], dA.pextents[0:dA.prank], dA.pstrides[0:dA.prank], dA)
#pragma acc exit data delete(dL.pdata[0:dL.pdatalength], dL.pextents[0:dL.prank], dL.pstrides[0:dL.prank], dL)
#pragma acc exit data delete(step_size)
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


        mdspan<T, CA> tempA(adata,A.pdatastruct.prowmayor, n,n);

        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                // Extract submatrix R = L[c:n, z:c-1]

                auto R = L.subspanmatrix(c, z,u, c - z);

                // Compute S = RR^T using a fast matrix multiplication algorithm
                mdspan<T, CA> S(sdata,R.pdatastruct.prowmayor, u,u);
                mdspan<T,CA> RT=R.transpose();

                switch (algorithm.algorithm_version)
                {
                case Naive:
                    matrix_multiply_dot(R,RT,S,gpu_multiplication);
                    break;
                case Strassen:
                    strassen_multiply(R,RT,S,algorithm,gpu_multiplication);
                    break;
                case WinogradVariant:
                    winograd_multiply(R,RT,S,algorithm,gpu_multiplication);
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
                              bool gpu_upload=false, bool gpu_multoffload=false  )
{
    assert(A.extent(0) == A.extent(1) && "Matrix must be square");


    if (gpu_upload==true)
    {
        datastruct<T>dA=A.pdatastruct, dL=L.pdatastruct, dU=U.pdatastruct;

#pragma acc enter data copyin(dA)
#pragma acc enter data copyin(dA.pdata[0:dA.pdatalength])
#pragma acc enter data copyin(dA.pextents[0:dA.prank])
#pragma acc enter data copyin(dA.pstrides[0:dA.prank])

#pragma acc enter data copyin(dL)
#pragma acc enter data create(dL.pdata[0:dL.pdatalength])
#pragma acc enter data copyin(dL.pextents[0:dL.prank])
#pragma acc enter data copyin(dL.pstrides[0:dL.prank])

#pragma acc enter data copyin(dU)
#pragma acc enter data create(dU.pdata[0:dU.pdatalength])
#pragma acc enter data copyin(dU.pextents[0:dU.prank])
#pragma acc enter data copyin(dU.pstrides[0:dU.prank])

#pragma acc enter data copyin (step_size)

#pragma acc kernels present(dA,dL,dU,step_size)
        do
        {
            gpu_lu_decomposition( dA,  dL, dU, (T*) nullptr,(T*) nullptr,step_size);
        }
        while(false);
#pragma acc update self(dL.pdata[0:dL.pdatalength])
#pragma acc update self(dU.pdata[0:dU.pdatalength])

#pragma acc exit data delete(dA.pdata[0:dA.pdatalength], dA.pextents[0:dA.prank], dA.pstrides[0:dA.prank], dA)
#pragma acc exit data delete(dL.pdata[0:dL.pdatalength], dL.pextents[0:dL.prank], dL.pstrides[0:dL.prank], dL)
#pragma acc exit data delete(dU.pdata[0:dU.pdatalength], dU.pextents[0:dU.prank], dU.pstrides[0:dU.prank], dU)
#pragma acc exit data delete(step_size)

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
        mdspan<T, CA> tempA(adata,nn,A.pdatastruct.prowmayor, {A.pdatastruct.pextents[0],A.pdatastruct.pextents[1]}, {A.pdatastruct.pstrides[0], A.pdatastruct.pstrides[1]});

        size_t z=0;
        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                auto RL = L.subspanmatrix(c, z, u,v);
                auto RU = U.subspanmatrix(z, c, v, u);
                mdspan<T, CA> S(sdata,RU.pdatastruct.prowmayor, u,u);
                switch (algorithm.algorithm_version)
                {
                case Naive:
                    matrix_multiply_dot(RL,RU,S,gpu_multoffload);
                    break;
                case Strassen:
                    strassen_multiply(RL,RU,S,algorithm,gpu_multoffload);
                    break;
                case WinogradVariant:
                    winograd_multiply(RL,RU,S,algorithm,gpu_multoffload);
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
                U(c,i)=temp;
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
                L(i,c)=temp;
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
                             bool gpu_upload=false,  bool gpu_multoffload=false )
{

    if (gpu_upload==true)
    {
        datastruct<T> dA= A.pdatastruct;
        datastruct<T> dQ=Q.pdatastruct;
        datastruct<T> dR=R.pdatastruct;

#pragma acc enter data copyin(dA)
#pragma acc enter data copyin(dA.pdata[0:dA.pdatalength])
#pragma acc enter data copyin(dA.pextents[0:dA.prank])
#pragma acc enter data copyin(dA.pstrides[0:dA.prank])

#pragma acc enter data copyin(dQ)
#pragma acc enter data create(dQ.pdata[0:dQ.pdatalength])
#pragma acc enter data copyin(dQ.pextents[0:dQ.prank])
#pragma acc enter data copyin(dQ.pstrides[0:dQ.prank])

#pragma acc enter data copyin(dR)
#pragma acc enter data create(dR.pdata[0:dR.pdatalength])
#pragma acc enter data copyin(dR.pextents[0:dR.prank])
#pragma acc enter data copyin(dR.pstrides[0:dR.prank])

#pragma acc enter data copyin(step_size)

#pragma acc kernels present(dA,dA.pdata[0:dA.pdatalength],dA.pextents[0:dA.prank],dA.pstrides[0:dA.prank], \
                             dQ, dQ.pdata[0:dQ.pdatalength],dQ.pextents[0:dQ.prank],dQ.pstrides[0:dQ.prank],\
                             dR, dR.pdata[0:dR.pdatalength],dR.pextents[0:dR.prank],dR.pstrides[0:dR.prank], step_size)
        do
        {
            gpu_qr_decomposition(dA,dQ,dR,(T*)nullptr,(T*)nullptr,(T*)nullptr,step_size);
        }
        while(false);
#pragma acc update self(dQ.pdata[0:dQ.pdatalength])
#pragma acc update self(dR.pdata[0:dR.pdatalength])

#pragma acc exit data delete  (dA.pdata[0:dA.pdatalength], dA.pextents[0:dA.prank], dA.pstrides[0:dA.prank], dA)
#pragma acc exit data delete  (dQ.pdata[0:dQ.pdatalength], dQ.pextents[0:dQ.prank], dQ.pstrides[0:dQ.prank], dQ)
#pragma acc exit data delete   (dR.pdata[0:dR.pdatalength], dR.pextents[0:dR.prank], dR.pstrides[0:dR.prank], dR)
#pragma acc exit data delete   (step_size)

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
        mdspan<T, CA> M(tempM,A.pdatastruct.pdatalength,A.pdatastruct.prowmayor, {A.pdatastruct.pextents[0],A.pdatastruct.pextents[1]}, {A.pdatastruct.pstrides[0],A.pdatastruct.pstrides[1]}); // Copy of A

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
                auto C = mdspan<T, CA>(tempC, BM.pdatastruct.prowmayor,cz, mc);

                auto BQT=BQ.transpose();
                switch (algorithm.algorithm_version)
                {
                case Naive:
                    matrix_multiply_dot(BQT,BM,C,gpu_multoffload);
                    break;
                case Strassen:
                    strassen_multiply(BQT,BM,C,algorithm,gpu_multoffload);
                    break;
                case WinogradVariant:
                    winograd_multiply(BQT,BM,C,algorithm,gpu_multoffload);
                }


                // Compute S = BQ * C
                auto S = mdspan<T, CA>(tempS, BQ.pdatastruct.prowmayor, n, mc);

                switch (algorithm.algorithm_version)
                {
                case Naive:
                    matrix_multiply_dot(BQ,C,S,gpu_multoffload);
                    break;
                case Strassen:
                    strassen_multiply(BQ,C,S,algorithm,gpu_multoffload);
                    break;
                case WinogradVariant:
                    winograd_multiply(BQ,C,S,algorithm,gpu_multoffload);
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
        case Naive:
            matrix_multiply_dot(QT,A,R,gpu_multoffload);
            break;
        case Strassen:
            strassen_multiply(QT,A,R,algorithm,gpu_multoffload);
            break;
        case WinogradVariant:
            winograd_multiply(QT,A,R,algorithm,gpu_multoffload);
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
// Transfer dC to the GPU
#pragma acc enter data copyin(dC) // Allocate memory for the struct
#pragma acc enter data create(dC.pdata[0:dC.pdatalength]) // Allocate pdata on device
#pragma acc enter data copyin(dC.pextents[0:dC.prank])    // Copy extents
#pragma acc enter data copyin(dC.pstrides[0:dC.prank])    // Copy strides

// Transfer dB to the GPU
#pragma acc enter data copyin(dB)
#pragma acc enter data copyin(dB.pdata[0:dB.pdatalength])
#pragma acc enter data copyin(dB.pextents[0:dB.prank])
#pragma acc enter data copyin(dB.pstrides[0:dB.prank])

// Transfer dA to the GPU
#pragma acc enter data copyin(dA)
#pragma acc enter data copyin(dA.pdata[0:dA.pdatalength])
#pragma acc enter data copyin(dA.pextents[0:dA.prank])
#pragma acc enter data copyin(dA.pstrides[0:dA.prank])

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
#pragma acc update self(dC.pdata[0:dC.pdatalength])
#pragma acc exit data delete(dC.pdata[0:dC.pdatalength], dC.pextents[0:dC.prank], dC.pstrides[0:dC.prank], dC)
#pragma acc exit data delete(dB.pdata[0:dB.pdatalength], dB.pextents[0:dB.prank], dB.pstrides[0:dB.prank], dB)
#pragma acc exit data delete(dA.pdata[0:dA.pdatalength], dA.pextents[0:dA.prank], dA.pstrides[0:dA.prank], dA)
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
