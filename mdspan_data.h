#ifndef MDSPAN_DATAH
#define MDSPAN_DATAH

#include "mdspan_omp.h"


template<typename T>
class datastruct;



template <typename T, typename Container>
class mdspan_data: public mdspan<T,Container>
{
public:
    friend class datastruct<T>;

    mdspan_data() {};

    mdspan_data(const size_t datalength, const bool rowm, const bool memmap, const Container& extents, const Container& strides);
    mdspan_data(const bool rowm,const  bool memmap, const Container& extents, const Container& strides);
    mdspan_data(const bool rowm,const  bool memmap, const Container& extents);
    mdspan_data(const bool rowm, const bool memmap, const size_t rows, const size_t cols);

    ~mdspan_data();

    mdspan_data<T, Container> subspan(const Container& offsets, const Container& sub_extents) ;
    mdspan_data<T, Container> subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols);
    mdspan_data<T, Container> column(const size_t col_index);
    mdspan_data<T, Container> row(const size_t row_index);
    mdspan_data<T, Container> transpose();

    mdspan_data<T, Container> copy(bool memmap);
    mdspan_data<T, Container> subspan_copy(const Container& offsets, const Container& sub_extents, bool with_memmap) ;
    mdspan_data<T, Container> subspan_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols, bool with_memmap);
    mdspan_data<T, Container> column_copy(const size_t col_index,bool with_memmap);
    mdspan_data<T, Container> row_copy(const size_t row_index, bool with_memmap );
    mdspan_data<T, Container> transpose_copy( bool with_memmap);

    mdspan_data<T, Container> collapsed_view();


protected:
    bool owns_data=false;
    bool pmemmap=false;
};

template<typename T, typename Tag>
using mdspan_data_t = mdspan<T, typename container_for_tag<Tag>::type>;



template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data( const bool rowm,const bool memmap, const Container& extents, const Container& strides )
    : mdspan<T,Container>(nullptr, rowm,  extents,strides)
{
    if (memmap)
        this->dpdata = Datastruct_Host_Memory_Functions<T>::create_temp_mmap(this->dpdatalength);
    else
        this->dpdata = new T[this->dpdatalength];
    pmemmap=memmap;
    owns_data = true;

}






template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(const bool rowm,const bool memmap,const  Container& extents)
    :  mdspan<T,Container>(nullptr, rowm,  extents)
{

    if (memmap)
        this->dpdata = Datastruct_Host_Memory_Functions<T>::create_temp_mmap(this->dpdatalength);
    else
        this->dpdata = new T[this->dpdatalength];

    pmemmap=memmap;
    owns_data = true;
}


template <typename T, typename Container>
mdspan_data<T, Container>::~mdspan_data()
{
    if(owns_data)
    {
        for (auto &pr : this->pis_offloaded)
        {
            if (pr.second)
            {
                Datastruct_GPU_Memory_Functions<T>::exit_struct(*this, pr.first);
                pr.second = false;
            }
        }
        if (pmemmap)
            Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(this->dpdata, this->dpdatalength);
        else
            delete[] this->dpdata;
        owns_data=false;
    }
}


template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::copy(bool memmap)
{

    mdspan_data<T,Container>m(this->rowm,memmap,this->pextents,this->pstrides);
    memcpy(m.dpdata,this->pd,sizeof(T)*this->pdatalength);

    m.pis_offloaded=this->pis_offloaded;
    for(auto& p : this->pis_offloaded)
    {
        if(p.second==true)
        {
            Datastruct_Functions<T>::create_in_struct(*m,p.first);
        }
    }
    m.owns_data = true;
    return m;
}



template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(const bool rowm,const bool memmap, const size_t rows, const size_t cols)
    :   mdspan<T,Container>(nullptr, rowm,  rows,cols)
{
    if (memmap)
        this->dpdata = Datastruct_Host_Memory_Functions<T>::create_temp_mmap(this->dpdatalength);
    else
        this->dpdata = new T[this->dpdatalength];
    pmemmap=memmap;
    owns_data = true;

}






template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::subspan_copy(const Container& offsets, const Container& sub_extents,bool with_memmap)
{
    size_t size=1;
    #pragma omp simd reduction (*:size)
    for (size_t i=0; i< sub_extents.size(); i++)
        size*=sub_extents[i];

    mdspan_data<T, Container>  sub(this->pdrowmajor,with_memmap,size);
    sub= mdspan<T, Container>::subspan(offsets, sub_extents, this->dpdata);
    sub.owns_data = true;
    return sub;
}





template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::subspan_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,tile_rows*tile_cols);
    return mdspan<T, Container>::subspan(row, col, tile_rows, tile_cols, this->dpdata);
     sub.owns_data = true;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::column_copy(const size_t col_index,bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,this->pdextents[0]);
    sub= mdspan<T, Container>::column(col_index, this->dpdata);
    sub.owns_data = true;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::row_copy(const size_t row_index,bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,this->pdextents[1]);
    sub= mdspan<T, Container>::row(row_index, this->dpdata);
    sub.owns_data = true;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::transpose_copy(bool with_memmap)
{
    mdspan_data<T, Container> sub(this->pdrowmajor,with_memmap,this->dpextents[1]*this->dpextents[0]);
    sub= mdspan<T, Container>::transpose(this->dpdata);
    sub.owns_data = true;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container> ::subspan(const Container& offsets, const Container& sub_extents)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::subspan(offsets, sub_extents);
    sub.owns_data = false;
    return sub;
}


template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::subspan(row, col, tile_rows, tile_cols);
    sub.owns_data = false;
    return sub;
}


template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::column(const size_t col_index)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::column(col_index);
    sub.owns_data = false;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::row(const size_t row_index)
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::row(row_index);
    sub.owns_data = false;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container>mdspan_data<T, Container>::transpose()
{
    mdspan_data<T, Container> sub= mdspan<T, Container>::transpose();
    sub.owns_data = false;
    return sub;
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::collapsed_view()
{
    mdspan_data<T, Container> sub= mdspan<T, Container>:: collapsed_view();
    sub.owns_data = false;
    return sub;
}

#endif

