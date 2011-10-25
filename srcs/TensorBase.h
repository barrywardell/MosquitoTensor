#ifndef TENSORBASE_H_
#define TENSORBASE_H_

namespace Mosquito {

  /**
   * Base class of tensor objects. Stores components, allows indexing.
   */
  class TensorBase {

    public:
      /**
       * Named types for the indices. There are synonyms: UP are
       * CONTRAVARIANT (vector type) while DOWN are COVARIANT indices
       * (covector type).
       */
      enum IndexType {
        COVARIANT = -1,   /**< Specifies a covariant tensor index */
        CONTRAVARIANT = 1,/**< Specifies a contravariant tensor index */
        DOWN = -1,        /**< Specifies a covariant tensor index */
        UP = 1            /**< Specifies a contravariant tensor index */
      };

      /**
       * \brief Returns the rank of this tensor.
       *
       * \retval rank The rank of this tensor.
       */
      int getRank() const;

      /**
       * \brief Returns a const pointer to the index types of this Tensor.
       *
       * \retval types The index types of this tensor.
       */
      const IndexType* getTypes() const;

      /**
       * \brief Returns a pointer to the components.
       *
       * Since one can individually change the components via the index,
       * this routine is provided so one can get all the data for
       * printing, editing, evolving or whatnot.
       * \retval components A pointer to the array storing the components.
       */
      double* getComponents() const;

      /**
       * \brief Copy all components to an array of doubles
       *
       * This routine is provided so one can quickly output all the data to
       * a C array of doubles. It is assumed that the array has been allocated
       * and is at least as large as the number of tensor components.
       * \param array A pointer to a double array for the data
       * \retval num The number of components copied
       */
      int getComponents(double* array) const;

      /**
       * \brief The number of components in the tensor
       *
       * \retval num The number of components
       */
      int getNumComponents() const;

      /**
       * \brief Copy all components from an array of doubles
       *
       * This routine is provided so one can quickly set all the data from
       * a C array of doubles. It is assumed that the array has been allocated
       * and is at least as large as the number of tensor components.
       * \param array A pointer to a double array containing the data
       * \retval num The number of components copied
       */
      int setComponents(const double* array);

      /**
       * \brief Returns a reference to the scalar value.
       *
       * The object must be a scalar or this will fail.
       * \retval value The scalar value.
       */
      double & operator()() const;

      /**
       * \brief Returns a reference to the indexed component.
       * \param i1 The first index.
       * \param ... The next indices.
       * \retval component The indexed component.
       */
      double & operator()(int i1, ...) const;

      /**
       * \brief Returns a reference to the indexed component.
       *
       * It is sometimes preferable to work with an array of indices than
       * to work with them directly.
       * \param indices The array of the indices specifying the component.
       * \retval component The indexed component.
       */
      double & operator()(int* indices) const;

      /**
       * \brief An indexing function. 
       *
       * To abstract away the storage model. Converts n=rank indices into a
       * single 1-d index. This is used to get the actual component from
       * the 1d storage array.
       * \param indices An array of indices.
       */
      int index(const int* indices) const;

      /**
       * \brief An indexing function. 
       *
       * To abstract away the storage model. Converts n=rank indices into a
       * single 1-d index. This is used to get the actual component from
       * the 1d storage array.
       * \param i1 The first index.
       * \param ... The next rank-1 indices.
       */
      int index(int i1, ...) const;

      /**
       * \brief Converts 1d index to rank-d.
       *
       * Converts the 1d index (as returned from index()) into an
       * array of indices from 0-DIMENSION-1.
       * \param index The 1d index.
       * \param indices The indices array to set.
       */
      void indexToIndices(int index, int* indices) const;

    protected:

      /**
       * \brief Integer power function.
       *
       * Used because one sums from i = 0 .. D^rank.
       * \param i The base of the power.
       * \param j The exponent.
       * \retval k = i^j
       */
      int ipow(int i, int j) const;

      /**
       * \brief The types of the tensor indexes.
       */
      IndexType* types;

      /**
       * \brief The components of the tensor.
       */
      double* components;

      /**
       * \brief The rank of the tensor.
       */
      int rank;

  };

};

#endif
