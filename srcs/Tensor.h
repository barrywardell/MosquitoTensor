// Copyright Aaryn Tonita, 2011
// Distributed under the Gnu general public license
#ifndef TENSOR_H_
#define TENSOR_H_

#include <cstdarg>

#include "TensorBase.h"
#include "IndexedTensor.h"

namespace Mosquito {

  /**
   * \brief The class Tensor encapsulates the components of a tensor at a
   * given point (not the entire manifold) and also the logic necessary to
   * perform algebra with Tensors. 
   *
   * The dimension is assumed to be 4.
   * Indexing of components and indices begin at zero. So for Z^a_b, a is
   * the 0th index and runs from 0-3.
   */
  class Tensor : public TensorBase {
    public:

      /**
       * \brief Constructor.
       *
       * Creates the point tensor. Sets the rank, creates storage, zeros
       * the components and sets the contravariant/covariant type of the
       * indices.
       * \param Rank The rank of the tensor.
       * \param ... The types of the indices.
       */
      Tensor(int Rank = 0, ...);

      /**
       * \brief Constructor.
       *
       * Creates the point tensor. Sets the rank, creates storage, zeros
       * the components and sets the contravariant/covariant type of the
       * indices.
       * \param Rank The rank of the tensor.
       * \param Types The types of the indices.
       */
      Tensor(int Rank, const IndexType* Types);

      /**
       * \brief Constructor from character array.
       * Sets the rank, type of index, and name of index at construction
       * time. Index string must be of form "^a^b_c^d", that is, it must
       * be one of "^" or "_" followed by another character. These
       * characters should be unique, so that a contraction is not
       * performed over the initialised tensor. See contract() for
       * information about indexes.
       * If the data pointer is not given or is null then a new array of the
       * appropriate size is created and initialized.
       * \param indexString The character array defining the tensor type.
       * \param data Pointer to an array where the components are stored
       */
      Tensor(const char* indexString, double *data = 0);

      /**
       * Copy constructor.
       * \param original The original Tensor, to copy.
       */
      Tensor(const Tensor &original);

      /**
       * Copy constructor from IndexedTensor.
       * \param original The original Tensor, to copy.
       */
      Tensor(const IndexedTensor &original);

      /**
       * \brief Contracts this tensor over chosen indices.
       *
       * Contract tensor index1 with tensor index2. Computes a trace.
       * That is result = A^[i1..index1..in]_[j1..index2..jm]
       * \param index1 The first index to contract with.
       * \param index2 The second index to contract with.
       * \retval result The tensor to store the result in.
       */
      Tensor contract(int index1, int index2) const;

      /**
       * \brief Names the indices, creating an IndexedTensor.
       *
       * Initial values are null.
       * \param names The list of indices.
       * \retval this The indexed tensor, possibly with indexes contracted.
       */
      IndexedTensor operator[](const char* names);

      /**
       * \brief Destructor.
       */
      virtual ~Tensor();

      /**
       * \brief Scalar inplace multiplication.
       *
       * \param scalar The scalar to multiply by.
       * \retval this A reference to this tensor.
       */
      Tensor & operator*=(const double scalar);

      /**
       * \brief Scalar inplace division.
       *
       * \param scalar The scalar to divide by.
       * \retval this A reference to this tensor.
       */
      Tensor & operator/=(const double scalar) {
        (*this) *= 1./scalar;
        return *this;
      }

      /**
       * \brief Assignment.
       * Only works on tensors of the same type, when the index types
       * are ordered indentically.
       * \param tensor The tensor to set this one to.
       * \retval *this A reference to this tensor.
       */
      Tensor &operator=(const Tensor &tensor);

      /**
       * \brief Scalar multiplication.
       *
       * \param scalar The scalar to multiply by.
       * \retval A new tensor which is equal to this one multiplied by
       * scalar.
       */
      Tensor operator*(const double scalar) const;

      /**
       * \brief Tensor multiplication.
       *
       * After multiplication, contractions are performed (if any).
       * \param tensor The tensor to multiply by.
       * \retval A new tensor which is equal to this one multiplied by
       * the supplied tensor.
       */
      Tensor operator*(const Tensor& tensor) const;

      /**
       * \brief Division.
       * \param scalar The scalar to divide by.
       * \retval A new tensor equal to this one divided by the argument.
       */
      Tensor operator/(double scalar) const {
        return operator*(1./scalar);
      };

      /**
       * \brief Scalar multiplication is commutative.
       * \param scalar The scalar to multiply by.
       * \param tensor The tensor to be multiplied.
       * \retval result The tensor multiplied by the scalar.
       */
      friend Tensor operator*(const double scalar, 
          const Tensor &tensor) { return tensor*scalar;};

    private:

      /**
       * \brief Allocates storage.
       *
       * Used by the constructors to allocate memory.
       * \param Rank The rank of the tensor.
       * \param Types The types of the indices.
       */
      void init(int Rank, const IndexType* Types);

      /**
       * \brief Whether to delete components array in destructor
       *
       * The components array may be either allocated when the object is created
       * or it may be passed as an argument to the constructor. When the
       * destructor is called, this array should only be deleted in the former
       * case. This flags whether the deletion should happen or not
       */
      bool deleteComponents;
  };
};

#endif
