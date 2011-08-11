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
      Tensor(int Rank, ...);

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
       * \param indexString The character array defining the tensor type.
       */
      Tensor(const char* indexString);

      /**
       * Copy constructor.
       * \param original The original Tensor, to copy.
       */
      Tensor(const Tensor &original);

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
       * \brief Names the indices.
       *
       * Indexing operation. Sets the abstract indexes. Indexes which are
       * null or '.' or '-' are not summed over.
       * Identical indexes are contracted so long as one is contravariant
       * and the other is covariant. If one has delta^a_b
       * and then calls delta(a,a), then contraction will be performed
       * while delta^{ab} with delta['aa'] is inconsistent and will assert
       * and abort.
       *
       * Initial values are null.
       * \param i1 The first index or NULL, to remove all indices.
       * \param ... The va_list of other indexes.
       * \retval this The indexed tensor, possibly with indexes contracted.
       */
      IndexedTensor it(char* names);

      /**
       * \brief Names the indices.
       *
       * Indexing operation. Sets the abstract indexes. Indexes which are
       * null or '.' or '-' are not summed over.
       * Identical indexes are contracted so long as one is contravariant
       * and the other is covariant. If one has delta^a_b
       * and then calls delta(a,a), then contraction will be performed
       * while delta^{ab} with delta['aa'] is inconsistent and will assert
       * and abort.
       *
       * Initial values are null.
       * \param i1 The first index or NULL, to remove all indices.
       * \param ... The va_list of other indexes.
       * \retval this The indexed tensor, possibly with indexes contracted.
       */
      Tensor operator[](char* names);

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
       * \brief Inplace addition.
       *
       * \param tensor The tensor to add to this tensor.
       * \retval this A reference to this tensor after the addtion.
       */
      Tensor & operator+=(const Tensor &tensor);

      /**
       * \brief Inplace subtraction.
       *
       * \param tensor The tensor to subtract from this tensor.
       * \retval this A reference to this tensor after the subtraction.
       */
      Tensor & operator-=(const Tensor &tensor) {
        (*this) += (-1)*tensor;
        return *this;
      }

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
       * \brief Addition.
       * \param tensor The tensor to add to this one.
       * \retval A new tensor equal to this one with the argument added to
       * it.
       */
      Tensor operator+(const Tensor &tensor) const;

      /**
       * \brief Subtraction.
       * \param tensor The tensor to subtract from this one.
       * \retval A new tensor equal to this one with the argument
       * subtracted.
       */
      Tensor operator-(const Tensor &tensor) const {
        return operator+((-1)*tensor);
      };

      /**
       * \brief Division.
       * \param scalar The scalar to divide by.
       * \retval A new tensor equal to this one divided by the argument.
       */
      Tensor operator/(double scalar) const {
        return operator*(1./scalar);
      };

      /**
       * \brief Assignment operator.
       * Tensor to be assigned must have same rank and permutable indices.
       * The index types must be the same.
       * \param tensor The tensor to assign to the one to the left hand
       * side.
       * \retval lhs A reference to the tensor on the left hand side.
       */
      Tensor & operator=(const Tensor &tensor);

      /**
       * \brief Scalar multiplication is commutative.
       * \param scalar The scalar to multiply by.
       * \param tensor The tensor to be multiplied.
       * \retval result The tensor multiplied by the scalar.
       */
      friend Tensor operator*(const double scalar, 
          const Tensor &tensor) { return tensor*scalar;};

    protected:

      /**
       * \brief Contracts.
       *
       * If there are indices which should be contracted, it contracts
       * them and returns the result.
       */
      Tensor contract() const;

      /**
       * \brief The indices, for summation.
       */
      char* indexes;

      /**
       * \brief Constructor which names indices.
       *
       * Creates the point tensor. Sets the rank, creates storage, zeros
       * the components and sets the contravariant/covariant type of the
       * indices.
       * \param Rank The rank of the tensor.
       * \param Types The types of the indices.
       * \param Indexes The indexes used for the tensor.
       */
      Tensor(int Rank, const IndexType* Types, const char* Indexes);

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
       * \brief Returns the permutation vector which defines how to
       * rearrange indices.
       *
       * Used so that one may properly perform Ricci calculus. In
       * particular one wants to be able to perform operations like:
       * \f[
       *  h_{ab} = T^a{}_b+U_b{}^a
       * \f]
       * \param indexes2 The indexes of the other (second) tensor.
       * \param permute The vector to store the permutation in.
       * \retval permutable A boolean which states whether indexes2 is a
       * permutation of this tensor's indexes or not.
       */
      bool permutation(char* indexes2, int* permute) const;
  };
}

#endif
