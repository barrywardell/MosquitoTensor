#ifndef INDEXEDTENSOR_H_
#define INDEXEDTENSOR_H_

#include "TensorBase.h"

namespace Mosquito {

  /**
   * \brief An indexed tensor, for computational purposes.
   *
   * This class should never be instantiated directly. Instead one
   * should use the Tensor class, and create this object via indexing
   * and then only as part of a computation.
   *
   * Uses binary tree computation to delay calculations and save memory.
   * Indexing of a computed type works left to right. For instance:
   * \f[
   *  A_{ab} + B_{ba} \rightarrow C_{ab}
   * \f]
   * where C is the derived type. For multiplications with contractions,
   * the same principle applies
   * \f[
   *  A_{acb}B^{c}{}_d \rightarrow C_{abd}
   * \f]
   * However, assignment overrides this implementation detail in a
   * sensible fashion:
   * \f[
   * A^{a}_{b} = \sigma^c{}_b\sigma^a{}_c
   * \f]
   * and in fact one is only able to assign to a variable by defining
   * the indexing chosen.
   */
  class IndexedTensor : public TensorBase {
    public:
      /**
       * \brief Constructs a IndexedTensor, possibly contracting.
       * \param Rank The rank of the tensor.
       * \param Types The IndexType's of the indices.
       * \param Components A pointer to the components.
       * \param Labels The labels for the indices.
       */
      IndexedTensor(int Rank, IndexType* Types, double* Components,
          char* Labels);

      /**
       * \brief Assignment.
       *
       * The assignment operator, overwrites the data in components. By
       * doing so, it alters the data of the Tensor object that created
       * this IndexedTensor.
       * \param tensor The indexed tensor to assign to this one.
       * \retval *this Although that is somewhat useless.
       */
      IndexedTensor &operator=(const IndexedTensor &tensor);

      /**
       * \brief Destructor. Might need to delete some leaves.
       */
      ~IndexedTensor();

      /**
       * \brief Computes the indexed component.
       * 
       * When this indexed tensor has type TENSOR then the component is
       * simply returned. This is where all of the computations are
       * performed, via a recursive binary tree evaluation.
       * \param indices The indices of the component to compute.
       * \retval component The computed component.
       */
      double computeComponent(const int *indices) const;

      /**
       * \brief Scalar multiplication.
       * \param scalar The scalar to multiply by.
       * \retval The product (*this)*scalar.
       */
      IndexedTensor operator*(const double scalar) const;

      /**
       * \brief Scalar multiplication.
       * \param scalar The scalar to multiply by.
       * \param tensor The tensor to multiply.
       * \retval result The product scalar*tensor.
       */
      friend IndexedTensor operator*(const double scalar, 
          const IndexedTensor &tensor) {return tensor*scalar;};

      /**
       * \brief Addition of tensors.
       * \param tensor The tensor to add to this one.
       * \retval result The sum of the two tensors.
       */
      IndexedTensor operator+(const IndexedTensor &tensor) const;

      /**
       * \brief Tensor subtraction.
       * \param tensor The tensor to subtract from this one.
       * \retval result The resut (*this)-tensor.
       */
      IndexedTensor operator-(const IndexedTensor &tensor) const 
      { return operator+(tensor*(-1));};

      /**
       * \brief Tensor multiplication.
       *
       * May also perform contractions across tensors.
       * \param tensor The tensor to multiply by this.
       * \retval The product (*this)*tensor.
       */
      IndexedTensor operator*(const IndexedTensor &tensor) const;

    private:
      /**
       * \brief Defines whether this is an actual tensor or a node in
       * the operation tree.
       */
      enum TensorType {
        MULTIPLICATION = 0,       /**< This is a multiplication node. */
        CONTRACTION = 1,          /**< This is a contraction node. */
        ADDITION = 2,             /**< This is an addition node. */
        TENSOR = 3,               /**< This is a leaf: actual data. */
        SCALARMULTIPLICATION = 4  /**< Scalar multiplication. */
      };

      /**
       * \brief Labels for the indices.
       */
      char* labels;

      /**
       * \brief The type of this IndexedTensor.
       */
      TensorType indexedType;

      /**
       * \brief The left tensor in the operation tree.
       */
      const IndexedTensor *left;

      /**
       * \brief The right tensor in the operation tree.
       */
      const IndexedTensor *right;

      /**
       * \brief Special storage for contraction type nodes, so as not to throw
       * this information away.
       */
      int leftContractionIndex;

      /**
       * \brief Special storage for contraction type nodes, so as not to throw
       * this information away.
       */
      int rightContractionIndex;

      /**
       * \brief A scalar to multiply by.
       */
      double multiplicand;

      /**
       * \brief Returns the permutation vector which defines how to
       * rearrange indices.
       *
       * Used so that one may properly perform Ricci calculus. In
       * particular one wants to be able to perform operations like:
       * \f[
       *  h_{ab} = T^a{}_b+U_b{}^a
       * \f]
       * \param labels2 The indexes of the other (second) tensor.
       * \param permute The vector to store the permutation in.
       * \retval permutable A boolean which states whether labels2 is a
       * permutation of this tensor's indexes or not.
       */
      bool permutation(const char* labels2, int* permute) const;

      /**
       * \brief Builds the branch of contractions.
       *
       * Assumes that this tensor has complete type information (labels,
       * types and rank) and builds the nodes above it recursively.
       * \param index1 The first index to contract on.
       * \param index2 The second index to contract on.
       * \param contractionsNeeded Used to flag termination.
       */
      IndexedTensor *contract(int index1, int index2, int contractionsNeeded);

      /**
       * \brief Constructs a blank IndexedTensor.
       *
       * Used internally so that branches can be constructed.
       */
      IndexedTensor() {};
  };
};

#endif
