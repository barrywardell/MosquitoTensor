#include "IndexedTensor.h"
#include <cstdlib>
#include <cassert>

#define DIMENSION 4

using namespace Mosquito;

IndexedTensor::~IndexedTensor() {
  delete[] labels;
  if (indexedType == CONTRACTION) {
    delete left;
    delete[] types;
  } else if (indexedType == MULTIPLICATION) {
    delete[] types;
  }
}

IndexedTensor::IndexedTensor(const IndexedTensor &tensor) {
  rank = tensor.rank;
  indexedType = tensor.indexedType;
  labels = copyLabels(tensor.labels);
  if (indexedType == CONTRACTION || indexedType == MULTIPLICATION) {
    types = new IndexType[rank];
    for (int i = 0; i < rank; i++) {
      types[i] = tensor.types[i];
    }
  } else {
    types = tensor.types;
  }
  if (indexedType == CONTRACTION) {
    left = new IndexedTensor(*tensor.left);
  } else {
    left = tensor.left;
  }
  right = tensor.right;
  multiplicand = tensor.multiplicand;
  leftContractionIndex = tensor.leftContractionIndex;
  rightContractionIndex = tensor.rightContractionIndex;
  components = tensor.components;
}

IndexedTensor::IndexedTensor() {
  nullify();
}

void IndexedTensor::nullify() {
  rank = -1;
  rightContractionIndex = -1;
  leftContractionIndex = -1;
  left = NULL;
  right = NULL;
  components = NULL;
}

IndexedTensor::IndexedTensor(int Rank, IndexType* Types, 
    double* Components, const char* Labels) {
  nullify();
  // Determine if we need to contract.
  int contractionsNeeded = 0;
  int index2 = -1, index1 = -1;
  for (int i = 0; i < Rank; i++) {
    int found = 0; // Check that index appears once or twice only.
    for (int j = i+1; j < Rank; j++) {
      if (Labels[i] == Labels[j]) {
        found += 1;
        assert(Types[i] != Types[j]); // Contractions must be over up-down.
        contractionsNeeded++;
        if (index1 == -1) {
          index1 = i;
          index2 = j;
        }
      }
    }
    assert(found == 0 || found == 1);
  }

  // Make this the required type.
  if (contractionsNeeded > 0) {
    // First build the leaf and recursively the branch...
    IndexedTensor *leaf = new IndexedTensor();
    leaf->components = Components;
    leaf->types = Types;
    leaf->rank = Rank;
    leaf->labels = leaf->copyLabels(Labels);
    leaf->indexedType = TENSOR;
    left = leaf->contract(index1, index2, contractionsNeeded);

    // construct labels and types to finish building this tensor
    rank = Rank - 2*contractionsNeeded;
    indexedType = CONTRACTION;
    labels = new char[rank];
    types = new IndexType[rank];
    int runningIndex = 0;
    leftContractionIndex = -1;
    for (int i = 0; i < rank+2; i++) {
      for (int j = i+1; j < rank+2; j++) {
        if (left->labels[i] == left->labels[j]) {
          assert(left->types[i] != left->types[j]);
          leftContractionIndex = i;
          rightContractionIndex = j;
        }
      }
      if (i != leftContractionIndex &&
          i != rightContractionIndex) {
        labels[runningIndex] = left->labels[i];
        types[runningIndex++] = left->types[i];
      }
    }
  } else {
    rank = Rank;
    labels = copyLabels(Labels);
    types = Types;
    components = Components;
    indexedType = TENSOR;
  }
}

IndexedTensor &IndexedTensor::operator=(const IndexedTensor &tensor) {
  assert(indexedType == TENSOR);
  if (rank == 0) {
    components[0] = tensor.computeComponent(0);
    return *this;
  }
  int permute[rank];
  bool permutable = permutation(tensor.labels, permute);
  assert(permutable);
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    int indices[rank];
    indexToIndices(i, indices);
    int permutedIndices[rank];
    for (int j = 0; j < rank; j++) {
      permutedIndices[permute[j]] = indices[j];
    }

    components[i] = tensor.computeComponent(permutedIndices);
  }
  return *this;
}

double IndexedTensor::computeComponent(const int* indices) const {
  if (indexedType == TENSOR) {
    return components[index(indices)];
  } else if (indexedType == ADDITION) {
    // Indexing is left prioritizing... so this's labels is left's labels
    // TODO: Make a function to get permuted indices directly?
    int permute[rank];
    bool permutable = permutation(right->labels, permute);
    assert(permutable);
    int permutedIndices[rank];
    for (int i = 0; i < rank; i++) {
      permutedIndices[permute[i]] = indices[i];
    }
    return  left ->computeComponent(indices) +
            multiplicand*right->computeComponent(permutedIndices);

  } else if (indexedType == MULTIPLICATION) {
    return left->computeComponent(indices)* // Only uses the first few
           right->computeComponent(&indices[left->rank]);
  } else if (indexedType == CONTRACTION) {
    // Build the constant indices;
    int indicesLeft[rank + 2];
    int runningIndex = 0;
    for (int i = 0; i < rank + 2; i++) {
      if (i != leftContractionIndex &&
          i != rightContractionIndex) {
        indicesLeft[i] = indices[runningIndex];
        runningIndex++;
      }
    }
    // Perform the contraction.
    double value = 0;
    for (int i = 0; i < DIMENSION; i++) {
      indicesLeft[leftContractionIndex] = i;
      indicesLeft[rightContractionIndex] = i;
      value += left->computeComponent(indicesLeft);
    }
    return value;
  } else if (indexedType == SCALARMULTIPLICATION) {
    return multiplicand*left->computeComponent(indices);
  }
  return 0;
}

bool IndexedTensor::permutation(const char* labels2, int* permute) const {
  for (int i = 0; i < rank; i++) {
    bool indexFound = false;
    assert(labels[i]); // No NULL labels!
    for (int j = 0; !indexFound && labels2[j] != '\0' ; j++) {
      if (labels[i] == labels2[j]) {
        permute[i] = j;
        indexFound = true;
      }
    }
    if (!indexFound) return false;
  }
  return true;
}

IndexedTensor * IndexedTensor::contract(int index1, int index2,
    int contractionsNeeded) {
  if (contractionsNeeded > 1) {
    // By assumption this IndexedTensor is completely setup and is 
    // a multiplication or a tensor or a contraction.
    IndexedTensor *node = new IndexedTensor();
    node->rank = rank - 2;
    node->indexedType = CONTRACTION;
    node->left = this;
    node->types = new IndexType[rank-2];
    node->labels = new char[rank-2];
    node->leftContractionIndex = index1;
    node->rightContractionIndex = index2;

    // Build node's labels and types.
    int runningIndex = 0;
    for (int i = 0; i < rank; i++) {
      if (i != index1 && i != index2) {
        node->types[runningIndex] = types[i];
        node->labels[runningIndex++] = labels[i];
      }
    }

    // Determine indexes if another contraction is needed.
    for (int i = 0; i < rank-2; i++) {
      for (int j = i+1; j < rank-2; j++) {
        if (node->labels[i] == node->labels[j]) {
          assert(node->types[i] != node->types[j]);
          index1 = i;
          index2 = j;
        }
      }
    }
    return node->contract(index1, index2, contractionsNeeded - 1);
  }
  return this;
}

IndexedTensor IndexedTensor::operator*(const double scalar) const {
  IndexedTensor result;
  result.indexedType = SCALARMULTIPLICATION;
  result.types = types;
  result.rank = rank;
  result.labels = result.copyLabels(labels);
  result.left = this;
  result.multiplicand = scalar;
  return result;
}

IndexedTensor IndexedTensor::operator+(const IndexedTensor &tensor) const {
  return arithmetic(tensor, 1);
}

IndexedTensor IndexedTensor::operator-(const IndexedTensor &tensor) const {
  return arithmetic(tensor, -1);
}

IndexedTensor IndexedTensor::arithmetic(const IndexedTensor &tensor,
    int sign) const {
  // Ensure consistency of addition.
  assert(rank == tensor.getRank());
  int permute[rank];
  bool permutable = permutation(tensor.labels, permute);
  assert(permutable);
  for (int i = 0; i < rank; i++) {
    assert(types[i] == tensor.types[permute[i]]);
  }

  // Set up addition node.
  IndexedTensor result;
  result.indexedType = ADDITION;
  result.types = types;
  result.rank = rank;
  result.labels = result.copyLabels(labels);
  result.left = this;
  result.right = &tensor;
  result.multiplicand = sign;
  return result;
}

IndexedTensor IndexedTensor::operator*(const IndexedTensor &tensor) const {
  // Build product data.
  int prodRank = rank + tensor.getRank();
  char *prodLabels = new char[prodRank];
  IndexType *prodTypes = new IndexType[prodRank];
  for (int i = 0; i < rank; i++) {
    prodLabels[i] = labels[i];
    prodTypes[i] = types[i];
  }
  for (int i = 0; i < tensor.getRank(); i++) {
    prodLabels[rank + i] = tensor.labels[i];
    prodTypes[rank + i] = tensor.types[i];
  }

  // Determine if contractions are needed.
  int index1 = -1, contractionsNeeded = 0, index2 = -1;
  for (int i = 0; i < prodRank; i++) {
    int found = 0;
    for (int j = i + 1; j < prodRank; j++) {
      if (prodLabels[i] == prodLabels[j]) {
        found++;
        assert(prodTypes[i] != prodTypes[j]);
        if (index1 == -1) {
          index1 = i;
          index2 = j;
        }
        contractionsNeeded++;
      }
    }
    assert(found == 0 || found == 1);
  }

  // Build product.
  IndexedTensor *product = new IndexedTensor();
  product->rank = prodRank;
  product->indexedType = MULTIPLICATION;
  product->left = this;
  product->right = &tensor;
  product->types = prodTypes;
  product->labels = prodLabels;

  // Determine what type to return:
  if (contractionsNeeded == 0) {
    return *product;
  } else {
    IndexedTensor *result = new IndexedTensor();
    result->indexedType = CONTRACTION;
    result->rank = prodRank - 2*contractionsNeeded;
    result->left = product->contract(index1, index2, contractionsNeeded);
    result->labels = new char[result->rank];
    result->types = new IndexType[result->rank];
    result->leftContractionIndex = -1;
    int runningIndex = 0;
    for (int i = 0; i < result->rank + 2; i++) {
      for (int j = i + 1; j < result->rank + 2; j++) {
        if (result->left->labels[i] == result->left->labels[j]) {
          result->leftContractionIndex = i;
          result->rightContractionIndex = j;
        }
      }
      if (i != result->leftContractionIndex &&
          i != result->rightContractionIndex) {
        result->labels[runningIndex] = result->left->labels[i];
        result->types[runningIndex++] = result->left->types[i];
      }
    }
    return *result;
  }
}

char* IndexedTensor::copyLabels(const char* Labels) const {
  assert(rank >= 0);
  char *copy = new char[rank];
  for (int i = 0; i < rank; i++) {
    copy[i] = Labels[i];
  }
  return copy;
}
