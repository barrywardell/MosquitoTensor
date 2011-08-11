#include "IndexedTensor.h"
#include <cassert>

#define DIMENSION 4

using namespace Mosquito;

IndexedTensor::~IndexedTensor() {
  if (indexedType == CONTRACTION) {
    delete left;
    delete[] types;
    delete[] labels;
  }
}

IndexedTensor::IndexedTensor(int Rank, IndexType* Types, 
    double* Components, char* Labels) {
  // Determine if we need to contract.
  int contractionsNeeded = 0;
  int index2, index1 = -1;
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
    leaf->labels = Labels;
    leaf->rank = Rank;
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
    labels = Labels;
    rank = Rank;
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
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    int indices[rank];
    indexToIndices(i, indices);
    int permutedIndices[rank];
    for (int j = 0; j < rank; j++) {
      permutedIndices[j] = indices[permute[j]];
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
    permutation(right->labels, permute);
    int permutedIndices[rank];
    for (int i = 0; i < rank; i++) {
      permutedIndices[i] = indices[permute[i]];
    }
    return  left ->computeComponent(indices) +
            right->computeComponent(permutedIndices);

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
  }
  return 0;
}

bool IndexedTensor::permutation(const char* labels2, int* permute) const {
  for (int i = 0; i < rank; i++) {
    bool indexFound = false;
    assert(labels[i]);
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
  // By assumption this IndexedTensor is completely setup.
  IndexedTensor *node = new IndexedTensor();
  node->indexedType = CONTRACTION;
  node->left = this;
  node->types = new IndexType[rank-2];
  node->labels = new char[rank-2];

  // Build node's labels and types.
  int runningIndex = 0;
  for (int i = 0; i < rank; i++) {
    if (i != index1 && i != index2) {
      node->types[runningIndex] = types[i];
      node->labels[runningIndex++] = labels[i];
    }
  }

  // Determine indexes if another contraction is needed.
  if (contractionsNeeded > 1) {
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
