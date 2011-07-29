// Copyright Aaryn Tonita, 2011
// Distributed under the Gnu general public license
#include <cstdlib>
#include <cassert>
#include "Tensor.h"
#define DIMENSION 4

int Tensor::ipow(int i, int j) const {
  int retValue = 1;
  for (int k = 0; k < j; k++) {
    retValue *= i;
  }
  return retValue;
}

Tensor::Tensor(const char* indexString) {
  // Determine rank.
  rank = -1;
  for (int i = 0; i < 33 && rank < 0; i++) { 
    if (indexString[i] == '\0') {
      rank = i/2;
    }
  }
  assert(rank > 0);

  // Initialise.
  types = new IndexType[rank];
  indexes = new char[rank];
  components = new double[ipow(DIMENSION, rank)];
  for (int i = 0; i < ipow(DIMENSION, rank); i++) components[i] = 0;

  // Determine index type and label.
  for (int i = 0; i < rank; i++) {
    if (indexString[2*i] == '^') {
      types[i] = UP;
      indexes[i] = indexString[2*i + 1];
    } else if (indexString[2*i] == '_') {
      types[i] = DOWN;
      indexes[i] = indexString[2*i + 1];
    } else {
      assert(false);
    }
  }
}

Tensor::Tensor(int Rank, ...) {
  if (Rank == 0) {
    init(Rank,NULL);
  } else {
    IndexType Types[Rank];
    va_list listPointer;
    va_start(listPointer, Rank);
    for (int i = 0; i < Rank; i++) {
      Types[i] = (IndexType)va_arg(listPointer, int);
    }
    va_end(listPointer);
    init(Rank, Types);
  }
}

Tensor::Tensor(int Rank, const IndexType* Types) {
  init(Rank,Types);
}

void Tensor::init(int Rank, const IndexType* Types) {
  rank = Rank;
  if (rank > 0) {
    types = new IndexType[rank];
    components = new double[ipow(DIMENSION,rank)];
    indexes = new char[rank];
  } else {
    types = new IndexType[1]; // To avoid double free...
    components = new double[1];
    indexes = new char[rank];
  }
  for (int i = 0; i < rank; i++) {
    types[i] = Types[i];
    indexes[i] = 0;
  }
  for (int i = 0; i < ipow(DIMENSION,rank); i++) components[i] = 0.0;
}

Tensor::Tensor(int Rank, const IndexType* Types, const char* Indexes) {
  init(Rank,Types);
  for (int i = 0; i < rank; i++) {
    indexes[i] = Indexes[i];
  }
}

Tensor::Tensor(const Tensor &original) {
  rank = original.rank;
  if (rank > 0) {
    types = new IndexType[rank];
    components = new double[ipow(DIMENSION,rank)];
    indexes = new char[rank];
  } else {
    types = new IndexType[1]; // To avoid double free...
    components = new double[1];
    indexes = new char[1];
  }
  for (int i = 0; i < rank; i++) {
    types[i] = original.types[i];
    indexes[i] = original.indexes[i];
  }
  for (int i = 0; i < ipow(DIMENSION,rank); i++) {
    components[i] = original.components[i];
  }
}

int Tensor::getRank() const {
  return rank;
}

const Tensor::IndexType* Tensor::getTypes() const {
  return types;
}

double & Tensor::operator()(int* indices) const {
  return components[index(indices)];
}

double & Tensor::operator()(int i1, ...) const {
  if (rank == 0) {
    return components[0];
  } else {
    int indices[rank];
    indices[0] = i1;
    va_list listPointer;
    va_start(listPointer, i1);
    for (int i = 1; i < rank; i++) {
      indices[i] = va_arg(listPointer, int);
    }
    va_end(listPointer);
    return components[index(indices)];
  }
}

double * Tensor::getComponents() const {
  return components;
}

int Tensor::index(int i1, ...) const {
  if (rank == 0) {
    return 0;
  } else {
    int indices[rank];
    indices[0] = i1;
    va_list listPointer;
    va_start(listPointer, i1);
    for (int i = 1; i < rank; i++) {
      indices[i] = va_arg(listPointer, int);
    }
    va_end(listPointer);
    return index(indices);
  }
}

int Tensor::index(int* indices) const {
  int index = 0;
  int factor = 1;
  for (int j = rank-1; j >= 0; j--) {
    index += factor*indices[j];
    factor *= DIMENSION;
  }
  return index;
}

Tensor::~Tensor() {
  delete[] types;
  delete[] components;
}

void Tensor::indexToIndices(int index, int* indices) const {
  for (int i = rank-1; i >= 0; i--) {
    indices[i] = index%DIMENSION;
    index /= DIMENSION;
  }
}

Tensor Tensor::contract(int index1, int index2) const {
  // Build the result type.
  assert(types[index1] != types[index2]);
  // Won't use the last two, but need to allocate more than 0...
  Tensor::IndexType resultTypes[rank];
  char resultIndexes[rank];
  int runningIndex = 0;
  for (int i = 0; i < rank; i++) {
    if (i != index1 && i != index2) {
      resultTypes[runningIndex] = types[i];
      resultIndexes[runningIndex++] = indexes[i];
    }
  }
  Tensor result(rank-2, resultTypes);
  for (int i = 0; i < rank-2; i++) {
    result.indexes[i] = resultIndexes[i];
  }

  int indices[rank];
  int resultIndices[result.getRank()];
  for (int i = 0; i < ipow(DIMENSION, result.getRank()); i++) {
    result.indexToIndices(i,resultIndices);
    runningIndex = 0;
    // The free indices...
    for (int j = 0; j < rank; j++) {
      if (j != index1 && j!= index2) {
        indices[j] = resultIndices[runningIndex];
        runningIndex++;
      }
    }

    // Sum over the contracting indices.
    double value = 0.0;
    for (int j = 0; j < DIMENSION; j++) {
      indices[index1] = j;
      indices[index2] = j;
      value += components[index(indices)];
    }
    result.components[i] = value;
  }
  return result;
}

Tensor & Tensor::operator*=(const double scalar) {
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    components[i] *= scalar;
  }
  return *this;
}

Tensor & Tensor::operator+=(const Tensor &tensor) {
  // Check indexing.
  int permute[rank];
  int indices[rank];
  int permutedIndices[rank];
  bool permutable = permutation(tensor.indexes, permute);
  assert(permutable);

  // Check to make sure permuted indexes are equal.
  const Tensor::IndexType* tensorTypes = tensor.getTypes();
  for (int i = 0; i < rank; i++) {
    assert(types[i] == tensorTypes[permute[i]]);
  }

  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    for (int j = 0; j < rank; j++) permutedIndices[j] = indices[permute[j]];
    components[i] += tensor(permutedIndices);
  }
  return *this;
}

Tensor Tensor::operator*(const double scalar) const {
  Tensor result = *this;
  result *= scalar;
  return result;
}

Tensor Tensor::operator+(const Tensor &tensor) const {
  Tensor result = *this;
  result += tensor;
  return result;
}

Tensor Tensor::operator[](const char* names) {
  for (int i = 0; i < rank; i++) {
    indexes[i] = names[i];
    assert(names[i] != '\0'); // Ensure that there are enough indices.
  }
  Tensor copy = *this;
  return copy;
}

Tensor Tensor::contract() const {
  for (int i = 0; i < rank; i++) {
    char index = indexes[i];
    if (index != 0 && index != '.' && index != '-') {
      for (int j = i+1; j < rank; j++) {
        if (index == indexes[j]) {
          return contract(i,j).contract();
        }
      }
    }
  }
  Tensor copy = *this;
  return copy;
}

Tensor Tensor::operator*(const Tensor& tensor) const {
  // tensoruild the result type...
  IndexType resultTypes[rank + tensor.getRank()];
  char resultIndexes[rank + tensor.getRank()];
  const Tensor::IndexType* bTypes = tensor.getTypes();
  for (int i = 0; i < rank; i++) {
    resultTypes[i] = types[i];
    resultIndexes[i] = indexes[i];
  }
  for (int i = 0; i < tensor.getRank(); i++) {
    resultTypes[rank+i] = bTypes[i];
    resultIndexes[rank+i] = tensor.indexes[i];
  }
  Tensor result(rank+tensor.getRank(), resultTypes, resultIndexes);

  int resultIndices[result.getRank()];
  int indices[rank];
  int tensorIndices[tensor.getRank()];
  for (int i = 0; i < ipow(DIMENSION, result.getRank()); i++) {
    result.indexToIndices(i, resultIndices);
    for (int j = 0; j < rank; j++) indices[j] = resultIndices[j];
    for (int j = 0; j < tensor.getRank(); j++) tensorIndices[j] = resultIndices[rank+j];
    double value = components[index(indices)]
      *tensor.components[tensor.index(tensorIndices)];
    result.components[i] = value;
  }
  return result.contract();
}

bool Tensor::permutation(char* indexes2, int* permute) const {
  for (int i = 0; i < rank; i++) {
    bool indexFound = false;
    assert(indexes[i]);
    for (int j = 0; !indexFound && indexes2[j] != '\0' ; j++) {
      if (indexes[i] == indexes2[j]) {
        permute[i] = j;
        indexFound = true;
      }
    }
    if (!indexFound) return false;
  }
  return true;
}

Tensor & Tensor::operator=(const Tensor & tensor) {
  if (this == &tensor) return *this;
  assert(rank == tensor.getRank());
  int permute[rank];
  bool permutable = permutation(tensor.indexes, permute);
  assert(permutable);
  const Tensor::IndexType* tensorTypes = tensor.getTypes();
  for (int i = 0; i < rank; i++) {
    assert(types[i] == tensorTypes[permute[i]]);
  }

  // Now assign components.
  int indices[rank], permutedIndices[rank];
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    for (int j = 0; j < rank; j++) permutedIndices[j] = indices[permute[j]];
    components[i] = tensor(permutedIndices);
  }
  return *this;
}

int Tensor::setComponents(const double* v)
{
  int i;
  for (i = 0; i < ipow(DIMENSION, rank); i++) {
    components[i] = v[i];
  }
  return i+1;
}

int Tensor::getComponents(double* v)
{
  int i;
  for (i = 0; i < ipow(DIMENSION, rank); i++) {
    v[i] = components[i];
  }
  return i+1;
}
