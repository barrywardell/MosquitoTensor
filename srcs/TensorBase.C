#include "TensorBase.h"
#include <cstdarg>
#include <cassert>

#define DIMENSION 4

using namespace Mosquito;

double & TensorBase::operator()(int* indices) const {
  return components[index(indices)];
}

double & TensorBase::operator()() const {
  assert(rank == 0);
  return components[0];
}

double & TensorBase::operator()(int i1, ...) const {
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

int TensorBase::index(int i1, ...) const {
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

int TensorBase::index(const int* indices) const {
  int index = 0;
  int factor = 1;
  for (int j = rank-1; j >= 0; j--) {
    index += factor*indices[j];
    factor *= DIMENSION;
  }
  return index;
}

void TensorBase::indexToIndices(int index, int* indices) const {
  for (int i = rank-1; i >= 0; i--) {
    indices[i] = index%DIMENSION;
    index /= DIMENSION;
  }
}

int TensorBase::setComponents(const double* v)
{
  int i;
  for (i = 0; i < ipow(DIMENSION, rank); i++) {
    components[i] = v[i];
  }
  return i+1;
}

int TensorBase::getComponents(double* v) const
{
  int i;
  for (i = 0; i < ipow(DIMENSION, rank); i++) {
    v[i] = components[i];
  }
  return i+1;
}

double * TensorBase::getComponents() const {
  return components;
}

int TensorBase::getNumComponents() const
{
  return ipow(DIMENSION, rank);
}

int TensorBase::ipow(int i, int j) const {
  int retValue = 1;
  for (int k = 0; k < j; k++) {
    retValue *= i;
  }
  return retValue;
}

int TensorBase::getRank() const {
  return rank;
}

const TensorBase::IndexType* TensorBase::getTypes() const {
  return types;
}
