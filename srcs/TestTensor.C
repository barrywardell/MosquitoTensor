// Copyright Aaryn Tonita, 2011
// Distributed under the Gnu general public license
#include <iostream>
#include <cassert>
#define private public
#define protected public
#include "Tensor.h"

#define DIMENSION 4

int ipow(int i, int j);

class TestTensor: public Tensor {
  public:
    TestTensor(int Rank, IndexType* Types) : 
      Tensor(Rank,Types) {};
    void runTests();

  private:
    void runIndexingTest();
    void runLinearCombinationTest();
    void runScalarMultiplyTest();
    void runTensorMultiplyTest();
    void runContractionTest();
    double abs(double x);
};

void TestTensor::runContractionTest() {
  Tensor::IndexType vector = Tensor::UP;
  Tensor u(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    u.setComponent(&i, i+1);
  }
  vector = Tensor::DOWN;
  Tensor v(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    v.setComponent(&i, ipow(-1,i)/(double)(i+1));
  }

  Tensor::IndexType indexTypes[2];
  indexTypes[0] = Tensor::UP;
  indexTypes[1] = Tensor::DOWN;
  Tensor uv = u*v;

  Tensor scalar1 = uv.contract(0, 1);
  assert(scalar1.getComponent(0) == 0.);

  for (int i = 0; i < DIMENSION; i++) {
    u.setComponent(&i, 1);
    v.setComponent(&i, 1);
  }
  Tensor uv2 = u*v;
  Tensor scalar3 = uv2.contract(0,1);
  assert(scalar3.getComponent(0) == 4.);

  Tensor scalar4 = u('a')*v('a');
  assert(scalar4.getRank() == 0);
  assert(scalar4.getComponent(0) == 4.);
}

void TestTensor::runTensorMultiplyTest() {
  Tensor::IndexType vector = Tensor::UP;
  Tensor u(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    u.setComponent(&i, i+1);
  }
  Tensor::IndexType indexTypes[2];
  indexTypes[0] = Tensor::UP;
  indexTypes[1] = Tensor::UP;
  Tensor uSquared = u*u;
  int indices[2];
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      indices[0] = i;
      indices[1] = j;
      assert(abs((i+1.)*(j+1.) - uSquared.getComponent(indices)) < 1.0e-16);
    }
  }
}

double TestTensor::abs(double x) {
  if (x < 0) return -x;
  return x;
}

void TestTensor::runScalarMultiplyTest() {
  Tensor tempA(rank, types);
  int indices[3];
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    tempA.setComponent(indices, i);
  }

  Tensor tempD = tempA;
  tempD *= 2.5;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    std::cout.precision(16);
    // This is exact in binary arithmetic since 2.5 = 2 + 2^-1
    assert(abs(2.5*tempA.getComponent(indices)
          - tempD.getComponent(indices))
        < 1.0e-19);
  }

  Tensor tempE = 2.5*tempA;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    std::cout.precision(16);
    // This is exact in binary arithmetic since 2.5 = 2 + 2^-1
    assert(abs(2.5*tempA.getComponent(indices)
          - tempE.getComponent(indices))
        < 1.0e-19);
  }
}

void TestTensor::runTests() {
  std::cout << "Running tests on class Tensor.\n";
  int nTests = 0;

  runLinearCombinationTest();
  nTests++; std::cout << ".\n";

  runIndexingTest();
  nTests++; std::cout << ".\n";

  runScalarMultiplyTest();
  nTests++; std::cout << ".\n";

  runTensorMultiplyTest();
  nTests++; std::cout << ".\n";

  runContractionTest();
  nTests++; std::cout << ".\n";

  std::cout << "Complete. Ran " << nTests << " tests successfully.\n";
}

void TestTensor::runIndexingTest() {
  int indices[rank];
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i, indices);
    assert(i == index(indices));
  }
}

void TestTensor::runLinearCombinationTest() {
  Tensor tempA(rank, types);
  int indices[3];
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    tempA.setComponent(indices, i);
  }

  Tensor tempB =  3.0*tempA - 2.0*tempA;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    double a = tempA.getComponent(indices);
    double c = tempB.getComponent(indices);
    assert(a == c);
  }

  Tensor tempD = tempA;
  tempD += (-1)*tempA;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    double c = tempD.getComponent(indices);
    assert(c==0);
  }

  Tensor tempE = tempA + (-1)*tempA;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    double c = tempE.getComponent(indices);
    assert(c==0);
  }

  Tensor tempF = tempA - tempA;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    double c = tempF.getComponent(indices);
    assert(c==0);
  }

  Tensor tempG = tempA;
  tempG -= tempA;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    indexToIndices(i,indices);
    double c = tempG.getComponent(indices);
    assert(c==0);
  }
}

int main() {
  Tensor::IndexType indexTypes[3];
  indexTypes[0] = Tensor::CONTRAVARIANT;
  indexTypes[1] = Tensor::COVARIANT;
  indexTypes[2] = Tensor::COVARIANT;
  TestTensor test(3, indexTypes);
  test.runTests();
  return 0;
}
