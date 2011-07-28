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
    u.get(i) = i+1;
  }
  vector = Tensor::DOWN;
  Tensor v(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    v.get(i) =  ipow(-1,i)/(double)(i+1);
  }

  Tensor::IndexType indexTypes[2];
  indexTypes[0] = Tensor::UP;
  indexTypes[1] = Tensor::DOWN;
  Tensor uv = u*v;

  Tensor scalar1 = uv.contract(0, 1);
  assert(scalar1.get(0) == 0.);

  for (int i = 0; i < DIMENSION; i++) {
    u.get(i) = 1;
    v.get(i) = 1;
  }
  Tensor uv2 = u*v;
  Tensor scalar3 = uv2.contract(0,1);
  assert(scalar3.get(0) == 4.);

  Tensor scalar4 = u('a')*v('a');
  assert(scalar4.getRank() == 0);
  assert(scalar4.get(0) == 4.);

  IndexType up = Tensor::UP;
  IndexType down = Tensor::DOWN;
  Tensor test0(2, up, down);
  test0.get(0,0) = -1;
  test0.get(1,1) = -6;
  test0.get(2,2) = 3;
  test0.get(3,3) = 4;
  test0.get(3,4) = -100000;
  Tensor scalar5 = test0('a','a');
  assert(scalar5.get(0) == 0.);
}

void TestTensor::runTensorMultiplyTest() {
  Tensor::IndexType vector = Tensor::UP;
  Tensor u(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    u.get(i) =  i+1;
  }
  Tensor::IndexType indexTypes[2];
  indexTypes[0] = Tensor::UP;
  indexTypes[1] = Tensor::UP;
  Tensor uSquared = u*u;
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      assert(abs((i+1.)*(j+1.) - uSquared.get(i,j)) < 1.0e-16);
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
  double * components = tempA.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    components[i] = i;
  }

  Tensor tempD = tempA;
  tempD *= 2.5;
  double * componentsA = tempA.getComponents();
  double * componentsD = tempD.getComponents();
  std::cout.precision(16);
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    // This is exact in binary arithmetic since 2.5 = 2 + 2^-1
    assert(abs(2.5*componentsA[i] - componentsD[i]) < 1.0e-19);
  }

  Tensor tempE = 2.5*tempA;
  double * componentsE = tempE.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    // This is exact in binary arithmetic since 2.5 = 2 + 2^-1
    assert(abs(2.5*componentsA[i] - componentsE[i]) < 1.0e-19);
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

  int permute[rank];
  Tensor::IndexType up = Tensor::UP;
  Tensor::IndexType down = Tensor::DOWN;
  Tensor gamma(3,up,down,down);
  gamma('a','b','c');
  gamma.permutation(gamma.indexes, permute);
  for (int i = 0; i < 3; i++) {
    assert(i == permute[i]);
  }
}

void TestTensor::runLinearCombinationTest() {
  Tensor tempA(rank, types);
  double* componentsA = tempA.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    componentsA[i] = i;
  }

  Tensor tempB =  3.0*tempA - 2.0*tempA;
  double* componentsB = tempB.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double a = componentsA[i];
    double c = componentsB[i];
    assert(a == c);
  }

  Tensor tempD = tempA;
  tempD += (-1)*tempA;
  double* componentsD = tempD.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsD[i];
    assert(c == 0);
  }

  Tensor tempE = tempA + (-1)*tempA;
  double* componentsE = tempE.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsE[i];
    assert(c==0);
  }

  Tensor tempF = tempA - tempA;
  double* componentsF = tempF.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsF[i];
    assert(c==0);
  }

  Tensor tempG = tempA;
  tempG -= tempA;
  double* componentsG = tempG.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsG[i];
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
