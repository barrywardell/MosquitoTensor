// Copyright Aaryn Tonita, 2011
// Distributed under the Gnu general public license
#include <iostream>
#include <cassert>
#define private public
#define protected public
#include "Tensor.h"

#define DIMENSION 4

using namespace Mosquito;

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
    u(i) = i+1;
  }
  vector = Tensor::DOWN;
  Tensor v(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    v(i) =  ipow(-1,i)/(double)(i+1);
  }

  Tensor::IndexType indexTypes[2];
  indexTypes[0] = Tensor::UP;
  indexTypes[1] = Tensor::DOWN;
  Tensor uv = u*v;

  Tensor scalar1 = uv.contract(0, 1);
  assert(scalar1(0) == 0.);

  for (int i = 0; i < DIMENSION; i++) {
    u(i) = 1;
    v(i) = 1;
  }
  Tensor uv2 = u*v;
  Tensor scalar3("");
  scalar3[""] = uv2["aa"];
  assert(scalar3(0) == 4.);

  Tensor scalar4 = u["a"]*v["a"];
  assert(scalar4.getRank() == 0);
  assert(scalar4(0) == 4.);

  IndexType up = Tensor::UP;
  IndexType down = Tensor::DOWN;
  Tensor test0(2, up, down);
  test0(0,0) = -1;
  test0(1,1) = -6;
  test0(2,2) = 3;
  test0(3,3) = 4;
  test0(3,4) = -100000;
  scalar4[""] = test0["aa"];
  assert(scalar4(0) == 0.);
}

void TestTensor::runTensorMultiplyTest() {
  Tensor::IndexType vector = Tensor::UP;
  Tensor u(1, &vector);
  for (int i = 0; i < DIMENSION; i++) {
    u(i) =  i+1;
  }
  Tensor::IndexType indexTypes[2];
  indexTypes[0] = Tensor::UP;
  indexTypes[1] = Tensor::UP;
  Tensor uSquared = u*u;
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      assert(abs((i+1.)*(j+1.) - uSquared(i,j)) < 1.0e-16);
    }
  }

  Tensor v("_a");
  for (int i = 0; i < DIMENSION; i++) {
    u(i) = 1;
    v(i) = 1;
  }
  Tensor uv = u["a"]*v["b"];
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      assert(uv(i,j) == 1);
    }
  }
  Tensor scalar = uv["aa"];
  assert(scalar(0) == 4);
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

  Tensor tempF = 2.*tempA;
  Tensor tempAA = tempF/2.;
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    assert(tempAA.components[i] == tempA.components[i]);
  }

  tempAA["abc"] = 0.5*tempF["abc"];
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    assert(tempAA.components[i] == tempA.components[i]);
  }
}

void TestTensor::runTests() {
  std::cout << "Running tests on class Tensor.\n";
  int nTests = 0;

  runIndexingTest();
  nTests++; std::cout << ".\n";

  runLinearCombinationTest();
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

  Tensor christoffel("^a_b_c");
  Tensor somethingElse("_b^a_c");
  christoffel["abc"] = somethingElse["bac"];
}

void TestTensor::runLinearCombinationTest() {
  Tensor tempA(rank, types);
  double* componentsA = tempA.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    componentsA[i] = i;
  }

  Tensor temp("^a_b_c");

  temp["abc"] =  3.0*tempA["abc"] - 2.0*tempA["abc"];
  double* componentsB = temp.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double a = componentsA[i];
    double c = componentsB[i];
    assert(a == c);
  }

  temp["abc"] = tempA["abc"] - tempA["abc"];
  double* componentsD = temp.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsD[i];
    assert(c == 0);
  }

  temp["abc"] = tempA["abc"] + (-1)*tempA["abc"];
  double* componentsE = temp.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsE[i];
    assert(c==0);
  }

  Tensor tempF = tempA["abc"] - tempA["abc"];
  double* componentsF = tempF.getComponents();
  for (int i = 0; i < ipow(DIMENSION, rank); i++) {
    double c = componentsF[i];
    assert(c==0);
  }

  IndexType down = Tensor::DOWN;
  IndexType up = Tensor::UP;
  Tensor tempH(2, down, up);
  Tensor tempI(2, up, down);
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      tempH(i,j) = ipow((i+1),j);
      tempI(i,j) = ipow((j+1),i);
    }
  }
  Tensor tempJ = tempH["ab"] - tempI["ba"];
  for (int i = 0; i < ipow(DIMENSION, 2); i++) {
    assert(tempJ.components[i] == 0);
  }
  // This fails. As it should.
  // Tensor tempJJ = tempH('b','a') - tempI('b','a');

  // build an antisymmetric tensor.
  Tensor tempK("_a_b");
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < i; j++) {
      tempK(i,j) = ipow(i+1, j);
    }
  }
  for (int j = 0; j < DIMENSION; j++) {
    for (int i = 0; i < j; i++) {
      tempK(i,j) = -tempK(j,i);
    }
  }
  Tensor tempL = tempK["ba"];
  Tensor tempM = tempL["ba"] + tempK["ab"];
  for (int i = 0; i < DIMENSION; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      assert(tempM(i,j) == 0);
    }
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
