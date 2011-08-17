/* A class for managing a list of tensor objects.
 *
 * Copyright (C) 2011 Barry Wardell
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 */

#include "TensorList.h"
#include <cassert>

using namespace std;
using namespace Mosquito;

TensorList::TensorList()
 : numComponents(0)
{
}

Tensor& TensorList::operator[](const char* name)
{
  assert(tensors.count(name));
  return tensors[name];
}

void TensorList::append(const char *name, const char *indexString)
{
  tensors.insert(map<string, Tensor>::value_type(name, indexString));
  numComponents += tensors[name].getNumComponents();
}

void TensorList::append(const char *name)
{
  tensors.insert(map<string, Tensor>::value_type(name, 0));
  numComponents += tensors[name].getNumComponents();
}

int TensorList::getComponents(double* array)
{
  int components = 0;
  for (map<string,Tensor>::iterator it = tensors.begin(); it != tensors.end(); ++it)
  {
    components += it->second.getComponents(array);
    array += it->second.getNumComponents();
  }

  return components;
}

int TensorList::setComponents(const double* array)
{
  int components = 0;
  for (map<string,Tensor>::iterator it = tensors.begin(); it != tensors.end(); ++it)
  {
    components += it->second.setComponents(array);
    array += it->second.getNumComponents();
  }

  return components;
}

int TensorList::getNumComponents() const
{
  return numComponents;
}
