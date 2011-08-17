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
#ifndef TENSORLIST_H
#define TENSORLIST_H

#include <map>
#include <string>
#include "Tensor.h"

namespace Mosquito {
  class TensorList {
    public:
      /**
       * Constructor.
       *
       * Creates an empty TensorList object.
       */
      TensorList();

      /**
       * \brief Get a named tensor
       *
       * Retrieve the named tensor from the list.
       *
       * \param name The name of the tensor.
       * \retval this The named tensor object.
       */
      Tensor& operator[](const char* name);

      /**
       * \brief Append a tensor to the list
       *
       * \param name The name of the tensor.
       * \param indexString The character array defining the tensor type.
       */
      void append(const char* name, const char* indexString);

      /**
       * \brief Append a scalar Tensor object to the list
       *
       * \param name The name of the scalar.
       */
      void append(const char* name);

      /**
       * \brief Copy components of all tensors to an array of doubles
       *
       * This routine is provided so one can quickly output all the data to
       * a C array of doubles. It is assumed that the array has been allocated
       * and is at least as large as the total number of tensor components.
       * \param array A pointer to a double array for the data
       * \retval num The number of components copied
       */
      int getComponents(double* array);

      /**
       * \brief Copy components of all tensors from an array of doubles
       *
       * This routine is provided so one can quickly set all the data from
       * a C array of doubles. It is assumed that the array has been allocated
       * and is at least as large as the number of tensor components.
       * \param array A pointer to a double array containing the data
       * \retval num The number of components copied
       */
      int setComponents(const double* array);

      /**
       * \brief Return the total number of components in all tensors
       *
       * \retval num The total number of components
       */
      int getNumComponents() const;

    private:
      /**
       * The Tensor objects in the TensorList
       */
      std::map<std::string, Tensor> tensors;

      /**
       * Total number of components of all tensors
       */
      int numComponents;
  };
};

#endif
