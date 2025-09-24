#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_int y;
  
  __pp_vec_int zeros = _pp_vset_int(0);
  __pp_vec_int ones = _pp_vset_int(1);
  const static float max = 9.999999f;
  __pp_vec_float vmax = _pp_vset_float(max);
  __pp_mask maskNotDone, maskExpIsGtZero, maskExpIsNotGtZero, maskResultIsGtMax, maskResultIsNotGtMax;
  __pp_mask maskAll = _pp_init_ones();
  __pp_mask maskNotAll = _pp_init_ones(0);
  __pp_mask maskOutOffBound = _pp_init_ones(N % VECTOR_WIDTH);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll);
    _pp_vload_int(y, exponents + i, maskAll);
    __pp_vec_float result = _pp_vset_float(1.0f);
    maskNotDone = maskAll;
    maskExpIsGtZero = maskResultIsGtMax = maskNotAll;
    do {
      // subtract exp by 1 if exp > 0
      _pp_vgt_int(maskExpIsGtZero, y, zeros, maskNotDone);
      _pp_vsub_int(y, y, ones, maskExpIsGtZero);
      // perform multiply
      _pp_vmult_float(result, result, x, maskExpIsGtZero);
      // set result to max if result > max
      _pp_vgt_float(maskResultIsGtMax, result, vmax, maskExpIsGtZero);
      _pp_vset_float(result, max, maskResultIsGtMax);
      maskResultIsNotGtMax = _pp_mask_not(maskResultIsGtMax);
      maskNotDone = _pp_mask_and(maskExpIsGtZero, maskResultIsNotGtMax);
    } while (_pp_cntbits(maskNotDone) > 0);
    if (i == (N / VECTOR_WIDTH) * VECTOR_WIDTH)
    {
      _pp_vstore_float(output + i, result, maskOutOffBound);
    }
    else
    {
      _pp_vstore_float(output + i, result, maskAll);
    }
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}