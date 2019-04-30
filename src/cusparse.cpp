#include "pangolin/cusparse.hpp"

static const char *cusparseStatusGetString(cusparseStatus_t status) {
  switch (status) {
  case CUSPARSE_STATUS_SUCCESS:
    return "success";
  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "not initialized";
  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "alloc failed";
  case CUSPARSE_STATUS_INVALID_VALUE:
    return "invalid value";
  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "arch mismatch";
  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "mapping error";
  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "execution failed";
  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "internal error";
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "matrix type not supported";
  case CUSPARSE_STATUS_ZERO_PIVOT:
    return "zero pivot";
  default: {
    LOG(error, "unhandled cusparseStatus value");
    return "";
  }
  }
}

void checkCusparse(cusparseStatus_t result, const char *file, const int line) {
  if (result != CUSPARSE_STATUS_SUCCESS) {
    printf("cusparse Error: %s in %s : %d\n", cusparseStatusGetString(result),
           file, line);
    exit(-1);
  }
}