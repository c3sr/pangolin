#include <algorithm>
#include <map>
#include <random>

#include "pangolin/dense/vector.hu"

namespace pangolin {

template <uint8_t PARTITION>
__global__ void __launch_bounds__(1024, 2)
    repartition_kernel(uint8_t *orig_P, uint8_t *new_P, uint32_t *cardi, uint32_t *new_cardi, uint32_t *coo_col,
                       uint32_t *coo_col_ptr, const uint64_t num_nodes, const float h, uint64_t part_size,
                       bool is_divisible) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_nodes) {
    uint64_t cur_node = idx;
    uint64_t connected_and_in_curpart[PARTITION] = {0};
    uint64_t degree = (coo_col_ptr[cur_node + 1] - coo_col_ptr[cur_node]);
    uint32_t start_cur_col_ptr = coo_col_ptr[cur_node];

    // calculate the nodes that are connected to cur_node and in same current partition
    for (uint64_t j = 0; j < degree; j++) {
      uint32_t cur_col_ptr = start_cur_col_ptr + j;
      uint32_t cur_col = coo_col[cur_col_ptr];
      for (int cur_part = 0; cur_part < PARTITION; cur_part++) {
        connected_and_in_curpart[cur_part] += (orig_P[cur_col] == cur_part);
      }
    }

    float cost[PARTITION] = {0};
    for (int i = 0; i < PARTITION; i++) {
      cost[i] =
          h * connected_and_in_curpart[i] + (1 - h) * (num_nodes - (cardi[i] + degree - connected_and_in_curpart[i]));
    }

    // initialize arg_sort array
    int arg_sort[PARTITION];
    for (int i = 0; i < PARTITION; i++)
      arg_sort[i] = i;

    // perform arg_sort with cost
    for (int i = 0; i < PARTITION - 1; i++) {
      for (int j = 0; j < PARTITION - i - 1; j++) {
        if (cost[j] < cost[j + 1]) {
          float temp = cost[j];
          cost[j] = cost[j + 1];
          cost[j + 1] = temp;
          int temp2 = arg_sort[j];
          arg_sort[j] = arg_sort[j + 1];
          arg_sort[j + 1] = temp2;
        }
      }
    }

    unsigned int old_size;
    for (int i = 0; i < PARTITION; i++) {
      if (!is_divisible) {
        if (arg_sort[i] == PARTITION - 1) {
          old_size = atomicAdd((unsigned int *)&new_cardi[arg_sort[i]], (unsigned int)1);
          if (old_size < part_size + (num_nodes % PARTITION)) {
            new_P[cur_node] = arg_sort[i];
            break;
          }
        } else {
          old_size = atomicAdd((unsigned int *)&new_cardi[arg_sort[i]], (unsigned int)1);
          if (old_size < part_size) {
            new_P[cur_node] = arg_sort[i];
            break;
          }
        }
      } else {
        old_size = atomicAdd((unsigned int *)&new_cardi[arg_sort[i]], (unsigned int)1);
        if (old_size < part_size) {
          new_P[cur_node] = arg_sort[i];
          break;
        }
      }
    }
  }
}

template <uint8_t PARTITION>
__global__ void evalEdges(uint8_t *P, uint32_t *coo_row, uint32_t *coo_col, uint64_t *edges_per_part,
                          uint64_t coo_size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < coo_size) {
    uint32_t src = coo_row[idx];
    uint32_t dest = coo_col[idx];
    if (src != dest) {
      uint8_t src_part = P[src];
      uint8_t dest_part = P[dest];
      atomicAdd((unsigned int *)&edges_per_part[src_part * PARTITION + dest_part], (unsigned int)1);
    }
  }

  return;
}

/* ! Cross-Decomposition Class definition
If the Size of PARTITION gets larger and data type must be changed,
change the type of PARTITION in template
*/
template <typename pangolinCOO, uint8_t PARTITION> class CrossDecomp {
private:
  float h;
  bool is_divisible = true;
  uint32_t num_nodes, part_size;
  uint32_t fixed_array[PARTITION]; // this array contains partition sizes to initialize cardi arrays
  Vector<uint8_t> row_P, col_P;
  Vector<uint32_t> row_cardi, row_new_cardi, col_cardi, col_new_cardi;
  Vector<uint64_t> edges_per_part;
  std::map<uint32_t, uint32_t> renamed_map;

public:
  CrossDecomp(){};
  void CrossDecompInit(pangolinCOO COO);
  void Host_repartition_async(bool one_iter, int num_iter, pangolinCOO COO);
  void Host_repartition_sync(bool one_iter, int num_iter, pangolinCOO COO);
  void initParts(uint8_t *P, uint32_t *cardi);
  void reset_cardi();
  void Host_evalEdges_async(pangolinCOO COO);
  void Host_evalEdges_sync(pangolinCOO COO);
  void printEdgesPerPar();
  void rename();
};

/* ! Initialization function for Cross-Decomposition
    This function must be called after instantiating CrossDecomp obj
*/
template <typename pangolinCOO, uint8_t PARTITION>
void CrossDecomp<pangolinCOO, PARTITION>::CrossDecompInit(pangolinCOO COO) {
  using namespace std;
  h = 0.9;
  num_nodes = COO.num_nodes();

  row_P.resize(num_nodes);
  col_P.resize(num_nodes);
  row_cardi.resize(PARTITION, 0);
  row_new_cardi.resize(PARTITION, 0);
  col_cardi.resize(PARTITION, 0);
  col_new_cardi.resize(PARTITION, 0);
  edges_per_part.resize(PARTITION * PARTITION, 0);

  initParts(row_P.data(), row_cardi.data());
  initParts(col_P.data(), col_cardi.data());

  part_size = floor(num_nodes / PARTITION);
  uint32_t rem = num_nodes % PARTITION;
  std::fill(fixed_array, fixed_array + PARTITION, part_size);
  if (rem != 0) {
    cout << "Size of Each Partition is: " << part_size << endl;
    cout << "Size of Last Partition is: " << part_size + rem << endl;
    is_divisible = false;
    fixed_array[PARTITION - 1] = part_size + rem;
  } else {
    cout << "N is divided perfectly" << endl;
    cout << "Size of Each Partition is: " << part_size << endl;
  }
}

/* ! Initialize the partitions for each node in the graph
        with uniform random distribution
*/
template <typename pangolinCOO, uint8_t PARTITION>
void CrossDecomp<pangolinCOO, PARTITION>::initParts(uint8_t *P, uint32_t *cardi) {
  using namespace std;
  random_device rd;
  mt19937 mt(rd());
  // mt19937 mt(1);
  uniform_int_distribution<int> dist(0, PARTITION - 1);
  for (size_t i = 0; i < num_nodes; i++) {
    uint64_t part = dist(mt);
    P[i] = part;
    cardi[part]++;
    // P[i] = i%PARTITION;
    // cardi[i%PARTITION]++;
  }
}

/* ! This function invokes kernel for Cross-Decomposition asynchronous
 */
template <typename pangolinCOO, uint8_t PARTITION>
void CrossDecomp<pangolinCOO, PARTITION>::Host_repartition_async(bool one_iter, int num_iter, pangolinCOO COO) {
  dim3 dimGrid(ceil(((float)num_nodes) / 1024), 1, 1);
  dim3 dimBlock(1024, 1, 1); // 1024

  if (one_iter) {
    repartition_kernel<PARTITION><<<dimGrid, dimBlock>>>(col_P.data(), row_P.data(), col_cardi.data(),
                                                         col_new_cardi.data(), COO.colInd_.data(), COO.rowPtr_.data(),
                                                         num_nodes, h, part_size, is_divisible);
    CUDA_RUNTIME(cudaGetLastError());
  } else {
    for (int i = 0; i < num_iter; i++) {
      repartition_kernel<PARTITION><<<dimGrid, dimBlock>>>(row_P.data(), col_P.data(), row_cardi.data(),
                                                           row_new_cardi.data(), COO.colInd_.data(), COO.rowPtr_.data(),
                                                           num_nodes, h, part_size, is_divisible);
      CUDA_RUNTIME(cudaGetLastError());
      repartition_kernel<PARTITION><<<dimGrid, dimBlock>>>(col_P.data(), row_P.data(), col_cardi.data(),
                                                           col_new_cardi.data(), COO.colInd_.data(), COO.rowPtr_.data(),
                                                           num_nodes, h, part_size, is_divisible);
      CUDA_RUNTIME(cudaGetLastError());
      reset_cardi();
    }
  }

  return;
}

/* ! This function invokes kernel for Cross-Decomposition synchronously
 */
template <typename pangolinCOO, uint8_t PARTITION>
void CrossDecomp<pangolinCOO, PARTITION>::Host_repartition_sync(bool one_iter, int num_iter, pangolinCOO COO) {
  auto start = std::chrono::system_clock::now();
  Host_repartition_async(one_iter, num_iter, COO);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "Cross Decomposition Runtime {}s", elapsed);
  return;
}

/* ! For multiple iterations of Cross-Decomposition,
resetting cardinality arrays after each iteration is necessary.
This function simply resets cardinality arrays with correct values.
*/
template <typename pangolinCOO, uint8_t PARTITION> void CrossDecomp<pangolinCOO, PARTITION>::reset_cardi() {
  for (int i = 0; i < PARTITION; i++) {
    row_cardi[i] = fixed_array[i];
    col_cardi[i] = fixed_array[i];
    row_new_cardi[i] = 0;
    col_new_cardi[i] = 0;
  }
  return;
}

/* ! Host function to invoke kernels asynchronous
to evaluate edges inside and between partitions
*/
template <typename pangolinCOO, uint8_t PARTITION>
void CrossDecomp<pangolinCOO, PARTITION>::Host_evalEdges_async(pangolinCOO COO) {
  std::fill(edges_per_part.begin(), edges_per_part.end() + PARTITION * PARTITION, 0);
  dim3 dimGrid0(ceil(((float)COO.nnz()) / 1024), 1, 1);
  dim3 dimBlock0(1024, 1, 1); // 1024

  evalEdges<PARTITION>
      <<<dimGrid0, dimBlock0>>>(row_P.data(), COO.rowInd_.data(), COO.colInd_.data(), edges_per_part.data(), COO.nnz());
  CUDA_RUNTIME(cudaGetLastError());
  return;
}

/* ! Host function to invoke kernels synchronous
to evaluate edges inside and between partitions
*/
template <typename pangolinCOO, uint8_t PARTITION>
void CrossDecomp<pangolinCOO, PARTITION>::Host_evalEdges_sync(pangolinCOO COO) {
  Host_evalEdges_async(COO);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  printEdgesPerPar();
  return;
}

/* ! This function simply prints out the result of evalEdges to terminal
 */
template <typename pangolinCOO, uint8_t PARTITION> void CrossDecomp<pangolinCOO, PARTITION>::printEdgesPerPar() {
  using namespace std;
  cout << "********************************************************" << endl;
  map<pair<uint8_t, uint8_t>, bool> track;
  for (uint8_t i = 0; i < PARTITION; i++) {
    for (uint8_t j = 0; j < PARTITION; j++) {
      if (i == j) {
        cout << "Internal Edges for Partition " << (int)i << " :" << (edges_per_part[i * PARTITION + j]) << endl;
      } else if (track.find(make_pair(i, j)) == track.end()) {
        cout << "Between "
             << "PARTITION " << (int)i << " and " << (int)j << " Edges: " << edges_per_part[i * PARTITION + j] << endl;
        track[make_pair(i, j)] = true;
        track[make_pair(j, i)] = true;
      }
    }
  }
  cout << "********************************************************" << endl;
  return;
}

/*! Rename the nodes based on result of Cross-Decomposition Partition
 */
template <typename pangolinCOO, uint8_t PARTITION> void CrossDecomp<pangolinCOO, PARTITION>::rename() {
  uint32_t start[PARTITION];
  start[0] = 0;
  for (int i = 1; i < PARTITION; i++) {
    start[i] = start[i - 1] + part_size;
  }

  std::cout << "Rename Starts at ";
  for (int i = 0; i < PARTITION; i++) {
    std::cout << start[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < num_nodes; i++) {
    renamed_map[i] = start[row_P[i]];
    start[row_P[i]]++;
  }
  return;
}

} // namespace pangolin
