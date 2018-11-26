 #include "graph/gpu_triangle_counter.hpp"
 #include "graph/logger.hpp"
 #include "utilities.h"

 #include "graph/dag2019.hpp"

 __global__ void kernel_triangleCounter_tc(long long int *cpu_tc, int *cpu_edgeids_src, int *cpu_edgeids_dest, long long int *cpu_rowptrs, long long int size, long long int offset){
     
    long long int blockId = (long long int)blockIdx.y * (long long int)gridDim.x + (long long int)blockIdx.x;
    long long int id = blockId * ((long long int)blockDim.x * (long long int)blockDim.y) + ((long long int)threadIdx.y * (long long int)blockDim.x) + (long long int) threadIdx.x;

    if (id < size) {

        long long int count = 0;
        
        int u, v;
        //long long int e = working_edgelist[id + offset];
        //decodeEdge(u, v, e);
        u = cpu_edgeids_src[id+offset];
        v = cpu_edgeids_dest[id+offset];

        long long int u_ptr = cpu_rowptrs[u];
        long long int u_end = cpu_rowptrs[u + 1];

        long long int v_ptr = cpu_rowptrs[v];
        long long int v_end = cpu_rowptrs[v + 1];
        int v_u, v_v;

        // if (u_ptr >= size) {
        //     printf("u_ptr too large\n");
        // }
        v_u = cpu_edgeids_dest[u_ptr];
        // if (v_ptr >= size) {
        //     printf("v_ptr too large: %lu >= %lu\n", v_ptr, size);
        // }
        v_v = cpu_edgeids_dest[v_ptr];

        while (u_ptr < u_end && v_ptr < v_end){

            //long long int e_u = cpu_edgeids[u_ptr];
            //long long int e_v = cpu_edgeids[v_ptr];

            //int u_u, v_u, u_v, v_v;
            //u_u = cpu_edgeids[u_ptr+1];
            //u_v = cpu_edgeids[v_ptr+1];

            //decodeEdge(u_u, v_u, e_u);
            //decodeEdge(u_v, v_v, e_v);

            if (v_u == v_v) {
                ++count;
                v_u = cpu_edgeids_dest[++u_ptr];
                v_v = cpu_edgeids_dest[++v_ptr];
            }
            else if (v_u < v_v){
                v_u = cpu_edgeids_dest[++u_ptr];
                //++u_ptr;
            }
            else {
                v_v = cpu_edgeids_dest[++v_ptr];
                //++v_ptr;
            }
        }
        cpu_tc[id + offset] = count;
    }
}

__global__ void kernel_tc(size_t *triangleCounts, const Int *edgeSrc, const Int *edgeDst, const Int *nodes, const size_t edgeOffset, const size_t numEdges){
     
    const Int gx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (Int i = gx + edgeOffset; i < edgeOffset + numEdges; i += blockDim.x * gridDim.x) {

        // get the src and dst node for this edge
        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];

        Int src_edge = nodes[src];
        const Int src_edge_end = nodes[src + 1];

        Int dst_edge = nodes[dst];
        const Int dst_edge_end = nodes[dst + 1];

        size_t count = 0;
        while (src_edge < src_edge_end && dst_edge < dst_edge_end){

            Int u = edgeDst[src_edge];
            Int v = edgeDst[dst_edge];

            // the two nodes that make up this edge both have a common dst
            if (u == v) {
                ++count;
                ++src_edge;
                ++dst_edge;
            }
            else if (u < v){
                ++src_edge;
            }
            else {
                ++dst_edge;
            }
        }
        triangleCounts[i] = count;
    }
}

GPUTriangleCounter::GPUTriangleCounter() {
    LOG(debug, "ctor GPU triangle counter, sizeof(Int) = {}", sizeof(Int));
    }

GPUTriangleCounter::~GPUTriangleCounter() {
    LOG(debug, "dtor GPU triangle counter");
}

void GPUTriangleCounter::read_data(const std::string &path) {

    LOG(info, "reading {}", path);
    auto edgeList = EdgeList::read_tsv(path);
    LOG(debug, "building DAG");
    hostDAG_ = DAG2019::from_edgelist(edgeList);

    LOG(info, "{} nodes", hostDAG_.num_nodes());
    LOG(info, "{} edges", hostDAG_.num_edges());
}

size_t GPUTriangleCounter::count() {
    
    const size_t edgeBytes = hostDAG_.edgeSrc_.size() * sizeof(Int);
    const size_t nodeBytes = hostDAG_.nodes_.size() * sizeof(Int);
    size_t *triangleCounts;
    const size_t countBytes = hostDAG_.num_edges() * sizeof(*triangleCounts);

    LOG(debug, "registering {}B", edgeBytes);
    CUDA_RUNTIME(cudaHostRegister(hostDAG_.edgeSrc_.data(), edgeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    LOG(debug, "registering {}B @ ", edgeBytes);
    CUDA_RUNTIME(cudaHostRegister(hostDAG_.edgeDst_.data(), edgeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    LOG(debug, "registering {}B", nodeBytes);
    CUDA_RUNTIME(cudaHostRegister(hostDAG_.nodes_.data(), nodeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));

    LOG(debug, "alloc/mapping {}B", countBytes);
    CUDA_RUNTIME(cudaHostAlloc(&triangleCounts, countBytes, cudaHostAllocMapped));

    dim3 dimBlock(256);
    dim3 dimGrid((hostDAG_.num_edges() + dimBlock.x - 1) / dimBlock.x);

    Int *edgeSrc_d, *edgeDst_d, *nodes_d;
    CUDA_RUNTIME(cudaHostGetDevicePointer(&edgeSrc_d, hostDAG_.edgeSrc_.data(), 0));
    CUDA_RUNTIME(cudaHostGetDevicePointer(&edgeDst_d, hostDAG_.edgeDst_.data(), 0)); 
    CUDA_RUNTIME(cudaHostGetDevicePointer(&nodes_d, hostDAG_.nodes_.data(), 0));

    kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts, edgeSrc_d, edgeDst_d, nodes_d, 0, hostDAG_.num_edges());
    CUDA_RUNTIME(cudaDeviceSynchronize());

    size_t total = 0;
    for(size_t i = 0; i < hostDAG_.num_edges(); ++i) {
        total += triangleCounts[i];
    }

    LOG(debug, "unregistering/freeing CUDA memory");
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.edgeSrc_.data()));
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.edgeDst_.data()));
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.nodes_.data()));
    CUDA_RUNTIME(cudaFreeHost(triangleCounts));
    return total;
}

 void GPUTriangleCounter::execute(const char* filename, int omp_numthreads) {
 
     for (int devid = 0; devid < NUMDEV; devid++){
         LOG(debug, "initializing device {}", devid);
         CUDA_RUNTIME(cudaSetDevice(devid));
         CUDA_RUNTIME(cudaFree(0));
         CUDA_RUNTIME(cudaDeviceSynchronize());
     }
         
 
#ifdef USE_OPENMP
     omp_set_num_threads(omp_numthreads);
#endif // USE_OPENMP
 
 //	printf("graph\tn\tm\ttc\tread_time\tprocessing_time\ttc_time\ttotal_time\n");
 
     double time1 = omp_get_wtime();
 
     LOG(info, "reading {}", filename);
 #ifdef CSR_READ
     readGraph_DARPA_CSR(filename, edge_vec_src, edge_vec_dest, row_ptrs_vec, edgecount, nodecount);
 #else
     readGraph_DARPA(filename, edge_vec, edgecount, nodecount);
 #endif
     
 //#ifdef SNAP_READ
 //	readGraph_SNAP_CSR(filename, edge_vec_src, edge_vec_dest, row_ptrs_vec, edgecount, nodecount);
 //#endif
 
 
 
     double time2 = omp_get_wtime();
     double read_time = time2 - time1;
     LOG(info, "took {}s", read_time);
    
    LOG(info, "{} nodes", nodecount);
    LOG(info, "{} edges", edgecount);
     printf("%s\t%d\t%lld\t", filename, nodecount, edgecount);
 

 
 //	std::cout << "------------------------" << std::endl;
 //	std::cout << "Triangle counting" << std::endl;
 //	std::cout << "Node count: " << nodecount << std::endl;
 //	std::cout << "Edge count: " << edgecount << std::endl;
 //	std::cout << "Graph read time: " << time2 - time1 << " s" << std::endl;
 
    LOG(debug, "allocate arrays");
     allocArrays();
 
 #ifndef CSR_READ
     LOG(debug, "calculating row pointers (1/2)");
     calcRowPtrs();
     
     LOG(debug, "preparing adjacency list");
     prepAdj();
 
     LOG(debug, "calculating row pointers (2/2)");
     calcRowPtrs();
     
     LOG(debug, "preparing working edge list");
     prepWorkingEdgeList();
 #endif
 
 #ifdef CSR_READ
     working_edgecount = edgecount;
 #endif
 
     double time3 = omp_get_wtime();
 
     double processing_time = time3 - time2;
 
 //	std::cout << "Preprocessing time: " << time3 - time2 << " s" << std::endl;
 
     long long int tc = countTriangles();
 
     double time4 = omp_get_wtime();
 
     double tc_time = time4 - time3;
 
 //	std::cout << "Triangle count: " << tc << std::endl;
 //	std::cout << "Triangle counting time: " << time4 - time3 << " s" << std::endl;
 
     freeArrays();
 
     double time5 = omp_get_wtime();
 
     double total_time = time5 - time1;
 
     printf("%lld\t%f\t%f\t%f\t%f\n", tc, read_time, processing_time, tc_time, total_time);
 
 //	std::cout << "Total time: " << time5 - time1 << " s" << std::endl;
 //	std::cout << "------------------------" << std::endl;
 
 }
 
 long long int GPUTriangleCounter::countTriangles(void) {
 
     calculateWorkSplit(working_edgecount);
 
     long long int cpu_itrator_begin = edge_split[NUMDEV];
     long long int cpu_itrator_end = edge_split[NUMDEV + 1];
 
     /******************************************* stream compaction ******************************************************/
 
     long long int tc;
 
 #ifndef CSR_READ
 CUDA_RUNTIME(cudaHostAlloc((void**)&cpu_tc, working_edgecount * sizeof(long long int), cudaHostAllocMapped));
     std::fill(cpu_tc, cpu_tc + working_edgecount, 0);
 #endif
 
     // predicate construction
     if(WORK_FRAC_GPU > 0.01) {
 #pragma omp parallel for // count triangles for each edge in parallel
         for (int devid = 0; devid < NUMDEV; devid++) {
 
             CUDA_RUNTIME(cudaSetDevice(devid));
 
             long long int gpu_iterator_begin = edge_split[devid];
             long long int gpu_iterator_end = edge_split[devid + 1];
             long long int gpu_edgecount = gpu_iterator_end - gpu_iterator_begin;
 
             dim3 blocks_per_grid;
             dim3 threads_per_block;
             long long int total_blocks = 0;
 
             calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, gpu_edgecount);
 #ifdef CSR_READ
            LOG(debug, "Kernel gridDim={} blockDim={}, gpu edges={}, gpu iter={}", blocks_per_grid.x, threads_per_block.x, gpu_edgecount, gpu_iterator_begin);
             kernel_triangleCounter_tc << <blocks_per_grid, threads_per_block >> >(cpu_tc, graph.edgeids_src, graph.edgeids_dest, graph.rowptrs, gpu_edgecount, gpu_iterator_begin);
 #endif
 #ifndef CSR_READ
             kernel_triangleCounter_tc << <blocks_per_grid, threads_per_block >> >(cpu_tc, working_edgelist, (long int *) graph.edgeids, graph.rowptrs, gpu_edgecount, gpu_iterator_begin);
 #endif
         }
     }
 
 #pragma omp parallel for // count triangles for each edge in parallel
     for (long long int i = cpu_itrator_begin; i < cpu_itrator_end; i++) {
         
         long long int count = 0;
 
         int u = graph.edgeids_src[i];
         int v = graph.edgeids_dest[i];
         //long long int e = working_edgelist[i];
         //decodeEdge(u, v, e);
 
         long long int u_ptr = graph.rowptrs[u];
         long long int u_end = graph.rowptrs[u + 1];
 
         long long int v_ptr = graph.rowptrs[v];
         long long int v_end = graph.rowptrs[v + 1];
         
         int v_u = graph.edgeids_dest[u_ptr];
         int v_v = graph.edgeids_dest[v_ptr];
 
         while (u_ptr < u_end && v_ptr < v_end){
 
             //long long int e_u = graph.edgeids[u_ptr];
             //long long int e_v = graph.edgeids[v_ptr];
 
             //int u_u, v_u, u_v, v_v;
             //decodeEdge(u_u, v_u, e_u);
             //decodeEdge(u_v, v_v, e_v);
 
             if (v_u == v_v) {
                 ++count;
                 v_u = graph.edgeids_dest[++u_ptr];
                 v_v = graph.edgeids_dest[++v_ptr];
             }
             else if (v_u < v_v){
                 v_u = graph.edgeids_dest[++u_ptr];
             }
             else {
                 v_v = graph.edgeids_dest[++v_ptr];
             }
         }
         cpu_tc[i] = count;
     }
     
     if(WORK_FRAC_GPU > 0.01) {
         for (int devid = 0; devid < NUMDEV; devid++) {
             CUDA_RUNTIME(cudaSetDevice(devid));
             CUDA_RUNTIME(cudaDeviceSynchronize());
         }
     }
 
     tc = thrust::reduce(cpu_tc, cpu_tc + working_edgecount);
 
     return tc;
 }
 
 void GPUTriangleCounter::allocArrays(void) {
 
 #ifdef CSR_READ
 CUDA_RUNTIME(cudaHostAlloc((void**)&cpu_tc, edgecount * sizeof(long long int), cudaHostAllocMapped));
 //	std::fill(cpu_tc, cpu_tc + edgecount, 0);
     
 CUDA_RUNTIME(cudaHostAlloc((void**)&graph.edgeids_src, edgecount * sizeof(int), cudaHostAllocMapped));
 CUDA_RUNTIME(cudaHostAlloc((void**)&graph.edgeids_dest, edgecount * sizeof(int), cudaHostAllocMapped));
//  LOG(debug, "mapping...");
//  CUDA_RUNTIME(cudaHostRegister(edge_vec_src.data(), edge_vec_src.size() * sizeof(edge_vec_src[0]), cudaHostRegisterMapped));
//  CUDA_RUNTIME(cudaHostRegister(edge_vec_dest.data(), edge_vec_dest.size() * sizeof(edge_vec_dest[0]), cudaHostRegisterMapped));
//  graph.edgeids_src = edge_vec_src.data();
//  graph.edgeids_dest = edge_vec_dest.data();
//  LOG(debug, "mapping done");

 CUDA_RUNTIME(cudaHostAlloc((void**)&graph.rowptrs, (nodecount + 1) * sizeof(long long int), cudaHostAllocMapped));
     std::copy(edge_vec_src.begin(), edge_vec_src.end(), graph.edgeids_src);
     std::copy(edge_vec_dest.begin(), edge_vec_dest.end(), graph.edgeids_dest);
 #endif
 
 #ifndef CSR_READ
 CUDA_RUNTIME(cudaHostAlloc((void**)&graph.edgeids, edgecount * sizeof(long long int), cudaHostAllocMapped));
 CUDA_RUNTIME(cudaHostAlloc((void**)&graph.rowptrs, (nodecount + 1) * sizeof(long long int), cudaHostAllocMapped));
     std::copy(edge_vec.begin(), edge_vec.end(), graph.edgeids);
 #endif
 
 #ifdef CSR_READ
     std::copy(row_ptrs_vec.begin(), row_ptrs_vec.end(), graph.rowptrs);
     row_ptrs_vec.clear();
     edge_vec_src.clear();
     edge_vec_dest.clear();
 #endif
 
 #ifndef CSR_READ
     edge_vec.clear();
 #endif
 }
 
 void GPUTriangleCounter::freeArrays(void){
 #ifdef CSR_READ
 CUDA_RUNTIME(cudaFreeHost(graph.edgeids_src));
 CUDA_RUNTIME(cudaFreeHost(graph.edgeids_dest));
 #endif
 CUDA_RUNTIME(cudaFreeHost(graph.rowptrs));
 CUDA_RUNTIME(cudaFreeHost(cpu_tc));
 #ifndef CSR_READ
 CUDA_RUNTIME(cudaFreeHost(graph.edgeids));
 CUDA_RUNTIME(cudaFreeHost(working_edgelist));
 #endif
 }
 
 void GPUTriangleCounter::calculateWorkSplit(long long int count){
     // calculate edge fractions to be processed by each resource
 
     long long int gpu_frac, cpu_split_val, gpu_split_val, overflow;
 
     if (WORK_FRAC_GPU < 0.01) {
         gpu_split_val = 0;
         cpu_split_val = count;
     }
     else if (count > (NUMDEV + 1) * MAX_EDGES_GPU) {
         gpu_split_val = MAX_EDGES_GPU;
         cpu_split_val = count - NUMDEV * gpu_split_val;
     }
 
     else {
         gpu_frac = (WORK_FRAC_GPU < 1) ? WORK_FRAC_GPU * count : count;
         cpu_split_val = count - gpu_frac;
 
         gpu_split_val = gpu_frac / (NUMDEV > 0 ? NUMDEV : 1);
         overflow = gpu_frac % (NUMDEV > 0 ? NUMDEV : 1);
 
         if(WORK_FRAC_GPU < 1)
             cpu_split_val += overflow;
         else
             cpu_split_val = 0;
     }
 
     for (int i = 0; i < NUMDEV; i++)
         edge_split[i] = gpu_split_val;
 
     edge_split[NUMDEV] = cpu_split_val;
 
     thrust::exclusive_scan(edge_split, edge_split + NUMDEV + 1, edge_split);
     edge_split[NUMDEV + 1] = count;
 
 }
 

 
 