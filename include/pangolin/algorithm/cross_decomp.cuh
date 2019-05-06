#include <algorithm>
#include <map>


// using namespace std::chrono;
using namespace std;

//if you change PARTITION Size
//make sure to change the arg_sort in repartition kerenel and temp array in InitCrossDecomp
#define PARTITION 4 

typedef pangolin::Vector<uint8_t> pangolinVec8;
typedef pangolin::Vector<uint32_t> pangolinVec32;
typedef pangolin::Vector<uint64_t> pangolinVec64;

namespace pangolin {

__global__ void __launch_bounds__(1024,2) repartition_kernel(   uint8_t* orig_P, uint8_t* new_P, 
                                                                uint32_t* cardi, uint32_t* new_cardi,
                                                                uint32_t* coo_col, uint32_t* coo_col_ptr,
                                                                const uint64_t num_nodes, 
                                                                const float h, uint64_t part_size,
                                                                bool is_divisible)
{
    uint64_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    
	if(idx < num_nodes)
	{		
        uint64_t cur_node = idx;
        uint64_t connected_and_in_curpart[PARTITION] = {0};
        uint64_t degree = (coo_col_ptr[cur_node+1] - coo_col_ptr[cur_node]);
        uint32_t start_cur_col_ptr = coo_col_ptr[cur_node];

        for(uint64_t j=0; j<degree; j++){
            uint32_t cur_col_ptr = start_cur_col_ptr + j;
            uint32_t cur_col = coo_col[cur_col_ptr];
            for (int cur_part = 0; cur_part < PARTITION; cur_part++) {
               connected_and_in_curpart[cur_part] += (orig_P[cur_col] == cur_part);
            }
        }
        
        float cost[PARTITION] = {0};
        for (int i = 0; i < PARTITION; i++){
            cost[i] = h* connected_and_in_curpart[i] + 
                        (1-h)*(num_nodes - (cardi[i] + degree - connected_and_in_curpart[i]));
        } 

        int arg_sort[PARTITION] = {0,1,2,3};
        for (int i = 0; i < PARTITION-1; i++){       
            for (int j = 0; j < PARTITION-i-1; j++){
                if (cost[j] < cost[j+1]){
                    float temp = cost[j];
                    cost[j] = cost[j+1];
                    cost[j+1] = temp;
                    int temp2 = arg_sort[j];
                    arg_sort[j] = arg_sort[j+1];
                    arg_sort[j+1] = temp2;
                }
            }
        }

        unsigned int old_size;
        for (int i = 0; i < PARTITION; i++){
            if(!is_divisible){
                if(arg_sort[i]==PARTITION-1){
                    old_size = atomicAdd((unsigned int*)&new_cardi[arg_sort[i]],(unsigned int) 1);
                    if(old_size < part_size+(num_nodes%PARTITION)){
                        new_P[cur_node] = arg_sort[i];
                        break;
                    }
                }
                else{
                    old_size = atomicAdd((unsigned int*)&new_cardi[arg_sort[i]],(unsigned int) 1);
                    if(old_size < part_size){
                        new_P[cur_node] = arg_sort[i];
                        break;
                    }
                }
            }
            else{
                old_size = atomicAdd((unsigned int*)&new_cardi[arg_sort[i]],(unsigned int) 1);
                if(old_size < part_size){
                    new_P[cur_node] = arg_sort[i];
                    break;
                }
            }
        }
    }
}

__global__ void evalEdges(uint8_t* P, 
                          uint32_t* coo_row, uint32_t* coo_col, 
                          uint64_t* edges_per_part, uint64_t coo_size)
{

    uint64_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx==0)
        printf("Evaluating Edges kernel launched!@\n");
    
    if(idx<coo_size){
        uint32_t src = coo_row[idx];
        uint32_t dest = coo_col[idx];

        if(src != dest){
            uint8_t src_part = P[src];
            uint8_t dest_part = P[dest];
            atomicAdd( (unsigned int*)&edges_per_part[src_part*PARTITION+dest_part], (unsigned int) 1);
        }
    }
    return;

} 



template<typename pangolinCOO>
class CrossDecomp{
    private:
        float h;
        bool is_divisible = true;
        uint32_t num_nodes, part_size;
        uint32_t fixed_array[PARTITION];//this array contains partition sizes to initialize cardi arrays
        pangolinVec8 row_P, col_P;
        pangolinVec32 row_cardi, row_new_cardi, col_cardi, col_new_cardi;
        pangolinVec64 edges_per_part;
        map<uint32_t, uint32_t> renamed_map;
        
        
    public:
        CrossDecomp(){};
        void CrossDecompInit(pangolinCOO COO);
        void Host_repartition(bool one_iter, int num_iter, pangolinCOO COO);
        void initParts(uint8_t* P, uint32_t* cardi);
        void reset_cardi();
        void Host_evalEdges(pangolinCOO COO);
        void printEdgesPerPar();
        void rename();
};



template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::CrossDecompInit(pangolinCOO COO)
{
    h = 0.9;
    num_nodes = COO.num_nodes();

    row_P.resize(num_nodes);
    col_P.resize(num_nodes);
    row_cardi.resize(PARTITION,0);
    row_new_cardi.resize(PARTITION,0);
    col_cardi.resize(PARTITION,0);
    col_new_cardi.resize(PARTITION,0);
    edges_per_part.resize(PARTITION*PARTITION,0);

    initParts(row_P.data(), row_cardi.data());
    initParts(col_P.data(), col_cardi.data());

    part_size = floor(num_nodes/PARTITION);
    uint32_t rem = num_nodes%PARTITION;
    uint32_t temp[PARTITION] = {part_size, part_size, part_size, part_size};
    if(rem!=0){
        cout<<"Size of Each Partition is: "<<part_size<<endl;
        cout<<"Size of Last Partition is: "<<part_size+rem<<endl;
        is_divisible = false;
        temp[PARTITION-1] = part_size+rem;
    }
    else{
        cout<<"N is divided perfectly"<<endl;
        cout<<"Size of Each Partition is: "<<part_size<<endl;
    }
    std::memcpy(fixed_array, temp, PARTITION);
}


template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::initParts(uint8_t* P, uint32_t* cardi)
{
	random_device rd;
	mt19937 mt(rd());
    // mt19937 mt(1);
	uniform_int_distribution<int> dist(0, PARTITION - 1);
    for (size_t i=0; i< num_nodes; i++)    {   
        uint64_t part = dist(mt);
        P[i] = part;
        cardi[part]++;
		// P[i] = i%PARTITION;
		// cardi[i%PARTITION]++;
    }
}

template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::Host_repartition(bool one_iter, int num_iter, pangolinCOO COO)
{
    dim3 dimGrid(ceil(((float)num_nodes)/1024),1,1);
    dim3 dimBlock(1024,1,1); //1024

    if(one_iter){
        repartition_kernel<<<dimGrid, dimBlock>>>(  col_P.data(), row_P.data(), 
                                                    col_cardi.data(), col_new_cardi.data(),
                                                    COO.colInd_.data(), COO.rowPtr_.data(), 
                                                    num_nodes, h, part_size,
                                                    is_divisible);
    }
    else{
        for(int i=0; i<num_iter; i++){
            repartition_kernel<<<dimGrid, dimBlock>>>(  row_P.data(), col_P.data(), 
                                                        row_cardi.data(), row_new_cardi.data(),
                                                        COO.colInd_.data(), COO.rowPtr_.data(), 
                                                        num_nodes, h, part_size,
                                                        is_divisible);

            repartition_kernel<<<dimGrid, dimBlock>>>(  col_P.data(), row_P.data(), 
                                                        col_cardi.data(), col_new_cardi.data(),
                                                        COO.colInd_.data(), COO.rowPtr_.data(), 
                                                        num_nodes, h, part_size,
                                                        is_divisible);
            reset_cardi();
        }
    }
    CUDA_RUNTIME(cudaDeviceSynchronize());
    return;
}

template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::reset_cardi()
{
    for(int i=0; i<PARTITION; i++)
    {
        row_cardi[i]=fixed_array[i];
        col_cardi[i]=fixed_array[i];
        row_new_cardi[i] = 0;
        col_new_cardi[i] = 0;
    }
}


template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::Host_evalEdges(pangolinCOO COO)
{
    std::fill(edges_per_part.begin(), edges_per_part.end() + PARTITION*PARTITION, 0);
    dim3 dimGrid0(ceil(((float)COO.nnz())/1024),1,1);
    dim3 dimBlock0(1024,1,1); //1024

    evalEdges<<<dimGrid0, dimBlock0>>>(row_P.data(), COO.rowInd_.data(), COO.colInd_.data(), edges_per_part.data(), COO.nnz());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    printEdgesPerPar();
}

template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::printEdgesPerPar()
{
    cout<<"********************************************************"<<endl;
    map<pair<uint8_t,uint8_t>, bool> track;
    for(uint8_t i=0; i<PARTITION; i++)
    {
        for(uint8_t j=0; j<PARTITION; j++){
            if(i==j){
                cout<<"Internal Edges for Partition "<<(int)i<<" :"<<(edges_per_part[i*PARTITION+j])<<endl;
            }
            else if ( track.find(make_pair(i,j)) == track.end()){
                cout<<"Between "<<"PARTITION "<<(int)i<<" and "<<(int)j<<" Edges: "<<edges_per_part[i*PARTITION+j]<<endl;
                track[make_pair(i,j)] = true;
                track[make_pair(j,i)] = true;
            }
        }
    }
    cout<<"********************************************************"<<endl;
    return;
}

template<typename pangolinCOO>
void CrossDecomp<pangolinCOO>::rename()
{
    uint64_t start[PARTITION];
    start[0] = 0;
    for(int i=1; i<PARTITION; i++){
        start[i] = start[i-1] + part_size;
    }
    cout<<"Rename Starts at ";
    for(int i=0; i<PARTITION; i++){
       cout<<start[i]<<" ";
    }
    cout<<endl;

    for(int i=0; i<num_nodes; i++){
        renamed_map[i] = start[row_P[i]];
        start[row_P[i]]++;
    }
}



}//namespace Pangolin

