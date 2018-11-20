/* Author: Ketan Date 
           Vikram Sharma Mailthdoy
 */

#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif 

#include "clara.hpp"
#include "graph/logger.hpp"
#include "graph/gpu_triangle_counter.hpp"


#include "cutrianglecounter.h"

int main(int argc, char** argv){
	

	std::string adjacencyListPath;
	int numThreads = 1;
	int numGPUs = 1;
	bool help = false;
	bool debug = false;
	bool verbose = false;

	clara::Parser cli;

	cli = cli | clara::Opt( numThreads, "int" )
        ["-c"]["--num_cpu"]
        ("How many CPU threads?");
	cli = cli | clara::Opt( numGPUs, "int" )
        ["-g"]["--num_gpu"]
        ("How many GPUs?");
	cli = cli | clara::Opt( debug )
                ["--debug"]	
		( "log debug messages" );
	cli = cli | clara::Opt( verbose )
                ["--verbose"]	
		( "log verbose messages" );
	cli = cli | clara::Help(help);
	cli = cli | clara::Arg( adjacencyListPath, "file" )
        ("Path to adjacency list");


	const char* adj_filename = argv[1];
	
	auto result = cli.parse( clara::Args( argc, argv ) );
	if( !result ) {
		LOG(error, "Error in command line: {}", result.errorMessage());
		exit(1);
	}

	if (help) {
		std::cout << cli;
		return 0;
	}

	if (debug) {
		logger::console->set_level(spdlog::level::debug);
	} else if (verbose) {
		logger::console->set_level(spdlog::level::trace);
	}

	if (numThreads <= 0)
		numThreads = 1;
	if (numThreads >= omp_get_max_threads())
		numThreads = omp_get_max_threads();

	LOG(info, "{} gpus", numGPUs);
	LOG(info, "{} cpus", numThreads);

	TriangleCounter *tc;
	tc = new GPUTriangleCounter();
	tc->execute(adjacencyListPath.c_str(), numThreads);

	delete tc;
	return 0;
}





