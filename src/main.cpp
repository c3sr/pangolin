/* Author: Ketan Date 
           Vikram Sharma Mailthdoy
 */

#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif 

#include "clara.hpp"

#include "cutrianglecounter.h"

int main(int argc, char** argv){
	
	std::string adjacencyListPath;
	int numThreads = 1;
	int numGPUs = 1;
	bool help = false;

	clara::Parser cli;

	cli = cli | clara::Opt( numThreads, "int" )
        ["-c"]["--num_cpu"]
        ("How many CPU threads?");
	cli = cli | clara::Opt( numGPUs, "int" )
        ["-g"]["--num_gpu"]
        ("How many GPUs?");
	cli = cli | clara::Help(help);
	cli = cli | clara::Arg( adjacencyListPath, "file" )
        ("Path to adjacency list");

	const char* adj_filename = argv[1];
	
	auto result = cli.parse( clara::Args( argc, argv ) );
	if( !result ) {
		std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
		exit(1);
	}

	if (help) {
		std::cout << cli;
		return 0;
	}

	if (numThreads <= 0)
		numThreads = 1;
	if (numThreads >= omp_get_max_threads())
		numThreads = omp_get_max_threads();

	CuTriangleCounter cutc;
	cutc.execute(adjacencyListPath.c_str(), numThreads);

	return 0;
}





