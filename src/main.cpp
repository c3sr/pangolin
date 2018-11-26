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
#include "graph/config.hpp"

#include "cutrianglecounter.h"

int main(int argc, char **argv)
{

	Config config;

	std::string adjacencyListPath;
	bool help = false;
	bool debug = false;
	bool verbose = false;

	clara::Parser cli;

	cli = cli | clara::Opt(config.numCPUThreads_, "int")
					["-c"]["--num_cpu"]("How many CPU threads?");
	cli = cli | clara::Opt(config.numGPUs_, "int")
					["-g"]["--num_gpu"]("How many GPUs?");
	cli = cli | clara::Opt(debug)
					["--debug"]("log debug messages");
	cli = cli | clara::Opt(verbose)
					["--verbose"]("log verbose messages");
	cli = cli | clara::Help(help);
	cli = cli | clara::Opt(config.type_, "type gpu|nvgraph")["-t"]["--type"]("Triangle counting method").required();
	cli = cli | clara::Arg(adjacencyListPath, "graph file")("Path to adjacency list").required();

	auto result = cli.parse(clara::Args(argc, argv));
	if (!result)
	{
		LOG(error, "Error in command line: {}", result.errorMessage());
		exit(1);
	}

	if (help)
	{
		std::cout << cli;
		return 0;
	}

	if (debug)
	{
		logger::console->set_level(spdlog::level::debug);
	}
	else if (verbose)
	{
		logger::console->set_level(spdlog::level::trace);
	}

	if (config.type_.empty())
	{
		LOG(critical, "type must be provided");
		std::cout << cli;
		return -1;
	}

	if (adjacencyListPath.empty())
	{
		LOG(critical, "graph file must be provided");
		std::cout << cli;
		return -1;
	}

	if (config.numCPUThreads_ <= 0)
		config.numCPUThreads_ = 1;
	if (config.numCPUThreads_ >= omp_get_max_threads())
		config.numCPUThreads_ = omp_get_max_threads();

	LOG(info, "{} gpus", config.numGPUs_);
	LOG(info, "{} cpus", config.numCPUThreads_);

	TriangleCounter *tc;
	tc = TriangleCounter::CreateTriangleCounter(config);
	tc->read_data(adjacencyListPath);
	const auto numTriangles = tc->count();
	tc->execute(adjacencyListPath.c_str(), config.numCPUThreads_);

	delete tc;
	return 0;
}
