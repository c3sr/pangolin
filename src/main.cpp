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
	cli = cli | clara::Help(help);
	cli = cli | clara::Opt(debug)
					["--debug"]("print debug messages to stderr");
	cli = cli | clara::Opt(verbose)
					["--verbose"]("print verbose messages to stderr");
	cli = cli | clara::Opt(config.numCPUThreads_, "int")
					["-c"]["--num_cpu"]("number of cpu threads (default = automatic)");
	cli = cli | clara::Opt(config.numGPUs_, "int")
					["-g"]["--num_gpu"]("number of gpus");
	cli = cli | clara::Opt(config.type_, "cpu|gpu|nvgraph")["-m"]["--method"]("method (default = gpu)").required();
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

	if (verbose)
	{
		logger::console->set_level(spdlog::level::trace);
	}
	else if (debug)
	{
		logger::console->set_level(spdlog::level::debug);
	}

	TriangleCounter *tc;
	tc = TriangleCounter::CreateTriangleCounter(config);
	tc->read_data(adjacencyListPath);
	const auto numTriangles = tc->count();
	LOG(info, "{} triangles", numTriangles);
	tc->execute(adjacencyListPath.c_str(), config.numCPUThreads_);

	delete tc;
	return 0;
}
