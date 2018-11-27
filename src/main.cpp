#include <iostream>
#include <fmt/format.h>

#include "clara.hpp"
#include "graph/logger.hpp"
#include "graph/triangle_counter.hpp"
#include "graph/config.hpp"

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
	cli = cli | clara::Opt(config.type_, "cpu|cudamemcpy|nvgraph|um|zc")["-m"]["--method"]("method (default = um)").required();
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

	auto start = std::chrono::system_clock::now();
	tc->setup_data();
	double elapsed = (std::chrono::system_clock::now() - start).count()/1e9;
	LOG(debug, "setup_data time {}s", elapsed);

	start = std::chrono::system_clock::now();
	const auto numTriangles = tc->count();
	elapsed = (std::chrono::system_clock::now() - start).count()/1e9;

	fmt::print("{} {} {} {}\n", adjacencyListPath, numTriangles, elapsed, numTriangles/elapsed);

	delete tc;
	return 0;
}
